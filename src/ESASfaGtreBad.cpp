/*
    sfacpp - Estimation of TRE/GTRE SFA models
    Copyright (C) 2025 Edmund Haacke
    Copyright (C) 2025 NHS England

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/**
 * @file ESASfaGtreBad.cpp
 */

#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include "sfa/ESASfaGtreBad.hpp"
#include "sfa/ESASfaCross.hpp"
#include "data/ESADataPanel.hpp"
#include "data/ESADataCross.hpp"
#include "math/esamath.hpp"
#include "math/esandist.hpp"
#include "utils/esautils.hpp"
#include "utils/finitediff.hpp"
#include "optim/esaoptimization.hpp"
#include "optim/optimparams.hpp"
#include "optim/ESAGlobalOptimParams.hpp"
#include "regression/ESARandEff.hpp"
#include "thread_cache/ESASfaGtreBadTC.hpp"
#include "utils/ThreadContext.hpp"
#include "utils/excepts.hpp"

ESASfaGtreBad::ESASfaGtreBad(
    const std::shared_ptr<ESADataBase> dataObjPtr,
    const int s,
    const int nsim,
    const int seed,
    const HaltonSettings hsetting
) : ESASfaTreBase(dataObjPtr, s, nsim, seed, hsetting)
{   

}

ESASfaGtreBad::ESASfaGtreBad(
    const std::shared_ptr<ESADataBase> dataObjPtr,
    const int s,
    const int nsim,
    const int seed,
    const std::shared_ptr<arma::dmat> haltonDrawPtr,
    const HaltonSettings hsetting
) : ESASfaTreBase(dataObjPtr, s, nsim, seed, haltonDrawPtr, hsetting)
{
    
}


/// Objective function to minimize/maximize
double ESASfaGtreBad::operator()(const arma::dcolvec& params, const bool exceptNotFinite) const
{
    // dereference pointer to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    arma::dmat& haltonMat = *haltonDraws;
    // ESASfaModelType mT = dataObj.getModelType();
    auto inner = [this, &params, &haltonMat, &exceptNotFinite](
        const unsigned int idx,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zui0,
        const auto& zvi0
    ){
        // check required terms available
        if (!zuit || !zvit || !zui0 || !zvi0) throw std::invalid_argument("missing one of 'zuit', 'zvit', 'zui0', 'zvi0'");
        arma::dmat dens;
        this->densityHalfNormal(idx, params, y, x, zuit.value(), zvit.value(), zui0.value(), zvi0.value(), haltonMat, dens);
        // log sum exp trick
        // calculate ln of density
        arma::dmat lnDen = arma::log(dens);
        // calculate S_r - sum over t; (1 x nsim)
        arma::dmat Sr = esamath::colSum(lnDen);
        // find max value, Smax
        double Smax = Sr.max();
        // iterate thru r
        arma::dmat Kr = arma::exp(Sr - Smax);
        double Ktotal = arma::accu(Kr);
        double logSum = Smax + std::log(Ktotal);
        double ll1 = logSum - std::log(nsim);
        if (exceptNotFinite) {
            if (!std::isfinite(ll1)) {
                std::string m = "Likelihood not finite for " + std::to_string(idx) + " for GTRE";
                throw esaexcepts::DensityNotFinite(m.c_str());
            }
        }
        return ll1;
    };
    // check whether or not to use threading
    ESAGlobalOptimParams *globalOpts = ESAGlobalOptimParams::GetInstance();
    double llScore = 0.0;
    if (globalOpts->optimThreaded) {
        llScore = dataObj.panelCallableSumThreaded(inner);
    } else {
        llScore = dataObj.panelCallableSum(inner);
    }
    return llScore;
}

/// Calculate both gradient and hessian matrix simulatenously (preferred method)
void ESASfaGtreBad::gradHess(
    const arma::dcolvec& params,
    const bool exceptNotFinite,
    arma::dmat* gradOut,
    arma::dmat* hessOut,
    arma::dmat* jacOut
) const
{
    // dereference pointer to halton matrix
    arma::dmat& haltonMat = *haltonDraws;
    // dereference pointer to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    ESASfaModelType mT = dataObj.getModelType();
    if (mT != ESASfaModelType::GTRE_HNORM_ZUIT_ZUI0) {
        throw std::runtime_error("only support Gtre half normal");
    }
    ESASfaModelDistribution mD = ESAEnums::getDistribution(mT);
    // global optimization options
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    HessianCalcMethod hessMethod = globalOptimParams->hessianMethod;
    bool threaded = globalOptimParams->optimThreaded;
    int accuracy = globalOptimParams->hessianNumApproxAcc;

    arma::dmat jac, h;
    auto inner = [this, &params, &haltonMat, &hessMethod, &accuracy, &mT, &mD, &exceptNotFinite](
        const unsigned int idx,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zui0,
        const auto& zvi0,
        arma::dmat* j1,
        arma::dmat* h1
    ) {
        // always must have zuit, zvit, zui0, zvi0
        if (!zuit || !zvit || !zui0 || !zvi0) {
            throw std::invalid_argument("'zuit', 'zvit', 'zui0', and 'zvi0' must be present");
        }
        // check zmuit for truncated normal distribution
        if (mD == ESASfaModelDistribution::TNORM && !zmuit) {
            throw std::invalid_argument("'zmuit' required for trunc normal distribution");
        }
        this->gradHessInner(
            idx, // index of current firm
            mT, // model type
            params, // column vector of parameters
            y,
            x,
            zmuit,
            zuit.value(),
            zvit.value(),
            zui0.value(),
            zvi0.value(),
            haltonMat, // halton draws
            hessMethod, // hessian calculation method
            accuracy, // num approx hess accuracy
            j1, // ptr out to mtrx for jacobian
            h1 // ptr out for mtrx for hessian
        );
        if (exceptNotFinite && j1) {
            // check that the jacobian is finite
            if (!(*j1).is_finite()) {
                std::string m = "Gradient not finite for " + std::to_string(idx) + " for GTRE";
                throw esaexcepts::GradientNotFinite(m.c_str());
            }
        }
        if (exceptNotFinite && h1) {
            // check that the hessian is finite
            if (!(*h1).is_finite()) {
                std::string m = "Hessian not finite for " + std::to_string(idx) + " for GTRE";
                throw esaexcepts::HessianNotFinite(m.c_str());
            }
        }
    };
    if (threaded){
        dataObj.panelCallableThreaded(inner, &jac, &h, false, true);
    } else {
        dataObj.panelCallable(inner, &jac, &h, false, true);
    }
    double nobs = dataObj.getNids();
    arma::dmat g = esamath::colSum(jac);
    // g = g / nobs;
    // h = h / nobs;
    if (gradOut) *gradOut = g;
    if (hessOut) *hessOut = h;
    if (jacOut) *jacOut = jac;
}

/**
 * @brief Return struct containing sigma, lambda parameters
 * @param par Column vector of parameters
 * @return Instance of ESASigmaParams
 */
ESASigmaParams ESASfaGtreBad::getSigmaParams(const arma::dcolvec& par) const
{
    // dereference point to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    arma::dmat sigmas = dataObj.dataCallable(
        [&par, &dataObj](
            const auto& y,
            const auto& x,
            const auto& zmuit,
            const auto& zuit,
            const auto& zvit,
            const auto& zui0,
            const auto& zvi0
        ) {
            if (!zuit || !zvit || !zvi0 || !zui0) {
                throw std::runtime_error("missing 'zuit', 'zvit', 'zui0', or 'zvi0");
            }
            // TODO: implement support for truncated normal distribtion!
            if (!dataObj.paramZuit(par)) throw std::invalid_argument("missing 'b_zuit'");
            arma::dcolvec b_zuit = dataObj.paramZuit(par).value();
            if (!dataObj.paramZvit(par)) throw std::invalid_argument("missing 'b_zvit'");
            arma::dcolvec b_zvit = dataObj.paramZvit(par).value();
            if (!dataObj.paramZvi0(par)) throw std::invalid_argument("missing 'b_zvi0'");
            arma::dcolvec b_zvi0 = dataObj.paramZvi0(par).value();
            if (!dataObj.paramZui0(par)) throw std::invalid_argument("missing 'b_zui0'");
            arma::dcolvec b_zui0 = dataObj.paramZui0(par).value();
            // allocate 3 column matrix for uit; vit; vi0;
            arma::dmat out(y.n_rows, 4);
            // sigma2uit - variance of the time-varying inefficiency
            arma::dmat sigma2uit = esautils::processSig2Term(b_zuit, zuit.value());
            // sigma2vit - variance of the stochastic noise component
            arma::dmat sigma2vit = esautils::processSig2Term(b_zvit, zvit.value());
            // sigma2vi0 - variance of the firm effect
            arma::dmat sigma2vi0 = esautils::processSig2Term(b_zvi0, zvi0.value());
            // sigma2ui0 - variance of the time-invariant inefficiency
            arma::dmat sigma2ui0 = esautils::processSig2Term(b_zui0, zui0.value());
            // set columns in output matrix
            out.col(0) = sigma2uit;
            out.col(1) = sigma2vit;
            out.col(2) = sigma2vi0;
            out.col(3) = sigma2ui0;
            return out;
        }
    );
    // construct the instance of sigma params
    ESASigmaParams sigparams(
        sigmas.col(0), // sigma2uit
        sigmas.col(1), // sigma2vit
        sigmas.col(2), // sigma2vi0
        sigmas.col(3) //sigma2ui0
    );
    return sigparams;
}



/// Density for the half, truncated normal distribution
template <typename TY, typename TX, typename TZuit, typename TZvit, typename TZui0, typename TZvi0>
void ESASfaGtreBad::densityHalfNormal(
    const unsigned int idx,
    const arma::dcolvec& par, 
    const arma::Base<double, TY>& yIn,
    const arma::Base<double, TX>& xIn,
    const arma::Base<double, TZuit>& zuitIn,
    const arma::Base<double, TZvit>& zvitIn,
    const arma::Base<double, TZui0>& zui0In,
    const arma::Base<double, TZvi0>& zvi0In,
    const arma::dmat& draws,
    arma::dmat& outDens
) const
{
    // dereference pointer to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    // persistent variables on specific thread - initialised once per thread, then reused forever
    ThreadContext* ctx = getContext();
    if (!ctx->gtreDensPanel) {
        ctx->gtreDensPanel = std::make_unique<thread_cache_gtre::WSDensityHalfNormal>();
    }
    thread_cache_gtre::WSDensityHalfNormal& ws = *ctx->gtreDensPanel;
    // draws depend on whether use the same for everyone, or different for each individual
    int drawBase = this->obsUseSameHaltonDraw ? 0 : idx;
    int nIds = dataObj.getNids();
    int ui0DrawBase = this->obsUseSameHaltonDraw ? 1 : nIds + idx;
    // check not going to go out of bounds on the draws
    if (draws.n_rows < ui0DrawBase){
        throw std::invalid_argument("Expecting draws to be at least " + std::to_string(ui0DrawBase) + " rows");
    }
    if (draws.n_cols != nsim){
        throw std::invalid_argument("draws must have the same number of columns as nsim");
    }
    // check s is either -1 or 1
    if (s != -1 && s != 1){
        throw std::invalid_argument("s must be either -1 or 1");
    }
    // extract the coefficients
    arma::dcolvec b_x = dataObj.paramX(par);
    if (!dataObj.paramZuit(par)) throw std::invalid_argument("missing 'zuit'");
    arma::dcolvec b_zuit = dataObj.paramZuit(par).value();
    if (!dataObj.paramZvit(par)) throw std::invalid_argument("missing 'zvit'");
    arma::dcolvec b_zvit = dataObj.paramZvit(par).value();
    if (!dataObj.paramZui0(par)) throw std::invalid_argument("missing 'zui0");
    arma::dcolvec b_zui0 = dataObj.paramZui0(par).value();
    if (!dataObj.paramZvi0(par)) throw std::invalid_argument("missing 'zvi0'");
    arma::dcolvec b_zvi0 = dataObj.paramZvi0(par).value();
    // unwrap armadillo references
    const auto& y = yIn.get_ref();
    const auto& x = xIn.get_ref();
    const auto& zuit = zuitIn.get_ref();
    const auto& zvit = zvitIn.get_ref();
    const auto& zui0 = zui0In.get_ref();
    const auto& zvi0 = zvi0In.get_ref();
    // calculate sizes
    int nT = y.n_rows;
    int maxT = dataObj.getMaxT();
    // resize when necessary
    ws.ensureSize(maxT, nsim);
    // views for xb, sigmas, since their storage matricies can be >nT
    arma::span spanNt(0, nT - 1), spanZero(0, 0);
    arma::subview<double> xb = ws.xb.submat(spanNt, spanZero);
    arma::subview<double> sigma2uit = ws.sigma2uit.submat(spanNt, spanZero);
    arma::subview<double> sigmauit = ws.sigmauit.submat(spanNt, spanZero);
    arma::subview<double> sigma2vit = ws.sigma2vit.submat(spanNt, spanZero);
    arma::subview<double> sigmavit = ws.sigmavit.submat(spanNt, spanZero);
    arma::subview<double> sigma2ui0 = ws.sigma2ui0.submat(spanNt, spanZero);
    arma::subview<double> sigmaui0 = ws.sigmaui0.submat(spanNt, spanZero);
    arma::subview<double> sigma2vi0 = ws.sigma2vi0.submat(spanNt, spanZero);
    arma::subview<double> sigmavi0 = ws.sigmavi0.submat(spanNt, spanZero);
    arma::subview<double> sigma2 = ws.sigma2.submat(spanNt, spanZero);
    arma::subview<double> sigma = ws.sigma.submat(spanNt, spanZero);
    arma::subview<double> lambda = ws.lambda.submat(spanNt, spanZero);
    // calculate xb
    xb = x * b_x;
    // sigma2uit (variance of the time-varying inefficency)
    sigma2uit = esautils::processSig2Term(b_zuit, zuit);
    sigmauit = arma::sqrt(sigma2uit);
    // sigma2vit (variance of the stochastic noise component)
    sigma2vit = esautils::processSig2Term(b_zvit, zvit);
    sigmavit = arma::sqrt(sigma2vit);
    // sigma2ui0 (variance of the time-invariant inefficency)
    sigma2ui0 = esautils::processSig2Term(b_zui0, zui0);
    sigmaui0 = arma::sqrt(sigma2ui0);
    // sigma2vi0 (variance of the firm effect)
    sigma2vi0 = esautils::processSig2Term(b_zvi0, zvi0);
    sigmavi0 = arma::sqrt(sigma2vi0);
    // sigma2
    sigma2 = sigma2uit + sigma2vit;
    // sigma
    sigma = arma::sqrt(sigma2);
    // lambda - sigu/sigv
    lambda = sigmauit / sigmavit;
    // subview for draws for inefficiency and firm effe
    const arma::subview<double> draw_vi0 = draws.submat(arma::span(drawBase, drawBase), arma::span(0, nsim - 1));
    const arma::subview<double> draw_ui0 = draws.submat(arma::span(ui0DrawBase, ui0DrawBase), arma::span(0, nsim - 1));
    // subviews to buffer for halton-draw related
    // nb draw_vi0_norm, draw_ui0_norm deleted from struct
    // arma::subview<double> draw_vi0_norm = ws.draw_vi0_norm.submat(arma::span(0, 0), arma::span(0, nsim - 1));
    // arma::subview<double> draw_ui0_norm = ws.draw_ui0_norm.submat(arma::span(0, 0), arma::span(0, nsim - 1));
    arma::subview<double> draw_vi0_scaled = ws.draw_vi0_scaled.submat(spanNt, arma::span(0, nsim - 1));
    arma::subview<double> draw_ui0_scaled = ws.draw_ui0_scaled.submat(spanNt, arma::span(0, nsim - 1));
    // map both these draws to the normal distribution using the inverse of the normal CDF
    // arma::dmat draw_vi0_norm = esamath::ppf_internal(draw_vi0, 0.0, 1.0);
    // draw_vi0_norm = esandist::ppf(draw_vi0, 0.0, 1.0); // removed 2025.12.29
    // double vi0Mean = arma::mean(arma::vectorise(draw_vi0_norm));
    // draw_vi0_norm -= vi0Mean;
    // for the time-invariant inefficiency, we need to map the draws to the half-normal distribution - take the absolute value
    // arma::dmat draw_ui0_norm = dlib::abs(esamath::ppf_internal(draw_ui0, 0.0, 1.0));
    // draw_ui0_norm = esandist::ppf(draw_ui0, 0.0, 1.0); // removed 2025.12.29
    // double ui0Mean = arma::mean(arma::vectorise(draw_ui0_norm));
    // draw_ui0_norm -= ui0Mean;
    // take absolute value for half normal distribution
    // draw_ui0_norm = arma::abs(draw_ui0_norm); // removed 2025.12.29
    // calculate vi0 by variance, which should be a nT x nsim matrix
    // draw_vi0_scaled = sigmavi0 * draw_vi0_norm; // removed 2025.12.29
    draw_vi0_scaled = sigmavi0 * draw_vi0;
    // calculate ui0 by variance, which should be a nT x nsim matrix
    // draw_ui0_scaled = sigmaui0 * draw_ui0_norm; // removed 2025.12.29
    draw_ui0_scaled = sigmaui0 * arma::abs(draw_ui0);
    // calculate eps
    arma::subview<double> epsr = ws.epsr.submat(spanNt, arma::span(0, nsim - 1));
    epsr = -draw_vi0_scaled;
    // epsr -= (s * draw_ui0_scaled); // previous - this was wrong
    epsr += (s * draw_ui0_scaled);
    epsr.each_col() += y;
    epsr.each_col() -= xb;
    // double* epsrMem = ws.epsr.memptr();
    // const double* yMem = y.colptr(0);
    // const double* xbMem = ws.xb.memptr();
    // for (int t = 0; t < nT; t++) {
    //     double val = xbMem[t] - yMem[t];
    //     // hoping that this gets SIMD optimized over nsim, should be more efficient
    //     for (int r = 0; r < nsim; r++) {
    //         epsrMem[t + r * maxT] -= val;
    //     }
    // }
    // density terms
    arma::subview<double> c1In = ws.c1In.submat(spanNt, arma::span(0, nsim - 1));
    arma::subview<double> c2In = ws.c2In.submat(spanNt, arma::span(0, nsim - 1));
    c1In = epsr;
    c2In = epsr;
    c1In.each_col() /= sigma;
    c2In.each_col() %= lambda;
    c2In *= -this->s;
    c2In.each_col() /= sigma;
    // double* c1Mem = ws.c1In.memptr();
    // double* c2Mem = ws.c2In.memptr();
    // const double* sigMem = ws.sigma.memptr();
    // const double* lamMem = ws.lambda.memptr();
    // for (int t = 0; t < nT; t++) {
    //     double sigVal = sigMem[t];
    //     double lamVal = lamMem[t];
    //     for (int r = 0; r < nsim; r++) {
    //         c1Mem[t + r * maxT] /= sigVal;
    //         // c2
    //         c2Mem[t + r * maxT] *= lamVal;
    //         c2Mem[t + r * maxT] *= -this->s;
    //         c2Mem[t + r * maxT] /= sigVal;
    //     }
    // }
    // view to write into destination memory 
    if (outDens.n_rows < nT || outDens.n_cols < nsim) {
        // only resize if the destination buffer is too small
        // should really be maxT, nsim, but just want to make sure there is enough space
        outDens.set_size(nT, nsim);
    }
    arma::subview<double> outView = outDens.submat(spanNt, arma::span(0, nsim - 1));
    arma::subview<double> cdfView = ws.cdfVal.submat(spanNt, arma::span(0, nsim - 1));
    outView = arma::normpdf(c1In, 0.0, 1.0);
    cdfView = arma::normcdf(c2In, 0.0, 1.0);
    // (2.0 / sigma) * pdf * cdf
    outView %= cdfView;
    outView.each_col() /= sigma;
    // double* outMem = outDens.memptr();
    // for (int t = 0; t < nT; t++) {
    //     double sigVal = sigMem[t];
    //     for (int r = 0; r < nsim; r++) {
    //         outMem[t + r * maxT] /= sigVal;
    //     }
    // }
    outView *= 2.0;
    // arma::dmat eps_r = y_r - xb_r - draw_vi0_scaled - s * draw_ui0_scaled;
    // // calculate c1 and c2
    // arma::dmat c1In = eps_r / sigma_r;
    // arma::dmat c2In = ( - s * ((eps_r % lambda_r)) / sigma_r );
    // // calculate the density
    // arma::dmat ld_r = (2.0 / sigma_r) % arma::normpdf(c1In, 0.0, 1.0) %  arma::normcdf(c2In, 0.0, 1.0);
    // return ld_r;
}


struct WSGradHessInner {
    arma::dmat jac;
    arma::dmat hess;
    arma::dmat gir;
    arma::dmat dens;

    void ensureSize(unsigned int maxT, unsigned int nSim, unsigned int nParams) {

    }
};

template <typename TY, typename TX, typename TZmuit = arma::dmat, typename TZuit, typename TZvit, typename TZui0, typename TZvi0>
void ESASfaGtreBad::gradHessInner(
    const unsigned int idx,
    const ESASfaModelType mT,
    const arma::dcolvec& par,
    const arma::Base<double, TY>& yIn,
    const arma::Base<double, TX>& xIn,
    const std::optional<TZmuit>& zmuitIn,
    const arma::Base<double, TZuit>& zuitIn,
    const arma::Base<double, TZvit>& zvitIn,
    const arma::Base<double, TZui0>& zui0In,
    const arma::Base<double, TZvi0>& zvi0In,
    const arma::dmat& draws,
    const HessianCalcMethod hessMethod,
    const unsigned int accuracy,
    arma::dmat* jacOut,
    arma::dmat* hessOut
) const
{
    if (hessMethod != HessianCalcMethod::ANALYTICAL){
        throw std::runtime_error("not implement gtre for non-analytical");
    }
    // buffer for persistent variables per thread
    ThreadContext* ctx = getContext();
    if (!ctx->gtreGradHessPanel) {
        ctx->gtreGradHessPanel = std::make_unique<thread_cache_gtre::WSGradHessInner>();
    }
    thread_cache_gtre::WSGradHessInner& ws = *ctx->gtreGradHessPanel;

    // unwrap armadillo matricies
    const auto& y = yIn.get_ref();
    const auto& x = xIn.get_ref();
    const auto& zuit = zuitIn.get_ref();
    const auto& zvit = zvitIn.get_ref();
    const auto& zui0 = zui0In.get_ref();
    const auto& zvi0 = zvi0In.get_ref();
    // dereference ptr to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    // number of time periods
    int nT = y.n_rows;
    int maxT = dataObj.getMaxT();
    int ns = this->nsim;
    int nParams = dataObj.nParams();
    // ensure correct dims for persistent vars
    ws.ensureSize(maxT, ns, nParams);
    ws.zeros();
    // view to the relevant part of dens
    arma::dmat gir, dens;
    // calculate density (nT x nsim)
    if (mT == ESASfaModelType::GTRE_HNORM_ZUIT_ZUI0){
        this->densityHalfNormal(idx, par, y, x, zuit, zvit, zui0, zvi0, draws, dens);
    } else {
        throw std::invalid_argument("unsupported gtre method");
    }
    // create weights based on the densities (1 x nsim)
    arma::dmat Qir(1, this->nsim);
    this->weightFromDensForPanel(dens, Qir);
    arma::subview<double> QirView = Qir.submat(arma::span(0, 0), arma::span(0, ns - 1));
    // variables to store jacobian, gradient, and hessian in
    // vector of vector of matricies for hessians over i, t, and r
    // std::vector<std::vector<arma::dmat>> hess_itr(this->nsim, std::vector<arma::dmat>(nT));
    arma::dmat jac(nT, nParams);
    arma::dmat hess(nParams, nParams);
    if (hessMethod == HessianCalcMethod::ANALYTICAL && mT == ESASfaModelType::GTRE_HNORM_ZUIT_ZUI0){
        std::vector<arma::dmat> jacSims;
        this->internalAnalyticJacHess(
            idx,
            mT,
            par,
            y,
            x,
            zmuitIn,
            zuit,
            zvit,
            zvi0,
            draws, // halton draws
            std::make_optional(zui0), // optional type zui0
            QirView,
            &jac,
            &hess
        );
    } else {
        // call method to numerically approximate gradient
        throw std::invalid_argument("not implemented for non-analytical");
    }
    if (jacOut) *jacOut = jac;
    if (!hessOut) return;
    if (hessOut) *hessOut = hess;
}

// ===

template void ESASfaGtreBad::densityHalfNormal<
    arma::dmat, // TY
    arma::dmat, // TX
    arma::dmat, // TZuit,
    arma::dmat, // TZvit,
    arma::dmat, // TZui0,
    arma::dmat  // TZvi0
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::dmat&,
    arma::dmat&
) const;
template void ESASfaGtreBad::densityHalfNormal<
    arma::subview<double>, // TY
    arma::subview<double>, // TX
    arma::subview<double>, // TZuit
    arma::subview<double>, // TZvit
    arma::subview<double>, // TZui0
    arma::subview<double> // TZvi0
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    arma::dmat&
) const;

template void ESASfaGtreBad::gradHessInner<
    arma::dmat, // TY
    arma::dmat, // TX
    arma::dmat, // TZmuit
    arma::dmat, // TZuit
    arma::dmat, // TZvit
    arma::dmat, // TZui0 NOTE THIS IS DIFFERENCE ORDER TO TRE
    arma::dmat // TZvi0
>(
    const unsigned int,
    const ESASfaModelType,
    const arma::dcolvec&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const std::optional<arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::dmat&,
    const HessianCalcMethod,
    const unsigned int,
    arma::dmat*,
    arma::dmat*
) const;
template void ESASfaGtreBad::gradHessInner<
    arma::subview<double>, // TY
    arma::subview<double>, // TX
    arma::subview<double>, // TZmuit
    arma::subview<double>, // TZuit
    arma::subview<double>, // TZvit
    arma::subview<double>, // TZui0
    arma::subview<double> //TZvi0
>(
    const unsigned int,
    const ESASfaModelType,
    const arma::dcolvec&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const std::optional<arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    const HessianCalcMethod,
    const unsigned int,
    arma::dmat*,
    arma::dmat*
) const;