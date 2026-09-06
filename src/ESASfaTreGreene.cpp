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
 * @file ESASfaTreGreene.cpp
 * @brief Class for estimating the True Random Effects model using method from Greene (2005)
 * @date 2025-09-02
 * @author Edmund Haacke
 */


#include <iostream>
#include <math.h>
#include <vector>
#include <stdexcept>
#include <typeinfo>
#include <memory>
#include <string>
#include <algorithm>
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "utils/excepts.hpp"
#include "utils/esautils.hpp"
#include "utils/finitediff.hpp"
#include "math/esamath.hpp"
#include "math/esandist.hpp"
#include "data/ESADataCross.hpp"
#include "data/ESADataBase.hpp"
#include "data/ESADataPanel.hpp"
#include "sfa/ESASfaCross.hpp"
#include "sfa/ESASfaTreGreene.hpp"
#include "regression/ESARandEff.hpp"
#include "thread_cache/ESASfaTreGreeneTC.hpp"
#include "utils/ThreadContext.hpp"

// ---- optimization ----
#include "optim/esaoptimization.hpp"
#include "optim/ESAOptimResult.hpp"
#include "optim/optimparams.hpp"
#include "optim/ESAGlobalOptimParams.hpp"

// *******************************************************
// ****************** Public Methods *********************
// *******************************************************
/// Constructor
ESASfaTreGreene::ESASfaTreGreene(
    const std::shared_ptr<ESADataBase> dataObjPtr,
    const int s,
    const int nsim,
    const int seed,
    const HaltonSettings hsetting
) : ESASfaTreBase(dataObjPtr, s, nsim, seed, hsetting)
{

}

ESASfaTreGreene::ESASfaTreGreene(
    const std::shared_ptr<ESADataBase> dataObjPtr,
    const int s,
    const int nsim,
    const int seed,
    const std::shared_ptr<arma::dmat> haltonDrawPtr,
    const HaltonSettings hsetting
) : ESASfaTreBase(dataObjPtr, s, nsim, seed, haltonDrawPtr, hsetting)
{
    
}

template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0, typename TZui0>
void ESASfaTreGreene::operatorInner(
    const arma::dcolvec& params,
    const ESASfaModelType mT,
    const arma::dmat& haltonMat,
    const unsigned int idx,
    const arma::Base<double, TY>& yIn,
    const arma::Base<double, TX>& xIn,
    const std::optional<TZmuit>& zmuit,
    const arma::Base<double, TZuit>& zuitIn,
    const arma::Base<double, TZvit>& zvitIn,
    const arma::Base<double, TZvi0>& zvi0In,
    const bool logLike,
    arma::dmat* out
) const
{
    // return if out is nullptr
    if (!out) return;
    // dereference ptr to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    // persistent variables on a specific thread - initialized once then reused forevers
    ThreadContext* ctx = getContext();
    // check if the pointer for treDensHNorm is empty
    if (!ctx->treOpInner) {
        ctx->treOpInner = std::make_unique<thread_cache::WSOperatorInner>();
    }
    thread_cache::WSOperatorInner& ws = *ctx->treOpInner;
    // auto wsPtr = std::make_unique<thread_cache::WSOperatorInner>();
    // thread_cache::WSOperatorInner& ws = *wsPtr;
    
    // unwrap armadillo references
    const auto& y = yIn.get_ref();
    const auto& x = xIn.get_ref();
    const auto& zuit = zuitIn.get_ref();
    const auto& zvit = zvitIn.get_ref();
    const auto& zvi0 = zvi0In.get_ref();
    int nT = y.n_rows;
    int ns = this->nsim;
    int maxT = dataObj.getMaxT();
    // allocate sizes if needed
    ws.ensureSize(maxT, ns);
    ws.zeros();
    if (mT == ESASfaModelType::TRE_HNORM_ZUIT) {
        // half-normal distribution
        // checks
        this->panelDensityHalfNormal(idx, params, y, x, zuit, zvit, zvi0, haltonMat, ws.ldr);
    } else if (mT == ESASfaModelType::TRE_TNORM_ZUIT) {
        // truncated-normal distribution
        // checks
        if (!zmuit) throw std::invalid_argument("Missing 'zmuit' for truncated normal distribution");
        this->panelDensityTruncNormal(idx, params, y, x, zmuit.value(), zuit, zvit, zvi0, haltonMat, ws.ldr);
    } else {
        throw std::runtime_error("Unsupported model type");
    }
    // view on the actual subset to process (nT x nsim)
    const arma::subview<double> ldrView = ws.ldr.submat(arma::span(0, nT - 1), arma::span(0, ns - 1));
    if (logLike) {
        arma::subview<double> lnDenView = ws.lnDen.submat(arma::span(0, nT - 1), arma::span(0, ns - 1));
        // use log-sum-exp trick
        // calculate ln of density
        lnDenView = arma::log(ldrView);
        // calculate S_r - sum over t; (1 x nsim)
        arma::subview<double> srView = ws.Sr.submat(arma::span(0, 0), arma::span(0, ns - 1));
        srView = esamath::colSum(lnDenView);
        // find max value, Smax
        double Smax = srView.max();
        // iterate thru r
        arma::subview<double> krView = ws.Kr.submat(arma::span(0, 0), arma::span(0, ns - 1));
        krView = arma::exp(srView - Smax);
        double Ktotal = arma::accu(krView);
        double logSum = Smax + std::log(Ktotal);
        double ll1 = logSum - std::log(nsim);
        if (out->n_rows != 1 || out->n_cols != 1){
            out->set_size(1, 1);
        }
        // arma::dmat ret(1, 1);
        // ret(0, 0) = ll1;
        // *out = ret;
        // fill buffer with single loglikelihood score for the panel
        out->at(0, 0) = ll1;
    } else {
        // ensure out is appropriately sized
        if (out->n_rows != nT || out->n_cols != nsim) {
            out->set_size(nT, nsim);
        }
        // fill buffer with log density
        *out = ldrView;
    }
}

/// Objective function to maximise/minimize
double ESASfaTreGreene::operator()(const arma::dcolvec& params, const bool exceptNotFinite) const
{
    // dereference pointer to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    arma::dmat& haltonMat = *haltonDraws;
    ESASfaModelType mT = dataObj.getModelType();
    // call appropriate method
    auto inner = [this, &params, &haltonMat, &mT, &exceptNotFinite](
        const unsigned int idx,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zui0,
        const auto& zvi0
    ){
        // persistent buffer for the thread - this is (1, 1) in dimension
        arma::dmat llOut(1, 1);
        if (!zuit || !zvit || !zvi0) throw std::invalid_argument("missing 'zuit' or 'zvit' or 'zvi0'");
        this->operatorInner(params, mT, haltonMat, idx, y, x, zmuit, zuit.value(), zvit.value(), zvi0.value(), true, &llOut);
        if (exceptNotFinite && !llOut.is_finite()) {
            std::string m = "Density it not finite for " + std::to_string(idx) + " in TRE";
            throw esaexcepts::DensityNotFinite(m.c_str());
        }
        return arma::accu(llOut);
    };

    ESAGlobalOptimParams *globalOpts = ESAGlobalOptimParams::GetInstance();
    double llScore = 0.0;
    if (globalOpts->optimThreaded) {
        llScore = dataObj.panelCallableSumThreaded(inner);
    } else {
        llScore = dataObj.panelCallableSum(inner);
    }

    return llScore;
}

double ESASfaTreGreene::operator()(const arma::dcolvec& params, const arma::Col<int>& subsetIdents) const
{
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    arma::dmat& haltonMat = *haltonDraws;
    ESASfaModelType mT = dataObj.getModelType();
    // call appropriate method
    double llScore = dataObj.panelCallableSum(
        subsetIdents,
        [this, &params, &haltonMat, &mT](
            const unsigned int idx,
            const auto& y,
            const auto& x,
            const auto& zmuit,
            const auto& zuit,
            const auto& zvit,
            const auto& zui0,
            const auto& zvi0
        ){
            // persistent buffer for the thread - (1, 1) in dimension
            arma::dmat llOut(1, 1);
            operatorInner(params, mT, haltonMat, idx, y, x, zmuit, zuit.value(), zvit.value(), zvi0.value(), true, &llOut);
            return arma::accu(llOut);
        }
    );
    return llScore;
}


ESASigmaParams ESASfaTreGreene::getSigmaParams(const arma::dcolvec& par) const
{
    // dereference point to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    arma::dmat sigmas = dataObj.dataCallable(
        [par, &dataObj](
            const auto& y,
            const auto& x,
            const auto& zmuit,
            const auto& zuit,
            const auto& zvit,
            const auto& zui0,
            const auto& zvi0
        ) {
            if (!zuit || !zvit || !zvi0) {
                throw std::runtime_error("missing 'zuit', 'zvit', or 'zvi0");
            }
            // views for coefficients
            // TODO: implement support for truncated normal distribtion!
            if (!dataObj.paramZuit(par)) throw std::invalid_argument("missing 'b_zuit'");
            arma::dcolvec b_zuit = dataObj.paramZuit(par).value();
            if (!dataObj.paramZvit(par)) throw std::invalid_argument("missing 'b_zvit'");
            arma::dcolvec b_zvit = dataObj.paramZvit(par).value();
            if (!dataObj.paramZvi0(par)) throw std::invalid_argument("missing 'b_zvi0'");
            arma::dcolvec b_zvi0 = dataObj.paramZvi0(par).value();
            // allocate 3 column matrix for uit; vit; vi0;
            arma::dmat out(y.n_rows, 3);
            // sigma2uit - variance of the time-varying inefficiency
            arma::dmat sigma2uit = esautils::processSig2Term(b_zuit, zuit.value());
            // sigma2vit - variance of the stochastic noise component
            arma::dmat sigma2vit = esautils::processSig2Term(b_zvit, zvit.value());
            // sigma2vi0 - variance of the firm effect
            arma::dmat sigma2vi0 = esautils::processSig2Term(b_zvi0, zvi0.value());
            // set columns in output matrix
            out.col(0) = sigma2uit;
            out.col(1) = sigma2vit;
            out.col(2) = sigma2vi0;
            return out;
        }
    );
    // construct the instance of sigma params
    ESASigmaParams sigparams(
        sigmas.col(0), // sigma2uit
        sigmas.col(1), // sigma2vit
        sigmas.col(2), // sigma2vi0
        std::nullopt //sigma2ui0
    );
    return sigparams;
}

template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0, typename TZui0>
void ESASfaTreGreene::gradientInner(
    const arma::dcolvec& params,
    const ESASfaModelType mT,
    const arma::dmat& haltonMat,
    const bool isAnalytical,
    const unsigned int idx,
    const arma::Base<double, TY>& yIn,
    const arma::Base<double, TX>& xIn,
    const std::optional<TZmuit>& zmuit,
    const arma::Base<double, TZuit>& zuitIn,
    const arma::Base<double, TZvit>& zvitIn,
    const arma::Base<double, TZvi0>& zvi0In,
    arma::dmat* outGrad,
    arma::dmat* outGir
) const
{
    // unwrap arma Base references
    // const auto& y = yIn.get_ref();
    // const auto& x = xIn.get_ref();
    // const auto& zuit = zuitIn.get_ref();
    // const auto& zvit = zvitIn.get_ref();
    // const auto& zvi0 = zvi0In.get_ref();
    // if (mT == ESASfaModelType::TRE_HNORM_ZUIT) {
    //     // half-normal distribution
    //     // whether analytical or numerically approximated
    //     if (isAnalytical){
    //         this->panelGradHalfNormAnalytical(idx, params, y, x, zuit, zvit, zvi0, haltonMat, outGrad, outGir);
    //     } else {
    //         this->panelGradHalfNormalNumApprox(idx, params, y, x, zuit, zvit, zvi0, haltonMat, outGrad, outGir);
    //     }
    // } else if (mT == ESASfaModelType::TRE_TNORM_ZUIT) {
    //     // truncated-normal distribution
    //     // only have numerically approximated...
    //     if (!zmuit) throw std::invalid_argument("missing 'zmuit' for truncated normal, in gradient");
    //     // only have numerically approximated
    //     this->panelGradTruncNormalNumApprox(idx, params, y, x, zmuit.value(), zuit, zvit, zvi0, haltonMat, outGrad, outGir);
    // } else {
    //     throw std::runtime_error("invalid");
    // }
}

/// Calculate both gradient and hessian matrix simulatenously
void ESASfaTreGreene::gradHess(
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
    // get the model type, check validity
    ESASfaModelType mT = dataObj.getModelType();
    if ((mT != ESASfaModelType::TRE_HNORM_ZUIT) && (mT != ESASfaModelType::TRE_TNORM_ZUIT)) {
        throw std::runtime_error("Model type not recognised '" + ESAEnums::strForModelType(mT) + "'");
    }
    // global optimization options
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    HessianCalcMethod hessMethod = globalOptimParams->hessianMethod;
    bool threaded = globalOptimParams->optimThreaded;
    int accuracy = globalOptimParams->hessianNumApproxAcc;
    // matricies to store outputs in
    arma::dmat jac, h;
    auto innerFn = [this, &params, &haltonMat, &hessMethod, &accuracy, &mT, &exceptNotFinite](
        const unsigned int idx,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zui0,
        const auto& zvi0,
        arma::dmat* g1,
        arma::dmat* h1
    ) {
        if (!zuit || !zvit || !zvi0) throw std::invalid_argument("missing one of 'zuit', 'zvit', or 'zvi0'");
        this->gradHessPanel(idx, mT, params, y, x, zmuit, zuit.value(), zvit.value(), zvi0.value(), haltonMat, hessMethod, accuracy, g1, h1);
        // check outputs if requested
        if (exceptNotFinite && (g1)) {
            if (!(*g1).is_finite()) {
                std::string m = "Gradient not finite for " + std::to_string(idx) + " in TRE";
                throw esaexcepts::GradientNotFinite(m.c_str());
            }
        }
        if (exceptNotFinite && (h1)) {
            if (!(*h1).is_finite()) {
                std::string m = "Hessian not finite for " + std::to_string(idx) + " in TRE";
                throw esaexcepts::HessianNotFinite(m.c_str());
            }
        }
    };
    if (threaded) {
        dataObj.panelCallableThreaded(innerFn, &jac, &h, false, true);
    } else {
        dataObj.panelCallable(innerFn, &jac, &h, false, true);
    }
    // dataObj.panelCallable(innerFn, &jac, &h, false, true);
    // sum up over rows
    arma::dmat g = esamath::colSum(jac);
    // divide thru the number of identifiers (which are firms)
    double nids = dataObj.getNids();
    // g = g / nids;
    // also divide the hessian by the same
    // h = h / nids;
    if (gradOut) *gradOut = g;
    if (hessOut) *hessOut = h;
    if (jacOut) *jacOut = jac;
}

/// calculate gradient & hessian matrix simulatenously, for a subset of identifiers
void ESASfaTreGreene::gradHess(
    const arma::dcolvec& params,
    const arma::Col<int>& subsetIdents,
    const double step,
    const bool analyticalGrad,
    const HessianCalcMethod hessMethod,
    const unsigned int accuracy,
    const bool threaded,
    arma::dmat* gradOut,
    arma::dmat* hessOut,
    arma::dmat* jacOut
) const
{
    throw std::invalid_argument("not implemented");
}

// *******************************************************
// ****************** Private Methods ********************
// *******************************************************

/// Density for half-normal distribution
template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0, typename TZui0>
void ESASfaTreGreene::panelDensityHalfNormal(
    const unsigned int idx,
    const arma::dcolvec& par,
    const arma::Base<double, TY>& yIn,
    const arma::Base<double, TX>& xIn,
    const arma::Base<double, TZuit>& zuitIn,
    const arma::Base<double, TZvit>& zvitIn,
    const arma::Base<double, TZvi0>& zvi0In,
    const arma::dmat& draws,
    arma::dmat& outDens
) const
{
    // dereference ptr to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    // persistent variables on a specific thread - initialized once, then reused forever
    // get the thread context
    ThreadContext* ctx = getContext();
    // check if the pointer for treDensHNorm is empty
    if (!ctx->treDensHNorm) {
        ctx->treDensHNorm = std::make_unique<thread_cache::WSPanelDensityHalfNormal>();
    }
    thread_cache::WSPanelDensityHalfNormal& ws = *ctx->treDensHNorm;

    arma::dcolvec b_x = dataObj.paramX(par);
    if (!dataObj.paramZuit(par)) throw std::invalid_argument("missing 'b_zuit'");
    arma::dcolvec b_zuit = dataObj.paramZuit(par).value();
    if (!dataObj.paramZvit(par)) throw std::invalid_argument("missing 'b_zvit'");
    arma::dcolvec b_zvit = dataObj.paramZvit(par).value();
    if (!dataObj.paramZvi0(par)) throw std::invalid_argument("missing 'b_zvi0'");
    arma::dcolvec b_zvi0 = dataObj.paramZvi0(par).value();
    // some numbers
    int nX = b_x.n_rows, nZuit = b_zuit.n_rows, nZvit = b_zvit.n_rows, nZvi0 = b_zvi0.n_rows; // nZui0 = 0;
    int nT = yIn.get_ref().n_rows;
    int ns = this->nsim;
    int maxT = dataObj.getMaxT();
    int requiredRows = (nT > maxT) ? nT : maxT; 
    // ensure buffer matricies are appropriately sized
    ws.ensureSize(requiredRows, ns, nX, nZuit, nZvit, nZvi0);
    // subviews to aligned buffers
    arma::subview<double> y = ws.y.rows(arma::span(0, nT - 1));
    // protect incase one of the variables has count 0, so -1 would lead to underflow
    arma::subview<double> x = ws.x.submat(arma::span(0, nT - 1), arma::span(0, std::max(nX - 1, 0)));
    arma::subview<double> zuit = ws.zuit.submat(arma::span(0, nT - 1), arma::span(0, std::max(nZuit - 1, 0)));
    arma::subview<double> zvit = ws.zvit.submat(arma::span(0, nT - 1), arma::span(0, std::max(nZvit - 1, 0)));
    arma::subview<double> zvi0 = ws.zvi0.submat(arma::span(0, nT - 1), arma::span(0, std::max(nZvi0 - 1, 0)));
    // copy into buffer (which is aligned)
    y = yIn.get_ref();
    x = xIn.get_ref();
    zuit = zuitIn.get_ref();
    zvit = zvitIn.get_ref();
    zvi0 = zvi0In.get_ref();
    // check that draws is appropriate length
    int drawIdxPos = this->obsUseSameHaltonDraw ? 0 : idx;
    if (draws.n_rows < drawIdxPos) throw std::invalid_argument("'draws' must be of length 'ident' at minimum");
    if (draws.n_cols < this->nsim) throw std::invalid_argument("'draws' must be of width 'nsim' at minimum");
    // incase maxT was miscalculatedv
    ws.zeros();
    // views for sigmas, since their storage matricies can be > nT
    arma::subview<double> xbView = ws.a1.submat(arma::span(0, nT - 1), arma::span(0, 0));
    arma::subview<double> sigma2uitView = ws.a2.submat(arma::span(0, nT - 1), arma::span(0, 0));
    arma::subview<double> sigmauitView = ws.a3.submat(arma::span(0, nT - 1), arma::span(0, 0));
    arma::subview<double> sigma2vitView = ws.a4.submat(arma::span(0, nT - 1), arma::span(0, 0));
    arma::subview<double> sigmavitView = ws.a5.submat(arma::span(0, nT - 1), arma::span(0, 0));
    arma::subview<double> sigma2vi0View = ws.a6.submat(arma::span(0, nT - 1), arma::span(0, 0));
    arma::subview<double> sigmavi0View = ws.a7.submat(arma::span(0, nT - 1), arma::span(0, 0));
    arma::subview<double> sigma2View = ws.a8.submat(arma::span(0, nT - 1), arma::span(0, 0));
    arma::subview<double> sigmaView = ws.a9.submat(arma::span(0, nT - 1), arma::span(0, 0));
    arma::subview<double> lambdaView = ws.a10.submat(arma::span(0, nT - 1), arma::span(0, 0));
    // ESALogger::logger()->info("DEBUG INFO:");
    // ESALogger::logger()->info("  - nT: {}", nT);
    // ESALogger::logger()->info("  - atlas size: {}x{}", ws.atlas.n_rows, ws.atlas.n_cols);
    // ESALogger::logger()->info("  - atlas mem_ptr: {}", (void*)ws.atlas.memptr());
    // ESALogger::logger()->info("  - xbView n_rows: {}", xbView.n_rows);
    // Note: accessing .memptr() on a subview forces evaluation or returns the parent pointer.
    // Use .colptr(0) to see where column 0 starts.
    // ESALogger::logger()->info("  - xbView colptr(0): {}", (void*)xbView.colptr(0));
    // calculate xb
    xbView = x * b_x;
    // sigma2uit - variance of the time-varying inefficiency
    sigma2uitView = esautils::processSig2Term(b_zuit, zuit);
    sigmauitView = arma::sqrt(sigma2uitView);
    // sigma2vit - variance of the stochastic noise component
    sigma2vitView = esautils::processSig2Term(b_zvit, zvit);
    sigmavitView = arma::sqrt(sigma2vitView);
    // sigma2vi0 - variance of the firm effect
    sigma2vi0View = esautils::processSig2Term(b_zvi0, zvi0);
    sigmavi0View = arma::sqrt(sigma2vi0View);
    // sigma2
    sigma2View = sigma2uitView + sigma2vitView;
    sigmaView = arma::sqrt(sigma2View);
    // lambda - sigu / sigv
    lambdaView = sigmauitView / sigmavitView;
    // subviews on the active part of memory (since allocated to max nT in the data)
    arma::subview<double> aiView = ws.a_i.submat(arma::span(0, nT - 1), arma::span(0, ns - 1));
    arma::subview<double> epsrView = ws.eps_r.submat(arma::span(0, nT - 1), arma::span(0, ns - 1));
    arma::subview<double> c1View = ws.c1In.submat(arma::span(0, nT - 1), arma::span(0, ns - 1));
    arma::subview<double> c2View = ws.c2In.submat(arma::span(0, nT - 1), arma::span(0, ns - 1));
    // map halton draws
    const arma::subview<double> drawView = draws.submat(arma::span(drawIdxPos, drawIdxPos), arma::span(0, ns - 1));
    // nb w_i deleted from struct
    // arma::subview<double> wiView = ws.w_i.submat(arma::span(0, 0), arma::span(0, ns - 1));
    // (1 x nsim)
    // wiView = esandist::ppf(drawView, 0.0, 1.0);
    // calculate a_i - nT x nsim matrix
    // aiView = sigmavi0View * wiView;
    aiView = sigmavi0View * drawView;
    // calculate eps_i = y_i - a_i - xb_r
    epsrView = -aiView;
    // y, xb are nT x 1 so apply to each column (avoid allocating repeated mtx as before)
    epsrView.each_col() += y;
    epsrView.each_col() -= xbView;
    // double* epsrMem = ws.eps_r.memptr();
    // const double* yMem = ws.y.memptr();
    // const double* xbMem = ws.a1.memptr();
    // for (int t = 0; t < nT; t++) {
    //     double val = xbMem[t] - yMem[t];
    //     // hopefully SIMD optimized
    //     for (int r = 0; r < nsim; r++) {
    //         epsrMem[t + r * maxT] -= val;
    //     }
    // }
    // density terms
    c1View = epsrView;
    // c2In = (- s * (eps_r % lambda_r)) / sigma_r;
    c2View = epsrView;
    // double* c1Mem = ws.c1In.memptr();
    // double* c2Mem = ws.c2In.memptr();
    // const double* sigmaMem = ws.a9.memptr();
    // const double* lambdaMem = ws.a10.memptr();
    // for (int t = 0; t < nT; t++) {
    //     double sigVal = sigmaMem[t];
    //     double lamVal = lambdaMem[t];
    //     for (int r = 0; r < nsim; r++) {
    //         c1Mem[t + r * maxT] /= sigVal;
    //         // for c2
    //         c2Mem[t + r * maxT] *= lamVal;
    //         c2Mem[t + r * maxT] *= -this->s;
    //         c2Mem[t + r * maxT] /= sigVal;
    //     }
    // }
    c1View.each_col() /= sigmaView;
    c2View.each_col() %= lambdaView;
    c2View *= -this->s;
    c2View.each_col() /= sigmaView;
    // view to write into destination memory the output and, cdf
    arma::subview<double> outView = outDens.submat(arma::span(0, nT - 1), arma::span(0, nsim - 1));
    arma::subview<double> cdfView = ws.cdfVal.submat(arma::span(0, nT - 1), arma::span(0, nsim - 1));
    // outView = arma::normpdf(c1View, 0.0, 1.0);
    outView = esandist::normpdf_boost(c1View, 0.0, 1.0);
    // cdfView = arma::normcdf(c2View, 0.0, 1.0);
    cdfView = esandist::normcdf_boost(c2View, 0.0, 1.0);
    // (2.0 / sigma) * pdf * cdf
    outView %= cdfView;
    outView.each_col() /= sigmaView;
    // double* outMem = outDens.memptr();
    // for (int t = 0; t < nT; t++) {
    //     double sigVal = sigmaMem[t];
    //     for (int r = 0; r < nsim; r++) {
    //         outMem[t + r * maxT] /= sigVal;
    //     }
    // }
    outView *= 2.0;
}

/// Density for the truncated normal distribution
template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0, typename TZui0>
void ESASfaTreGreene::panelDensityTruncNormal(
    const unsigned int idx,
    const arma::dcolvec& par,
    const arma::Base<double, TY>& yIn,
    const arma::Base<double, TX>& xIn,
    const arma::Base<double, TZmuit>& zmuitIn,
    const arma::Base<double, TZuit>& zuitIn,
    const arma::Base<double, TZvit>& zvitIn,
    const arma::Base<double, TZvi0>& zvi0In,
    const arma::dmat& draws,
    arma::dmat& outDens
) const
{
    // unwrap arma base references
    const auto& y = yIn.get_ref();
    const auto& x = xIn.get_ref();
    const auto& zmuit = zmuitIn.get_ref();
    const auto& zuit = zuitIn.get_ref();
    const auto& zvit = zvitIn.get_ref();
    const auto& zvi0 = zvi0In.get_ref();
    // dereference pointer to the data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    // check that draws is appropriate length
    int drawIdxPos = this->obsUseSameHaltonDraw ? 0 : idx;
    // if (draws.n_rows < drawIdxPos) throw std::invalid_argument("'draws' must be of length 'ident' at minimum");
    if (draws.n_rows <= drawIdxPos) { 
        throw std::invalid_argument(
            "draws.n_rows (" + std::to_string(draws.n_rows) + 
            ") must be > drawIdxPos (" + std::to_string(drawIdxPos) + ")"
        );
    }
    if (draws.n_cols < this->nsim) throw std::invalid_argument("'draws' must be of width 'nsim' at minimum");
    // extract coefficients
    // int nT = y.n_rows;
    // extract coefficients    
    arma::dcolvec b_x = dataObj.paramX(par);
    if (!dataObj.paramZuit(par)) throw std::invalid_argument("missing 'zuit'");
    arma::dcolvec b_zuit = dataObj.paramZuit(par).value();
    if (!dataObj.paramZvit(par)) throw std::invalid_argument("missing 'zvit'");
    arma::dcolvec b_zvit = dataObj.paramZvit(par).value();
    if (!dataObj.paramZvi0(par)) throw std::invalid_argument("missing 'zvi0'");
    arma::dcolvec b_zvi0 = dataObj.paramZvi0(par).value();
    if (!dataObj.paramZmuit(par)) throw std::invalid_argument("missing 'zmuit'");
    arma::dcolvec b_zmuit = dataObj.paramZmuit(par).value();
    // calculate xb
    arma::dmat xb = x * b_x;
    // calculate mu
    arma::dmat mu = zmuit * b_zmuit;
    // sigma2uit - variance of the time-varying inefficiency
    arma::dmat sigma2uit = esautils::processSig2Term(b_zuit, zuit);
    arma::dmat sigmauit = arma::sqrt(sigma2uit);
    // sigma2vit - variance of the stocahstic noise component
    arma::dmat sigma2vit = esautils::processSig2Term(b_zvit, zvit);
    arma::dmat sigmavit = arma::sqrt(sigma2vit);
    // sigma2vi0 - variance of the firm effect
    arma::dmat sigma2vi0 = esautils::processSig2Term(b_zvi0, zvi0);
    // arma::dmat sigma2vi0 = arma::dmat(nT, 1, arma::fill::ones) * b_zvi0;
    // sigma2
    arma::dmat sigma2 = sigma2uit + sigma2vit;
    // sig2star = (sig2vit * sig2uit) / sig2
    arma::dmat sigma2star = (sigma2vit % sigma2uit) / sigma2;
    arma::dmat sigmastar = arma::sqrt(sigma2star);
    // create number of repetitions to number of simulations
    arma::dmat y_r = esautils::repeatColVecAsCols<double>(y, nsim);
    arma::dmat mu_r = esautils::repeatColVecAsCols<double>(mu, nsim);
    arma::dmat xb_r = esautils::repeatColVecAsCols<double>(xb, nsim);
    arma::dmat sigmauit_r = esautils::repeatColVecAsCols<double>(sigmauit, nsim);
    arma::dmat sigma2vit_r = esautils::repeatColVecAsCols<double>(sigma2vit, nsim);
    arma::dmat sigma2uit_r = esautils::repeatColVecAsCols<double>(sigma2uit, nsim);
    arma::dmat sigma2_r = esautils::repeatColVecAsCols<double>(sigma2, nsim);
    arma::dmat sigmastar_r = esautils::repeatColVecAsCols<double>(sigmastar, nsim);
    arma::dmat sigma_r = arma::sqrt(sigma2vit_r + sigma2uit_r);
    // extract the halton draw (already mapped to N(0,1) during initialization via ppf)
    arma::dmat draw = draws.submat(arma::span(drawIdxPos, drawIdxPos), arma::span(0, nsim - 1));
    // calculate a_i, which should be nT x nsim matrix
    arma::dmat a_i = arma::sqrt(sigma2vi0) * draw;
    // calculate eps_i = y_i - a_i - xb_r
    arma::dmat eps_r = y_r - a_i - xb_r;
    // denominator of the density
    // calculate cdf(mu / sigu)
    arma::dmat cdf_mu_div_sigma = arma::normcdf((mu_r / sigmauit_r), 0.0, 1.0);
    // calculate mustar: mu* = (mu*sig2v - s*eps*sig2u) / sig2
    arma::dmat mu_star = ((mu_r % sigma2vit_r) - (this->s * eps_r % sigma2uit_r)) / sigma2_r;
    // calculate cdf( mustar / sigstar )
    arma::dmat cdf_mustar_div_sigstar = arma::normcdf((mu_star / sigmastar_r), 0.0, 1.0);
    // calculate the denominator
    arma::dmat denom = sigma_r % (cdf_mu_div_sigma / cdf_mustar_div_sigstar);
    // numerator of the density
    // pdf ((eps + s*mu) / sigma)  [A = (eps + s*mu) / sigma]
    arma::dmat numer = arma::normpdf(((eps_r + this->s * mu_r) / sigma_r), 0.0, 1.0);
    // calculate density
    arma::dmat den = numer / denom;
    // return den;
    outDens = den;
}

/// Numerically approximated gradient for half-normal distribution
template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0, typename TZui0>
void ESASfaTreGreene::panelGradHalfNormalNumApprox(
    const unsigned int idx,
    const arma::dcolvec& par,
    const arma::Base<double, TY>& yIn,
    const arma::Base<double, TX>& xIn,
    const arma::Base<double, TZuit>& zuitIn,
    const arma::Base<double, TZvit>& zvitIn,
    const arma::Base<double, TZvi0>& zvi0In,
    const arma::dmat& draws,
    arma::dmat* outGrad,
    arma::dmat* outGir
) const
{
    // unwrap armadillo base references
    const auto& y = yIn.get_ref();
    const auto& x = xIn.get_ref();
    const auto& zuit = zuitIn.get_ref();
    const auto& zvit = zvitIn.get_ref();
    const auto& zvi0 = zvi0In.get_ref();
    // for each individual, calculate Q_ir(theta) * g_ir(theta)
    // Q_ir(theta) = P_ir(theta) / sum(P_ir(theta)) (e.g., the weighted average of the conditional joint density for nT obs for each individual
    // (given the random individual specific effect))
    // P_ir(theta) = prod(P_irt(theta)) = prod(f(y_it) | w_ir) for t = 1, ..., nT
    // g_ir(theta) = sum_t( d ln(P_itr(theta)) / d theta ) = sum_t( d ln(f(y_it) | w_ir) / d theta )
    // start off with Q_ir(theta) 
    // calculate P_itr(theta) for each individual
    arma::dmat dens;
    panelDensityHalfNormal(idx, par, y, x, zuit, zvit, zvi0, draws, dens);
    // calculate P_it - take product of the densities
    arma::dmat prodDens = esamath::colProd<double>(dens);
    // calculate the sum of P_it
    double sumProdDens = arma::accu(prodDens);
    // calculate Q_ir(theta) = P_ir(theta) / sum(P_ir(theta))
    arma::dmat Qir = prodDens / sumProdDens;
    // calculate g_ir(theta) = sum_t( d ln(P_itr(theta)) / d theta ) = sum_t( d ln(f(y_it) | w_ir) / d theta )
    // matrix to store results in - should be 1 x k
    // empty matricies for both gir, grad
    arma::dmat gir(nsim, par.n_rows);
    arma::dmat grad(y.n_rows, par.n_rows);
    // iterate thru each variable
    for (size_t i = 0; i < par.n_rows; i++) {
        arma::dcolvec parPlus = par, parMinus = par;
        // calculate step size
        double stepSize = par.at(i) * std::pow(std::numeric_limits<double>::epsilon(), (1.0 / 3.0));
        parPlus.at(i) += stepSize;
        parMinus.at(i) -= stepSize;
        // calculate the density for the +, -
        arma::dmat densPlus, densMinus;
        this->panelDensityHalfNormal(idx, parPlus, y, x, zuit, zvit, zvi0, draws, densPlus);
        this->panelDensityHalfNormal(idx, parMinus, y, x, zuit, zvit, zvi0, draws, densMinus);
        // handle gir seperately to gradient
        if (outGir) {
            // multiply down the columns for joint density of the entire panel
            arma::dmat prodDensPlus = esamath::colProd<double>(densPlus);
            arma::dmat prodDensMinus = esamath::colProd<double>(densMinus);
            // take natural logarithms
            arma::dmat logProdDensPlus = arma::log(prodDensPlus);
            arma::dmat logProdDensMinus = arma::log(prodDensMinus);
            // calculate gradient
            arma::dmat gradSim = (logProdDensPlus - logProdDensMinus) / (2.0 * stepSize);
            // transpose, insert into gir
            gir.col(i) = gradSim.t();
        } 
        if (outGrad) {
            // take natural logarithms of the densities
            arma::dmat logDensPlus = arma::log(densPlus);
            arma::dmat logDensMinus = arma::log(densPlus);
            arma::dmat gradSim = (logDensPlus - logDensMinus) / (2.0 * stepSize);
            // pointwise multiply each gradient by the weights (Q_ir)
            for (size_t tt = 0; tt < y.n_rows; tt++) {
                grad(tt, i) = arma::accu((gradSim.row(tt) % Qir));
            }
        }
    }
    if (outGir) *outGir = gir;
    if (outGrad) *outGrad = grad;
}

/// Numerically approximated gradient for truncated-normal distribution
template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0, typename TZui0>
void ESASfaTreGreene::panelGradTruncNormalNumApprox(
    const unsigned int idx,
    const arma::dcolvec& par,
    const arma::Base<double, TY>& yIn,
    const arma::Base<double, TX>& xIn,
    const arma::Base<double, TZmuit>& zmuitIn,
    const arma::Base<double, TZuit>& zuitIn,
    const arma::Base<double, TZvit>& zvitIn,
    const arma::Base<double, TZvi0>& zvi0In,
    const arma::dmat& draws,
    arma::dmat* outGrad,
    arma::dmat* outGir
) const
{
    // unwrap armadillo base references
    const auto& y = yIn.get_ref();
    const auto& x = xIn.get_ref();
    const auto& zmuit = zmuitIn.get_ref();
    const auto& zuit = zuitIn.get_ref();
    const auto& zvit = zvitIn.get_ref();
    const auto& zvi0 = zvi0In.get_ref();
    // for each individual, calculate Q_ir(theta) * g_ir(theta)
    // Q_ir(theta) = P_ir(theta) / sum(P_ir(theta)) (e.g., the weighted average of the conditional joint density for nT obs for each individual
    // (given the random individual specific effect))
    // P_ir(theta) = prod(P_irt(theta)) = prod(f(y_it) | w_ir) for t = 1, ..., nT
    // g_ir(theta) = sum_t( d ln(P_itr(theta)) / d theta ) = sum_t( d ln(f(y_it) | w_ir) / d theta )
    // start off with Q_ir(theta) // start off with Q_ir(theta)// start off with Q_ir(theta)
    // calculate P_itr(theta) for each individual
    arma::dmat dens;
    panelDensityTruncNormal(idx, par, y, x, zmuit, zuit, zvit, zvi0, draws, dens);
    // calculate P_it - take the product of the densities
    arma::dmat prodDens = esamath::colProd<double>(dens);
    // sum of P_it
    double sumProdDens = arma::accu(prodDens);
    // calculate Q_ir(theta) = P_ir(theta) / sum(P_ir(theta))
    arma::dmat Qir = prodDens / sumProdDens;
    arma::dmat QirExplode = arma::dmat(y.n_rows, 1, arma::fill::ones) * Qir;
    // calculate g_ir(theta) = sum_t( d ln(P_itr(theta)) / d theta ) = sum_t( d ln(f(y_it) | w_ir) / d theta )
    // matrix to store results in - should be 1 x k
    // the gradient shares similar steps - so start both
    arma::dmat gir(nsim, par.n_rows);
    arma::dmat grad(y.n_rows, par.n_rows);
    // iterate thru parameters
    for (size_t i = 0; i < par.n_rows; i++) {
        // copy param vector -> +, - step
        arma::dcolvec parPlus = par, parMinus = par;
        // calculate step size
        double stepSize = par.at(i) * std::pow(std::numeric_limits<double>::epsilon(), (1.0 / 3.0));
        parPlus(i) += stepSize;
        parMinus(i) -= stepSize;
        // calculate density for +, -
        arma::dmat densPlus, densMinus;
        panelDensityTruncNormal(idx, parPlus, y, x, zmuit, zuit, zvit, zvi0, draws, densPlus);
        panelDensityTruncNormal(idx, parMinus, y, x, zmuit, zuit, zvit, zvi0, draws, densMinus);
        // gir and grad have slightly different calculations
        if (outGir) {
            // multiply down the columns for the joint density for each panel
            arma::dmat prodDensPlus = esamath::colProd<double>(densPlus);
            arma::dmat prodDensMinus = esamath::colProd<double>(densMinus);
            // take logs 
            arma::dmat logProdDensPlus = arma::log(prodDensPlus);
            arma::dmat logProdDensMinus = arma::log(prodDensMinus);
            arma::dmat gradSim = (logProdDensPlus - logProdDensMinus) / (2.0 * stepSize);
            // transpose, insert into gir
            gir.col(i) = gradSim.t();
        }
        if (outGrad) {
            // take natural logarithms
            arma::dmat logDensPlus = arma::log(densPlus);
            arma::dmat logDensMinus = arma::log(densMinus);
            arma::dmat gradSim2 = (logDensPlus - logDensMinus) / (2.0 * stepSize);
            // arma::dmat gradSimWeighted = (gradSim2 % QirExplode);
            for (size_t tt = 0; tt < y.n_rows; tt++) {
                grad(tt, i) = arma::accu((gradSim2.row(tt) % Qir));
            }
        }
    }
    if (outGir) *outGir = gir;
    if (outGrad) *outGrad = grad;
}

// ---- gradient or/and hessian calculation ----
template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0, typename TZui0>
void ESASfaTreGreene::gradHessPanel(
    const unsigned int idx,
    const ESASfaModelType mT,
    const arma::dcolvec& par,
    const arma::Base<double, TY>& y_in,
    const arma::Base<double, TX>& x_in,
    const std::optional<TZmuit>& zmuit_in,
    const arma::Base<double, TZuit>& zuit_in,
    const arma::Base<double, TZvit>& zvit_in,
    const arma::Base<double, TZvi0>& zvi0_in,
    const arma::dmat& draws,
    const HessianCalcMethod method,
    const unsigned int accuracy,
    arma::dmat* gradOut,
    arma::dmat* hessOut,
    double* llOut
) const
{
    if (method != HessianCalcMethod::ANALYTICAL) throw std::runtime_error("only support analytical gradHess (TRE)");
    // dereference ptr to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    // persistent variables on specific thread
    ThreadContext* ctx = getContext();
    // check if the pointer for treDensHNorm is empty
    if (!ctx->treGradHessPanel) {
        ctx->treGradHessPanel = std::make_unique<thread_cache::WSGradHessPanel>();
    }
    thread_cache::WSGradHessPanel& ws = *ctx->treGradHessPanel;
    // auto wsPtr = std::make_unique<thread_cache::WSGradHessPanel>();
    // thread_cache::WSGradHessPanel& ws = *wsPtr;
    // really ugly lol
    arma::dmat dummy(1, 1, arma::fill::zeros);
    // force return type to TZui0
    const auto& zmuitAcc = [&mT, &zmuit_in, &dummy]() -> TZmuit {
        if (mT == ESASfaModelType::TRE_TNORM_ZUIT){
            if (!zmuit_in) return zmuit_in.value();
        }
        if constexpr (std::is_same_v<TZmuit, arma::dmat> || std::is_same_v<TZmuit, arma::mat>) {
            return TZmuit(); // Returns empty matrix
        } else {
            return dummy.submat(0, 0, 0, 0);
        }
    }();
    const auto& y = y_in.get_ref();
    const auto& x = x_in.get_ref();
    const auto& zuit = zuit_in.get_ref();
    const auto& zvit = zvit_in.get_ref();
    const auto& zvi0 = zvi0_in.get_ref();
    // number of time periods
    unsigned int nT = y.n_rows;
    int maxT = dataObj.getMaxT();
    int ns = this->nsim;
    // number of parameters
    int nParam = dataObj.nParams();
    // resize matricies in workspace if needed - should be maxT, but incase it fails
    int requiredRows = (nT > maxT) ? nT : maxT; 
    ws.ensureSize(requiredRows, ns, nParam);
    // calculate the density (nT x nsim), stored in the persistent buffer
    if (mT == ESASfaModelType::TRE_HNORM_ZUIT) {
        this->panelDensityHalfNormal(idx, par, y, x, zuit, zvit, zvi0, draws, ws.dens);
    } else if (mT == ESASfaModelType::TRE_TNORM_ZUIT) {
        if (!zmuit_in) throw std::invalid_argument("expected zmuit, but missing optional");
        this->panelDensityTruncNormal(idx, par, y, x, zmuit_in.value(), zuit, zvit, zvi0, draws, ws.dens);
    } else {
        throw std::invalid_argument("unsupported TRE method");
    }
    // view to the relevant part of dens
    arma::subview<double> densView = ws.dens.submat(arma::span(0, nT - 1), arma::span(0, ns - 1));
    // multiply along the densities (1 x nsim)
    this->weightFromDensForPanel(densView, ws.Qir);
    arma::subview<double> QirView = ws.Qir.row(0);
    // optionally return panel log-likelihood (reuse Smax and Kr from weightFromDensForPanel workspace)
    if (llOut) {
        ThreadContext* ctxLL = getContext();
        auto& wsLL = *ctxLL->treBaseWeightDens;
        arma::subview<double> krView = wsLL.Kr.submat(arma::span(0, 0), arma::span(0, ns - 1));
        *llOut = wsLL.Smax + std::log(arma::accu(krView)) - std::log(static_cast<double>(ns));
    }
    // analytically derive gradient & hessian
    if (method == HessianCalcMethod::ANALYTICAL && mT == ESASfaModelType::TRE_HNORM_ZUIT) {
        this->internalAnalyticJacHess(
            idx, mT, par, y, x, zmuit_in, zuit, zvit, zvi0, draws,
            std::optional<TZui0>(std::nullopt), QirView, &ws.jac, &ws.hess
        );
    } else if (method == HessianCalcMethod::ANALYTICAL && mT == ESASfaModelType::TRE_TNORM_ZUIT) {
        if (!zmuit_in) throw std::invalid_argument("gradHessPanel: zmuit required for TN model");
        this->internalAnalyticJacHessTN(
            idx, par, y, x, zmuit_in.value(), zuit, zvit, zvi0, draws,
            QirView, &ws.jac, &ws.hess
        );
    } else {
        throw std::runtime_error("gradHessPanel: unsupported model type or method");
    }
    if (gradOut) {
        *gradOut = ws.jac.rows(0, nT - 1);
    }
    if (!hessOut) return;
    // vector of vector of matricies to store hessians over i, t, and r;
    // warning! hess_itr is nsim x maxT NOT nsim x nT!!

    // if calculating the analytical hessian, already filled in hess_itr so can
    // skip most of this logic, which is for the BHHH approach
    // if (method != HessianCalcMethod::ANALYTICAL) {
    //     // 2d vec of hessian matricies
    //     // std::vector<std::vector<arma::dmat>> hess_itr_tdom(nT, std::vector<arma::dmat>(this->nsim));
    //     // lambda function which mirrors gradient
    //     auto gfnWrap = [this, &idx, &y, &x, &zuit, &zmuit_in, &zvit, &zvi0, &draws, &mT, &numApproxGrad](const arma::dcolvec& par, const unsigned int tt){
    //         arma::dmat g;
    //         std::optional<TZmuit> zmuit_tt = std::optional<TZmuit>(std::nullopt);
    //         if (zmuit_in) zmuit_tt = zmuit_in.value().row(tt); 
    //         this->gradientInner(
    //             par,
    //             mT,
    //             draws,
    //             !numApproxGrad,
    //             idx,
    //             y.row(tt),
    //             x.row(tt),
    //             zmuit_tt,
    //             zuit.row(tt),
    //             zvit.row(tt),
    //             zvi0.row(tt),
    //             nullptr,
    //             &g
    //         );
    //         return g;
    //     };
    //     // iterate thru time periods
    //     for (unsigned int tt = 0; tt < nT; tt++) {
    //         std::vector<arma::dmat> hessSims;
    //         if (
    //             (method == HessianCalcMethod::NUM_APPROX) ||
    //             (method == HessianCalcMethod::NUM_APPROX_WITH_NUM_APPROX_GRAD)
    //         ) {
    //             auto gfnWrapCurried = [&tt, &gfnWrap](const arma::dcolvec& par){
    //                 return gfnWrap(par, tt);
    //             };
    //             hessSims = finitediff::calculateFiniteHessianSimsUsingGrad(par, gfnWrapCurried);
    //         } else if (
    //             (method == HessianCalcMethod::BHHH_APPROX_ANALYTICAL_GRAD) ||
    //             (method == HessianCalcMethod::BHHH_APPROX_NUM_APPROX_GRAD)
    //         ) {
    //             arma::dmat g = gfnWrap(par, tt);
    //             // sense check returned size
    //             if (g.n_rows != this->nsim || g.n_cols != par.n_rows) {
    //                 throw std::runtime_error("returned 'grad' for BHHH hess approx has incorrect dims");
    //             }
    //             hessSims.resize(this->nsim);
    //             for (unsigned int ns = 0; ns < this->nsim; ns++) {
    //                 // need to negate the outer product of the jacobian?
    //                 hessSims[ns] = (g.row(ns).t() * g.row(ns));
    //             }
    //         }
    //         ws.hess_itr_tdom[tt] = std::move(hessSims);
    //     }
    //     for (unsigned int i = 0; i < this->nsim; i++) {
    //         std::vector<arma::dmat>& innerVec = ws.hess_itr[i];
    //         innerVec.resize(nT);
    //         for (unsigned int tt = 0; tt < nT; tt++) {
    //             innerVec[tt] = std::move(ws.hess_itr_tdom[tt][i]);
    //         }
    //     }
    // }
    // calculate full hessian matrix
    // this->simulatedHessianForFirm(ws.hess_itr, ws.Qir, ws.gir, this->nsim, par.n_rows, nT, ws.hess);
    // since J.t() * J will be positive semi-definite, negate it?
    if (
        (method == HessianCalcMethod::BHHH_APPROX_ANALYTICAL_GRAD) ||
        (method == HessianCalcMethod::BHHH_APPROX_NUM_APPROX_GRAD)
    ) {
        ws.hess = -ws.hess;
    }
    if (hessOut) *hessOut = ws.hess;
}

// =========================================================
//                      EXPLICIT INSTANTIATIONS
// =========================================================

/// ---- template init operatorInner ----
template void ESASfaTreGreene::operatorInner<
    arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat
>(
    const arma::dcolvec&,
    const ESASfaModelType,
    const arma::dmat&,
    const unsigned int,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const std::optional<arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const bool,
    arma::dmat*
) const;
template void ESASfaTreGreene::operatorInner<
    arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>
>(
    const arma::dcolvec&,
    const ESASfaModelType,
    const arma::dmat&,
    const unsigned int,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const std::optional<arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const bool,
    arma::dmat*
) const;

/// ---- template init gradientInner ----
template void ESASfaTreGreene::gradientInner<
    arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat
>(
    const arma::dcolvec&,
    const ESASfaModelType,
    const arma::dmat&,
    const bool,
    const unsigned int,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const std::optional<arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    arma::dmat*,
    arma::dmat*
) const;
template void ESASfaTreGreene::gradientInner<
    arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>
>(
    const arma::dcolvec&,
    const ESASfaModelType,
    const arma::dmat&,
    const bool,
    const unsigned int,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const std::optional<arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    arma::dmat*,
    arma::dmat*
) const;

/// ---- template init gradHessPanel ----
template void ESASfaTreGreene::gradHessPanel<
    arma::dmat,     // TY
    arma::dmat,     // TX
    arma::dmat,     // TZmuit
    arma::dmat,     // TZuit
    arma::dmat,     // TZvit,
    arma::dmat,     // TZvi0
    arma::dmat      // TZui0
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
    const arma::dmat&,
    const HessianCalcMethod,
    const unsigned int,
    arma::dmat*,
    arma::dmat*,
    double*
) const;
// contiguous arrays
template void ESASfaTreGreene::gradHessPanel<
    arma::subview<double>,     // TY
    arma::subview<double>,     // TX
    arma::subview<double>,     // TZmuit
    arma::subview<double>,     // TZuit
    arma::subview<double>,     // TZvit,
    arma::subview<double>,     // TZvi0
    arma::subview<double>     // TZui0
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
    const arma::dmat&,
    const HessianCalcMethod,
    const unsigned int,
    arma::dmat*,
    arma::dmat*,
    double*
) const;

/// ---- template init operatorInner (mixed types for LCM) ----
template void ESASfaTreGreene::operatorInner<
    arma::dmat, arma::dmat, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::dmat
>(
    const arma::dcolvec&,
    const ESASfaModelType,
    const arma::dmat&,
    const unsigned int,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const std::optional<arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const bool,
    arma::dmat*
) const;

/// ---- template init gradHessPanel (mixed types for LCM) ----
template void ESASfaTreGreene::gradHessPanel<
    arma::dmat, arma::dmat, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::dmat
>(
    const unsigned int,
    const ESASfaModelType,
    const arma::dcolvec&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const std::optional<arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    const HessianCalcMethod,
    const unsigned int,
    arma::dmat*,
    arma::dmat*,
    double*
) const;

/// ---- template init panelDensityHalfNormal ----
template void ESASfaTreGreene::panelDensityHalfNormal<
    arma::dmat, // TY
    arma::dmat, // TX
    arma::dmat, // TZmuit
    arma::dmat, // TZuit
    arma::dmat, // TZvit
    arma::dmat, // TZvi0
    arma::dmat // TZui0
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::dmat&,
    arma::dmat&
) const;
template void ESASfaTreGreene::panelDensityHalfNormal<
    arma::subview<double>, // TY
    arma::subview<double>, // TX
    arma::subview<double>, // TZmuit
    arma::subview<double>, // TZuit
    arma::subview<double>, // TZvit
    arma::subview<double>, // TZvi0
    arma::subview<double> // TZui0
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    arma::dmat&
) const;

/// ---- template init panelDensityHalfNormal (mixed types for LCM) ----
template void ESASfaTreGreene::panelDensityHalfNormal<
    arma::dmat,            // TY
    arma::dmat,            // TX
    arma::dmat,            // TZmuit (default)
    arma::subview<double>, // TZuit
    arma::subview<double>, // TZvit
    arma::subview<double>, // TZvi0
    arma::dmat             // TZui0 (default)
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    arma::dmat&
) const;

/// ---- template init panelDensityTruncNormal (mixed types for LCM) ----
template void ESASfaTreGreene::panelDensityTruncNormal<
    arma::dmat,            // TY
    arma::dmat,            // TX
    arma::subview<double>, // TZmuit
    arma::subview<double>, // TZuit
    arma::subview<double>, // TZvit
    arma::subview<double>, // TZvi0
    arma::dmat             // TZui0 (default)
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    arma::dmat&
) const;

/// ---- template init panelDensityTruncNormal ----
template void ESASfaTreGreene::panelDensityTruncNormal<
    arma::dmat, // TY
    arma::dmat, // TX
    arma::dmat, // TZmuit
    arma::dmat, // TZuit,
    arma::dmat, // TZvit
    arma::dmat, // TZvi0
    arma::dmat // TZui0
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
template void ESASfaTreGreene::panelDensityTruncNormal<
    arma::subview<double>, // TY
    arma::subview<double>, // TX
    arma::subview<double>, // TZmuit
    arma::subview<double>, // TZuit
    arma::subview<double>, // TZvit
    arma::subview<double>, // TZvi0
    arma::subview<double> // TZui0
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


/// ---- template init panelGradHalfNormNumApprox ----
template void ESASfaTreGreene::panelGradHalfNormalNumApprox<
    arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&, 
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::dmat&,
    arma::dmat*,
    arma::dmat*
) const;
template void ESASfaTreGreene::panelGradHalfNormalNumApprox<
    arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    arma::dmat*,
    arma::dmat*
) const;

/// ---- template init panelGradTruncNormalNumApprox ----
template void ESASfaTreGreene::panelGradTruncNormalNumApprox<
    arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat
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
    arma::dmat*,
    arma::dmat*
) const;
template void ESASfaTreGreene::panelGradTruncNormalNumApprox<
    arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>, arma::subview<double>
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
    arma::dmat*,
    arma::dmat*
) const;