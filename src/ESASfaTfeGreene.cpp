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
 * @file ESASfaTfeGreene.cpp
 * @brief Class for estimating True Fixed Effects Model from Greene (2005) using LSDV approach
 * @date 2025-09-02
 * @author Edmund Haacke
 */

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <stdexcept>
#include "sfa/ESASfaTfeGreene.hpp"
#include "data/ESADataPanel.hpp"
#include "math/esamath.hpp"
#include "math/esandist.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "utils/excepts.hpp"
#include "optim/ESAGlobalOptimParams.hpp"


/// Constructor
ESASfaTfeGreene::ESASfaTfeGreene(const std::shared_ptr<ESADataBase> dataObjPtr, const double s) : ESASfaBase(dataObjPtr, s)
{
    // check whether panel data object
    if (!dynamic_cast<ESADataPanel *>(dataObjPtr.get())) {
        throw std::invalid_argument("Data object is not of type 'ESADataPanel'");
    }
}

/// Objective function to maximize/minimize
double ESASfaTfeGreene::operator()(const arma::dcolvec& params, const bool exceptNotFinite) const
{
    // dereference pointer to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    ESASfaModelType mT = dataObj.getModelType();
    arma::dmat dens = dataObj.dataCallable(
        [this, &params, &mT, &exceptNotFinite](
            const arma::colvec& y,
            const arma::dmat& x,
            const std::optional<arma::dmat>&zmuit,
            const std::optional<arma::dmat>&zuit,
            const std::optional<arma::dmat>&zvit,
            const std::optional<arma::dmat>&zui0,
            const std::optional<arma::dmat>&zvi0
        ) {
            // all of them need zuit, zvit, check the exist
            if (!zuit || !zvit) throw std::invalid_argument("missing 'zuit' or 'zvit'");
            arma::dmat d;
            if (mT == ESASfaModelType::TFE_HNORM_ZUIT) {
                // call density function for half normal
                d = this->densityHalfNormal(params, y, x, zuit.value(), zvit.value());
            } else if (mT == ESASfaModelType::TFE_TNORM_ZUIT) {
                // check zmuit exists
                if (!zmuit) throw std::invalid_argument("missing 'zmuit' for truncnormal");
                // call density function for truncated normal
                d = this->densityTruncNormal(params, y, x, zmuit.value(), zuit.value(), zvit.value());
            } else {
                throw std::invalid_argument("unsupported model type");
            }
            // 
            if (exceptNotFinite && !d.is_finite()) {
                std::string m = "Density not finite in TRE";
                throw esaexcepts::DensityNotFinite(m.c_str());
            }
            return d;
        }
    );
    // Calculate the log-likelihood
    // take the element-wise natural log of the densities
    arma::dmat ldens = arma::log(dens);
    // sum all the log-density values
    double ll = arma::accu(ldens);
    // return loglikelihood value
    return ll;
}

// /// Gradient at parameter vector
// arma::dmat ESASfaTfeGreene::gradient(const arma::dcolvec& params, const double step, const bool isAnalytical) const
// {
//     ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
//     ESASfaModelType mT = dataObj.getModelType();
//     arma::dmat jac = dataObj.dataCallable(
//         [this, &params, &mT, &isAnalytical, &jac](
//             const arma::colvec& y,
//             const arma::dmat& x,
//             const std::optional<arma::dmat>&zmuit,
//             const std::optional<arma::dmat>&zuit,
//             const std::optional<arma::dmat>&zvit,
//             const std::optional<arma::dmat>&zui0,
//             const std::optional<arma::dmat>&zvi0
//         ) {
//             return this->gradientInner(
//                 mT,
//                 isAnalytical,
//                 params,
//                 y,
//                 x,
//                 zmuit,
//                 zuit,
//                 zvit,
//                 zui0,
//                 zvi0
//             );
//         }
//     );
//     // calculate gradient (over all observations)
//     arma::dmat g = esamath::colSum(jac);
//     // divide thru number of observations
//     g = g / dataObjPtr->getNobs();
//     return g;
// }

// /// Hessian at parameter vector
// arma::dmat ESASfaTfeGreene::hessian(
//     const arma::dcolvec& params,
//     const HessianCalcMethod method,
//     const unsigned int accuracy,
//     const bool threaded
// ) const
// {
//     ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
//     ESASfaModelType mT = dataObj.getModelType();
//     arma::dmat hess = dataObj.dataCallable(
//         [this, &params, &mT, &method, &accuracy](
//             const arma::colvec& y,
//             const arma::dmat& x,
//             const std::optional<arma::dmat>&zmuit,
//             const std::optional<arma::dmat>&zuit,
//             const std::optional<arma::dmat>&zvit,
//             const std::optional<arma::dmat>&zui0,
//             const std::optional<arma::dmat>&zvi0
//         ) {
//             return this->hessianInner(
//                 mT,
//                 true,
//                 method,
//                 accuracy,
//                 params,
//                 y,
//                 x,
//                 zmuit,
//                 zuit,
//                 zvit,
//                 zui0,
//                 zvi0
//             );
//         }
//     );
//     // divide thru number of observations
//     hess = hess / dataObjPtr->getNobs();
//     return hess;
// }

/// Starting values for maximum likelihood estimation
arma::dcolvec ESASfaTfeGreene::startingValues() const
{
    // dereference ptr to underlying data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    // get matricies for zuit, zvit
    if (!dataObj.getZuit() || !dataObj.getZvit()){
        throw std::invalid_argument("zuit and zvit must be present (both half & trunc normal)");
    }
    arma::dmat zuit = dataObj.getZuit().value();
    arma::dmat zvit = dataObj.getZvit().value();
    // get number of columns for x, zmit, zuit, zvit
    int nX = dataObj.getX().n_cols, nZmuit = 0, nZuit = zuit.n_cols, nZvit = zvit.n_cols;
    // start off with estimating an OLS model, regressing x on y
    arma::dcolvec bX;
    try {
        bX = arma::pinv(dataObj.getX()) * dataObj.getY();
    } catch (const std::exception& e){
        ESALogger::logger()->warn("SfaTfeGreene - bX error inverting matrix to calculate starting values: {}", e.what());
        // set bX to 0
        bX = arma::dmat(nX, 1, arma::fill::zeros);
    }
    // calculate the residuals
    arma::dmat res = dataObj.getY() - dataObj.getX() * bX;
    // m2, m3
    double m2 = arma::accu(arma::pow(res, 2.0)) / res.n_rows;
    double m3 = arma::accu(arma::pow(res, 3.0)) / res.n_rows;
    double varuit = 0.0, varvit = 0.0;
    if ((this->s * m3) > 0){
        varuit = std::pow(std::abs((this->s * m3 * std::sqrt(M_PI / 2.0) / (1.0 - 4.0 / M_PI))), (2.0 / 3.0));
    } else {
        varuit = std::pow((this->s * m3 * std::sqrt(M_PI / 2.0) / (1.0 - 4.0 / M_PI)), (2.0 / 3.0));
    }
    if (m2 < ((M_PI -2.0) / M_PI) * varuit){
        varvit = std::abs((m2 - (1.0 - 2.0 / M_PI) * varuit));
    } else {
        varvit = m2 - (1.0 - 2.0 / M_PI) * varuit;
    }
    // dependent variable for u
    arma::dmat depU;
    if (nZuit > 1){
        depU = (1.0 / 2.0) * arma::log(arma::pow(((arma::pow(res, 2.0) - varvit) * M_PI / (M_PI - 2.0)), 2.0));
    } else {
        depU = arma::dmat(res.n_rows, 1, arma::fill::value(std::log(varuit)));
    }
    // depending again, whether there are any determinants
    arma::dmat depV;
    if (nZvit > 1){
        depV = (1.0 / 2.0) * arma::log(arma::pow((arma::pow(res, 2.0) - (1.0 - 2.0 / M_PI) * varuit), 2.0));
    } else {
        depV = arma::dmat(res.n_rows, 1, arma::fill::value(std::log(varvit)));
    }
    // regress zuit on depU
    arma::dcolvec bZuit;
    try {
        bZuit = arma::pinv(zuit) * depU;
    } catch (const std::exception& e){
        ESALogger::logger()->warn("SfaTfeGreene - bZuit error inverting matrix to calculate starting values: {}", e.what());
        // set bZuit to -0.1
        bZuit = arma::dmat(nZuit, 1, arma::fill::value(-0.1));
    }
    // replace any zeros with -0.1
    bZuit = esautils::replaceValuesPrecision<double>(bZuit, 0.0, -0.1);
    // regress zvit on depV
    arma::dcolvec bZvit;
    try {
        bZvit = arma::pinv(zvit) * depV;
    } catch (const std::exception& e){
        ESALogger::logger()->warn("SfaTfeGreene - bZvit error inverting matrix to calculate starting values: {}", e.what());
        // set bZvit to -0.1
        bZvit = arma::dmat(nZvit, 1, arma::fill::value(-0.1));
    }
    bZvit = esautils::replaceValuesPrecision<double>(bZvit, 0.0, -0.1);
    // if it's a truncated normal model, need to calculate something for mu
    arma::dmat bMuit;
    if (dataObj.getModelType() == ESASfaModelType::TFE_TNORM_ZUIT){
        // get mu from data object
        if (!dataObj.getZmuit()){
            throw std::invalid_argument("mu must be present for truncated normal distribution");
        }
        arma::dmat mu = dataObj.getZmuit().value();
        nZmuit = mu.n_cols;
        // regress mu on residuals
        try {
            bMuit = arma::pinv(mu) * res;
        } catch (const std::exception& e){
            ESALogger::logger()->warn("SfaTfeGreene - bMu error inverting matrix to calculate starting values: {}", e.what());
            // set bMu to -0.1
            bMuit = arma::dmat(nZmuit, 1, arma::fill::value(-0.1));
        }
        // replace any zeros with -0.1
        bMuit = esautils::replaceValuesPrecision<double>(bMuit, 0.0, -0.1);
    }
    // empty vector to store the starting values
    arma::dcolvec startVals(nX + nZmuit + nZuit + nZvit);
    // set the starting values
    startVals.submat(arma::span(0, (nX - 1)), arma::span(0, 0)) = bX;
    if (dataObj.getModelType() == ESASfaModelType::TFE_TNORM_ZUIT){
        startVals.submat(arma::span(nX, nX + nZmuit - 1), arma::span(0, 0)) = bMuit;
    }
    startVals.submat(arma::span(nX + nZmuit, nX + nZmuit + nZuit - 1), arma::span(0, 0)) = bZuit;
    startVals.submat(arma::span(nX + nZmuit + nZuit, nX + nZmuit + nZuit + nZvit - 1), arma::span(0, 0)) = bZvit;
    // adjust intercept in bX
    if (dataObj.xHasInterceptTerm()){
        startVals(0) = startVals(0) + (this->s * std::sqrt(varuit * 2.0 / M_PI));
    }
    return startVals;
}

/// density for half-normal distribution
arma::dmat ESASfaTfeGreene::densityHalfNormal(
    const arma::dcolvec& par,
    const arma::dcolvec& y,
    const arma::dmat& x,
    const arma::dmat& zuit,
    const arma::dmat& zvit
) const
{
    // dereference ptr to underlying data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    // extract the coefficients for each model
    arma::dcolvec bX = dataObj.paramX(par);
    // coefficient for zuit
    std::optional<arma::dcolvec> bZuitOpt = dataObj.paramZuit(par);
    if (!bZuitOpt) throw std::invalid_argument("could not get coefficient for zuit");
    arma::dcolvec bZuit = bZuitOpt.value();
    // coefficient for zvit
    std::optional<arma::dcolvec> bZvitOpt = dataObj.paramZvit(par);
    if (!bZvitOpt) throw std::invalid_argument("could not get coefficient for zvit");
    arma::dcolvec bZvit = bZvitOpt.value();
    // ---- calculate the log likelihood ----
    arma::dmat xb = x * bX;
    arma::dmat sigma2uit = esautils::processSig2Term(bZuit, zuit);
    arma::dmat sigmauit = arma::sqrt(sigma2uit);
    arma::dmat sigma2vit = esautils::processSig2Term(bZvit, zvit);
    arma::dmat sigmavit = arma::sqrt(sigma2vit);
    arma::dmat sigma2 = sigma2uit + sigma2vit;
    arma::dmat sigma = arma::sqrt(sigma2);
    arma::dmat lambda = sigmauit / sigmavit;
    // calculate epsilon = y - xb
    arma::dmat eps = y - xb;
    // calculate c1, c2
    arma::dmat c1In = eps / sigma;
    arma::dmat c2In = (-s * (eps % lambda)) / sigma;
    // calculate the density
    arma::dmat den = (2.0 / sigma) % (arma::normpdf(c1In, 0.0, 1.0) % arma::normcdf(c2In, 0.0, 1.0));
    return den;
}

/// density for truncated-normal distribution
arma::dmat ESASfaTfeGreene::densityTruncNormal(
    const arma::dcolvec& par,
    const arma::dcolvec& y,
    const arma::dmat& x,
    const arma::dmat& zmuit,
    const arma::dmat& zuit,
    const arma::dmat& zvit
) const
{
    // dereference ptr to underlying data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    // ---- extract coefficients ----
    arma::dcolvec bX = dataObj.paramX(par);
    // coefficient for zmuit
    std::optional<arma::dcolvec> bZmuitOpt = dataObj.paramZmuit(par);
    if (!bZmuitOpt) throw std::invalid_argument("could not get coefficient for zmuit");
    arma::dcolvec bZmuit = bZmuitOpt.value();
    // coefficient for zuit
    std::optional<arma::dcolvec> bZuitOpt = dataObj.paramZuit(par);
    if (!bZuitOpt) throw std::invalid_argument("could not get coefficient for zuit");
    arma::dcolvec bZuit = bZuitOpt.value();
    // coefficient for zvit
    std::optional<arma::dcolvec> bZvitOpt = dataObj.paramZvit(par);
    if (!bZvitOpt) throw std::invalid_argument("could not get coefficient for zvit");
    arma::dcolvec bZvit = bZvitOpt.value();
    // ---- calculations ----
    arma::dmat xb = x * bX;
    arma::dmat mu = zmuit * bZmuit;
    arma::dmat sigma2uit = esautils::processSig2Term(bZuit, zuit);
    arma::dmat sigmauit = arma::sqrt(sigma2uit);
    arma::dmat sigma2vit = esautils::processSig2Term(bZvit, zvit);
    arma::dmat sigmavit = arma::sqrt(sigma2vit);
    arma::dmat sigma2 = sigma2uit + sigma2vit;
    arma::dmat sigma = arma::sqrt(sigma2);
    // sigma2star
    arma::dmat sigma2star = (sigma2vit % sigma2uit) / sigma2;
    arma::dmat sigmastar = arma::sqrt(sigma2star);
    // calculate epsilon = y - xb
    // arma::dmat eps = s * (y - xb);
    // MODIFIED APR19
    arma::dmat eps = y - xb;
    // ---- denominator of the density ----
    arma::dmat muDivSigmau = mu / sigmauit;
    arma::dmat cdfMuDivSigmau = arma::normcdf(muDivSigmau, 0.0, 1.0);
    // calculate mustar
    arma::dmat muStarNumer = (mu % sigma2vit) - (eps % sigma2uit);
    arma::dmat muStar = muStarNumer / sigma2;
    // calculate cdf of mustar/sigstar
    arma::dmat muStarDivSigStar = muStar / sigmastar;
    arma::dmat cdfMuStarDivSigStar = arma::normcdf(muStarDivSigStar, 0.0, 1.0);
    // calculate the denominator
    arma::dmat denomC2 = cdfMuDivSigmau / cdfMuStarDivSigStar;
    arma::dmat denom = sigma % denomC2;
    // ---- numerator of the density ----
    arma::dmat epsmu = eps + mu;
    arma::dmat epsmuDivSigma = epsmu / sigma;
    arma::dmat numer = arma::normpdf(epsmuDivSigma, 0.0, 1.0);
    // calculate the density
    arma::dmat den = numer / denom;
    return den;
}

/// Gradient and Hessian at parameter vector
void ESASfaTfeGreene::gradHess(
    const arma::dcolvec& params,
    const bool exceptNotFinite,
    arma::dmat* gradOut,
    arma::dmat* hessOut,
    arma::dmat* jacOut
) const
{
   
    // dereference pointer to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    ESASfaModelType mT = dataObj.getModelType();
    // global optimization options
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    HessianCalcMethod hessMethod = globalOptimParams->hessianMethod;
    int accuracy = globalOptimParams->hessianNumApproxAcc;
    bool analyticalGrad = ESAEnums::isAnalyticalGrad(hessMethod);
    // empty matricies to store outputs for
    arma::dmat jac, grad, hess;
    dataObj.dataCallable(
        [this, &params, &mT, &analyticalGrad, &hessMethod, &accuracy, &exceptNotFinite](
            const arma::colvec& y,
            const arma::dmat& x,
            const std::optional<arma::dmat>&zmuit,
            const std::optional<arma::dmat>&zuit,
            const std::optional<arma::dmat>&zvit,
            const std::optional<arma::dmat>&zui0,
            const std::optional<arma::dmat>&zvi0,
            arma::dmat* g1,
            arma::dmat* h1,
            arma::dmat* j1
        ) {
            this->gradHessInner(
                mT, analyticalGrad, hessMethod, accuracy, params, y, x, zmuit, zuit, zvit, zui0, zvi0, g1, h1, j1
            );
            // check g1, h1
            if (exceptNotFinite && (g1)) {
                if (!(*g1).is_finite()) {
                    std::string m = "Gradient not finite in TRE";
                    throw esaexcepts::GradientNotFinite(m.c_str());
                }
            }
            if (exceptNotFinite && (h1)) {
                if (!(*h1).is_finite()) {
                    std::string m = "Hessian not finite in TRE";
                    throw esaexcepts::GradientNotFinite(m.c_str());
                }
            }
        },
        &grad,
        &hess,
        &jac
    );
    // divide gradient & hessian thru number of observations (not firms)
    int nobs = dataObjPtr->getNobs();
    grad = grad / nobs;
    hess = hess / nobs;
    if (jacOut) *jacOut = jac;
    if (gradOut) *gradOut = grad;
    if (hessOut) *hessOut = hess;
}

arma::dmat ESASfaTfeGreene::gradientInner(
    const ESASfaModelType mT,
    const bool analyticalGrad,
    const arma::dcolvec& par,
    const arma::colvec& y,
    const arma::dmat& x,
    const std::optional<arma::dmat>&zmuit,
    const std::optional<arma::dmat>&zuit,
    const std::optional<arma::dmat>&zvit,
    const std::optional<arma::dmat>&zui0,
    const std::optional<arma::dmat>&zvi0
) const
{
    arma::dmat g;
    this->gradHessInner(
        mT,
        analyticalGrad,
        HessianCalcMethod::ANALYTICAL, // irrelevant - placeholder
        0, // accuracy - irrelevant; placeholder
        par,
        y,
        x,
        zmuit,
        zuit,
        zvit,
        zui0,
        zvi0,
        &g, // gradient
        nullptr, // hessian
        nullptr // jacobian
    );
    return g;
}

arma::dmat ESASfaTfeGreene::hessianInner(
    const ESASfaModelType mT,
    const bool analyticalGrad,
    const HessianCalcMethod hessMethod,
    const unsigned int accuracy,
    const arma::dcolvec& par,
    const arma::colvec& y,
    const arma::dmat& x,
    const std::optional<arma::dmat>&zmuit,
    const std::optional<arma::dmat>&zuit,
    const std::optional<arma::dmat>&zvit,
    const std::optional<arma::dmat>&zui0,
    const std::optional<arma::dmat>&zvi0
) const
{
    arma::dmat h;
    this->gradHessInner(
        mT,
        analyticalGrad,
        hessMethod,
        accuracy,
        par,
        y,
        x,
        zmuit,
        zuit,
        zvit,
        zui0,
        zvi0, 
        nullptr, // gradient
        &h, // hessian
        nullptr // jacobian
    );
    return h;
}

void ESASfaTfeGreene::gradHessInner(
    const ESASfaModelType mT,
    const bool analyticalGrad,
    const HessianCalcMethod hessMethod,
    const unsigned int accuracy,
    const arma::dcolvec& par,
    const arma::colvec& y,
    const arma::dmat& x,
    const std::optional<arma::dmat>&zmuit,
    const std::optional<arma::dmat>&zuit,
    const std::optional<arma::dmat>&zvit,
    const std::optional<arma::dmat>&zui0,
    const std::optional<arma::dmat>&zvi0,
    arma::dmat* gradOut,
    arma::dmat* hessOut,
    arma::dmat* jacOut
) const
{
    // always must have zuit, zvit
    if (!zuit || !zvit) throw std::invalid_argument("'zuit' and 'zvit' required");
    ESASfaModelDistribution mD = ESAEnums::getDistribution(mT);
    // check zmuit is there for truncated normal
    if (mD == ESASfaModelDistribution::TNORM && !zmuit) {
        throw std::invalid_argument("'zmuit' required for trunc norm");
    }
    // always do an analytical grad
    arma::dmat jac, grad, hess;
    if (
        (hessOut) &&
        (hessMethod == HessianCalcMethod::ANALYTICAL) && 
        (mD == ESASfaModelDistribution::HNORM)
    ) {
        // half normal & analytical gradient + hessian
        analyticJacHessHalfNormal(par, y, x, zuit.value(), zvit.value(), &jac, &hess);
    } else if (
        hessOut &&
        (hessMethod == HessianCalcMethod::ANALYTICAL) &&
        (mD == ESASfaModelDistribution::TNORM)
    ) {
        // trunc normal & analytical gradient + hessian
        throw std::invalid_argument("not implemented");
    } else if (mD == ESASfaModelDistribution::HNORM) {
        // only bother getting analytical gradient for half normal
        analyticJacHessHalfNormal(par, y, x, zuit.value(), zvit.value(), &jac, nullptr);
    } else if (mD == ESASfaModelDistribution::TNORM) {
        // only bother getting analytical gradient for trunc-normal
        throw std::invalid_argument("not implemented");
    }
    // calculate gradient (over all observations)
    arma::dmat g = esamath::colSum(jac);
    if (jacOut) *jacOut = jac;
    if (gradOut) *gradOut = g;
    if (hessOut) *hessOut = hess;
}

/// Analytical gradient and Hessian for half-normal distribution
/// based off the code from SfaR package - and Coelli 1995.
/// https://github.com/hdakpo/sfaR/blob/main/R/sfacross-hnormal.R
void ESASfaTfeGreene::analyticJacHessHalfNormal(
    const arma::dcolvec& par,
    const arma::dcolvec& y,
    const arma::dmat& x,
    const arma::dmat& zuit,
    const arma::dmat& zvit,
    arma::dmat* jacOut,
    arma::dmat* hessOut
) const
{
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    int prodCost = this->s;
    // get params for x, zuit, zvit
    if (!dataObj.getZuit() || !dataObj.getZvit()) {
        throw std::invalid_argument("'zuit', 'zvit' must be present for half-normal dist");
    }
    arma::dcolvec b_x = dataObj.paramX(par);
    arma::dcolvec b_zu = dataObj.paramZuit(par).value();
    arma::dcolvec b_zv = dataObj.paramZvit(par).value();
    unsigned int nX = b_x.n_rows, nZu = b_zu.n_rows, nZv = b_zv.n_rows, nobs = y.n_rows;
    // weights matrix, set to 1
    arma::dcolvec wHvar(nobs, 1, arma::fill::ones);
    // calculate xb
    arma::dmat xb = x * b_x;
    arma::dmat wu = esautils::processSig2Term(b_zu, zuit, true);
    arma::dmat wv = esautils::processSig2Term(b_zv, zvit, true);
    // calculate sigma2
    arma::dmat sigma2 = wu + wv;
    arma::dmat sigma = arma::sqrt(sigma2);
    // calculate residual
    arma::dmat eps = y - xb;
    arma::dmat wvsq = wv / sigma2;
    arma::dmat mustar = -wu % (prodCost * (eps / sigma2));
    arma::dmat sigmastar = arma::sqrt((wu % wvsq));
    arma::dmat musig = mustar / sigmastar;
    arma::dmat pmusig = arma::normcdf(musig, 0.0, 1.0);
    arma::dmat dmusig = arma::normpdf(musig, 0.0, 1.0);
    arma::dmat sigx2 = sigma2 % sigmastar;
    arma::dmat dmusigwu = dmusig % wu;
    arma::dmat pmusig2 = pmusig % sigmastar;
    arma::dmat depsiIn = prodCost * (eps / sigma);
    arma::dmat depsi = arma::normpdf(depsiIn, 0.0, 1.0);
    arma::dmat depsisq = depsi % arma::pow(sigma2, 2.0);
    arma::dmat depsisqx2 = 0.5 * (prodCost * depsi) % (eps / depsisq);
    arma::dmat sigx3 = (
        0.5 * ((1.0 - wvsq)) % (wu / sigmastar)
    ) + sigmastar;
    // ((s^2 * eps^2 / sigma2 - 1) /depsisq) 
    arma::dmat sigx4c1 = (
        (std::pow(prodCost, 2.0) * (arma::pow(eps, 2.0) / sigma2) - 1.0) /
        depsisq
    );
    // s^2 * depsi * sigma2 * eps^2 / depsisq^2
    arma::dmat sigx4c2 = (
        std::pow(prodCost, 2.0) * (depsi % sigma2 % arma::pow(eps, 2.0)) /
        arma::pow(depsisq, 2.0)
    );
    arma::dmat sigx4 = (0.5 * (sigx4c1 - sigx4c2) - (0.5 / depsisq));
    // (s^2 * eps^2 / (depsi * sigma2^4))
    arma::dmat sigx5c1 = 0.5 * (std::pow(prodCost, 2.0) * (
        arma::pow(eps, 2.0) / (depsi % arma::pow(sigma2, 4.0))
    ));
    arma::dmat sigx5c2num = (
        0.5 * (std::pow(prodCost, 2.0) * (depsi % arma::pow(eps, 2.0))) +
        2.0 * (depsi % sigma2)
    );
    arma::dmat sigx5c2 = (sigx5c2num / arma::pow(depsisq, 2.0));
    arma::dmat sigx5 = sigx5c1 - sigx5c2;
    arma::dmat sigx5epsi = prodCost * (sigx5 % depsi % eps);
    arma::dmat wusq = (wu / sigma2);
    arma::dmat wuwvsq = ((1.0 - wusq) % wv);
    arma::dmat sig2wuc1 = 0.5 * (wuwvsq / sigmastar) + sigmastar;
    arma::dmat sig2wuc2 = (wu / arma::pow(sigx2, 2.0));
    arma::dmat sig2wu = (1.0 / sigx2) - (sig2wuc1 % sig2wuc2);
    arma::dmat sigx6 = (sig2wu % (dmusig / pmusig));

    // calculate multipliers for x, zu, zv
    arma::dmat multiX = prodCost * ( ((dmusigwu / pmusig2) + prodCost*eps) / sigma2 );
    // zu
    arma::dmat multiZuc1 = prodCost * ((depsisqx2 - sigx6) % eps) - 0.5 / sigma2;
    arma::dmat multiZu = (wu % multiZuc1);
    // zv
    //((sigx3) * dmusigwu/((sigx2)^2 * pmusig) + depsisqx2)
    arma::dmat multiZvc1 = ( sigx3 % (dmusigwu / (arma::pow(sigx2, 2.0) % pmusig)) ) + depsisqx2;
    arma::dmat multiZvIn = prodCost * (multiZvc1 % eps) - 0.5 / sigma2;
    arma::dmat multiZv = (wv % multiZvIn);

    // apply to each of the respective matricies
    arma::dmat grad(dataObj.getNobs(), (nX + nZu + nZv));
    arma::dmat gradX = esautils::sweepMatrixElementwise(x, 1, multiX, "*");
    arma::dmat gradZu = esautils::sweepMatrixElementwise(zuit, 1, multiZu, "*");
    arma::dmat gradZv = esautils::sweepMatrixElementwise(zvit, 1, multiZv, "*");
    // set the columns of the gradient matrix
    grad.submat(arma::span(0, nobs - 1), arma::span(0, nX - 1)) = gradX;
    grad.submat(arma::span(0, nobs - 1), arma::span(nX, nX + nZu - 1)) = gradZu;
    grad.submat(arma::span(0, nobs - 1), arma::span(nX + nZu, nX + nZu + nZv - 1)) = gradZv;
    if (jacOut) *jacOut = grad;
    // check if not supposed to calculate hessian; return
    if (!hessOut) return;
    // ---- calculate analytical hessian ----
    arma::dmat hessll = arma::dmat(nX + nZu + nZv, nX + nZu + nZv);
    arma::dmat hessXXvarsMulti = (
        std::pow(prodCost, 2) * wHvar % (
            dmusigwu % wu % (prodCost * (eps)/(wv % pmusig2) - dmusig/arma::pow(pmusig2, 2))/(sigma2) - 1.0
        )/sigma2
    );
    arma::dmat hessXXvarsLHS = esautils::sweepMatrixElementwise(x, 1, hessXXvarsMulti, "*");
    arma::dmat hessXXvars = hessXXvarsLHS.t() * x;
    // set in hessian matrix
    if (hessXXvars.n_rows != hessXXvars.n_cols && hessXXvars.n_rows != nX) {
        throw std::runtime_error("hess Xvars not correct shape");
    }
    hessll.submat(arma::span(0, nX - 1), arma::span(0, nX - 1)) = hessXXvars;
    arma::dmat hessXZuMulti = (
        prodCost * wHvar % (
            sigx6 + prodCost * (
                (sigx4) % depsi - (sig2wu) % dmusigwu % (prodCost * (eps)/wv - dmusig/(pmusig2))/((sigma2) % pmusig)
            ) % (eps)
        ) % wu
    );
    arma::dmat hessXZuLHS = esautils::sweepMatrixElementwise(x, 1, hessXZuMulti, "*");
    arma::dmat hessXZu = hessXZuLHS.t() * zuit;
    // set in hessian matrix
    if (hessXZu.n_rows != nX && hessXZu.n_cols != nZu) {
        throw std::runtime_error("hess zuvars not correct shape");
    }
    hessll.submat(arma::span(0, nX - 1), arma::span(nX, nX + nZu - 1)) = hessXZu;
    hessll.submat(arma::span(nX, nX + nZu - 1), arma::span(0, nX - 1)) = hessXZu.t();
    // ---- xvars x zv component ----
    arma::dmat hessXZvMulti = (
        prodCost * wHvar % wv % (
            prodCost * (
                sigx3 % dmusigwu % wu % (
                    prodCost * eps/(arma::pow(sigx2, 2.0) % wv % pmusig) -
                    arma::pow(sigx2, 2.0) % dmusig/( arma::pow((arma::pow(sigx2, 2.0) % pmusig), 2.0) % sigmastar)
                )/sigma2 + sigx4 % depsi
            ) % eps - sigx3 % dmusigwu/(arma::pow(sigx2, 2.0) % pmusig)
        )
    );
    arma::dmat hessXZvLHS = esautils::sweepMatrixElementwise(x, 1, hessXZvMulti, "*");
    arma::dmat hessXZv = hessXZvLHS.t() * zvit;
    // set in hessian matrix
    hessll.submat(arma::span(0, nX - 1), arma::span(nX + nZu, nX + nZu + nZv - 1)) = hessXZv;
    hessll.submat(arma::span(nX + nZu, nX + nZu + nZv - 1), arma::span(0, nX - 1)) = hessXZv.t();
    // ---- zu x zu component ----
    arma::dmat hessZuZuMulti = (
        wHvar % (
            (0.5/arma::pow(sigma2, 2.0) +
            prodCost * (0.5 * sigx5epsi - dmusig % (
                prodCost * arma::pow(sig2wu, 2.0) % (dmusig/pmusig - prodCost * wu % eps/sigx2) % eps - (
                (0.5 * wusq + 1.0 - 0.5 * (0.5 * (1.0 - wusq) + wusq)) % wuwvsq/sigmastar +
                (
                    2.0 - 2.0 * (arma::pow(0.5 * (wuwvsq/sigmastar) + sigmastar, 2.0) % wu % sigma2/arma::pow(sigx2, 2.0)) % sigmastar
                ) / arma::pow(sigx2, 2.0))/pmusig
            ) % eps) % wu +
            prodCost * (depsisqx2 - sigx6) % eps) - 0.5/(sigma2)
        ) % wu
    );
    arma::dmat hessZuZuLHS = esautils::sweepMatrixElementwise(zuit, 1, hessZuZuMulti, "*");
    arma::dmat hessZuZu = hessZuZuLHS.t() * zuit;
    hessll.submat(arma::span(nX, nX + nZu - 1), arma::span(nX, nX + nZu - 1)) = hessZuZu;

    // ---- zv x zv component ----
    arma::dmat hessZvZvMulti = (
        wHvar % ((0.5 * wvsq - 0.5)/sigma2 + prodCost * (
            (
                (
                    (0.5 * (wvsq) - 0.5 * (0.5 * (1.0 - wvsq) + wvsq)) % (1 - wvsq) +
                    std::pow(prodCost, 2.0) * arma::pow(sigx3, 2.0) % wu % wv % arma::pow(eps, 2.0)/(arma::pow(sigx2, 2.0) % sigma2)
                ) % wu/(arma::pow(sigx2, 2.0) % pmusig2) +
                sigx3 % (1.0/(arma::pow(sigx2, 2.0) % pmusig) - sigx3 % (2.0 * (sigma2 % pmusig2) +
                prodCost * dmusigwu % eps) % wv/arma::pow((arma::pow(sigx2, 2.0) % pmusig), 2.0))) % dmusigwu +
                prodCost * (0.5 * (sigx5 % wv) + 0.5/(depsisq)) % depsi % eps
        ) % eps) % wv
    );
    arma::dmat hessZvZvLHS = esautils::sweepMatrixElementwise(zvit, 1, hessZvZvMulti, "*");
    arma::dmat hessZvZv = hessZvZvLHS.t() * zvit;
    hessll.submat(arma::span(nX + nZu, nX + nZu + nZv - 1), arma::span(nX + nZu, nX + nZu + nZv - 1)) = hessZvZv;
    // ---- zu x zv component ----
    arma::dmat hessZuZvMulti = (
        wHvar % (0.5/arma::pow(sigma2, 2.0) + prodCost * (
            (
                (
                    (
                        (0.5 * wuwvsq - std::pow(prodCost, 2.0) * sigx3 % (sig2wu) % wu % arma::pow(eps, 2.0))/sigma2 +
                        0.5 * ((wusq - 1.0) % wvsq + 1.0 - 0.5 * ((1.0 - wusq) % (1.0 - wvsq))) +
                        0.5 * (1.0 - wvsq)
                    ) % wu/sigmastar + sigmastar
                ) / (arma::pow(sigx2, 2.0) % pmusig) -
                sigx3 % (
                    2.0 * ((0.5 * (wuwvsq/sigmastar) + sigmastar) % (sigma2) % pmusig2) -
                    prodCost * arma::pow(sigx2, 2.0) % sig2wu % dmusig % eps
                ) % wu/arma::pow((arma::pow(sigx2, 2.0) % pmusig), 2.0)
            ) % dmusig +
            0.5 * sigx5epsi
        ) % eps) % wu % wv
    );
    arma::dmat hessZuZvLHS = esautils::sweepMatrixElementwise(zuit, 1, hessZuZvMulti, "*");
    arma::dmat hessZuZv = hessZuZvLHS.t() * zvit;
    hessll.submat(arma::span(nX, nX + nZu - 1), arma::span(nX + nZu, nX + nZu + nZv - 1)) = hessZuZv;
    hessll.submat(arma::span(nX + nZu, nX + nZu + nZv - 1), arma::span(nX, nX + nZu - 1)) = hessZuZv.t();
    if (hessOut) *hessOut = hessll;
}