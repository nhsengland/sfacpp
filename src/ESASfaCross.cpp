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
 * @file ESASfaCross.cpp
 * @brief ESASfaCross class implementation file
 * @date 2025-02-01
 * @author Edmund Haacke
 */

#include <math.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "sfa/ESASfaCross.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "utils/finitediff.hpp"
#include "utils/esautils.hpp"
#include "math/esamath.hpp"
#include "math/esandist.hpp"
#include "optim/ESAGlobalOptimParams.hpp"

/// Constructor
ESASfaCross::ESASfaCross(const std::shared_ptr<ESADataBase>& dataObjPtr, const double s) : ESASfaBase(dataObjPtr, s)
{
    // check can cast to cross sectional - not panel
    if (!dynamic_cast<ESADataCross*>(dataObjPtr.get())){
        throw std::invalid_argument("data object is not of type ESADataCross (cross sectional)");
    }
}

/// Objective function to maximise
double ESASfaCross::operator()(const arma::dcolvec& params, const bool exceptNotFinite) const
{
    // dereference ptr to data object
    ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
    if (dataObj.getModelType() == ESASfaModelType::CROSS_HNORM_ZUIT){
        return this->logLikelihoodCrossHalfNormal(params);
    } else if (dataObj.getModelType() == ESASfaModelType::CROSS_TNORM_ZUIT){
        return this->logLikelihoodCrossTruncNormal(params);
    }
    throw std::runtime_error("Model type not recognised '" + ESAEnums::strForModelType(dataObj.getModelType()) + "'");
}

// /// Gradient across individuals
// arma::dmat ESASfaCross::jacobian(
//     const arma::dcolvec& params,
//     const double step,
//     const bool isAnalytical
// ) const
// {
//     /// dereference ptr to data object
//     ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
//     arma::dmat grad = dataObj.data_callable(
//         [this, &params, &dataObj](
//             const arma::dcolvec& y,
//             const arma::dmat& x,
//             const std::optional<arma::dmat>& mu,
//             const std::optional<arma::dmat>& zu,
//             const std::optional<arma::dmat>& zv
//         ){
//             // whether half-normal or truncated normal
//             if (dataObj.getModelType() == ESASfaModelType::CROSS_HNORM_ZUIT){
//                 // check zu, zv are present
//                 if (!zu || !zv){
//                     throw std::invalid_argument("zu and zv must be present for half-normal distribution");
//                 }
//                 return this->gradientDerivCrossHalfNormal(params, y, x, zu.value(), zv.value(), this->s);
//             } else if (dataObj.getModelType() == ESASfaModelType::CROSS_TNORM_ZUIT){
//                 // check mu, zu, zv are present
//                 if (!mu || !zu || !zv){
//                     throw std::invalid_argument("mu, zu and zv must be present for truncated normal distribution");
//                 }
//                 return this->gradientDerivCrossTruncNormal(params, y, x, mu.value(), zu.value(), zv.value(), this->s);

//             }
//             throw std::invalid_argument("Unsupported model type");
//         }
//     );
//     return grad;
// }

// arma::dmat ESASfaCross::gradient(const arma::dcolvec& params, const double step, const bool isAnalytical) const
// {
//     // since it is based off the jacobian, calculate the jacobian
//     arma::dmat j = this->jacobian(params, step, isAnalytical);
//     // sum up and divide through the number of observations
//     arma::dmat g = esamath::colSum(j);
//     int nobs = this->dataObjPtr->getNobs();
//     arma::dmat g_avg = g / nobs;
//     if (g_avg.n_rows != 1) throw std::runtime_error("expect g_avg to be n=1 rows");
//     return g_avg;
// }

// /// Hessian matrix
// arma::dmat ESASfaCross::hessian(
//     const arma::dcolvec& params,
//     const HessianCalcMethod method,
//     const unsigned int accuracy,
//     const bool threaded
// ) const
// {
//     /// dereference ptr to data object
//     ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
//     arma::dmat hess = dataObj.data_callable(
//         [this, &params, &method, &accuracy, &dataObj](
//             const arma::dcolvec& y,
//             const arma::dmat& x,
//             const std::optional<arma::dmat>& mu,
//             const std::optional<arma::dmat>& zu,
//             const std::optional<arma::dmat>& zv
//         ){
//             arma::dmat hessOut;
//             // whether half-normal or truncated normal
//             if (dataObj.getModelType() == ESASfaModelType::CROSS_HNORM_ZUIT){
//                 // check zu, zv are present
//                 if (!zu || !zv){
//                     throw std::invalid_argument("zu and zv must be present for half-normal distribution");
//                 }
//                 // if BHHH approximation, calculate the gradient (whether analytically, or numerically)
//                 if (method == HessianCalcMethod::BHHH_APPROX_ANALYTICAL_GRAD || method == HessianCalcMethod::BHHH_APPROX_NUM_APPROX_GRAD){
//                     arma::dmat grad = this->gradientDerivCrossHalfNormal(params, y, x, zu.value(), zv.value(), this->s);
//                     // outer product of the gradient
//                     hessOut = -(grad.t() * grad);
//                 } else if (method == HessianCalcMethod::ANALYTICAL) {
//                     // analytical hessian matrix
//                     hessOut = this->hessianDerivCrossHalfNormal(params, y, x, zu.value(), zv.value(), this->s);
//                 } else if (method == HessianCalcMethod::NUM_APPROX){
//                     // numerically approximated hessian matrix
//                     std::function<double(const arma::dcolvec&)> lmda = [this](const arma::dcolvec& params){
//                         return this->operator()(params);
//                     };
//                     hessOut = finitediff::calculateFiniteHessian(params, lmda, accuracy);
//                 }
//             } else if (dataObj.getModelType() == ESASfaModelType::CROSS_TNORM_ZUIT){
//                 // check mu, zu, zv are present
//                 if (!mu || !zu || !zv){
//                     throw std::invalid_argument("mu, zu and zv must be present for truncated normal distribution");
//                 }
//                 // if BHHH approximation, calculate the gradient (whether analytically, or numerically)
//                 if (method == HessianCalcMethod::BHHH_APPROX_ANALYTICAL_GRAD || method == HessianCalcMethod::BHHH_APPROX_NUM_APPROX_GRAD){
//                     arma::dmat grad = this->gradientDerivCrossTruncNormal(params, y, x, mu.value(), zu.value(), zv.value(), this->s);
//                     // outer product of the gradient
//                     hessOut = -(grad.t() * grad);
//                 } else if (method == HessianCalcMethod::BHHH_APPROX_ANALYTICAL_GRAD) {
//                     // analytical hessian matrix
//                     hessOut = this->hessianDerivCrossTruncNormal(
//                         params, y, x, mu.value(), zu.value(), zv.value(), this->s
//                     );
//                 } else if (method == HessianCalcMethod::BHHH_APPROX_NUM_APPROX_GRAD){
//                     // numerically approximated hessian matrix
//                     std::function<double(const arma::dcolvec&)> lmda = [this](const arma::dcolvec& params){
//                         return this->operator()(params);
//                     };
//                     hessOut = finitediff::calculateFiniteHessian(params, lmda, accuracy);
//                 }
//             } else {
//                 // unsupported model type
//                 throw std::invalid_argument("Unsupported model type");
//             }
//             return hessOut;
//         }
//     );
//     return hess;
// }

/// Gradient and hessian matrix together
void ESASfaCross::gradHess(
    const arma::dcolvec& params,
    const bool exceptNotFinite,
    arma::dmat* gradOut,
    arma::dmat* hessOut,
    arma::dmat* jacOut
) const
{
    // dereference pointer to data object
    ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
    // get the model type
    ESASfaModelType mT = dataObj.getModelType();
    if (
        (mT != ESASfaModelType::CROSS_HNORM_ZUIT) &&
        (mT != ESASfaModelType::CROSS_TNORM_ZUIT)
    ) {
        throw std::runtime_error("Model type not recognised '" + ESAEnums::strForModelType(mT) + "'");
    }
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    HessianCalcMethod hessMethod = globalOptimParams->hessianMethod;
    bool threaded = globalOptimParams->optimThreaded;
    // arma::dmat g, h;
    auto innerFn = [this, mT, params, hessMethod](
        const arma::dcolvec& y,
        const arma::dmat& x,
        const std::optional<arma::dmat>& mu,
        const std::optional<arma::dmat>& zu,
        const std::optional<arma::dmat>& zv,
        arma::dmat* g,
        arma::dmat* h
    ){
        // both require zv, zu
        if (!zu || !zv) throw std::invalid_argument("'zu' and 'zv' are missing, but expected");
        if (mT == ESASfaModelType::CROSS_HNORM_ZUIT) {
            // half normal inefficiency component
            *g = this->gradientDerivCrossHalfNormal(params, y, x, zu.value(), zv.value(), this->s);
        } else if (mT == ESASfaModelType::CROSS_TNORM_ZUIT) {
            // truncated normal inefficiency component
            // check for mu presence
            if (!mu) throw std::invalid_argument("'mu' is missing, but required for trunc-normal");
            *g = this->gradientDerivCrossTruncNormal(params, y, x, mu.value(), zu.value(), zv.value(), this->s);
        }
        // calculate hessian matrix
        if (
            (hessMethod == HessianCalcMethod::BHHH_APPROX_ANALYTICAL_GRAD) ||
            (hessMethod == HessianCalcMethod::BHHH_APPROX_NUM_APPROX_GRAD)
        ) {
            // BHHH approximation using the analytical gradient (irrespective of whether or not user specifies)
            *h = ((*g).t() * (*g));
        } else if (hessMethod == HessianCalcMethod::ANALYTICAL) {
            // analytical hessian matrix
            if (mT == ESASfaModelType::CROSS_HNORM_ZUIT) {
                // half-normal inefficiency component
                *h = this->hessianDerivCrossHalfNormal(params, y, x, zu.value(), zv.value(), this->s);
            } else if (mT == ESASfaModelType::CROSS_TNORM_ZUIT) {
                // truncated-normal inefficiency component
                *h = this->hessianDerivCrossTruncNormal(params, y, x, mu.value(), zu.value(), zv.value(), this->s);
            }
        } else if (
            (hessMethod == HessianCalcMethod::NUM_APPROX) ||
            (hessMethod == HessianCalcMethod::NUM_APPROX_WITH_NUM_APPROX_GRAD)
        ) {
            // also ignore if should use a numerically approximated gradient - since have the analytical ones setup
            std::function<double(const arma::dcolvec&)> lmda = [this](const arma::dcolvec& p) {
                return this->operator()(p);
            };
            *h = finitediff::calculateFiniteHessian(params, lmda, 3);
        } else {
            throw std::invalid_argument("Enum in HessianCalcMethod has not been implemented");
        }
    };
    // ignore whether or not using threading...
    arma::dmat e1, e2;
    dataObj.data_callable(innerFn, &e1, &e2);
    // 
    int nobs = dataObj.getNobs();
    // calculate the average gradient
    arma::dmat grad = esamath::colSum(e1) / nobs;
    // calculate the average hessian
    arma::dmat hess = e2 / nobs;
    if (
        (hessMethod == HessianCalcMethod::BHHH_APPROX_ANALYTICAL_GRAD) ||
        (hessMethod == HessianCalcMethod::BHHH_APPROX_NUM_APPROX_GRAD)
    ) {
        hess = -hess;
    }
    if (gradOut) *gradOut = grad;
    if (hessOut) *hessOut = hess;
    if (jacOut) *jacOut = e1;
}

/// Starting values for ML estimation
arma::dcolvec ESASfaCross::startingValues() const
{
    /// dereference ptr to data object
    ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
    // get matricies for zu, zv 
    if (!dataObj.getZu() || !dataObj.getZv()){
        throw std::invalid_argument("zu and zv must be present (both trunc & half normal)");
    }
    arma::dmat zu = dataObj.getZu().value();
    arma::dmat zv = dataObj.getZv().value();
    // get number of columns for x, mu, zu, zv
    int nX = dataObj.getX().n_cols, nMu = 0, nZu = zu.n_cols, nZv = zv.n_cols;
    // start off with estimating OLS by regressing x on y
    arma::dcolvec b_x;
    try {
        b_x = arma::pinv(dataObj.getX()) * dataObj.getY();
    } catch (const std::exception& e){
        ESALogger::logger()->warn("SfaCross - b_x error inverting matrix to calculate starting values: {}", e.what());
        // set b_x to 0
        b_x = arma::dmat(dataObj.getX().n_cols, 1, arma::fill::zeros);
    }
    // calculate residuals
    arma::dmat res = dataObj.getY() - dataObj.getX() * b_x;
    // calculate m2, m3
    double m2 = arma::accu(arma::pow(res, 2.0)) / res.n_rows;
    double m3 = arma::accu(arma::pow(res, 3.0)) / res.n_rows;
    double varu = 0.0, varv = 0.0;
    if ((this->s * m3) > 0){
        varu = std::pow(std::abs((this->s * m3 * std::sqrt(M_PI/2.0) / (1.0 - 4.0/M_PI))), (2.0/3.0));
    } else {
        varu = std::pow((this->s * m3 * std::sqrt(M_PI/2.0) / (1.0 - 4.0/M_PI)), (2.0/3.0));
    }
    if (m2 < ((M_PI - 2.0)/M_PI * varu)){
        varv = std::abs((m2 - (1.0 - 2.0/M_PI) * varu));
    } else {
        varv = m2 - (1.0 - 2.0/M_PI) * varu;
    }
    // calculate dependent variable for u and v
    arma::dmat depU;
    // depending on whether or not there are any determinants
    //if (nZu > 1){
    // depU = (1.0/2.0) * arma::log(arma::pow(((arma::pow(res, 2.0) - varv) * M_PI / (M_PI - 2.0)), 2.0));
    //} else {
    //    depU = arma::dmat(res.n_rows, 1, arma::fill::value(std::log(varu)));
    //}
    if (nZu > 1) {
        depU = (1.0/2.0) * arma::log(arma::pow(((arma::pow(res, 2.0) - varv) * M_PI / (M_PI - 2.0)), 2.0));
    } else {
        depU = arma::dmat(res.n_rows, 1, arma::fill::value(varu));
    }
    // again, depending on whether or not there are ny determiannts
    arma::dmat depV;
    //if (nZv > 1){
    // depV = (1.0/2.0) * arma::log(arma::pow((arma::pow(res, 2.0) - (1.0 - 2.0/M_PI) * varu), 2.0));
    depV = (1.0/2.0) * arma::log(arma::pow((arma::pow(res, 2.0) - (1.0 - 2.0/M_PI) * varu), 2.0));
    //} else {
    //    depV = arma::dmat(res.n_rows, 1, arma::fill::value(std::log(varv)));
    //}
    // if (nZv > 1) {
    //     depV = (1.0/2.0) * arma::log(arma::pow((arma::pow(res, 2.0) - (1.0 - 2.0/M_PI) * varu), 2.0));
    // } else {
    //     depV = arma::dmat(res.n_rows, 1, arma::fill::value(std::exp(varv)));
    // }
    // regress zu on depU
    arma::dcolvec b_zu;
    try {
        b_zu = arma::pinv(zu) * depU;
    } catch (const std::exception& e){
        ESALogger::logger()->warn("SfaCross - b_zu error inverting matrix to calculate starting values: {}", e.what());
        // set b_zu to 0
        b_zu = arma::dmat(zu.n_cols, 1, arma::fill::value(-0.1));
    }
    // replace any zeros with -0.1
    esautils::replaceValuesPrecision<double>(b_zu, 0.0, -0.1);
    // regress zv on depV
    arma::dcolvec b_zv;
    try {
        b_zv = arma::pinv(zv) * depV;
    } catch (const std::exception& e){
        ESALogger::logger()->warn("SfaCross - b_zv error inverting matrix to calculate starting values: {}", e.what());
        // set b_zv to 0.1
        b_zv = arma::dmat(zv.n_cols, 1, arma::fill::value(-0.1));
    }
    // replace any zeros with -0.1
    esautils::replaceValuesPrecision<double>(b_zv, 0.0, -0.1);
    // if the model is truncated normal, also calculate for mu
    arma::dcolvec b_mu;
    if (dataObj.getModelType() == ESASfaModelType::CROSS_TNORM_ZUIT){
        // get mu from data object
        if (!dataObj.getMu()){
            throw std::invalid_argument("mu must be present for truncated normal distribution");
        }
        arma::dmat mu = dataObj.getMu().value();
        nMu = mu.n_cols;
        // regress mu on residuals
        try {
            b_mu = arma::pinv(mu) * res;
        } catch (const std::exception& e){
            ESALogger::logger()->warn("SfaCross - b_mu error inverting matrix to calculate starting values: {}", e.what());
            // set b_mu to -0.1
            b_mu = arma::dmat(mu.n_cols, 1, arma::fill::value(-0.1));
        }
        // replace any zeros with -0.1
        esautils::replaceValuesPrecision<double>(b_mu, 0.0, -0.1);
    }
    // empty vector to store starting values
    arma::dcolvec startVals(nX + nMu + nZu + nZv);
    // can add the starting values for x to the startVals
    startVals.rows(0, nX - 1) = b_x;
    if (dataObj.getModelType() == ESASfaModelType::CROSS_TNORM_ZUIT){
        // set mu in the starting values vector
        startVals.rows(nX, nX + nMu - 1) = b_mu;
    }
    // set zu in the starting values vector
    startVals.rows(nX + nMu, nX + nMu + nZu - 1) = b_zu;
    // set zv in the starting values vector
    startVals.rows(nX + nMu + nZu, nX + nMu + nZu + nZv - 1) = b_zv;
    // adjust the intercept in b_x component
    startVals(0) = startVals(0) + (this->s * std::sqrt(varu * 2.0/M_PI));
    return startVals;
 }

/// ------------------- Half Normal Distribution -------------------

/// density of the half-normal distribution
arma::dmat ESASfaCross::densityCrossHalfNormal(
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& zu,
    const arma::dmat& zv,
    const int s
) const
{
    /// dereference ptr to data object
    ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
    // check s either 1, -1
    if (s != -1 && s != 1){
        throw std::invalid_argument("s must be either -1 or 1");
    }
    // get parameters for x, zu, zv
    arma::dcolvec b_x = dataObj.paramX(par);
    arma::dcolvec b_zu = dataObj.paramZu(par);
    arma::dcolvec b_zv = dataObj.paramZv(par);
    // calculate xb
    arma::dmat xb = x * b_x;
    // calculate sigma2u
    arma::dmat sigma2u = esautils::processSig2Term(b_zu, zu);
    arma::dmat sigmau = arma::sqrt(sigma2u);
    // calculate sigma2v
    arma::dmat sigma2v = esautils::processSig2Term(b_zv, zv);
    arma::dmat sigmav = arma::sqrt(sigma2v);
    // calculate sigma2
    arma::dmat sigma2 = sigma2u + sigma2v;
    arma::dmat sigma = arma::sqrt(sigma2);
    // calculate lambda
    arma::dmat lambda = sigmau / sigmav;
    // calculate eps
    arma::dmat eps = y - xb;
    // calculate c1, c2
    arma::dmat c1In = eps / sigma;
    arma::dmat c2In = - this->s * (eps % lambda) / sigma;
    // calculate the density
    arma::dmat ld = (2.0 / sigma) % arma::normpdf(c1In, 0.0, 1.0) % arma::normcdf(c2In, 0.0, 1.0);
    return ld;
}

/// Log-likelihood of the half-normal distribution
double ESASfaCross::logLikelihoodCrossHalfNormal(
    const arma::dcolvec& par
) const
{
    /// dereference ptr to data object
    ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
    arma::dmat lls = dataObj.data_callable(
        [this, &par](
            const arma::dcolvec& y,
            const arma::dmat& x,
            const std::optional<arma::dmat>& mu,
            const std::optional<arma::dmat>& zu,
            const std::optional<arma::dmat>& zv
        ){
            // check zu, zv are present
            if (!zu || !zv){
                throw std::invalid_argument("zu and zv must be present for half-normal distribution");
            }
            // calculate density of the half-normal distribution
            arma::dmat dens = this->densityCrossHalfNormal(par, y, x, zu.value(), zv.value(), this->s);
            // take logs of the density
            arma::dmat logDens = arma::log(dens);
            return logDens;
        }
    );
    return arma::accu(lls);
}

/// Analytical gradient of the half-normal distribution
/// based off the code from SfaR package - and Coelli 1995.
/// https://github.com/hdakpo/sfaR/blob/main/R/sfacross-hnormal.R
arma::dmat ESASfaCross::gradientDerivCrossHalfNormal(
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& zu,
    const arma::dmat& zv,
    const int s
) const
{
    // dereference ptr to data object
    ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
    int prodCost = this->s;
    // get parameters for x, zu, zv
    if (!dataObj.getZu() || !dataObj.getZv()){
        throw std::invalid_argument("zu and zv must be present for half-normal distribution");
    }
    arma::dcolvec b_x = dataObj.paramX(par);
    arma::dcolvec b_zu = dataObj.paramZu(par);
    arma::dcolvec b_zv = dataObj.paramZv(par);
    unsigned int nX = b_x.n_rows, nZu = b_zu.n_rows, nZv = b_zv.n_rows, nobs = y.n_rows;
    // calculate xb
    arma::dmat xb = x * b_x;
    arma::dmat wu = esautils::processSig2Term(b_zu, zu, true);
    arma::dmat wv = esautils::processSig2Term(b_zv, zv, true);
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
    arma::dmat gradOut(dataObj.getNobs(), (nX + nZu + nZv));
    arma::dmat gradX = esautils::sweepMatrixElementwise(x, 1, multiX, "*");
    arma::dmat gradZu = esautils::sweepMatrixElementwise(zu, 1, multiZu, "*");
    arma::dmat gradZv = esautils::sweepMatrixElementwise(zv, 1, multiZv, "*");
    // set the columns of the gradient matrix
    gradOut.submat(arma::span(0, nobs - 1), arma::span(0, nX - 1)) = gradX;
    gradOut.submat(arma::span(0, nobs - 1), arma::span(nX, nX + nZu - 1)) = gradZu;
    gradOut.submat(arma::span(0, nobs - 1), arma::span(nX + nZu, nX + nZu + nZv - 1)) = gradZv;
    // final sweep
    return gradOut;
}

/// Hessian matrix for the half-normal distribution
/// Originally by https://github.com/hdakpo/sfaR/blob/main/R/sfacross-hnormal.R
arma::dmat ESASfaCross::hessianDerivCrossHalfNormal(
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& zu,
    const arma::dmat& zv,
    const int s
) const
{
    // dereference ptr to data object
    ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
    int prodCost = this->s;
    // get parameters for x, zu, zv
    if (!dataObj.getZu() || !dataObj.getZv()){
        throw std::invalid_argument("zu and zv must be present for half-normal distribution");
    }
    
    arma::dcolvec b_x = dataObj.paramX(par);
    arma::dcolvec b_zu = dataObj.paramZu(par);
    arma::dcolvec b_zv = dataObj.paramZv(par);
    unsigned int nX = b_x.n_rows, nZu = b_zu.n_rows, nZv = b_zv.n_rows, nobs = y.n_rows;
    // weights matrix - set to 1
    arma::dcolvec wHvar(nobs, 1, arma::fill::ones);
    // calculate xb
    arma::dmat xb = x * b_x;
    // inefficiency & statistical noise components
    arma::dmat wu = esautils::processSig2Term(b_zu, zu, true);
    arma::dmat wv = esautils::processSig2Term(b_zv, zv, true);
    //exp(Wv) is wv here; exp(Wu) is wu here;
    arma::dmat sigma2 = wu + wv;
    arma::dmat sigma = arma::sqrt(sigma2);
    // calculate residual
    arma::dmat eps = y - xb;
    arma::dmat wvsq = wv / sigma2;
    arma::dmat mustar = - wu % (prodCost * (eps / sigma2));
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
    // create the hessian matrix
    arma::dmat hessll = arma::dmat(nX + nZu + nZv, nX + nZu + nZv);
    // ---- xvars x xvars component ----
    // arma::dmat hessXXvarsMulti = (
    //     std::pow(prodCost, 2.0) * wHvar % (dmusigwu % wu %
    //     (prodCost * (eps)/(wv % pmusig2) - dmusig/arma::pow(pmusig2, 2.0))/(sigma2) - 1.0)/(sigma2)
    // );
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
    // ---- xvars x zu component ----
    // arma::dmat hessXZuMulti = (
    //     prodCost * wHvar % (
    //         sigx6 + prodCost * (
    //             (sigx4) % depsi -
    //             (sig2wu) % dmusigwu % (prodCost * eps/wv - dmusig/pmusig2) /
    //             (sigma2 % pmusig)
    //         ) % eps
    //     ) % wu
    // );
    arma::dmat hessXZuMulti = (
        prodCost * wHvar % (
            sigx6 + prodCost * (
                (sigx4) % depsi - (sig2wu) % dmusigwu % (prodCost * (eps)/wv - dmusig/(pmusig2))/((sigma2) % pmusig)
            ) % (eps)
        ) % wu
    );
    arma::dmat hessXZuLHS = esautils::sweepMatrixElementwise(x, 1, hessXZuMulti, "*");
    arma::dmat hessXZu = hessXZuLHS.t() * zu;
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
    arma::dmat hessXZv = hessXZvLHS.t() * zv;
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
    arma::dmat hessZuZuLHS = esautils::sweepMatrixElementwise(zu, 1, hessZuZuMulti, "*");
    arma::dmat hessZuZu = hessZuZuLHS.t() * zu;
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
    arma::dmat hessZvZvLHS = esautils::sweepMatrixElementwise(zv, 1, hessZvZvMulti, "*");
    arma::dmat hessZvZv = hessZvZvLHS.t() * zv;
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
    arma::dmat hessZuZvLHS = esautils::sweepMatrixElementwise(zu, 1, hessZuZvMulti, "*");
    arma::dmat hessZuZv = hessZuZvLHS.t() * zv;
    hessll.submat(arma::span(nX, nX + nZu - 1), arma::span(nX + nZu, nX + nZu + nZv - 1)) = hessZuZv;
    hessll.submat(arma::span(nX + nZu, nX + nZu + nZv - 1), arma::span(nX, nX + nZu - 1)) = hessZuZv.t();
    return hessll;
}

//// ------------------- Truncated Normal Distribution ----------------

/// density of the truncated normal distribution
arma::dmat ESASfaCross::densityCrossTruncNormal(
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& mu,
    const arma::dmat& zu,
    const arma::dmat& zv,
    const int s
) const
{
    /// dereference ptr to data object
    ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
    // check s either 1, -1
    if (s != -1 && s != 1){
        throw std::invalid_argument("s must be either -1 or 1");
    }
    int prodCost = this->s;
    // get parameters for x, mu, zu, zv
    arma::dcolvec b_x = dataObj.paramX(par);
    arma::dcolvec b_mu = dataObj.paramMu(par);
    arma::dcolvec b_zu = dataObj.paramZu(par);
    arma::dcolvec b_zv = dataObj.paramZv(par);
    // calculate xb
    arma::dmat xb = x * b_x;
    // calculate mu
    arma::dmat wmu = mu * b_mu;
    // calculate sigma2u
    arma::dmat sigma2u = esautils::processSig2Term(b_zu, zu);
    arma::dmat sigmau = arma::sqrt(sigma2u);
    // calculate sigma2v
    arma::dmat sigma2v = esautils::processSig2Term(b_zv, zv);
    arma::dmat sigmav = arma::sqrt(sigma2v);
    // calculate sigma2
    arma::dmat sigma2 = sigma2u + sigma2v;
    arma::dmat sigma = arma::sqrt(sigma2);
    arma::dmat sigma2star = (sigma2v % sigma2u) / sigma2;
    arma::dmat sigmastar = arma::sqrt(sigma2star);
    // calculate eps
    arma::dmat eps = y - xb;
    arma::dmat muDivSigma = wmu / sigmau;
    arma::dmat cdfMuDivSigma = arma::normcdf(muDivSigma, 0.0, 1.0);
    // mustar
    arma::dmat muStarNumer = (wmu % sigma2v) - prodCost * (eps % sigma2u);
    arma::dmat muStar = muStarNumer / sigma2;
    // cdf of mustar/sigstar
    arma::dmat muStarDivSigStar = muStar / sigmastar;
    arma::dmat cdfMuStarDivSigStar = arma::normcdf(muStarDivSigStar, 0.0, 1.0);
    // denominator of density
    arma::dmat denomC2 = cdfMuDivSigma / cdfMuStarDivSigStar;
    arma::dmat denom = sigma % denomC2;
    // numerator of density
    arma::dmat epsmu = eps + wmu;
    arma::dmat espmuDivSigma = epsmu / sigma;
    arma::dmat numer = arma::normpdf(espmuDivSigma, 0.0, 1.0);
    // density
    arma::dmat den = numer / denom;
    return den;
}

/// Log-likelihood of the truncated normal distribution
double ESASfaCross::logLikelihoodCrossTruncNormal(
    const arma::dcolvec& par
) const
{
    /// dereference ptr to data object
    ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
    arma::dmat lls = dataObj.data_callable(
        [this, &par](
            const arma::dcolvec& y,
            const arma::dmat& x,
            const std::optional<arma::dmat>& mu,
            const std::optional<arma::dmat>& zu,
            const std::optional<arma::dmat>& zv
        ){
            // check mu, zu, zv present
            if (!mu || !zu || !zv){
                throw std::invalid_argument("mu, zu and zv must be present for truncated normal distribution");
            }
            // calculate density of the truncated normal distribution
            arma::dmat dens = this->densityCrossTruncNormal(par, y, x, mu.value(), zu.value(), zv.value(), this->s);
            // take logs of the density
            arma::dmat logDens = arma::log(dens);
            return logDens;
        }
    );
    return arma::accu(lls);
}

/// Analytical gradient of the truncated normal distribution
arma::dmat ESASfaCross::gradientDerivCrossTruncNormal(
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& mu,
    const arma::dmat& zu,
    const arma::dmat& zv,
    const int s
) const
{
    /// dereference ptr to data object
    ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
    int prodCost = this->s;
    // get parameters for x, mu, zu, zv
    if (!dataObj.getMu() || !dataObj.getZu() || !dataObj.getZv()){
        throw std::invalid_argument("mu, zu and zv must be present for truncated normal distribution");
    }
    arma::dcolvec b_x = dataObj.paramX(par);
    arma::dcolvec b_mu = dataObj.paramMu(par);
    arma::dcolvec b_zu = dataObj.paramZu(par);
    arma::dcolvec b_zv = dataObj.paramZv(par);
    unsigned int nX = b_x.n_rows, nMu = b_mu.n_rows, nZu = b_zu.n_rows, nZv = b_zv.n_rows, nobs = y.n_rows;
    // calculate xb, wmu, wu, wv
    arma::dmat xb = x * b_x;
    arma::dmat wmu = mu * b_mu;
    arma::dmat wu = esautils::processSig2Term(b_zu, zu);
    arma::dmat wv = esautils::processSig2Term(b_zv, zv);
    // sigma2 
    arma::dmat sigma2 = wu + wv;
    arma::dmat sigma = arma::sqrt(sigma2);
    // residual - epsilon
    arma::dmat eps = dataObj.getY() - xb;
    // mustar 
    arma::dmat mustar = ((wmu % wv) - (prodCost * (wu % eps))) / sigma2;
    // sigmastar
    arma::dmat sigmastarIn = (wu % wv) / sigma2;
    arma::dmat sigmastar = arma::sqrt(sigmastarIn);
    // musig; pdf, cdf of musig
    arma::dmat musig = mustar / sigmastar;
    arma::dmat pmusig = arma::normcdf(musig, 0.0, 1.0);
    arma::dmat dmusig = arma::normpdf(musig, 0.0, 1.0);
    // mustar2, pdf of mustar2
    arma::dmat mustar2 = (wmu + prodCost * eps) / sigma;
    arma::dmat dmustar2 = arma::normpdf(mustar2, 0.0, 1.0);
    arma::dmat dmustar2epsi = dmustar2 % arma::pow((wmu + prodCost*eps), 2.0);
    arma::dmat dmustar2epsix2 = dmustar2epsi / (dmustar2 % sigma2);
    arma::dmat dmustar2epsix3 = (0.5 * dmustar2epsix2) - 0.5 /  sigma2;
    arma::dmat sigx2 = sigma2 % sigmastar;
    arma::dmat sigx3 = ( 0.5 * (1.0 - (wv / sigma2)) % (wu / sigmastar) ) + sigmastar;
    arma::dmat wusq = wu / sigma2;
    arma::dmat sigx4 = ((wmu % wv) - (prodCost * (wu % eps))) / arma::pow(sigx2, 2.0);
    arma::dmat sigx5 = (0.5 * ((1.0 - wusq) % (wv / sigmastar)) + sigmastar) % sigx4;
    arma::dmat sigx6 = sigx5 + prodCost * (eps / sigx2);
    arma::dmat mustar3 = wmu + prodCost * eps;
    arma::dmat wudiv2 = arma::exp((zu * b_zu) / 2.0);
    arma::dmat mudivwu = wmu / wudiv2;
    arma::dmat dmu = arma::normpdf(mudivwu, 0.0, 1.0);
    arma::dmat pmu = arma::normcdf(mudivwu, 0.0, 1.0);
    arma::dmat sigmu = (wmu % sigx2) - (sigx3 % sigx4);
    arma::dmat pmusigx2 = pmusig % sigmastar;
    arma::dmat pmuwu = wudiv2 % pmu;
    // multipliers for x, mu, zu, zv
    arma::dmat multiX = prodCost * ( ((dmusig % (wu / pmusigx2)) + mustar3) / sigma2);
    arma::dmat multiMu = ( (dmusig % (wv / pmusigx2)) - mustar3 / sigma2 ) - (dmu / pmuwu);
    arma::dmat multiZu = ( dmustar2epsix3 - (sigx6 % (dmusig / pmusig)) % wu ) + (0.5 * (wmu % (dmu / pmuwu)));
    arma::dmat multiZv = (dmustar2epsix3 + (dmusig % (sigmu / pmusig))) % wv;
    // apply multiplies to respective matricies
    arma::dmat gradOut(dataObj.getNobs(), (nX + nMu + nZu + nZv));
    arma::dmat gradX = esautils::sweepMatrixElementwise(x, 1, multiX, "*");
    arma::dmat gradMu = esautils::sweepMatrixElementwise(mu, 1, multiMu, "*");
    arma::dmat gradZu = esautils::sweepMatrixElementwise(zu, 1, multiZu, "*");
    arma::dmat gradZv = esautils::sweepMatrixElementwise(zv, 1, multiZv, "*");
    // set the columns of the gradient matrix
    gradOut.submat(arma::span(0, nobs - 1), arma::span(0, nX - 1)) = gradX;
    gradOut.submat(arma::span(0, nobs - 1), arma::span(nX, nX + nMu - 1)) = gradMu;
    gradOut.submat(arma::span(0, nobs - 1), arma::span(nX + nMu, nX + nMu + nZu - 1)) = gradZu;
    gradOut.submat(arma::span(0, nobs - 1), arma::span(nX + nMu + nZu, nX + nMu + nZu + nZv - 1)) = gradZv;
    return gradOut;
}

/// Hessian matrix for the truncated normal distribution
arma::dmat ESASfaCross::hessianDerivCrossTruncNormal(
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& mu,
    const arma::dmat& zu,
    const arma::dmat& zv,
    const int s
) const
{
    throw std::runtime_error("Not implemented hessian for cross-sectional truncated normal");
    // dereference ptr to data object
    // ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
    // int prodCost = this->s;
    // get parameters for x, zu, zv
    // if (!dataObj.getZu() || !dataObj.getZv()){
    //     throw std::invalid_argument("zu and zv must be present for half-normal distribution");
    // }
    // arma::dcolvec b_x = dataObj.paramX(par);
    // arma::dcolvec b_zu = dataObj.paramZu(par);
    // arma::dcolvec b_zv = dataObj.paramZv(par);
    // unsigned int nX = b_x.n_rows, nZu = b_zu.n_rows, nZv = b_zv.n_rows, nobs = y.n_rows;
}