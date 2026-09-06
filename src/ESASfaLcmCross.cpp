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

#include "sfa/ESASfaLcmCross.hpp"
#include "data/ESADataLCM.hpp"
#include "math/esamath.hpp"
#include "utils/esautils.hpp"
#include "utils/log/logs.hpp"

#include <cmath>

// ---- Constructor ----

ESASfaLcmCross::ESASfaLcmCross(
    std::shared_ptr<ESADataPanelLCM> lcmDataPtr,
    const int s
) : ESASfaBase(std::static_pointer_cast<ESADataBase>(lcmDataPtr), static_cast<double>(s)),
    lcmDataPtr(lcmDataPtr)
{
}

// ---- Static Density ----

double ESASfaLcmCross::densityHalfNormal(double eps, double sigma2u, double sigma2v, int s)
{
    double sigma2 = sigma2u + sigma2v;
    double sigma = std::sqrt(sigma2);
    double lambda = std::sqrt(sigma2u) / std::sqrt(sigma2v);
    double c1 = eps / sigma;
    double c2 = -s * eps * lambda / sigma;
    // 2/sigma * phi(c1) * Phi(c2)
    double pdf_val = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * c1 * c1);
    double cdf_val = 0.5 * std::erfc(-c2 / std::sqrt(2.0));
    return (2.0 / sigma) * pdf_val * cdf_val;
}

// ---- Per-observation gradient of log-density (half-normal) ----
// Derivation: f = (2/sigma) * phi(eps/sigma) * Phi(c2)
//   where eps = y - x*beta, sigma2 = sigma2u + sigma2v, lambda = sigmau/sigmav
//   c2 = -s*eps*lambda/sigma
//   sigma2u = exp(zuit*b_zuit), sigma2v = exp(zvit*b_zvit)
// log f = log(2) - 0.5*log(sigma2) - 0.5*eps^2/sigma2 + log(Phi(c2))

arma::rowvec ESASfaLcmCross::gradLogDensityHalfNormal(
    double y_val,
    const arma::rowvec& x_row,
    const arma::rowvec& zuit_row,
    const arma::rowvec& zvit_row,
    const arma::dcolvec& b_x,
    const arma::dcolvec& b_zuit,
    const arma::dcolvec& b_zvit,
    int s
) const
{
    int nX = b_x.n_rows;
    int nZuit = b_zuit.n_rows;
    int nZvit = b_zvit.n_rows;
    int k = nX + nZuit + nZvit;

    double xb = arma::as_scalar(x_row * b_x);
    double eps = y_val - xb;
    double sigma2u = std::exp(arma::as_scalar(zuit_row * b_zuit));
    double sigma2v = std::exp(arma::as_scalar(zvit_row * b_zvit));
    double sigma2 = sigma2u + sigma2v;
    double sigma = std::sqrt(sigma2);
    double sigmau = std::sqrt(sigma2u);
    double sigmav = std::sqrt(sigma2v);
    double lambda = sigmau / sigmav;

    double c2 = -s * eps * lambda / sigma;

    // Mills ratio M = phi(c2)/Phi(c2)
    double phi_c2 = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * c2 * c2);
    double Phi_c2 = 0.5 * std::erfc(-c2 / std::sqrt(2.0));
    double M = (Phi_c2 > 1e-300) ? (phi_c2 / Phi_c2) : 0.0;

    // d log f / d beta = (eps/sigma2 + s*M*lambda/sigma) * x
    double dBeta_scalar = eps / sigma2 + s * M * lambda / sigma;

    // d log f / d b_zu = sigma2u * d(log f)/d(sigma2u)
    // d(log f)/d(sigma2u) = -1/(2*sigma2) + eps^2/(2*sigma2^2) + M * dc2/d(sigma2u)
    // dc2/d(sigma2u) = -s*eps*sigmav / (2*sigma^3*sigmau)
    double dc2_dsigma2u = -s * eps * sigmav / (2.0 * sigma * sigma * sigma * sigmau);
    double dlogf_dsigma2u = -1.0 / (2.0 * sigma2) + eps * eps / (2.0 * sigma2 * sigma2) + M * dc2_dsigma2u;
    double dZu_scalar = sigma2u * dlogf_dsigma2u;

    // d log f / d b_zv = sigma2v * d(log f)/d(sigma2v)
    // dc2/d(sigma2v) = s*eps*lambda/(2*sigma) * (1/sigma2v + 1/sigma2)
    double dc2_dsigma2v = s * eps * lambda / (2.0 * sigma) * (1.0 / sigma2v + 1.0 / sigma2);
    double dlogf_dsigma2v = -1.0 / (2.0 * sigma2) + eps * eps / (2.0 * sigma2 * sigma2) + M * dc2_dsigma2v;
    double dZv_scalar = sigma2v * dlogf_dsigma2v;

    // Assemble gradient row
    arma::rowvec grad(k);
    grad.cols(0, nX - 1) = dBeta_scalar * x_row;
    grad.cols(nX, nX + nZuit - 1) = dZu_scalar * zuit_row;
    grad.cols(nX + nZuit, k - 1) = dZv_scalar * zvit_row;
    return grad;
}

// ---- Log-Likelihood ----

double ESASfaLcmCross::operator()(const arma::dcolvec& params, const bool exceptNotFinite) const
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    unsigned int nClasses = lcmData.getNClasses();
    unsigned int nSeg = lcmData.getNSeg();
    arma::dcolvec segParams = lcmData.paramSeg(params);
    int nobs = lcmData.getNobs();

    const arma::dcolvec& y = lcmData.getY();
    const arma::dmat& x = lcmData.getX();
    const arma::dmat& seg = lcmData.getSeg();

    double ll = 0.0;

    for (int i = 0; i < nobs; i++) {
        // Class probs for observation i
        arma::rowvec z_i = seg.row(i);
        arma::dcolvec pi_i = lcmutils::computeClassProbs(z_i, segParams, nClasses);

        // Per-class densities
        arma::dcolvec logLik_c(nClasses);
        for (unsigned int c = 0; c < nClasses; c++) {
            arma::dcolvec b_x = lcmData.paramX(params, c);
            auto b_zuit_opt = lcmData.paramZuit(params, c);
            auto b_zvit_opt = lcmData.paramZvit(params, c);
            if (!b_zuit_opt || !b_zvit_opt) {
                throw std::invalid_argument("LC-Cross: missing zuit or zvit params");
            }
            arma::dcolvec b_zuit = b_zuit_opt.value();
            arma::dcolvec b_zvit = b_zvit_opt.value();

            double xb = arma::as_scalar(x.row(i) * b_x);
            double eps = y(i) - xb;
            // For cross-sectional: zuit and zvit are from the LCM data
            auto zuitOpt = lcmData.getZuit();
            auto zvitOpt = lcmData.getZvit();
            double sigma2u = std::exp(arma::as_scalar(zuitOpt.value().row(i) * b_zuit));
            double sigma2v = std::exp(arma::as_scalar(zvitOpt.value().row(i) * b_zvit));
            double dens = densityHalfNormal(eps, sigma2u, sigma2v, static_cast<int>(this->s));
            logLik_c(c) = std::log(std::max(dens, 1e-300));
        }

        arma::dcolvec logJoint = arma::log(pi_i) + logLik_c;
        double logLi = lcmutils::logSumExp(logJoint);

        if (exceptNotFinite && !std::isfinite(logLi)) {
            throw std::runtime_error("LC-Cross: log-likelihood not finite for obs " + std::to_string(i));
        }
        ll += logLi;
    }
    return ll;
}

double ESASfaLcmCross::operator()(const arma::dcolvec& params, const arma::Col<int>& subsetIdents) const
{
    return this->operator()(params, false);
}

// ---- Gradient and BHHH Hessian ----

void ESASfaLcmCross::gradHess(
    const arma::dcolvec& params,
    const bool exceptNotFinite,
    arma::dmat* gradOut,
    arma::dmat* hessOut,
    arma::dmat* jacOut
) const
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    unsigned int nClasses = lcmData.getNClasses();
    unsigned int nSeg = lcmData.getNSeg();
    arma::dcolvec segParams = lcmData.paramSeg(params);
    int nobs = lcmData.getNobs();
    int totalParams = params.n_rows;

    int nSegParams = (nClasses - 1) * nSeg;
    int nTransition = lcmData.getNTransition();
    int nX = lcmData.getNX();
    int nZmuit = lcmData.getNZmuit();
    int nZuit = lcmData.getNZuit();
    int nZvit = lcmData.getNZvit();
    int nZvi0 = lcmData.getNZvi0();
    int perClassParams = nX + nZmuit + nZuit + nZvit + nZvi0;

    const arma::dcolvec& y = lcmData.getY();
    const arma::dmat& x = lcmData.getX();
    const arma::dmat& seg = lcmData.getSeg();
    auto zuitOpt = lcmData.getZuit();
    auto zvitOpt = lcmData.getZvit();

    arma::dmat jac(nobs, totalParams, arma::fill::zeros);

    for (int i = 0; i < nobs; i++) {
        arma::rowvec z_i = seg.row(i);
        arma::dcolvec pi_i = lcmutils::computeClassProbs(z_i, segParams, nClasses);

        // Per-class log-likelihoods and gradients
        arma::dcolvec logLik_c(nClasses);
        std::vector<arma::rowvec> classGrads(nClasses);

        for (unsigned int c = 0; c < nClasses; c++) {
            arma::dcolvec b_x = lcmData.paramX(params, c);
            auto b_zuit_opt = lcmData.paramZuit(params, c);
            auto b_zvit_opt = lcmData.paramZvit(params, c);
            arma::dcolvec b_zuit = b_zuit_opt.value();
            arma::dcolvec b_zvit = b_zvit_opt.value();

            double xb = arma::as_scalar(x.row(i) * b_x);
            double eps_i = y(i) - xb;
            double sigma2u = std::exp(arma::as_scalar(zuitOpt.value().row(i) * b_zuit));
            double sigma2v = std::exp(arma::as_scalar(zvitOpt.value().row(i) * b_zvit));
            double dens = densityHalfNormal(eps_i, sigma2u, sigma2v, static_cast<int>(this->s));
            logLik_c(c) = std::log(std::max(dens, 1e-300));

            // Per-class gradient of log f w.r.t. class-c params
            classGrads[c] = gradLogDensityHalfNormal(
                y(i), x.row(i), zuitOpt.value().row(i), zvitOpt.value().row(i),
                b_x, b_zuit, b_zvit, static_cast<int>(this->s)
            );
        }

        // Posteriors
        arma::dcolvec logPi_i = arma::log(pi_i);
        arma::dcolvec tau_i = lcmutils::computePosteriors(logPi_i, logLik_c);

        // Assemble gradient row
        arma::rowvec g_i(totalParams, arma::fill::zeros);

        // Seg gradient: (tau_ic - pi_ic) * z_i for c = 0..C-2
        for (unsigned int c = 0; c < nClasses - 1; c++) {
            double diff = tau_i(c) - pi_i(c);
            g_i.cols(c * nSeg, (c + 1) * nSeg - 1) = diff * z_i;
        }

        // Per-class frontier gradient weighted by posterior
        int nTransTotal = nClasses * (nClasses - 1) * nTransition;
        int classParamStart = nSegParams + nTransTotal;
        for (unsigned int c = 0; c < nClasses; c++) {
            int offset = classParamStart + c * perClassParams;
            // Only fill the [beta | zuit | zvit] portion (cross-sectional has no zvi0, zmuit typically)
            int activeParams = nX + nZuit + nZvit;
            g_i.cols(offset, offset + activeParams - 1) = tau_i(c) * classGrads[c];
        }

        jac.row(i) = g_i;
    }

    arma::dmat grad = esamath::colSum(jac);
    arma::dmat hess = -(jac.t() * jac);

    if (gradOut) *gradOut = grad;
    if (hessOut) *hessOut = hess;
    if (jacOut) *jacOut = jac;
}

void ESASfaLcmCross::gradHess(
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
    this->gradHess(params, false, gradOut, hessOut, jacOut);
}

// ---- Starting Values ----

arma::dcolvec ESASfaLcmCross::startingValues() const
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    unsigned int nClasses = lcmData.getNClasses();
    unsigned int nSeg = lcmData.getNSeg();
    unsigned int nTransition = lcmData.getNTransition();
    int nX = lcmData.getNX();
    int nZmuit = lcmData.getNZmuit();
    int nZuit = lcmData.getNZuit();
    int nZvit = lcmData.getNZvit();
    int nZvi0 = lcmData.getNZvi0();
    int perClassParams = nX + nZmuit + nZuit + nZvit + nZvi0;
    int nTransTotal = nClasses * (nClasses - 1) * nTransition;
    int totalParams = (nClasses - 1) * nSeg + nTransTotal + nClasses * perClassParams;

    arma::dcolvec startVals(totalParams, arma::fill::zeros);

    // OLS for beta starting values
    const arma::dcolvec& y = lcmData.getY();
    const arma::dmat& x = lcmData.getX();
    arma::dcolvec b_ols = arma::solve(x.t() * x, x.t() * y);

    int classParamStart = (nClasses - 1) * nSeg + nTransTotal;
    for (unsigned int c = 0; c < nClasses; c++) {
        int offset = classParamStart + c * perClassParams;
        // Frontier: OLS + perturbation
        double perturbation = 0.1 * (static_cast<double>(c) - static_cast<double>(nClasses - 1) / 2.0);
        arma::dcolvec classStart(perClassParams, arma::fill::zeros);
        classStart.rows(0, nX - 1) = b_ols * (1.0 + perturbation);
        // Variance params start at log(0.1) = small initial variances
        if (nZuit > 0) classStart(nX + nZmuit) = std::log(0.1);
        if (nZvit > 0) classStart(nX + nZmuit + nZuit) = std::log(0.1);
        startVals.rows(offset, offset + perClassParams - 1) = classStart;
    }

    return startVals;
}

// ---- Sigma Parameters ----

ESASigmaParams ESASfaLcmCross::getSigmaParams(const arma::dcolvec& par) const
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    // Use class 0 for representative sigma params
    auto b_zuit_opt = lcmData.paramZuit(par, 0);
    auto b_zvit_opt = lcmData.paramZvit(par, 0);
    arma::dcolvec b_zuit = b_zuit_opt.value_or(arma::dcolvec({0.0}));
    arma::dcolvec b_zvit = b_zvit_opt.value_or(arma::dcolvec({0.0}));

    auto zuitOpt = lcmData.getZuit();
    auto zvitOpt = lcmData.getZvit();

    arma::dmat s2uit = esautils::processSig2Term(b_zuit, zuitOpt.value());
    arma::dmat s2vit = esautils::processSig2Term(b_zvit, zvitOpt.value());

    return ESASigmaParams(s2uit, s2vit);
}

double ESASfaLcmCross::getN() const
{
    return static_cast<double>(lcmDataPtr->getNobs());
}
