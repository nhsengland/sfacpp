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

#include "sfa/ESASfaLcTre.hpp"
#include "data/ESADataPanel.hpp"
#include "data/ESADataLCM.hpp"
#include "math/esamath.hpp"
#include "math/esandist.hpp"
#include "math/GaussHermite.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "utils/globalstate.hpp"
#include "utils/ThreadContext.hpp"
#include "thread_cache/ESASfaTreBaseTC.hpp"
#include "regression/ESARandEff.hpp"
#include "utils/kmeans.hpp"
#include "optim/ESAGlobalOptimParams.hpp"
#include "utils/esautils.hpp"

// AGHQ helpers
namespace {

/**
 * @brief Evaluate log ∏_t f(y_it | a_t = σ_vi0(t)·vStd, θ_c) for the half-normal TRE model.
 *
 * replicates the per-period density formula from panelDensityHalfNormal for a single
 * scalar draw vStd of the standardized random effect v_i0 ~ N(0,1).
 *
 * @param vStd standardized draw: actual v_i0 = σ_vi0(t) * vStd  (drawn from N(0,1))
 * @param nT number of periods for this firm
 * @param s +1 production, -1 cost
 */
static double panelLogLikHNormAt(
    double vStd,
    int nT, double s,
    const arma::dcolvec& y,
    const arma::dmat& x,
    const arma::dmat& zuit,
    const arma::dmat& zvit,
    const arma::dmat& zvi0,
    const arma::dcolvec& b_x,
    const arma::dcolvec& b_zuit,
    const arma::dcolvec& b_zvit,
    const arma::dcolvec& b_zvi0
)
{
    boost::math::normal_distribution<double> stdNorm(0.0, 1.0);
    double logL = 0.0;
    for (int t = 0; t < nT; ++t) {
        // calculate scalars for each of the main components in the model (xb, sigmas)
        double xb = arma::dot(x.row(t), b_x);
        double sig2uit = std::exp(arma::dot(zuit.row(t), b_zuit));
        double sig2vit = std::exp(arma::dot(zvit.row(t), b_zvit));
        double sig2vi0t = std::exp(arma::dot(zvi0.row(t), b_zvi0));
        // calculate derived variables from those scalars
        double siguit = std::sqrt(sig2uit);
        double sigvit = std::sqrt(sig2vit);
        double sigvi0t = std::sqrt(sig2vi0t);
        double sig2 = sig2uit + sig2vit;
        double sig = std::sqrt(sig2);
        double lambda = siguit / sigvit;
        // the random effect
        double a_t = sigvi0t * vStd;
        double eps_t = y(t) - a_t - xb;
        double c1 = eps_t / sig;
        double c2 = -s * eps_t * lambda / sig;
        // calculate density
        double pdf_val = boost::math::pdf(stdNorm, c1);
        double cdf_val = boost::math::cdf(stdNorm, c2);
        double density = 2.0 * pdf_val * std::max(cdf_val, 1e-300) / sig;
        logL += std::log(std::max(density, 1e-300));
    }
    return logL;
}

/**
 * @brief Find AGHQ posterior mode and standard deviation for one firm, one class.
 *
 * maximize the unnormalized log-posterior:
 *   log q(v) = log ∏_t f(y_it | σ_vi0(t)·v, θ_c) − v²/2
 * where v is the standardized v_i0 ~ N(0,1).
 *
 * use Newton's method with central finite differences; the clamped step prevents
 * overshooting; iteration stops when |gradient| < 1e-8 or the Hessian is not negative-definite.
 *
 * @return {v_star, sigma_star}  where sigma_star = 1/sqrt(posterior_precision), floored at 0.1.
 */
static std::pair<double, double> aghqModeAndSigma(
    int nT, double s,
    const arma::dcolvec& y,
    const arma::dmat& x,
    const arma::dmat& zuit,
    const arma::dmat& zvit,
    const arma::dmat& zvi0,
    const arma::dcolvec& b_x,
    const arma::dcolvec& b_zuit,
    const arma::dcolvec& b_zvit,
    const arma::dcolvec& b_zvi0,
    int maxIter   = 10,
    double delta  = 1e-3
)
{
    double v = 0.0;
    double h_final = -1.0; // posterior curvature (should be negative)

    for (int iter = 0; iter < maxIter; ++iter) {
        double lp = (
            panelLogLikHNormAt(
                v + delta, nT, s, y, x, zuit, zvit, zvi0,
                b_x, b_zuit, b_zvit, b_zvi0
            ) - 0.5 * (v + delta) * (v + delta)
        );
        double l0 = (
            panelLogLikHNormAt(
                v, nT, s, y, x, zuit, zvit, zvi0,
                b_x, b_zuit, b_zvit, b_zvi0
            ) - 0.5 * v * v
        );
        double lm = (
            panelLogLikHNormAt(
                v - delta, nT, s, y, x, zuit, zvit, zvi0,
                b_x, b_zuit, b_zvit, b_zvi0
            ) - 0.5 * (v - delta) * (v - delta)
        );
        // central finite differences for first and second order derivative
        double g = (lp - lm) / (2.0 * delta); // ∂ log q / ∂v
        double h = (lp - 2.0 * l0 + lm) / (delta * delta); // ∂² log q / ∂v²
        h_final = h;
        // check if convergence
        if (h >= 0.0 || std::abs(g) < 1e-8) break;
        double step = g / h; // Newton step (h < 0, so this heads toward the mode)
        if (step >  2.0) step =  2.0;
        if (step < -2.0) step = -2.0;
        v -= step;
    }
    // posterior precision κ = −h - floor to keep σ* finite (since div thru h)
    double kappa = std::max(-h_final, 0.01);
    double sigma_star = 1.0 / std::sqrt(kappa);
    // floor sigma_star to avoid nodes become essentially identical
    sigma_star = std::max(sigma_star, 0.1);
    return {v, sigma_star};
}

} // anon namespace

// ---- Constructors ----

ESASfaLcTre::ESASfaLcTre(
    std::shared_ptr<ESADataPanelLCM> lcmDataPtr,
    std::shared_ptr<ESADataPanel> helperDataPtr,
    const int s,
    const int nsim,
    const int seed,
    const HaltonSettings hsetting
) : ESASfaTreGreene(helperDataPtr, s, nsim, seed, hsetting),
    lcmDataPtr(lcmDataPtr)
{
}

ESASfaLcTre::ESASfaLcTre(
    std::shared_ptr<ESADataPanelLCM> lcmDataPtr,
    std::shared_ptr<ESADataPanel> helperDataPtr,
    const int s,
    const int nsim,
    const int seed,
    const std::shared_ptr<arma::dmat> haltonDrawPtr,
    const HaltonSettings hsetting
) : ESASfaTreGreene(helperDataPtr, s, nsim, seed, haltonDrawPtr, hsetting),
    lcmDataPtr(lcmDataPtr)
{
}

// ---- log-likelihood ----

double ESASfaLcTre::operator()(const arma::dcolvec& params, const bool exceptNotFinite) const
{
    // this is still effectively MSL not quadrature - but only using EM won't route
    // here - that is via the ESASfaLCTreEM class.
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    arma::dmat& haltonMat = *haltonDraws;
    unsigned int nClasses = lcmData.getNClasses();
    arma::dcolvec segParams = lcmData.paramSeg(params);
    ESASfaModelType mT = lcmData.getModelType();

    ESASfaModelType innerMT;
    if (mT == ESASfaModelType::LC_TRE_HNORM) {
        innerMT = ESASfaModelType::TRE_HNORM_ZUIT;
    } else if (mT == ESASfaModelType::LC_TRE_TNORM) {
        innerMT = ESASfaModelType::TRE_TNORM_ZUIT;
    } else {
        throw std::runtime_error("Unsupported LC-TRE model type");
    }

    bool threaded = ESAGlobalOptimParams::GetInstance()->optimThreaded;
    unsigned int nFirms = static_cast<unsigned int>(lcmData.getNids());
    arma::dvec llVec(nFirms, arma::fill::zeros);

    auto inner = [this, &params, &haltonMat, &segParams, &nClasses, &innerMT, &exceptNotFinite, &llVec](
        const unsigned int idx,
        const auto& seg,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zvi0,
        const auto& transition
    ) {
        ESADataPanelLCM& lcmData = *this->lcmDataPtr;
        arma::rowvec z_i = arma::conv_to<arma::rowvec>::from(seg.row(0));
        // class probabilities
        arma::dcolvec pi_i = lcmutils::computeClassProbs(z_i, segParams, nClasses);
        arma::dcolvec logLik_c(nClasses);
        // iterate thru latent classes
        for (unsigned int c = 0; c < nClasses; c++) {
            arma::dcolvec par_c = lcmutils::buildClassParamVec(lcmData, params, c);
            arma::dmat llOut(1, 1);
            if (!zuit || !zvit || !zvi0) {
                throw std::invalid_argument("LC-TRE: missing zuit, zvit, or zvi0");
            }
            // log-likelihood score from TRE for this class
            this->operatorInner(
                par_c, innerMT, haltonMat, idx,
                y, x, zmuit,
                zuit.value(), zvit.value(), zvi0.value(),
                true, &llOut
            );
            logLik_c(c) = llOut(0, 0);
        }
        // joint likelihood
        arma::dcolvec logJoint = arma::log(pi_i) + logLik_c;
        double logLi = lcmutils::logSumExp(logJoint);
        // raise exception if the likelihood isn't finite
        if (exceptNotFinite && !std::isfinite(logLi)) {
            throw std::runtime_error("LC-TRE: log-likelihood not finite for firm " + std::to_string(idx));
        }
        // add log like to column vector of log likelihood scores
        llVec(idx) = logLi;
        arma::dmat dummy(1, 1, arma::fill::zeros);
        return dummy;
    };
    lcmData.dataCallable(inner, nullptr, nullptr, false, false, threaded);
    return arma::accu(llVec);
}

double ESASfaLcTre::operator()(const arma::dcolvec& params, const arma::Col<int>& subsetIdents) const
{
    return this->operator()(params, false);
}

// ---- Gradient, Louis analytical Hessian, and BHHH Jacobian ----
//
// when hessOut is non-null, computes the observed-data Hessian via the
// Louis (1982) identity:
//
//   H_obs = Σ_i { Σ_c τ_ic H_ic_expanded
//               − Σ_c τ_ic s_ic s_ic^T
//               + ĝ_i ĝ_i^T }
// where s_ic is the full-parameter score for firm i under class c and ĝ_i is the
// observed-data score (= Σ_c τ_ic s_ic).
//
// BHHH approximation (−JᵀJ) returned in jacOut for sandwich SEs.
void ESASfaLcTre::gradHess(
    const arma::dcolvec& params,
    const bool exceptNotFinite,
    arma::dmat* gradOut,
    arma::dmat* hessOut,
    arma::dmat* jacOut
) const
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    arma::dmat& haltonMat = *haltonDraws;
    unsigned int nClasses = lcmData.getNClasses();
    unsigned int nSeg = lcmData.getNSeg();
    unsigned int nTransition = lcmData.getNTransition();
    arma::dcolvec segParams = lcmData.paramSeg(params);
    ESASfaModelType mT = lcmData.getModelType();

    ESASfaModelType innerMT;
    if (mT == ESASfaModelType::LC_TRE_HNORM) {
        innerMT = ESASfaModelType::TRE_HNORM_ZUIT;
    } else if (mT == ESASfaModelType::LC_TRE_TNORM) {
        innerMT = ESASfaModelType::TRE_TNORM_ZUIT;
    } else {
        throw std::runtime_error("Unsupported LC-TRE model type");
    }
    if (nTransition != 0) {
        throw std::invalid_argument("LC-TRE model does not support transition parameters; use LC-Markov-TRE instead");
    }
    int totalParams = params.n_rows;
    int nSegParams  = static_cast<int>((nClasses - 1) * nSeg);
    int nX = lcmData.getNX();
    int nZmuit = lcmData.getNZmuit();
    int nZuit = lcmData.getNZuit();
    int nZvit = lcmData.getNZvit();
    int nZvi0 = lcmData.getNZvi0();
    int perClassParams = nX + nZmuit + nZuit + nZvit + nZvi0;
    bool hasIds = lcmData.getIdVec().has_value();
    int numIds = hasIds ? static_cast<int>(lcmData.getNids()) : 1;
    // check if global status of whether using threading
    bool threaded = ESAGlobalOptimParams::GetInstance()->optimThreaded;
    // the jacobian — per-firm rows
    arma::dmat jac(numIds, totalParams, arma::fill::zeros);
    // per-firm Louis Hessian slots (only populated when hessOut != nullptr)
    bool needLouis = (hessOut != nullptr);
    std::vector<arma::dmat> louisVec(numIds);
    // inline fn to calc grad
    auto gradInner = [
        this, &params, &haltonMat, &segParams, &nClasses, &nSeg,
        &innerMT, &perClassParams, &nSegParams, &totalParams,
        &exceptNotFinite, &jac, &louisVec, &needLouis
    ](
        const unsigned int idx,
        const auto& seg,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zvi0,
        const auto& transition
    ) {
        // de-reference ptr to the data structure for LC
        ESADataPanelLCM& lcmData = *this->lcmDataPtr;
        arma::rowvec z_i = arma::conv_to<arma::rowvec>::from(seg.row(0));
        // compute class probabilities
        arma::dcolvec pi_i = lcmutils::computeClassProbs(z_i, segParams, nClasses);
        arma::dcolvec logPi_i = arma::log(pi_i);
        if (!zuit || !zvit || !zvi0)
            throw std::invalid_argument("LC-TRE gradHess: missing zuit, zvit, or zvi0");
        // store log likelihood per class
        arma::dcolvec logLik_c(nClasses);
        // store summed score per class in full-param space
        std::vector<arma::drowvec> classScores(nClasses);
        // store analytical Hessian per class (full-param, only if needLouis)
        std::vector<arma::dmat> classHess(nClasses);
        // iterate thru the number of latent classes
        for (unsigned int c = 0; c < nClasses; c++) {
            arma::dcolvec par_c = lcmutils::buildClassParamVec(lcmData, params, c);
            arma::dmat jacC;
            arma::dmat hessC;
            double llC = 0.0;
            // call the underlying TRE gradient/hessian calculation method
            this->gradHessPanel(
                idx, innerMT, par_c,
                y, x, zmuit,
                zuit.value(), zvit.value(), zvi0.value(),
                haltonMat,
                HessianCalcMethod::ANALYTICAL, 0,
                &jacC, needLouis ? &hessC : nullptr, &llC
            );
            // update this classes' loglike score
            logLik_c(c) = llC;
            // summed score for class c, in full-param space (zeros outside the c-th class block)
            arma::drowvec s_ic(totalParams, arma::fill::zeros);
            arma::drowvec sumJacC = esamath::colSum(jacC);
            // figure out where to place row for this class (across all params)
            int classOffset = nSegParams + static_cast<int>(c) * perClassParams;
            s_ic.cols(classOffset, classOffset + perClassParams - 1) = sumJacC;
            classScores[c] = s_ic;
            // if calculating the hessian
            if (needLouis) {
                // place class-specific analytical Hessian into overall Hessian
                arma::dmat H_c_full(totalParams, totalParams, arma::fill::zeros);
                H_c_full.submat(classOffset, classOffset, classOffset + perClassParams - 1, classOffset + perClassParams - 1) = hessC;
                classHess[c] = H_c_full;
            }
        }
        // calculate the posterior
        arma::dcolvec tau_i = lcmutils::computePosteriors(logPi_i, logLik_c);
        // ---- build observed-data score g_i ----
        arma::rowvec g_i(totalParams, arma::fill::zeros);
        // segmentation block: (τ_ic − π_ic) z_i  for c = 0..C-2
        for (unsigned int c = 0; c < nClasses - 1; c++) {
            double diff = tau_i(c) - pi_i(c);
            int off = static_cast<int>(c) * static_cast<int>(nSeg);
            g_i.cols(off, off + static_cast<int>(nSeg) - 1) = diff * z_i;
        }
        // class blocks: τ_ic * s_ic (class score already zero outside its own block)
        for (unsigned int c = 0; c < nClasses; c++) {
            // the offset for this current class
            int off = nSegParams + static_cast<int>(c) * perClassParams;
            g_i.cols(off, off + perClassParams - 1) = (
                tau_i(c) * classScores[c].cols(off, off + perClassParams - 1)
            );
        }
        jac.row(idx) = g_i;
        // ---- louis hessian contribution from firm i ----
        if (needLouis) {
            arma::dmat H_i(totalParams, totalParams, arma::fill::zeros);
            // from the louis paper
            // Term 1: E[H_complete | Y_i] = Σ_c τ_ic H_ic_expanded
            // Term 2: −Σ_c τ_ic s_ic s_ic^T
            // Term 3: +ĝ_i^T ĝ_i
            //
            // Note: seg-block Hessian from the complete-data log-likelihood is the
            // multinomial-logit Hessian: H_seg_cd = -π_ic(δ_cd − π_id) z_i z_i^T
            // (same formula used in mStepSeg). This contributes to Term 1.
            // ----
            // >>> segmentation analytical Hessian block (Term 1 seg part)
            if (nSegParams > 0) {
                arma::dcolvec zz = z_i.t();
                arma::dmat zzT = zz * z_i;
                // iterate thru latent classes
                for (unsigned int c = 0; c < nClasses - 1; c++) {
                    int rc = static_cast<int>(c) * static_cast<int>(nSeg);
                    for (unsigned int d = 0; d < nClasses - 1; d++) {
                        int rd = static_cast<int>(d) * static_cast<int>(nSeg);
                        // diagonal indicator
                        double ind  = (c == d) ? 1.0 : 0.0;
                        double coef = -pi_i(c) * (ind - pi_i(d));
                        H_i.submat(
                            rc,
                            rd,
                            rc + static_cast<int>(nSeg) - 1,
                            rd + static_cast<int>(nSeg) - 1
                        ) += coef * zzT;
                    }
                }
            }
            // class analytical Hessian blocks
            arma::drowvec g_mean(totalParams, arma::fill::zeros);   // Σ_c τ_ic s_ic
            arma::dmat E_ssT(totalParams, totalParams, arma::fill::zeros); // Σ_c τ_ic s_ic s_ic^T
            // iterate thru latent classes
            for (unsigned int c = 0; c < nClasses; c++) {
                // multiply probability of class membership by analytical hessian for the class
                H_i += tau_i(c) * classHess[c];
                arma::drowvec s_ic = classScores[c];
                // add seg-score component to s_ic so the variance captures the full score
                for (unsigned int cc = 0; cc < nClasses - 1; cc++) {
                    double diff = (cc == c ? 1.0 : 0.0) - pi_i(cc);
                    int off = static_cast<int>(cc) * static_cast<int>(nSeg);
                    s_ic.cols(off, off + static_cast<int>(nSeg) - 1) += diff * z_i;
                }
                g_mean += tau_i(c) * s_ic;
                E_ssT += tau_i(c) * (s_ic.t() * s_ic);
            }
            // E[ss^T] - g_mean^T g_mean
            arma::dmat comp2 = E_ssT - (g_mean.t() * g_mean);
            H_i -= comp2;
            louisVec[idx] = std::move(H_i);
        }
        arma::dmat dummy(1, 1, arma::fill::zeros);
        return dummy;
    };
    lcmData.dataCallable(gradInner, nullptr, nullptr, false, false, threaded);

    arma::dmat grad = esamath::colSum(jac);
    if (gradOut) *gradOut = grad;
    if (hessOut) *hessOut = esautils::sumMatricies<double>(louisVec);
    if (jacOut)  *jacOut  = jac;
}

void ESASfaLcTre::gradHess(
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
    // doesn't work - bypass and just send overall; not needed anyhow
    this->gradHess(params, false, gradOut, hessOut, jacOut);
}

// ---- Starting Values ----

arma::dcolvec ESASfaLcTre::startingValues() const
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
    int nTransTotal = static_cast<int>(nClasses) * (static_cast<int>(nClasses) - 1) * static_cast<int>(nTransition);
    int totalParams = (static_cast<int>(nClasses) - 1) * static_cast<int>(nSeg) + nTransTotal + static_cast<int>(nClasses) * perClassParams;
    int classParamBase = (static_cast<int>(nClasses) - 1) * static_cast<int>(nSeg) + nTransTotal;
    // full vector of starting parameters
    arma::dcolvec startVals(totalParams, arma::fill::zeros);
    // shortcut - for the single class just use the TRE starting values
    if (nClasses == 1) {
        arma::dcolvec base = ESASfaTreBase::startingValues();
        startVals.rows(classParamBase, classParamBase + perClassParams - 1) = base;
        return startVals;
    }
    // ---- step 1: pooled RE on the helper ESADataPanel ----
    ESADataPanel& helperData = dynamic_cast<ESADataPanel&>(*dataObjPtr);
    const arma::Col<int>& idents = helperData.getIdVec();
    arma::Col<int> uniqIdents = arma::unique(idents);
    const int N = static_cast<int>(uniqIdents.n_rows);
    const int nobs = static_cast<int>(helperData.getNobs());
    // random effects estimation
    ESARandEff reMdl(helperData.getYPtr(), helperData.getXPtr(), helperData.getIdVecPtr());
    RandEffResult reRes = reMdl.fit();
    // ---- step 2: firm-level features from composite residual ----
    // Three features per firm: mean residual, within-firm variance, skewness
    arma::dmat firmFeatures(N, 3, arma::fill::zeros);
    for (int i = 0; i < N; ++i) {
        arma::uvec inds = arma::find(idents == uniqIdents(i));
        arma::dvec ri = reRes.reResidual.elem(inds);
        int Ti = static_cast<int>(ri.n_elem);
        double mean_i = arma::mean(ri);
        double var_i = (Ti > 1) ? arma::var(ri) : 0.0;
        double skew_i = 0.0;
        if (Ti >= 3 && var_i > 1e-20) {
            double sd_i = std::sqrt(var_i);
            double m3 = arma::mean(arma::pow(ri - mean_i, 3));
            skew_i = m3 / (sd_i * sd_i * sd_i);
        }
        firmFeatures(i, 0) = mean_i;
        firmFeatures(i, 1) = var_i;
        firmFeatures(i, 2) = skew_i;
    }
    // standardize columns (z-score), skip if std < 1e-10
    for (int d = 0; d < 3; ++d) {
        double mu  = arma::mean(firmFeatures.col(d));
        double sig = arma::stddev(firmFeatures.col(d));
        if (sig > 1e-10)
            firmFeatures.col(d) = (firmFeatures.col(d) - mu) / sig;
        else
            firmFeatures.col(d).zeros();
    }
    // ---- step 3: K-Means clustering of firms ----
    unsigned int minCS = static_cast<unsigned int>(std::max(2, N / (static_cast<int>(nClasses) * 5)));
    esautils::KMeansResult km = esautils::kmeans(firmFeatures, nClasses, 100, 1e-6, 42, minCS);
    // collect indices for each cluster (observation level)
    std::vector<arma::uvec> clusterObsInds(nClasses);
    std::vector<arma::uvec> clusterFirmInds(nClasses);
    {
        std::vector<std::vector<arma::uword>> obsVecs(nClasses), firmVecs(nClasses);
        for (int i = 0; i < N; ++i) {
            unsigned int c = km.assignments(i);
            arma::uvec inds = arma::find(idents == uniqIdents(i));
            firmVecs[c].push_back(static_cast<arma::uword>(i));
            for (arma::uword idx : inds) obsVecs[c].push_back(idx);
        }
        for (unsigned int c = 0; c < nClasses; ++c) {
            clusterObsInds[c]  = arma::conv_to<arma::uvec>::from(obsVecs[c]);
            clusterFirmInds[c] = arma::conv_to<arma::uvec>::from(firmVecs[c]);
        }
    }
    // pooled base starting values (fallback)
    arma::dcolvec baseStartVals = ESASfaTreBase::startingValues();
    const double scale = 0.8;
    // ---- step 4: per-cluster MoM decomposition ----
    std::vector<arma::dcolvec> classStartVals(nClasses);
    // loop thru the latent classes
    for (unsigned int c = 0; c < nClasses; ++c) {
        int nFirmsC = static_cast<int>(clusterFirmInds[c].n_elem);
        int nObsC = static_cast<int>(clusterObsInds[c].n_elem);
        // fallback for tiny clusters
        if (nFirmsC < 3 || nObsC < nX + 2) {
            arma::dcolvec cStart = baseStartVals;
            double perturbation = 0.05 * (static_cast<double>(c) - static_cast<double>(nClasses - 1) / 2.0);
            cStart.rows(0, nX - 1) *= (1.0 + perturbation);
            classStartVals[c] = cStart;
            continue;
        }
        // build sub-panel data for this cluster
        arma::dvec yC  = helperData.getY().elem(clusterObsInds[c]);
        arma::dmat xC  = helperData.getX().rows(clusterObsInds[c]);
        arma::Col<int> idC(nObsC);
        for (int j = 0; j < nObsC; ++j) idC(j) = idents(clusterObsInds[c](j));
        // cluster-level random effects regression
        arma::dmat yCmat(yC);
        ESARandEff reMdlC(&yCmat, &xC, &idC);
        RandEffResult reResC = reMdlC.fit();
        // decompose cluster residuals into transient and persistent
        arma::Col<int> uniqC = arma::unique(idC);
        int NC = static_cast<int>(uniqC.n_elem);
        arma::dvec residTransientC(nObsC, arma::fill::zeros);
        arma::dvec residPersistUniqC(NC, arma::fill::zeros);
        // iterate thru each panel
        for (int i = 0; i < NC; ++i) {
            arma::uvec inds_i = arma::find(idC == uniqC(i));
            double grpMean = arma::mean(reResC.reResidual.elem(inds_i));
            residPersistUniqC(i) = grpMean;
            residTransientC.elem(inds_i) = reResC.reResidual.elem(inds_i) - grpMean;
        }
        // MoM for transient component - time-varying inefficiency, and stochastic noise
        std::optional<arma::dmat> zuitC = std::nullopt, zvitC = std::nullopt;
        if (lcmData.getZuit().has_value()) {
            arma::dmat zuitSub = lcmData.getZuit().value().rows(clusterObsInds[c]);
            zuitC = zuitSub;
        }
        if (lcmData.getZvit().has_value()) {
            arma::dmat zvitSub = lcmData.getZvit().value().rows(clusterObsInds[c]);
            zvitC = zvitSub;
        }
        MoMResult transResC = getMoMComponents(residTransientC, zuitC, zvitC);
        // add a minimum floor
        if (transResC.sigma2u < 1e-4) transResC.sigma2u = 0.05;
        if (transResC.sigma2v < 1e-4) transResC.sigma2v = 0.05;
        // MoM for time-invariant firm effect component (vi0)
        double avgTC = static_cast<double>(nObsC) / static_cast<double>(NC);
        double vif = (transResC.sigma2u + transResC.sigma2v) / avgTC;
        double m2Pers = arma::accu(arma::pow(residPersistUniqC, 2)) / NC;
        double m2Ov = std::max(m2Pers - vif, arma::var(residPersistUniqC) * 0.2);
        // build collapsed zvi0 [firm effect] for this cluster
        std::optional<arma::dmat> zvi0CollapseC = std::nullopt;
        if (lcmData.getZvi0().has_value()) {
            arma::dmat zvi0ColC(NC, nZvi0, arma::fill::zeros);
            for (int i = 0; i < NC; ++i) {
                arma::uvec inds_i = arma::find(idC == uniqC(i));
                // map local inds_i back to cluster obs indices, then to global
                arma::uvec globalInds = clusterObsInds[c].elem(inds_i);
                zvi0ColC.row(i) = arma::mean(lcmData.getZvi0().value().rows(globalInds), 0);
            }
            zvi0CollapseC = zvi0ColC;
        }
        std::optional<arma::dmat> dummyZui0 = std::nullopt;
        // calculate method of moment components
        MoMResult persistResC = getMoMComponents(residPersistUniqC, dummyZui0, zvi0CollapseC, std::make_optional(m2Ov));
        // build class parameter vector in [beta | zmuit | zuit | zvit | zvi0] layout
        arma::dcolvec cStart(perClassParams, arma::fill::zeros);
        int cntr = 0;
        // beta from cluster RE, with intercept shift
        cStart.rows(0, nX - 1) = reResC.params;
        double meanUTrans = std::sqrt(2.0 / arma::datum::pi) * std::sqrt(transResC.sigma2u);
        cStart(0) += this->s * meanUTrans * scale;
        cntr += nX;
        // zmuit (zeros)
        if (nZmuit > 0) { cntr += nZmuit; }
        // zuit
        if (nZuit == 1) {
            cStart(cntr) = std::log(transResC.sigma2u) * scale;
        } else if (nZuit > 1) {
            cStart.rows(cntr, cntr + nZuit - 1) = transResC.b_zu;
            cStart(cntr) = std::log(transResC.sigma2u) * scale;
        }
        cntr += nZuit;
        // zvit — for TRE, intercept is log(sigma2_vi0); for others use transient
        if (nZvit == 1) {
            cStart(cntr) = std::log(arma::var(residPersistUniqC) + 1e-6) * scale;
        } else if (nZvit > 1) {
            cStart.rows(cntr, cntr + nZvit - 1) = transResC.b_zv;
            cStart(cntr) = std::log(transResC.sigma2v) * scale;
        }
        cntr += nZvit;
        // zvi0 - firm effect
        if (nZvi0 > 0) {
            cStart.rows(cntr, cntr + nZvi0 - 1) = persistResC.b_zv * scale;
        }
        classStartVals[c] = cStart;
    }
    // ---- step 5: segmentation initialisation (shortcut multinomial logit) ----
    // use hard cluster assignments as pseudo-posteriors: tau(i,c) = I(label_i == c)
    int totalSegParams = (static_cast<int>(nClasses) - 1) * static_cast<int>(nSeg);
    if (totalSegParams > 0) {
        arma::dcolvec segParams(totalSegParams, arma::fill::zeros);
        if (nSeg == 1) {
            // Intercept-only: alpha_c = log(n_c / n_{C-1})
            // Count firms per cluster
            std::vector<int> nc(nClasses, 0);
            for (int i = 0; i < N; ++i) nc[km.assignments(i)]++;
            int nRef = nc[nClasses - 1];
            if (nRef < 1) nRef = 1;
            for (unsigned int c = 0; c < nClasses - 1; ++c)
                segParams(c) = std::log(static_cast<double>(std::max(nc[c], 1)) / static_cast<double>(nRef));
        } else {
            // Newton-Raphson on multinomial logit — mirror mStepSeg with hard assignments
            const arma::dmat& segMat = lcmData.getSeg();
            // build hard-assignment tau matrix (N x C), one row per firm
            arma::dmat tau(N, nClasses, arma::fill::zeros);
            for (int i = 0; i < N; ++i) tau(i, km.assignments(i)) = 1.0;
            // NR iterations
            for (int iter = 0; iter < 20; ++iter) {
                arma::dcolvec grad(totalSegParams, arma::fill::zeros);
                arma::dmat hess(totalSegParams, totalSegParams, arma::fill::zeros);
                // build firm-level seg rows: first observation per firm
                for (int i = 0; i < N; ++i) {
                    // find first observation index for firm i in the full LCM panel
                    // use the firm's first seg row directly via clusterFirmInds mapping
                    // The LCM dataCallable yields seg rows per firm index; here we read
                    // the seg matrix directly using the global firm ordering.
                    // firms in ESADataPanelLCM are ordered as they appear in the id vector.
                    // map from cluster firm index back to global firm index.
                    // find first obs in LCM data matching uniqIdents[i]
                    const arma::Col<int>& lcmId = lcmData.getIdVec().value();
                    arma::uvec globalInds = arma::find(lcmId == uniqIdents(i));
                    if (globalInds.is_empty()) continue;
                    arma::rowvec z_i = segMat.row(globalInds(0));
                    // class probabilities
                    arma::dcolvec pi_i = lcmutils::computeClassProbs(z_i, segParams, nClasses);
                    // iterate thru C-1 latent classes
                    for (unsigned int c = 0; c < nClasses - 1; ++c) {
                        double diff = tau(i, c) - pi_i(c);
                        int off = static_cast<int>(c * nSeg);
                        grad.rows(off, off + static_cast<int>(nSeg) - 1) += diff * z_i.t();
                    }
                    arma::dcolvec zz = z_i.t();
                    arma::dmat zzT = zz * z_i;
                    for (unsigned int c = 0; c < nClasses - 1; ++c) {
                        int rc = static_cast<int>(c * nSeg);
                        for (unsigned int d = 0; d < nClasses - 1; ++d) {
                            int rd = static_cast<int>(d * nSeg);
                            double ind = (c == d) ? 1.0 : 0.0;
                            double coef = -pi_i(c) * (ind - pi_i(d));
                            hess.submat(
                                rc, rd,
                                rc + static_cast<int>(nSeg) - 1,
                                rd + static_cast<int>(nSeg) - 1
                            ) += coef * zzT;
                        }
                    }
                }
                arma::dmat hessReg = hess - 1e-8 * arma::eye(totalSegParams, totalSegParams);
                arma::dcolvec step;
                bool ok = arma::solve(step, hessReg, grad, arma::solve_opts::no_approx);
                if (!ok) step = grad * 0.01;
                segParams -= step;
                if (arma::norm(grad, 2) < 1e-6) break;
            }
        }
        startVals.rows(0, totalSegParams - 1) = segParams;
    }
    // ---- step 6: assemble class parameters ----
    for (unsigned int c = 0; c < nClasses; ++c) {
        int offset = classParamBase + static_cast<int>(c) * perClassParams;
        startVals.rows(offset, offset + perClassParams - 1) = classStartVals[c];
    }
    return startVals;
}

// ---- sigma params ----

ESASigmaParams ESASfaLcTre::getSigmaParams(const arma::dcolvec& par) const
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    unsigned int nClasses = lcmData.getNClasses();

    arma::dcolvec par_0 = lcmutils::buildClassParamVec(lcmData, par, 0);
    return ESASfaTreGreene::getSigmaParams(par_0);
}

double ESASfaLcTre::getN() const
{
    return lcmDataPtr->getNids();
}

// ---- posterior class probabilities ----

arma::dmat ESASfaLcTre::computePosteriors(const arma::dcolvec& params) const
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    arma::dmat& haltonMat = *haltonDraws;
    unsigned int nClasses = lcmData.getNClasses();
    arma::dcolvec segParams = lcmData.paramSeg(params);
    unsigned int nFirms = static_cast<unsigned int>(lcmData.getNids());
    ESASfaModelType mT = lcmData.getModelType();

    ESASfaModelType innerMT;
    if (mT == ESASfaModelType::LC_TRE_HNORM) {
        innerMT = ESASfaModelType::TRE_HNORM_ZUIT;
    } else if (mT == ESASfaModelType::LC_TRE_TNORM) {
        innerMT = ESASfaModelType::TRE_TNORM_ZUIT;
    } else {
        throw std::runtime_error("Unsupported LC-TRE model type for posterior computation");
    }

    arma::dmat posteriors(nFirms, nClasses, arma::fill::zeros);

    auto inner = [this, &params, &haltonMat, &segParams, &nClasses, &innerMT, &posteriors](
        const unsigned int idx,
        const auto& seg,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zvi0,
        const auto& transition
    ) {
        ESADataPanelLCM& lcmData = *this->lcmDataPtr;
        arma::rowvec z_i = arma::conv_to<arma::rowvec>::from(seg.row(0));
        arma::dcolvec pi_i = lcmutils::computeClassProbs(z_i, segParams, nClasses);
        arma::dcolvec logPi_i = arma::log(pi_i);
        // checks for sigma determinants
        if (!zuit || !zvit || !zvi0) {
            throw std::invalid_argument("LC-TRE posteriors: missing zuit, zvit, or zvi0");
        }
        // likelihood per latent class
        arma::dcolvec logLik_c(nClasses);
        for (unsigned int c = 0; c < nClasses; c++) {
            arma::dcolvec par_c = lcmutils::buildClassParamVec(lcmData, params, c);
            arma::dmat llOut(1, 1);
            this->operatorInner(
                par_c, innerMT, haltonMat, idx,
                y, x, zmuit,
                zuit.value(), zvit.value(), zvi0.value(),
                true, &llOut
            );
            logLik_c(c) = llOut(0, 0);
        }
        // compute posterior
        arma::dcolvec tau_i = lcmutils::computePosteriors(logPi_i, logLik_c);
        posteriors.row(idx) = tau_i.t();

        arma::dmat dummy(1, 1, arma::fill::zeros);
        return dummy;
    };
    lcmData.dataCallable(inner, nullptr, nullptr, false, false,
        ESAGlobalOptimParams::GetInstance()->optimThreaded);
    return posteriors;
}

// ---- GHQ helpers ----

arma::dmat ESASfaLcTre::computePosteriorsGHQ(
    const arma::dcolvec& params,
    const arma::drowvec& ghqLogWeights
) const
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    arma::dmat& ghqMat = *haltonDraws;
    unsigned int nClasses = lcmData.getNClasses();
    unsigned int nFirms = static_cast<unsigned int>(lcmData.getNids());
    arma::dcolvec segParams = lcmData.paramSeg(params);
    ESASfaModelType mT = lcmData.getModelType();
    int ns = this->nsim;

    ESASfaModelType innerMT;
    if (mT == ESASfaModelType::LC_TRE_HNORM) {
        innerMT = ESASfaModelType::TRE_HNORM_ZUIT;
    } else if (mT == ESASfaModelType::LC_TRE_TNORM) {
        innerMT = ESASfaModelType::TRE_TNORM_ZUIT;
    } else {
        throw std::runtime_error("computePosteriorsGHQ: unsupported model type");
    }
    arma::dmat posteriors(nFirms, nClasses, arma::fill::zeros);

    auto inner = [this, &params, &ghqMat, &ghqLogWeights, &segParams, &nClasses, &innerMT, &posteriors, &ns](
        const unsigned int idx,
        const auto& seg,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zvi0,
        const auto& /*transition*/
    ) {
        ESADataPanelLCM& lcmData = *this->lcmDataPtr;
        arma::rowvec z_i = arma::conv_to<arma::rowvec>::from(seg.row(0));
        arma::dcolvec pi_i = lcmutils::computeClassProbs(z_i, segParams, nClasses);
        arma::dcolvec logPi_i = arma::log(pi_i);

        if (!zuit || !zvit || !zvi0)
            throw std::invalid_argument("computePosteriorsGHQ: missing zuit/zvit/zvi0");

        ESADataPanel& helperData = (ESADataPanel&)*this->dataObjPtr;
        const auto& y_ref    = y.get_ref();
        const auto& x_ref    = x.get_ref();
        const auto& zuit_ref = zuit.value().get_ref();
        const auto& zvit_ref = zvit.value().get_ref();
        const auto& zvi0_ref = zvi0.value().get_ref();
        int nT = static_cast<int>(y_ref.n_rows);

        arma::dcolvec logLik_c(nClasses);
        // iterate thu the latent classes
        for (unsigned int c = 0; c < nClasses; c++) {
            arma::dcolvec par_c = lcmutils::buildClassParamVec(lcmData, params, c);
            double aghqCorr = 0.0;
            arma::drowvec nodeCorrRow(ns, arma::fill::zeros);
            // half-normal inefficiency distribution
            if (innerMT == ESASfaModelType::TRE_HNORM_ZUIT) {
                arma::dcolvec b_x = helperData.paramX(par_c);
                arma::dcolvec b_zuit = helperData.paramZuit(par_c).value();
                arma::dcolvec b_zvit = helperData.paramZvit(par_c).value();
                arma::dcolvec b_zvi0 = helperData.paramZvi0(par_c).value();
                // find v* and sigma* for adaptive part of GHQ
                auto [v_star, sigma_star] = aghqModeAndSigma(
                    nT, this->s, y_ref, x_ref, zuit_ref, zvit_ref, zvi0_ref,
                    b_x, b_zuit, b_zvit, b_zvi0);
                ghqMat.row(idx) = ghq::buildAGHQRow(v_star, sigma_star, ns);
                aghqCorr = ghq::aghqLogNormCorrection(sigma_star);
                // build per-node correction: z_k^2 - v*_ik^2 / 2
                const double* rawZ = ghq::rawNodes(ns);
                const double sqrt2 = std::sqrt(2.0);
                for (int k = 0; k < ns; ++k) {
                    double zk = rawZ[k];
                    double v_star_ik = v_star + sigma_star * sqrt2 * zk;
                    nodeCorrRow(k) = zk * zk - 0.5 * v_star_ik * v_star_ik;
                }
            }
            // log-likelihood calculation - nb using ghqMat instead of old halton draw
            arma::dmat densOut;
            this->operatorInner(
                par_c, innerMT, ghqMat, idx,
                y, x, zmuit,
                zuit.value(), zvit.value(), zvi0.value(),
                false, &densOut
            );
            // log density
            arma::drowvec lnDen = arma::sum(arma::log(densOut), 0);
            lnDen += nodeCorrRow;  // z_k^2 - v*_ik^2/2 correction
            double Smax = lnDen.max();
            arma::drowvec K_k = arma::exp(lnDen - Smax);
            double logWtdSum = std::log(arma::dot(arma::exp(ghqLogWeights), K_k));
            logLik_c(c) = aghqCorr + Smax + logWtdSum - ghq::logNorm();
        }
        // posterior
        arma::dcolvec tau_i = lcmutils::computePosteriors(logPi_i, logLik_c);
        posteriors.row(idx) = tau_i.t();
        // dummy return pointlessly
        arma::dmat dummy(1, 1, arma::fill::zeros);
        return dummy;
    };
    lcmData.dataCallable(inner, nullptr, nullptr, false, false,
        ESAGlobalOptimParams::GetInstance()->optimThreaded);
    return posteriors;
}

// ---- Frozen AGHQ nodes ----

ESASfaLcTre::FrozenAGHQNodes ESASfaLcTre::computeFrozenAGHQ(
    unsigned int c,
    const arma::dcolvec& params,
    const arma::drowvec& /*ghqLogWeights*/
) const
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    ESASfaModelType mT = lcmData.getModelType();
    bool is_hnorm = (mT == ESASfaModelType::LC_TRE_HNORM);
    // number of firms
    unsigned int nFirms = static_cast<unsigned int>(lcmData.getNids());
    int ns = this->nsim;
    FrozenAGHQNodes out;
    out.is_hnorm = is_hnorm;
    out.v_star = arma::dvec(nFirms, arma::fill::zeros);
    out.sigma_star = arma::dvec(nFirms, arma::fill::ones);
    out.aghq_corr = arma::dvec(nFirms, arma::fill::zeros);
    out.node_corr = arma::dmat(nFirms, ns, arma::fill::zeros);

    if (!is_hnorm) return out; // tnorm: standard GHQ, no adaptation

    arma::dcolvec par_c = lcmutils::buildClassParamVec(lcmData, params, c);
    ESADataPanel& helperData = (ESADataPanel&)*dataObjPtr;
    arma::dcolvec b_x    = helperData.paramX(par_c);
    arma::dcolvec b_zuit = helperData.paramZuit(par_c).value();
    arma::dcolvec b_zvit = helperData.paramZvit(par_c).value();
    arma::dcolvec b_zvi0 = helperData.paramZvi0(par_c).value();

    const double* rawZ = ghq::rawNodes(ns);
    const double sqrt2 = std::sqrt(2.0);

    auto inner = [this, &out, &b_x, &b_zuit, &b_zvit, &b_zvi0, ns, rawZ, sqrt2](
        const unsigned int idx,
        const auto& /*seg*/,
        const auto& y,
        const auto& x,
        const auto& /*zmuit*/,
        const auto& zuit,
        const auto& zvit,
        const auto& zvi0,
        const auto& /*transition*/
    ) {
        if (!zuit || !zvit || !zvi0)
            throw std::invalid_argument("computeFrozenAGHQ: missing zuit/zvit/zvi0");
        const auto& y_ref = y.get_ref();
        const auto& x_ref = x.get_ref();
        const auto& zuit_ref = zuit.value().get_ref();
        const auto& zvit_ref = zvit.value().get_ref();
        const auto& zvi0_ref = zvi0.value().get_ref();
        int nT = static_cast<int>(y_ref.n_rows);
        // calculate v*, sigma*
        auto [vs, ss] = aghqModeAndSigma(
            nT, this->s, y_ref, x_ref, zuit_ref, zvit_ref, zvi0_ref,
            b_x, b_zuit, b_zvit, b_zvi0
        );
        out.v_star(idx) = vs;
        out.sigma_star(idx) = ss;
        out.aghq_corr(idx) = ghq::aghqLogNormCorrection(ss);
        // populate per-node correction: z_k^2 - v*_ik^2/2
        for (int k = 0; k < ns; ++k) {
            double zk = rawZ[k];
            double v_star_ik = vs + ss * sqrt2 * zk;
            out.node_corr(idx, k) = zk * zk - 0.5 * v_star_ik * v_star_ik;
        }

        arma::dmat dummy(1, 1, arma::fill::zeros);
        return dummy;
    };
    lcmData.dataCallable(inner, nullptr, nullptr, false, false,
        ESAGlobalOptimParams::GetInstance()->optimThreaded);
    return out;
}

// ---- M-step combined LL + analytical gradient/Hessian ----

double ESASfaLcTre::weightedClassLLAndGradHess(
    unsigned int /*c*/,
    const arma::dcolvec& classParams,
    const arma::dvec& tau_c,
    const arma::drowvec& ghqLogWeights,
    const FrozenAGHQNodes& frozen,
    arma::dmat* gradOut,
    arma::dmat* hessOut
) const
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    arma::dmat& ghqMat = *haltonDraws;
    ESASfaModelType mT = lcmData.getModelType();
    ESASfaModelType innerMT;
    if (mT == ESASfaModelType::LC_TRE_HNORM)
        innerMT = ESASfaModelType::TRE_HNORM_ZUIT;
    else if (mT == ESASfaModelType::LC_TRE_TNORM)
        innerMT = ESASfaModelType::TRE_TNORM_ZUIT;
    else
        throw std::runtime_error("weightedClassLLAndGradHess: unsupported model type");

    int nFirms = static_cast<int>(lcmData.getNids());
    int nX = lcmData.getNX();
    int nZmuit = lcmData.getNZmuit();
    int nZuit = lcmData.getNZuit();
    int nZvit = lcmData.getNZvit();
    int nZvi0 = lcmData.getNZvi0();
    // number of class parameters
    int perClassParams = nX + nZmuit + nZuit + nZvit + nZvi0;
    bool threaded = ESAGlobalOptimParams::GetInstance()->optimThreaded;
    // variables to store likelihood, hessians, jacobian
    arma::dvec llVec(nFirms, arma::fill::zeros);
    arma::dmat jac(nFirms, perClassParams, arma::fill::zeros);
    std::vector<arma::dmat> hessVec(nFirms);
    auto inner = [this, &classParams, &ghqMat, &ghqLogWeights, &innerMT,
                  &tau_c, &jac, &llVec, &perClassParams, &frozen,
                  &hessVec, hessOut, gradOut](
        const unsigned int idx,
        const auto& /*seg*/,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zvi0,
        const auto& /*transition*/
    ) {
        if (!zuit || !zvit || !zvi0)
            throw std::invalid_argument("weightedClassLLAndGradHess: missing zuit/zvit/zvi0");
        // apply frozen AGHQ nodes — fixed data, derivative w.r.t. classParams is zero
        if (frozen.is_hnorm)
            ghqMat.row(idx) = ghq::buildAGHQRow(
                frozen.v_star(idx), frozen.sigma_star(idx), this->nsim
            );
        arma::dmat densOut;
        this->operatorInner(
            classParams, innerMT, ghqMat, idx,
            y, x, zmuit,
            zuit.value(), zvit.value(), zvi0.value(),
            false, &densOut
        );
        int ns = densOut.n_cols;

        // corrected AGHQ log-weights: S_ik + node_corr + log(w_k)
        arma::drowvec lnDen = arma::sum(arma::log(densOut), 0);
        lnDen += frozen.node_corr.row(idx);  // z_k^2 - v*_ik^2/2
        lnDen += ghqLogWeights;              // log(w_k)
        double Smax = lnDen.max();
        arma::drowvec K_k = arma::exp(lnDen - Smax);
        double sumK = arma::accu(K_k);
        // Q̃_ik = K_k / sumK  (AGHQ-corrected posterior weights) — local to this firm
        arma::dmat ghqQir(1, ns);
        ghqQir = K_k / sumK;
        double logLic = frozen.aghq_corr(idx) + Smax + std::log(sumK) - ghq::logNorm();
        llVec(idx) = tau_c(idx) * logLic;
        // analytical gradient (and optionally Hessian) via inlined internalAnalyticJacHess
        if (gradOut || hessOut) {
            arma::dmat jacC;
            arma::dmat hessC;
            this->internalAnalyticJacHess(
                idx, innerMT, classParams,
                y, x, zmuit,
                zuit.value(), zvit.value(), zvi0.value(),
                ghqMat,
                std::optional<arma::dmat>(std::nullopt),
                ghqQir.row(0),
                gradOut ? &jacC : nullptr,
                hessOut ? &hessC : nullptr
            );
            // if returning gradient
            if (gradOut) {
                arma::drowvec g_i = esamath::colSum(jacC);
                jac.row(idx) = tau_c(idx) * g_i;
            }
            // if returning hessian
            if (hessOut) {
                hessVec[idx] = tau_c(idx) * hessC;
            }
        }
        arma::dmat dummy(1, 1, arma::fill::zeros);
        return dummy;
    };
    lcmData.dataCallable(inner, nullptr, nullptr, false, false, threaded);

    if (gradOut) *gradOut = esamath::colSum(jac);
    if (hessOut) *hessOut = esautils::sumMatricies<double>(hessVec);
    return arma::accu(llVec);
}

double ESASfaLcTre::computeObservedLLGHQ(
    const arma::dcolvec& params,
    const arma::drowvec& ghqLogWeights
) const
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    arma::dmat& ghqMat = *haltonDraws;
    unsigned int nClasses = lcmData.getNClasses();
    arma::dcolvec segParams = lcmData.paramSeg(params);
    ESASfaModelType mT = lcmData.getModelType();
    ESASfaModelType innerMT;
    if (mT == ESASfaModelType::LC_TRE_HNORM)
        innerMT = ESASfaModelType::TRE_HNORM_ZUIT;
    else if (mT == ESASfaModelType::LC_TRE_TNORM)
        innerMT = ESASfaModelType::TRE_TNORM_ZUIT;
    else
        throw std::runtime_error("computeObservedLLGHQ: unsupported model type");

    unsigned int nFirms = static_cast<unsigned int>(lcmData.getNids());
    arma::dvec llVec(nFirms, arma::fill::zeros);
    // per-panel processing - lambda function for the dataCallable method
    auto inner = [this, &params, &ghqMat, &ghqLogWeights, &segParams, &nClasses, &innerMT, &llVec](
        const unsigned int idx,
        const auto& seg,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zvi0,
        const auto& /*transition*/
    ) {
        // dereference pt to LC data obj
        ESADataPanelLCM& lcmData = *this->lcmDataPtr;
        arma::rowvec z_i = arma::conv_to<arma::rowvec>::from(seg.row(0));
        // calculate class probabilities from the segmentation params - multinomial logit
        arma::dcolvec pi_i = lcmutils::computeClassProbs(z_i, segParams, nClasses);
        arma::dcolvec logPi_i = arma::log(pi_i);
        // checks to ensure zuit, zvit, zvi0 are present
        if (!zuit || !zvit || !zvi0) {
            throw std::invalid_argument("computeObservedLLGHQ: missing zuit/zvit/zvi0");
        }
        // dereference ptr to the ESADataPanel helper
        ESADataPanel& helperData = (ESADataPanel&)*this->dataObjPtr;
        const auto& y_ref = y.get_ref();
        const auto& x_ref = x.get_ref();
        const auto& zuit_ref = zuit.value().get_ref();
        const auto& zvit_ref = zvit.value().get_ref();
        const auto& zvi0_ref = zvi0.value().get_ref();
        int nT = static_cast<int>(y_ref.n_rows);
        int ns = this->nsim;
        // iterate thru the latent classes
        arma::dcolvec logLik_c(nClasses);
        for (unsigned int c = 0; c < nClasses; c++) {
            // build the parameter vector for this class
            arma::dcolvec par_c = lcmutils::buildClassParamVec(lcmData, params, c);
            // 
            double aghqCorr = 0.0;
            arma::drowvec nodeCorrRow(ns, arma::fill::zeros);

            if (innerMT == ESASfaModelType::TRE_HNORM_ZUIT) {
                arma::dcolvec b_x = helperData.paramX(par_c);
                arma::dcolvec b_zuit = helperData.paramZuit(par_c).value();
                arma::dcolvec b_zvit = helperData.paramZvit(par_c).value();
                arma::dcolvec b_zvi0 = helperData.paramZvi0(par_c).value();
                auto [v_star, sigma_star] = aghqModeAndSigma(
                    nT, this->s, y_ref, x_ref, zuit_ref, zvit_ref, zvi0_ref,
                    b_x, b_zuit, b_zvit, b_zvi0
                );
                ghqMat.row(idx) = ghq::buildAGHQRow(v_star, sigma_star, ns);
                aghqCorr = ghq::aghqLogNormCorrection(sigma_star);
                //
                const double* rawZ = ghq::rawNodes(ns);
                const double sqrt2 = std::sqrt(2.0);
                for (int k = 0; k < ns; ++k) {
                    double zk = rawZ[k];
                    double v_star_ik = v_star + sigma_star * sqrt2 * zk;
                    nodeCorrRow(k) = zk * zk - 0.5 * v_star_ik * v_star_ik;
                }
            }

            arma::dmat densOut;
            this->operatorInner(
                par_c, innerMT, ghqMat, idx,
                y, x, zmuit,
                zuit.value(), zvit.value(), zvi0.value(),
                false, &densOut
            );
            arma::drowvec lnDen = arma::sum(arma::log(densOut), 0);
            lnDen += nodeCorrRow;  // z_k^2 - v*_ik^2/2 correction
            double Smax = lnDen.max();
            arma::drowvec K_k = arma::exp(lnDen - Smax);
            logLik_c(c) = aghqCorr + Smax + std::log(arma::dot(arma::exp(ghqLogWeights), K_k))
                          - ghq::logNorm();
        }
        arma::dcolvec logJoint = logPi_i + logLik_c;
        double logLiMax = logJoint.max();
        llVec(idx) = logLiMax + std::log(arma::accu(arma::exp(logJoint - logLiMax)));

        arma::dmat dummy(1, 1, arma::fill::zeros);
        return dummy;
    };
    // get threaded status
    bool threaded = ESAGlobalOptimParams::GetInstance()->optimThreaded;
    lcmData.dataCallable(inner, nullptr, nullptr, false, false, threaded);
    return arma::accu(llVec);
}
