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

#ifndef ESA_UTILS_LCMUTILS_HPP
#define ESA_UTILS_LCMUTILS_HPP

#include <cmath>

// ---- armadillo ----
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif
// ---- end armadillo ----

#include "data/ESADataLCM.hpp"

namespace lcmutils {

/**
 * @brief Compute class membership probabilities via multinomial logit (softmax)
 * @param z_i Row vector of concomitant variables for firm i (1 x nSeg)
 * @param segParams Segmentation parameters as column vector, ordered as
 *        [alpha_0(nSeg) | alpha_1(nSeg) | ... | alpha_{C-2}(nSeg)]
 *        Last class (C-1) is reference with alpha = 0
 * @param nClasses Total number of classes C
 * @return Column vector of class probabilities (C x 1), sums to 1
 */
inline arma::dcolvec computeClassProbs(
    const arma::subview_row<double>& z_i,
    const arma::dcolvec& segParams,
    const unsigned int nClasses
)
{
    unsigned int nSeg = z_i.n_cols;
    arma::dcolvec logits(nClasses, arma::fill::zeros);
    for (unsigned int c = 0; c < nClasses - 1; c++) {
        arma::dcolvec alpha_c = segParams.rows(c * nSeg, (c + 1) * nSeg - 1);
        logits(c) = arma::as_scalar(z_i * alpha_c);
    }
    double maxLogit = logits.max();
    arma::dcolvec expLogits = arma::exp(logits - maxLogit);
    return expLogits / arma::accu(expLogits);
}

/**
 * @brief Overload for arma::rowvec (full matrix case)
 */
inline arma::dcolvec computeClassProbs(
    const arma::rowvec& z_i,
    const arma::dcolvec& segParams,
    const unsigned int nClasses
)
{
    unsigned int nSeg = z_i.n_cols;
    arma::dcolvec logits(nClasses, arma::fill::zeros);
    for (unsigned int c = 0; c < nClasses - 1; c++) {
        arma::dcolvec alpha_c = segParams.rows(c * nSeg, (c + 1) * nSeg - 1);
        logits(c) = arma::as_scalar(z_i * alpha_c);
    }
    double maxLogit = logits.max();
    arma::dcolvec expLogits = arma::exp(logits - maxLogit);
    return expLogits / arma::accu(expLogits);
}

/**
 * @brief Compute transition probabilities via multinomial logit
 * @param w_t Row of transition covariates at time t (1 x nTransition)
 * @param transParams Transition parameters as column vector
 *        Layout: for each source class c (c=0..C-1), (C-1) destination blocks of nTransition each
 *        For source c: destinations d != c get parameters, d == c is reference (gamma = 0)
 * @param fromClass Source class index
 * @param nClasses Total number of classes
 * @param nTransition Number of transition covariates
 * @return Column vector P(S_t = d | S_{t-1} = fromClass) for d = 0..C-1
 */
inline arma::dcolvec computeTransitionProbs(
    const arma::subview_row<double>& w_t,
    const arma::dcolvec& transParams,
    const unsigned int fromClass,
    const unsigned int nClasses,
    const unsigned int nTransition
)
{
    arma::dcolvec logits(nClasses, arma::fill::zeros);
    unsigned int classBlockSize = (nClasses - 1) * nTransition;
    unsigned int baseOffset = fromClass * classBlockSize;
    unsigned int destIdx = 0;
    for (unsigned int d = 0; d < nClasses; d++) {
        if (d == fromClass) continue;
        arma::dcolvec gamma_cd = transParams.rows(
            baseOffset + destIdx * nTransition,
            baseOffset + (destIdx + 1) * nTransition - 1
        );
        logits(d) = arma::as_scalar(w_t * gamma_cd);
        destIdx++;
    }
    double maxLogit = logits.max();
    arma::dcolvec expLogits = arma::exp(logits - maxLogit);
    return expLogits / arma::accu(expLogits);
}

/**
 * @brief Compute posterior class probabilities (Bayes' rule)
 * @param logPriors Log of prior class probs log(pi_ic), (C x 1)
 * @param logLikelihoods Log-likelihoods per class log(L_ic), (C x 1)
 * @return Column vector of posterior probs tau_ic (C x 1), sums to 1
 */
inline arma::dcolvec computePosteriors(
    const arma::dcolvec& logPriors,
    const arma::dcolvec& logLikelihoods
)
{
    arma::dcolvec logJoint = logPriors + logLikelihoods;
    double maxLJ = logJoint.max();
    arma::dcolvec expJoint = arma::exp(logJoint - maxLJ);
    return expJoint / arma::accu(expJoint);
}

/**
 * @brief Log-sum-exp for numerical stability
 * @param v Input vector
 * @return log(sum(exp(v)))
 */
inline double logSumExp(const arma::dcolvec& v)
{
    double maxVal = v.max();
    return maxVal + std::log(arma::accu(arma::exp(v - maxVal)));
}

/**
 * @brief Compute gradient of segmentation parameters for one firm
 * @details For class c: d(log L_i)/d(alpha_c) = (tau_ic - pi_ic) * z_i
 * @param tau Posterior class probabilities (C x 1)
 * @param pi Prior class probabilities (C x 1)
 * @param z_i Concomitant variables row (1 x nSeg)
 * @param nClasses Number of classes
 * @return Column vector gradient for seg params [(C-1)*nSeg x 1]
 */
inline arma::dcolvec segGradient(
    const arma::dcolvec& tau,
    const arma::dcolvec& pi,
    const arma::subview_row<double>& z_i,
    const unsigned int nClasses
)
{
    unsigned int nSeg = z_i.n_cols;
    arma::dcolvec grad((nClasses - 1) * nSeg);
    for (unsigned int c = 0; c < nClasses - 1; c++) {
        double diff = tau(c) - pi(c);
        grad.rows(c * nSeg, (c + 1) * nSeg - 1) = diff * z_i.t();
    }
    return grad;
}

/**
 * @brief Build a per-class parameter vector in ESADataPanel-compatible format
 * @details Layout: [beta | zmuit | zuit | zvit | zvi0]
 */
inline arma::dcolvec buildClassParamVec(
    const ESADataPanelLCM& data,
    const arma::dcolvec& fullParams,
    const unsigned int lc
)
{
    arma::dcolvec result = data.paramX(fullParams, lc);
    auto zmuit_c = data.paramZmuit(fullParams, lc);
    if (zmuit_c) result = arma::join_cols(result, zmuit_c.value());
    auto zuit_c = data.paramZuit(fullParams, lc);
    if (zuit_c) result = arma::join_cols(result, zuit_c.value());
    auto zvit_c = data.paramZvit(fullParams, lc);
    if (zvit_c) result = arma::join_cols(result, zvit_c.value());
    auto zvi0_c = data.paramZvi0(fullParams, lc);
    if (zvi0_c) result = arma::join_cols(result, zvi0_c.value());
    return result;
}

} // namespace lcmutils

#endif // ESA_UTILS_LCMUTILS_HPP
