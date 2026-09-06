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

#ifndef ESA_SFA_LC_TRE_EM_HPP
#define ESA_SFA_LC_TRE_EM_HPP

#include <memory>
#include <vector>

// ---- armadillo ----
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif

#include "sfa/ESASfaLcTre.hpp"
#include "sfa/ESASfaBase.hpp"
#include "math/GaussHermite.hpp"
#include "optim/optimparams.hpp"

/**
 * @brief Result from the EM algorithm
 */
struct EMResult {
    arma::dcolvec params;     // converged parameter vector
    double        logLike;    // observed-data log-likelihood at convergence
    int           nIter;      // number of EM iterations taken
    bool          converged;  // true if |ΔLL| < tol
    arma::dvec    llHistory;  // observed-data LL at each iteration (for diagnostics)
};

/**
 * @brief ESASfaBase-compatible wrapper for the τ-weighted class-c log-likelihood.
 *
 * Used by DlibWrapper to drive the dlib trust-region M-step for one latent class.
 * The wrapper presents the θ_c-parameterised objective
 *   f(θ_c) = Σ_i τ_ic · log L_ic(θ_c; AGHQ)
 * as a standard ESASfaBase so the existing DlibWrapper/dlib TR machinery is reused unchanged.
 *
 * gradHess() returns the analytical (not BHHH) gradient and Hessian via
 * weightedClassLLAndGradHess, which inlines the internalAnalyticJacHess loop
 * with AGHQ-corrected weights Q̃_ik.
 */
class ESASfaClassMStepObj : public ESASfaBase {
public:
    /**
     * @param ghqModel      LC-TRE model with nsim == nQuadPts
     * @param classIdx      Which latent class this M-step solves for
     * @param tau_c         (nFirms,) posterior weights τ_ic for class c
     * @param ghqLogWeights (1 x nQuadPts) log(w_k)
     * @param frozen        AGHQ nodes frozen at θ_old — immutable during TR iteration
     */
    ESASfaClassMStepObj(
        std::shared_ptr<ESASfaLcTre> ghqModel,
        unsigned int classIdx,
        arma::dvec tau_c,
        arma::drowvec ghqLogWeights,
        ESASfaLcTre::FrozenAGHQNodes frozen
    );

    double operator()(const arma::dcolvec& classParams, const bool exceptNotFinite = false) const override;

    void gradHess(
        const arma::dcolvec& classParams,
        const bool exceptNotFinite = false,
        arma::dmat* gradOut = nullptr,
        arma::dmat* hessOut = nullptr,
        arma::dmat* jacOut  = nullptr
    ) const override;

    arma::dcolvec startingValues() const override;

    double getN() const override;

private:
    std::shared_ptr<ESASfaLcTre> ghqModel_;
    unsigned int classIdx_;
    arma::dvec   tau_c_;
    arma::drowvec ghqLogWeights_;
    ESASfaLcTre::FrozenAGHQNodes frozen_;
    int perClassParams_;
};


/**
 * @brief EM algorithm with Gauss-Hermite Quadrature for the LC-TRE model.
 *
 * Algorithm (ECM variant — full convergence at each M-step):
 *   for iter = 1..maxIter:
 *     E-step:  τ_ic = π_ic L_ic / Σ_c π_ic L_ic  (AGHQ)
 *     M-step seg: maximize Σ_i Σ_c τ_ic log π_ic(γ) via Newton (multinomial logit)
 *     M-step class c (for c = 0..C-1):
 *               maximize Σ_i τ_ic log L_ic(θ_c; AGHQ, frozen nodes) via dlib trust-region
 *     check: |LL^(t) - LL^(t-1)| < tol → converged
 *
 * AGHQ integral:  L_ic = (1/√π) σ_i* Σ_k w_k exp(z_k^2 - v*_ik^2/2) ∏_t f(y_it | θ_c, node_ik)
 * This uses adaptive centering per firm, eliminating simulation noise from Halton MSL.
 *
 * Determinants of σ_vi0 (heteroscedastic random effect) are fully supported —
 * the AGHQ nodes are scaled firm-by-firm inside panelDensityHalfNormal/TruncNormal.
 */
class ESASfaLcTreEM {
public:
    /**
     * @param srcModel  The LC-TRE model to estimate (uses its data pointers and class structure)
     * @param params    EM and GHQ tuning parameters (from ESAOptimParams)
     * @param printLevel 0 = silent, 1 = per-iteration LL, 2 = verbose
     */
    ESASfaLcTreEM(
        std::shared_ptr<ESASfaLcTre> srcModel,
        const ESAOptimParams& params,
        unsigned int printLevel = 0
    );

    /**
     * @brief Run the EM algorithm starting from startVals.
     * @param startVals Full LC-TRE parameter vector (from ESASfaLcTre::startingValues())
     * @return EMResult with converged params, LL, iteration count, and LL history
     */
    EMResult run(const arma::dcolvec& startVals) const;

private:
    // E-step: compute (nFirms x nClasses) posterior matrix
    arma::dmat eStep(const arma::dcolvec& params) const;

    // Seg M-step: Newton on the multinomial logit Σ_i Σ_c τ_ic log π_ic(γ)
    arma::dcolvec mStepSeg(
        const arma::dcolvec& segParams,
        const arma::dmat& tau
    ) const;

    // Class M-step: dlib TR on Σ_i τ_ic log L_ic(θ_c; AGHQ, frozen nodes)
    arma::dcolvec mStepClass(
        unsigned int c,
        const arma::dcolvec& classParams,
        const arma::dmat& tau,
        const arma::dcolvec& fullParams
    ) const;

    // Observed-data log-likelihood (used for convergence check and llHistory)
    double observedLL(const arma::dcolvec& params) const;

    // GHQ model: a copy of srcModel with nsim == nQuadPts and ghqDraws injected
    std::shared_ptr<ESASfaLcTre> ghqModel_;
    arma::drowvec ghqLogWeights_;  // (1 x nQuadPts) log(w_k)
    unsigned int nClasses_;
    unsigned int nSeg_;
    unsigned int nQuadPts_;
    unsigned int maxIter_;
    unsigned int segMaxIter_;
    unsigned int classMaxIter_;
    double tol_;
    double segTol_;
    unsigned int printLevel_;
    ESAOptimParams optimParams_;
};

#endif // ESA_SFA_LC_TRE_EM_HPP
