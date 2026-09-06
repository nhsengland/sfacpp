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

#ifndef ESA_SFA_LC_TRE_HPP
#define ESA_SFA_LC_TRE_HPP

#include <memory>
#include <vector>
#include <optional>

// ---- armadillo ----
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif
// ---- end armadillo ---

#include "sfa/HaltonSettings.hpp"
#include "sfa/ESASfaTreGreene.hpp"
#include "utils/lcmutils.hpp"
#include "sfa/ESASigmaParams.hpp"
#include "data/ESADataLCM.hpp"
#include "data/ESADataPanel.hpp"
#include "utils/enums.hpp"
#include "math/GaussHermite.hpp"


class ESASfaLcTre : public ESASfaTreGreene {

public:

    /**
     * @brief Constructor for the Latent Class TRE model
     * @param lcmDataPtr Shared pointer to the LCM data object
     * @param helperDataPtr Shared pointer to an ESADataPanel built from the same data
     *        (used internally for per-class TRE density/gradient computation)
     * @param s Production (+1) or cost (-1) frontier
     * @param nsim Number of Halton simulations
     * @param seed Random seed
     * @param hsetting Halton sequence settings
     */
    ESASfaLcTre(
        std::shared_ptr<ESADataPanelLCM> lcmDataPtr,
        std::shared_ptr<ESADataPanel> helperDataPtr,
        const int s,
        const int nsim = 100,
        const int seed = 1234,
        const HaltonSettings hsetting = HaltonSettings()
    );

    /**
     * @brief Constructor with pre-built Halton draws
     */
    ESASfaLcTre(
        std::shared_ptr<ESADataPanelLCM> lcmDataPtr,
        std::shared_ptr<ESADataPanel> helperDataPtr,
        const int s,
        const int nsim = 100,
        const int seed = 1234,
        const std::shared_ptr<arma::dmat> haltonDrawPtr = nullptr,
        const HaltonSettings hsetting = HaltonSettings()
    );

    /**
     * @brief Log-likelihood for the LC-TRE model
     * @details Computes: sum_i log[ sum_c pi_ic * L_ic ]
     *          where L_ic is the TRE simulated likelihood for firm i under class c
     */
    double operator()(const arma::dcolvec& params, const bool exceptNotFinite = false) const override;
    double operator()(const arma::dcolvec& params, const arma::Col<int>& subsetIdents) const override;

    /**
     * @brief Gradient and BHHH Hessian approximation for the LC-TRE model
     * @details Per-firm gradient:
     *   g_i = [seg_grad | theta_0_grad | theta_1_grad | ... | theta_{C-1}_grad]
     *   where seg_grad_c = (tau_ic - pi_ic) * z_i
     *   and theta_c_grad = tau_ic * (d log L_ic / d theta_c)
     *   BHHH: H = -(jac' * jac)
     */
    void gradHess(
        const arma::dcolvec& params,
        const bool exceptNotFinite = false,
        arma::dmat* gradOut = nullptr,
        arma::dmat* hessOut = nullptr,
        arma::dmat* jacOut = nullptr
    ) const override;

    void gradHess(
        const arma::dcolvec& params,
        const arma::Col<int>& subsetIdents,
        const double step = 1e-8,
        const bool analyticalGrad = true,
        const HessianCalcMethod hessMethod = HessianCalcMethod::ANALYTICAL,
        const unsigned int accuracy = 0,
        const bool threaded = false,
        arma::dmat* gradOut = nullptr,
        arma::dmat* hessOut = nullptr,
        arma::dmat* jacOut = nullptr
    ) const override;

    arma::dcolvec startingValues() const override;

    ESASigmaParams getSigmaParams(const arma::dcolvec& par) const override;

    double getN() const override;

    const std::shared_ptr<ESADataPanelLCM> getLcmDataObj() const { return lcmDataPtr; }

    /**
     * @brief Compute posterior class probabilities for all firms
     * @param params Full LC-TRE parameter vector
     * @return N x C matrix of posterior probabilities (rows sum to 1)
     */
    arma::dmat computePosteriors(const arma::dcolvec& params) const;

    /**
     * @brief Compute posterior class probabilities using GHQ instead of Halton MSL.
     *
     * Must be called on a model constructed with nsim = nQuadPts and haltonDrawPtr = ghqDraws.
     * (The ESASfaLcTreEM class creates such a model internally.)
     *
     * @param params        Full LC-TRE parameter vector
     * @param ghqLogWeights (1 x nQuadPts) log(w_k) GHQ weight row vector
     * @return N x C matrix of posterior probabilities (rows sum to 1)
     */
    arma::dmat computePosteriorsGHQ(
        const arma::dcolvec& params,
        const arma::drowvec& ghqLogWeights
    ) const;

    /**
     * @brief Per-firm, per-class AGHQ node data frozen at the start of the M-step.
     *
     * Computed once from θ_old before the dlib trust-region optimizer is called.
     * All members are treated as fixed data during the M-step so the analytical
     * gradient is strictly consistent with the objective.
     *
     * Layout:
     *   v_star(i)      — posterior mode for firm i (standardised)
     *   sigma_star(i)  — posterior std-dev for firm i  (= 1/√κ_i)
     *   aghq_corr(i)   — log(σ_i*) correction added to log L_ic (no √2)
     *   node_corr(i,k) — z_k² − v*_ik²/2 per firm i and node k
     *   is_hnorm       — false → tnorm fallback (all corrections zero)
     */
    struct FrozenAGHQNodes {
        arma::dvec v_star;      // (nFirms,) posterior modes
        arma::dvec sigma_star;  // (nFirms,) posterior std-devs
        arma::dvec aghq_corr;   // (nFirms,) log(σ*) corrections
        arma::dmat node_corr;   // (nFirms × nQuadPts): z_k² − v*_ik²/2
        bool is_hnorm;
    };

    /**
     * @brief Compute frozen AGHQ nodes for class c using current θ_old params.
     *
     * Called once at M-step entry. Returns a FrozenAGHQNodes struct that is then
     * passed immutably to weightedClassLLAndGradHess.
     */
    FrozenAGHQNodes computeFrozenAGHQ(
        unsigned int c,
        const arma::dcolvec& params,
        const arma::drowvec& ghqLogWeights
    ) const;

    /**
     * @brief Compute τ-weighted complete-data class LL and optionally analytical gradient/Hessian.
     *
     * Replaces the old weightedClassLL + weightedClassLLGrad pair.
     * Returns Σ_i τ_ic · log L_ic(θ_c; AGHQ).
     *
     * When gradOut or hessOut are non-null the analytical (not BHHH) gradient and
     * Hessian are computed via the inlined internalAnalyticJacHess loop with
     * AGHQ-corrected weights Q̃_ik.
     *
     * @param c             Class index
     * @param classParams   Per-class parameter vector [beta | zmuit | zuit | zvit | zvi0]
     * @param tau_c         (nFirms,) vector of posterior weights τ_ic for class c
     * @param ghqLogWeights (1 x nQuadPts) log(w_k)
     * @param frozen        Frozen AGHQ nodes (from computeFrozenAGHQ, fixed during M-step)
     * @param gradOut       Optional: (1 × perClassParams) gradient row vector
     * @param hessOut       Optional: (perClassParams × perClassParams) analytical Hessian
     * @return Scalar weighted log-likelihood
     */
    double weightedClassLLAndGradHess(
        unsigned int c,
        const arma::dcolvec& classParams,
        const arma::dvec& tau_c,
        const arma::drowvec& ghqLogWeights,
        const FrozenAGHQNodes& frozen,
        arma::dmat* gradOut = nullptr,
        arma::dmat* hessOut = nullptr
    ) const;

    /**
     * @brief Observed log-likelihood Σ_i log Σ_c π_ic L_ic, computed with GHQ normalization.
     *
     * Must be called on a model constructed with nsim = nQuadPts and haltonDrawPtr = ghqDraws.
     *
     * @param params        Full LC-TRE parameter vector
     * @param ghqLogWeights (1 x nQuadPts) log(w_k)
     * @return Scalar observed log-likelihood
     */
    double computeObservedLLGHQ(
        const arma::dcolvec& params,
        const arma::drowvec& ghqLogWeights
    ) const;

private:

    std::shared_ptr<ESADataPanelLCM> lcmDataPtr;

    /**
     * @brief Compute per-firm log-likelihood contribution for the LC-TRE model
     */
    template <typename TSeg, typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0>
    double panelLogLikelihood(
        const unsigned int idx,
        const arma::Base<double, TSeg>& seg,
        const arma::Base<double, TY>& y,
        const arma::Base<double, TX>& x,
        const std::optional<TZmuit>& zmuit,
        const std::optional<TZuit>& zuit,
        const std::optional<TZvit>& zvit,
        const std::optional<TZvi0>& zvi0,
        const arma::dcolvec& params
    ) const;

    /**
     * @brief Compute per-firm gradient contribution for the LC-TRE model
     */
    template <typename TSeg, typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0>
    void panelGradient(
        const unsigned int idx,
        const arma::Base<double, TSeg>& seg,
        const arma::Base<double, TY>& y,
        const arma::Base<double, TX>& x,
        const std::optional<TZmuit>& zmuit,
        const std::optional<TZuit>& zuit,
        const std::optional<TZvit>& zvit,
        const std::optional<TZvi0>& zvi0,
        const arma::dcolvec& params,
        arma::dmat* gradOut
    ) const;
};

#endif // ESA_SFA_LC_TRE_HPP
