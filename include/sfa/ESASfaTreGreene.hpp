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


#ifndef ESA_SFA_TRE_GREENE_HPP
#define ESA_SFA_TRE_GREENE_HPP

#include <memory>
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
// ---- end armadillo ----

#include "utils/enums.hpp"
#include "sfa/HaltonSettings.hpp"
#include "sfa/ESASigmaParams.hpp"
#include "sfa/ESASfaTreBase.hpp"
#include "data/ESADataBase.hpp"


enum struct SfaTreGreenSimMode {
    halton = 1,
    haltonRandom = 2
};

class ESASfaTreGreene : public ESASfaTreBase {

public:

    /// @brief constructor
    /// @param dataObjPtr pointer to data object
    /// @param s Production or cost frontier
    /// @param nsim Number of simulations
    /// @param seed Seed for replicatability
    /// @param epsilon Epsilon value
    /// @param obsUseSameHaltonDraw Boolean whether observations should use same halton draw
    ESASfaTreGreene(
        const std::shared_ptr<ESADataBase> dataObjPtr,
        const int s,
        const int nsim = 100,
        const int seed = 1234,
        const HaltonSettings hsetting = HaltonSettings()
    );

    ESASfaTreGreene(
        const std::shared_ptr<ESADataBase> dataObjPtr,
        const int s,
        const int nsim = 100,
        const int seed = 1234,
        const std::shared_ptr<arma::dmat> haltonDrawPtr = nullptr,
        const HaltonSettings hsetting = HaltonSettings()
    );

    /// @brief Objective function to maximize/minimize
    /// @param params Column vector of parameters
    /// @return Value of the objective function at 'params'
    double operator()(const arma::dcolvec& params, const bool exceptNotFinite = false) const override;
    double operator()(const arma::dcolvec& params, const arma::Col<int>& subsetIdents) const override;
    
    // /**
    //  * @brief Calculate gradient across individuals (jacobian)
    //  * @param params Column vector of parameters
    //  * @param step Step size for numerical approximation
    //  * @param isAnalytical Whether or not to use the analytical or numerical approx of the individual gradient
    //  * @param (n x k) matrix of individual gradients (per observation, n = N x T)
    //  */
    // arma::dmat jacobian(const arma::dcolvec& params, const double step = 1e-8, const bool isAnalytical = true) const override;
    // arma::dmat jacobian(const arma::dcolvec& params, const arma::Col<int>& subsetIdents, const double step = 1e-8, const bool isAnalytical = true) const override;

    // /**
    //  * @brief Calculate overall gradient
    //  * @param params column vector of parameters
    //  * @param step Step size for numerical approximation (deprecated)
    //  * @param isAnalytical Whether to use analytical or numerical approximation of the gradient
    //  * @return (1 x k) matrix of the (average) gradient
    //  */
    // arma::dmat gradient(const arma::dcolvec& params, const double step = 1e-8, const bool isAnalytical = true) const override;
    // arma::dmat gradient(const arma::dcolvec& params, const arma::Col<int>& subsetIdents, const double step = 1e-8, const bool isAnalytical = true) const override;

    // /// @brief Calculate hessian matrix
    // /// @param params Column vector of parameters
    // /// @param method Enumeration from HessianCalcMethod, defining the method to take calculating the hessian
    // /// @param accuracy Unsigned integer defining the accuracy to use for numerical approximation of the hessian only
    // arma::dmat hessian(const arma::dcolvec& params, const HessianCalcMethod method, const unsigned int accuracy = 0, const bool threaded = false) const override;
    // arma::dmat hessian(const arma::dcolvec& params, const arma::Col<int>& subsetIdents, const HessianCalcMethod method, const unsigned int accuracy = 0, const bool threaded = false) const override;

    /**
     * @brief calculate both gradient and hessian matrix simulatenously
     * @param params
     * @param subsetIdents
     * @param step
     * @param analyticalGrad
     * @param hessMethod
     * @param accuracy
     * @param threaded
     * @param[out] gradOut Optional. A pointer to a destination arma::dmat. 
     * If provided (i.e., not nullptr), this matrix will be  populated with the calculated gradient vector.
     * @param[out] hessOut Optional. A pointer to a destination arma::dmat.
     * If provided (not nullptr), this matrix will be populated with the calculated Hessian matrix.
     * @param[out] jacOut  Optional. A pointer to a destination arma::dmat.
     * If provided (not nullptr), this matrix will be populated with the Jacobian (e.g., the observation-level gradient contributions).
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

    /**
     * @brief Calculate reduced form analytical gradient & hessian
     * @param params Column vector of parameters
     * @param threaded Whether or not to use threading in calculation
     * @param gradOut Pointer to gradient (out); m 
     */


    /// @brief Calculate suggested starting values for maximum likelihood estimation
    /// @return Column vector of starting values
    // arma::dcolvec startingValues() const override;

    /**
     * @brief Return struct containing sigma, lambda parameters
     */
    ESASigmaParams getSigmaParams(const arma::dcolvec& par) const override;

protected:

    template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0, typename TZui0 = arma::dmat>
    void operatorInner(
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
        const bool logLike = true,
        arma::dmat* out = nullptr
    ) const;

    template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0, typename TZui0 = arma::dmat>
    void gradHessPanel(
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
        double* llOut = nullptr
    ) const;

    template <typename TY, typename TX, typename TZmuit = arma::dmat, typename TZuit, typename TZvit, typename TZvi0, typename TZui0 = arma::dmat>
    void panelDensityHalfNormal(
        const unsigned int idx,
        const arma::dcolvec& par,
        const arma::Base<double, TY>& yIn,
        const arma::Base<double, TX>& xIn,
        const arma::Base<double, TZuit>& zuitIn,
        const arma::Base<double, TZvit>& zvitIn,
        const arma::Base<double, TZvi0>& zvi0In,
        const arma::dmat& draws,
        arma::dmat& outDens
    ) const;

    template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0, typename TZui0 = arma::dmat>
    void panelDensityTruncNormal(
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
    ) const;

private:

    template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0, typename TZui0 = arma::dmat>
    void gradientInner(
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
    ) const;

    /// @brief Analytical gradient for half-normal distribution
    template <typename TY, typename TX, typename TZmuit = arma::dmat, typename TZuit, typename TZvit, typename TZvi0, typename TZui0 = arma::dmat>
    void panelGradHalfNormAnalytical(
        const unsigned int idx,
        const arma::dcolvec& par,
        const arma::Base<double, TY>& yIn,
        const arma::Base<double, TX>& xIn,
        const arma::Base<double, TZuit>& zuitIn,
        const arma::Base<double, TZvit>& zvitIn,
        const arma::Base<double, TZvi0>& zvi0In,
        const arma::dmat& draws,
        arma::dmat* outGrad = nullptr,
        arma::dmat* outGir = nullptr
    ) const;

    /// @brief Numerically approximated gradient for half-normal distribution
    template <typename TY, typename TX, typename TZmuit = arma::dmat, typename TZuit, typename TZvit, typename TZvi0, typename TZui0 = arma::dmat>
    void panelGradHalfNormalNumApprox(
        const unsigned int idx,
        const arma::dcolvec& par,
        const arma::Base<double, TY>& yIn,
        const arma::Base<double, TX>& xIn,
        const arma::Base<double, TZuit>& zuitIn,
        const arma::Base<double, TZvit>& zvitIn,
        const arma::Base<double, TZvi0>& zvi0In,
        const arma::dmat& draws,
        arma::dmat* outGrad = nullptr,
        arma::dmat* outGir = nullptr
    ) const;

    /// @brief Numerically approximated gradient for truncated normal distribution
    template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0, typename TZui0 = arma::dmat>
    void panelGradTruncNormalNumApprox(
        const unsigned int idx,
        const arma::dcolvec& par,
        const arma::Base<double, TY>& yIn,
        const arma::Base<double, TX>& xIn,
        const arma::Base<double, TZmuit>& zmuitIn,
        const arma::Base<double, TZuit>& zuitIn,
        const arma::Base<double, TZvit>& zvitIn,
        const arma::Base<double, TZvi0>& zvi0In,
        const arma::dmat& draws,
        arma::dmat* outGrad = nullptr,
        arma::dmat* outGir = nullptr
    ) const;
};

#endif // ESA_SFA_TRE_GREENE_HPP