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

#ifndef ESA_SFA_GTRE_BAD_HPP
#define ESA_SFA_GTRE_BAD_HPP

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

#include "data/ESADataBase.hpp"
#include "sfa/ESASfaTreBase.hpp"
#include "sfa/ESASigmaParams.hpp"
#include "sfa/HaltonSettings.hpp"
#include "utils/enums.hpp"


class ESASfaGtreBad : public ESASfaTreBase {

public:

    /**
     * @brief Constructor
     * @param dataObjPtr Pointer to an ESADataBase object
     * @param s Whether production or cost frontier
     * @param nsim Number of simulations
     * @param seed Seed for replicatability
     * @param epsilon
     * @param obsUseSameHaltonDraw Boolean whether to use the same halton draw
     */
    ESASfaGtreBad(
        const std::shared_ptr<ESADataBase> dataObjPtr,
        const int s,
        const int nsim = 100,
        const int seed = 1234,
        const HaltonSettings hsetting = HaltonSettings()
    );

    ESASfaGtreBad(
        const std::shared_ptr<ESADataBase> dataObjPtr,
        const int s,
        const int nsim = 100,
        const int seed = 1234,
        const std::shared_ptr<arma::dmat> haltonDrawPtr = nullptr,
        const HaltonSettings hsetting = HaltonSettings()
    );

    /**
     * @brief Objective function to minimize/maximize
     * @param params Column vector of parameters
     * @return Value of the objective function at 'params'
     */
    double operator()(const arma::dcolvec& params, const bool exceptNotFinite = false) const override;

    /**
     * @brief Calculate both gradient and hessian matrix simulatenously (preferred method)
     * @param params Column vector of parameters
     * @param step epsilon for numerical approximation (ignored)
     * @param analyticalGrad Whether to use an analytical gradient or not
     * @param hessMethod Enumeration from HessianCalcMethod - defining the method to take calculating the hessian
     * @param accuracy Unsigned integer defining the accuracy to use for numerical approximation
     * @param threaded Whether or not to use threading
     * @param[out] gradOut Optional. A pointer to a destination arma::dmat
     *  If provided (i.e., not nullptr), this matrix will be populated with the calculated gradient vector
     * @param[out] hessOut Optional. A pointer to a destination arma::dmat
     *  If provided (i.e., not nullptr), this matrix will be populated with the calculated Hessian matrix
     * @param[out] jacOut Optional. A pointer to a destination arma::dmat
     *  If provided (i.e., not nullptr), this matrix will be poptulated with the Jacobian (e.g., obs level gradient)
     */
    void gradHess(
        const arma::dcolvec& params,
        const bool exceptNotFinite = false,
        arma::dmat* gradOut = nullptr,
        arma::dmat* hessOut = nullptr,
        arma::dmat* jacOut = nullptr
    ) const override;

    /**
     * @brief Return struct containing sigma, lambda parameters
     * @param par Column vector of parameters
     * @return Instance of ESASigmaParams
     */
    ESASigmaParams getSigmaParams(const arma::dcolvec& par) const override;

private:

    /**
     * @brief Density for the half, truncated normal distribution
     * @param par Column vector of parameters
     * @param y Column vector of the dependent variable
     * @param x Matrix of independent variables
     * @param zmuit Matrix of mean of truncated normal (trunc norm only)
     * @param zuit Matrix of determinants of inefficiency
     * @param zvit Matrix of determinants of random noise component
     * @param zui0 Matrix of determinants of time-invariant inefficiency
     * @param zvi0 Matrix of determinants of latent firm heterogeneity
     * @return Density for a given panel
     */
    template <typename TY, typename TX, typename TZuit, typename TZvit, typename TZui0, typename TZvi0>
    void densityHalfNormal(
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
    ) const;

    /** 
     * @brief Calculate hessian & gradient for an individual panel
    */
    template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZui0, typename TZvi0>
    void gradHessInner(
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
        arma::dmat* jacOut = nullptr,
        arma::dmat* hessOut = nullptr
    ) const;

};

#endif // ESA_SFA_GTRE_BAD_HPP