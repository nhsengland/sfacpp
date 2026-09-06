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

#ifndef ESASFATFEGREENE_HPP
#define ESASFATFEGREENE_HPP

#include <memory>

// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif
// --- end armadillo ---

#include "utils/enums.hpp"
#include "sfa/ESASfaBase.hpp"
#include "data/ESADataBase.hpp"


class ESASfaTfeGreene : public ESASfaBase {

public:
    /**
     * @brief Constructor
     * @param dataObjPtr Pointer to an ESADataBase object
     * @param s Whether production or cost frontier
     * @param epsilon
     * @param obsUseSameHaltonDraw Ignored
     */
    ESASfaTfeGreene(const std::shared_ptr<ESADataBase> dataObjPtr, const double s);

    /**
     * @brief Objective function to minimize/maximimze
     * @param params Column vector of parameters
     * @return Value of the objective function at 'params'
     */
    double operator()(const arma::dcolvec& params, const bool exceptNotFinite = false) const override;

    // /**
    //  * @brief Calculate gradient at parameter vector
    //  * @param params Column vector of parameters
    //  * @param step epsilon for numerical approximation
    //  * @param isAnalytical whether analytical gradient or not
    //  */
    // arma::dmat gradient(const arma::dcolvec& params, const double step = 1e-8, const bool isAnalytical = true) const override;

    // /**
    //  * @brief Calculate hessian at parameter vector
    //  * @note better to use gradHess when requiring both hessian and gradient
    //  * @param params Column vector of parameters
    //  * @param method Enumeration from HessianCalcMethod, defining the method to take calculating the Hessian
    //  * @param accuracy Unsigned integer defining the accuracy to use for numerical approximation of the hessian only
    //  * @param threaded Boolean whether to use threading to calculate hessian
    //  */
    // arma::dmat hessian(const arma::dcolvec& params, const HessianCalcMethod method, const unsigned int accuracy = 0, const bool threaded = false) const override;

    /**
     * @brief Gradient and Hessian at parameter vector
     * @param params Column vector of parameter to evaluate gradient and Hessian at
     * @param step Step size for numerical approximation (ignored)
     * @param analyticalGrad Boolean whether or not to use analytical gradient (default is true)
     * @param hessMethod Enumeration from HessianCalcMethod, defining the method to take calculating the Hessian
     * @param accuracy Unsigned integer defining the accuracy to use for numerical approximation of the hessian only
     * @param threaded Boolean whether to use threading to calculate Hessian
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

    /**
     * @brief Calculate suggested starting values for maximum likelihood estimation
     * @return Column vector of starting values
     */
    arma::dcolvec startingValues() const override;

private:

    /**
     * @brief Density for the half normal distribution
     * @param par Column vector of parameters
     * @param y Column vector of dependent variable
     * @param x Matrix of independent variables
     * @param zuit Matrix of determinants of inefficiency
     * @param zvit Matrix of determinants of random noise component
     */
    arma::dmat densityHalfNormal(
        const arma::dcolvec& par,
        const arma::dcolvec& y,
        const arma::dmat& x,
        const arma::dmat& zuit,
        const arma::dmat& zvit
    ) const;

    /**
     * @brief Density for the truncated normal distribution
     * @param par Column vector of parameters
     * @param y Column vector of dependent variable
     * @param x Matrix of independent variables
     * @param zmuit Matrix of determinants of pre-truncated mean of inefficiency
     * @param zuit Matrix of determinants of inefficiency
     * @param zvit Matrix of determinants of random noise component
     */
    arma::dmat densityTruncNormal(
        const arma::dcolvec& par,
        const arma::dcolvec& y,
        const arma::dmat& x,
        const arma::dmat& zmuit,
        const arma::dmat& zuit,
        const arma::dmat& zvit
    ) const;

    /**
     * @brief gradient inner
     * 
     */
    arma::dmat gradientInner(
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
    ) const;

    /**
     * @brief hessian inner
     */
    arma::dmat hessianInner(
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
    ) const;

    /**
     * @brief 
     * 
    */
    void gradHessInner(
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
        arma::dmat* gradOut = nullptr,
        arma::dmat* hessOut = nullptr,
        arma::dmat* jacOut = nullptr
    ) const;

    /**
     * @brief Analytical gradient and Hessian for half-normal distribution
     * @param par Column vector of parameters
     * @param y Column vector of dependent variable
     * @param x Matrix of independent variables
     * @param zuit Matrix of determinants of inefficiency
     * @param zvit Matrix of determinants of random noise component
     * @param[out] hessOut Optional. A pointer to a destination arma::dmat.
     * If provided (not nullptr), this matrix will be populated with the calculated Hessian matrix.
     * @param[out] jacOut  Optional. A pointer to a destination arma::dmat.
     * If provided (not nullptr), this matrix will be populated with the Jacobian (e.g., the observation-level gradient contributions).
     */
    void analyticJacHessHalfNormal(
        const arma::dcolvec& par,
        const arma::dcolvec& y,
        const arma::dmat& x,
        const arma::dmat& zuit,
        const arma::dmat& zvit,
        arma::dmat* jacOut,
        arma::dmat* hessOut
    ) const;
};

#endif // ESASFATFEGREENE_HPP