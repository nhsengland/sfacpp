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
 * @file ESASfaCross.hpp
 * @brief ESASfaCross class header file
 * @date 2025-02-01
 * @author Edmund Haacke
 */

#ifndef ESA_SFA_CROSS_HPP
#define ESA_SFA_CROSS_HPP

#include <memory>
#include <optional>

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
#include "data/ESADataCross.hpp"
#include "data/ESADataBase.hpp"

class ESASfaCross : public ESASfaBase {
    
public:
    /// Constructor
    ESASfaCross(const std::shared_ptr<ESADataBase>& dataObjPtr, const double s);

    /// @brief Objective function to maximise
    /// @param params Column vector of starting values
    /// @return double value of the log likelihood
    double operator()(const arma::dcolvec& params, const bool exceptNotFinite = false) const override;

    // /// @brief Calculate the gradient across individuals - Jacobian
    // /// @param Column vector of parameters
    // /// @param step Step size for numerical approximation
    // /// @param isAnalytical Whether to use analytical or numerical approximation of the gradient
    // /// @return (N = n x k) matrix of the gradient
    // arma::dmat jacobian(const arma::dcolvec& params, const double step = 1e-8, const bool isAnalytical = true) const override;

    // /**
    //  * @brief Calculate overall gradient
    //  * @param params
    //  * @param step
    //  * @param isAnalytical
    //  * @return Matrix (1 x k) of the gradient
    //  */
    // arma::dmat gradient(const arma::dcolvec& params, const double step = 1e-8, const bool isAnalytical = true) const override;

    // /// @brief Calculate the analytical hessian matrix
    // /// @param params Column vector of parameters
    // /// @param bhhhApproximation Whether to use the BHHH approximation
    // /// @param numApproxGrad Whether to use numerical approximation of the gradient
    // /// @return Matrix of the hessian
    // arma::dmat hessian(
    //     const arma::dcolvec& params,
    //     const HessianCalcMethod method,
    //     const unsigned int accuracy = 0,
    //     const bool threaded = false
    // ) const override;

    /**
     * @brief Calculate gradient and hessian matrix together
     * @param params Column vector of parameters
     * @param step Step size for numerical approximation
     * @param analyticalGrad Boolean whether or not to use analytical or numerical approximation of the gradient
     * @param hessMethod Method to calculate the hessian matrix
     * @param accuracy Accuracy to use when calculating the hessian matrix - for numerical approx only
     * @param gradOut Pointer to write gradient vector to
     * @param hessOut Pointer to write hessian matrix to
    */
    void gradHess(
        const arma::dcolvec& params,
        const bool exceptNotFinite = false,
        arma::dmat* gradOut = nullptr,
        arma::dmat* hessOut = nullptr,
        arma::dmat* jacOut = nullptr
    ) const override;

    /// @brief Calculate the suggested starting values for ML estimation
    /// @return Column vector of starting values
    arma::dcolvec startingValues() const override;

private:
    
    /// @brief Density of the half-normal distribution
    /// @param par Column vector of parameters
    /// @param y Column vector of the dependent variable
    /// @param x Matrix of the independent variables
    /// @param zu Matrix of the variables affecting the variance of the (time invariant) inefficiency
    /// @param zv Matrix of the variables affecting the variance of the random noise
    /// @param s Whether cost (-1) or prod (1)
    arma::dmat densityCrossHalfNormal(
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& zu,
        const arma::dmat& zv,
        const int s
    ) const;

    /// @brief Log-likelihood of the half-normal distribution
    /// @param par Column vector of parameters
    /// @param y Column vector of the dependent variable
    /// @param x Matrix of the independent variables
    /// @param zu Matrix of the variables affecting the variance of the (time invariant) inefficiency
    /// @param zv Matrix of the variables affecting the variance of the random noise
    /// @param s Whether cost (-1) or prod (1)
    double logLikelihoodCrossHalfNormal(const arma::dcolvec& par) const;

    /// @brief Analytical gradient of the half-normal distribution
    /// @param par Column vector of parameters
    /// @param y Column vector of the dependent variable
    /// @param x Matrix of the independent variables
    /// @param zu Matrix of the variables affecting the variance of the (time invariant) inefficiency
    /// @param zv Matrix of the variables affecting the variance of the random noise
    /// @param s Whether cost (-1) or prod (1)
    arma::dmat gradientDerivCrossHalfNormal(
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& zu,
        const arma::dmat& zv,
        const int s
    ) const;

    /// @brief Hessian of the half-normal distribution
    /// @param par Column vector of parameters
    /// @param y Column vector of the dependent variable
    /// @param x Matrix of the independent variables
    /// @param zu Matrix of the variables affecting the variance of the (time invariant) inefficiency
    /// @param zv Matrix of the variables affecting the variance of the random noise
    /// @param s Whether cost (-1) or prod (1)
    arma::dmat hessianDerivCrossHalfNormal(
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& zu,
        const arma::dmat& zv,
        const int s
    ) const;

    /// @brief Density of the truncated-normal distribution
    /// @param par Column vector of parameters
    /// @param y Column vector of the dependent variable
    /// @param x Matrix of the independent variables
    /// @param zu Matrix of the variables affecting the variance of the (time invariant) inefficiency
    /// @param zv Matrix of the variables affecting the variance of the random noise
    /// @param s Whether cost (-1) or prod (1)
    arma::dmat densityCrossTruncNormal(
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& mu,
        const arma::dmat& zu,
        const arma::dmat& zv,
        const int s
    ) const;

    /// @brief Log-likelihood of the truncated-normal distribution
    /// @param par Column vector of parameters
    /// @param y Column vector of the dependent variable
    /// @param x Matrix of the independent variables
    /// @param mu Matrix of the truncated mean of the inefficiency component
    /// @param zu Matrix of the variables affecting the variance of the (time invariant) inefficiency
    /// @param zv Matrix of the variables affecting the variance of the random noise
    /// @param s Whether cost (-1) or prod (1)
    double logLikelihoodCrossTruncNormal(const arma::dcolvec& par) const;

    /// @brief Analytical gradient of the truncated-normal distribution
    /// @param par Column vector of parameters
    /// @param y Column vector of the dependent variable
    /// @param x Matrix of the independent variables
    /// @param zu Matrix of the variables affecting the variance of the (time invariant) inefficiency
    /// @param zv Matrix of the variables affecting the variance of the random noise
    /// @param s Whether cost (-1) or prod (1)
    arma::dmat gradientDerivCrossTruncNormal(
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& mu,
        const arma::dmat& zu,
        const arma::dmat& zv,
        const int s
    ) const;

    /// @brief Hessian of the truncated-normal distribution
    /// @param par Column vector of parameters
    /// @param y Column vector of the dependent variable
    /// @param x Matrix of the independent variables
    /// @param mu Matrix of the truncated mean of the inefficiency component
    /// @param zu Matrix of the variables affecting the variance of the (time invariant) inefficiency
    /// @param zv Matrix of the variables affecting the variance of the random noise
    /// @param s Whether cost (-1) or prod (1)
    arma::dmat hessianDerivCrossTruncNormal(
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& mu,
        const arma::dmat& zu,
        const arma::dmat& zv,
        const int s
    ) const;
};

#endif // ESA_SFA_CROSS_HPP