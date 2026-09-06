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
 * @file ESASfaTfeChen.hpp
 * @author Edmund Haacke
 * @date 2025-03-06
 * @details Implementation of the TFE with skew-normal, using first differencing, proposed by Chen, Schmidt, Wang 2014
 */

#ifndef ESA_SFA_TFE_CHEN_HPP
#define ESA_SFA_TFE_CHEN_HPP

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

class ESASfaTfeChen : public ESASfaBase {

private:
    // ---- half-normal distribution ----
    /// @brief calculate the log-likelihood score for a panel, with half-normal distribution
    /// @param idx the current index position w.r.t. individuals
    /// @param par column vector of parameters
    /// @param y column vector of the dependent variable
    /// @param x matrix of independent variables
    /// @param zuit matrix of variables affecting variance of time-varying inefficiency
    /// @param zvit matrix of variables affecting variance of the stochastic noise component
    /// @param s whether cost (-1) or prod (+1) function
    /// @return double of the overall log-likelihood score
    double logLikeHalfNormal(
        const unsigned int idx
    ) const;

public:
    // Constructor method
    ESASfaTfeChen(std::shared_ptr<ESADataBase> dataObjPtr, const double s);

    /// @brief objective function to minimize
    /// @param params Column vector of parameters, to estimate value of objective function at
    /// @return double, the log likelihood score
    virtual double operator()(const arma::dcolvec& params, const bool exceptNotFinite = false) const override;

    // /// @brief calculate gradient across individuals
    // /// @param params Column vector of parameters
    // /// @param haltonMat Matrix of halton draws
    // /// @param step Step size for numerical approximation
    // /// @param isAnalytical Whether to use analytical or numerical approximation of the gradient
    // virtual arma::dmat gradient(
    //     const arma::dcolvec& params,
    //     const double step = 1e-8,
    //     const bool isAnalytical = true
    // ) const override;

    // /// @brief calculate the hessian matrix
    // /// @param params Column vector of the parameters
    // /// @param bhhhApproximation Whether to use the BHHH approximation
    // /// @param numApproxGrad Whether to use the numerical approximation of the gradient
    // /// @return matrix of the hessian
    // arma::dmat hessian(
    //     const arma::dcolvec& params,
    //     const HessianCalcMethod method,
    //     const unsigned int accuracy = 0,
    //     const bool threaded = false
    // ) const override;

    /// @brief calculate some starting values for the maximum likelihood estimation
    /// @return column vector of parameters to use as starting values
    arma::dcolvec startingValues() const override;
};

#endif // ESA_SFA_TFE_CHEN_HPP