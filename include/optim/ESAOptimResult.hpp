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
 * @
 * @file ESAOptim.hpp
 * @author Edmund Haacke
 * @date 2025-09-16
 * @details
 * Header to wrap external optimization functions in
 */

#ifndef ESA_OPTIM_HPP
#define ESA_OPTIM_HPP


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

#include "sfa/ESASfaBase.hpp"
#include "utils/enums.hpp"

class ESAOptimResult
{
public:
    ESAOptimResult(bool didConverge) : didConverge(didConverge) {}
 
    bool getDidConverge() const { return this->didConverge; }

    virtual ~ESAOptimResult() = default;
protected:
    const bool didConverge;
};

class ESAOptimResultFailed : public ESAOptimResult
{
public:
    ESAOptimResultFailed() : ESAOptimResult(false) {}
    ESAOptimResultFailed(const arma::dcolvec& x) : ESAOptimResult(false), x(x) {}

    arma::dcolvec getX() const { return this->x; }
private:
    arma::dcolvec x;
};

class ESAOptimResultSuccess : public ESAOptimResult
{
public:
    ESAOptimResultSuccess() : ESAOptimResult(true) {}
    ESAOptimResultSuccess(
        const ESASfaModelType mT,
        const arma::dcolvec& x,
        const double logLike,
        const arma::dmat& gradI,
        const arma::vec& grad,
        const arma::dmat& hess,
        const int N,
        const int nobs,
        const double gnorm = 0.0
    ) : ESAOptimResult(true),
        mT(mT),
        x(x),
        logLike(logLike),
        gradI(gradI),
        grad(grad),
        hess(hess),
        N(N),
        nobs(nobs),
        gnorm(gnorm)
    {}

    double getGnorm() const { return this->gnorm; }
    /**
     * @brief Return variance-covariance matrix
     * @details Inverse negative of the Hessian matrix
     * @param eigentol tolerance
     * @return square matrix of vcov
     */
    arma::dmat getVcov(double eigentol = 1e-12) const;

    /**
     * @brief Return parameter estimates
     * @return Column vector of estimated parameters
     */
    arma::dcolvec getX() const { return this->x; }

    /**
     * @brief Return Jacobian matrix e.g., first-order partial derivatives per observation
     * @return (nT x k) matrix of jacobian
     */
    arma::dmat getGradientIndividual() const { return this->gradI; }

    /**
     * @brief Return overall gradient
     * @return Vector of gradient
     */
    arma::vec getGradient() const { return this->grad; }

    /**
     * @brief Return Hessian matrix
     * @return (k x k) Hessian matrix
     */
    arma::dmat getHessian() const { return this->hess; }

    /**
     * @brief Return log-likelihood score
     * @return double of the LL score
     */
    double getLogLike() const { return this->logLike; }

    /**
     * @brief Return either nobs, or nfirms depending if cross sectional/panel
     * 
     */
    int getN() const { return this-> N; }

    /**
     * @brief Return number of observations
     */
    int getNobs() const { return this->nobs; }

    /**
     * @brief Return denominator for average hessian / gradient
     */
    int denomAvgGradHess() const {
        ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
        // for GTRE, TRE we divide through the number of FIRMS
        if (mF == ESASfaModelFamily::GTRE || mF == ESASfaModelFamily::TRE) {
            return this->N;
        }
        // otherwise divide thru the number of observations
        return this->nobs;
    }

    /**
     * @brief Return degrees of freedom
     */
    int degreesFreedom() const { return (this->nobs - this->x.n_rows); }


    /**
     * @brief Return summary including P values and confidence intervals
     * 
     */
    arma::dmat modelSummary(double conflvl = 0.95) const;

private:
    ESASfaModelType mT;
    arma::dcolvec x;
    double logLike;
    arma::dmat gradI;
    arma::vec grad;
    arma::dmat hess;
    int N;
    int nobs;
    double gnorm;
};

#endif // ESA_OPTIM_HPP