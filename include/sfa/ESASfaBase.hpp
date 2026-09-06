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

#ifndef ESASFABASE_HPP
#define ESASFABASE_HPP

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
#include "sfa/ESASigmaParams.hpp"
#include "utils/enums.hpp"
#include "sfa/HaltonSettings.hpp"

enum struct ESASfaDist {
    hnorm,
    tnorm
};

struct MoMResult {
    double sigma2u;
    double sigma2v;
    arma::dcolvec b_zu;
    arma::dcolvec b_zv;
};

class ESASfaBase {

public:
    // Constructor
    ESASfaBase( std::shared_ptr<ESADataBase> dataObjPtr, const double s);

    virtual ~ESASfaBase() = default;

    // objective function to minimize
    virtual double operator()(const arma::dcolvec& params, const bool exceptNotFinite = false) const;
    
    /// @brief Evaluate the objective function for params, for a subset of identifiers
    /// @param params Column vector of parameters
    /// @param subsetIdents Column vector of identifiers to calculate objective function for
    virtual double operator()(const arma::dcolvec& params, const arma::Col<int>& subsetIdents) const;

    /// @brief Calculate gradient and hessian matrix together
    /// @param params Column vector of parameters
    /// @param step Step size for numerical approximation
    /// @param analyticalGrad Boolean on whether or not to use analytical or numerical approximation of the gradient
    /// @param hessMethod Method used to calculate hessian matrix
    /// @param accuracy Accuracy to use when calculating hessian matrix - for numerical approx of 2nd deriv only
    /// @param gradOut 
    /// @param hessOut
    virtual void gradHess(
        const arma::dcolvec& params,
        const bool exceptNotFinite = false,
        arma::dmat* gradOut = nullptr,
        arma::dmat* hessOut = nullptr,
        arma::dmat* jacOut = nullptr
    ) const;
    virtual void gradHess(
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
    ) const;

    /**
     * @brief Calculate reduced from analytical gradient and hessian
     * @note This is only for the homoskedastic model!
     * @param params Column vector of parameters
     * @param threaded Whether or not to use threading in calculation
     * @param gradOut Pointer to gradient (out)
     * @param jacOut Pointer to jacobian (out)
     * @param hessOut Pointer to hessian (out)
     */
    virtual void gradHessReduced(
        const arma::dcolvec& params,
        const bool threaded = true,
        arma::dmat* gradOut = nullptr,
        arma::dmat* jacOut = nullptr,
        arma::dmat* hessOut = nullptr
    ) const;
    virtual void gradHessReduced(
        const arma::dcolvec& params,
        const arma::Col<int>& subsetIdents,
        const bool threaded = true,
        arma::dmat* gradOut = nullptr,
        arma::dmat* jacOut = nullptr,
        arma::dmat* hessOut = nullptr
    ) const;
    

    /// @brief Calculate suggested starting values for ML estimation
    /// @return Column vector of starting values
    virtual arma::dcolvec startingValues() const;

    // reference to the data object
    virtual const std::shared_ptr<ESADataBase> getDataObj() const {
        return this->dataObjPtr;
    }

    /**
     * @brief Return Model Type
     * @return Element from ESASfaModelType enumeration
     */
    virtual const ESASfaModelType getModelType() const {
        return this->dataObjPtr->getModelType();
    }

    /**
     * @brief Return sigma parameters
     */
    virtual ESASigmaParams getSigmaParams(const arma::dcolvec& par) const;

    /**
     * @brief Return N (e.g., firms, or nobs for cross sectional model)
     * @return integer denoting N
     */
    virtual double getN() const;

    const std::shared_ptr<arma::dmat> getHaltonDrawsPtr() const {
        if (!haltonDraws) return nullptr;
        return haltonDraws;
    }

    double getProdCost() const { return s; }

    virtual const HaltonSettings* getHaltonSettings() const { return nullptr; }

    /**
     * @brief Method of Moments for starting values
     * @param
     * @param
     * @param
     */
    template <typename TY, typename TZu, typename TZv>
    MoMResult getMoMComponents(
        const arma::Base<double, TY>& yIn,
        const std::optional<TZu>& zuOpt,
        const std::optional<TZv>& zvOpt,
        const std::optional<double>& m2override = std::nullopt
    ) const;

protected:
    // items any derived classes can use
    std::shared_ptr<ESADataBase> dataObjPtr;
    const double s;
    std::shared_ptr<arma::dmat> haltonDraws = nullptr;
};

#endif // ESASFABASE_HPP
