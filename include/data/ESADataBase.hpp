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
 * @file ESADataBase.hpp
 * @brief ESADataBase class header file
 * @date 2025-02-01
 * @author Edmund Haacke
 */


#ifndef ESA_DATA_BASE_HEADER_HPP
#define ESA_DATA_BASE_HEADER_HPP

// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif //RPACKAGE
// --- end armadillo ---
#include "utils/enums.hpp"
#include "utils/esautils.hpp"

class ESADataBase {

public:

    ESADataBase() : y(), x(), modelType(ESASfaModelType::MODEL_UNKNOWN), nX(0), nobs(0) {};

    /// @brief Constructor
    /// @param y Column vector of the dependent variable
    /// @param x Matrix of the independent variables
    /// @param modelType From ESASfaModelType enumeration defining model type in use
    ESADataBase(
        const arma::dcolvec* yIn,
        const arma::dmat* xIn,
        const ESASfaModelType modelType,
        const bool arraysContiguous = true
    ) : y(esautils::makeZeroCopyStrictViewColNoOpt<double>(yIn)),
        x(esautils::makeZeroCopyStrictViewNoOpt(xIn)),
        modelType(modelType),
        nX(this->findX()),
        nobs(y.n_rows),
        arraysContiguous(arraysContiguous)
    {
        // check numberof rows in x and y align
        if (x.n_rows != y.n_rows) throw std::invalid_argument("No. rows in 'x', 'y' must be equal");
    }

    /// @brief destructor
    virtual ~ESADataBase() = default;

    /// @brief Return the number of observations in total
    /// @return number of observations
    const double getNobs() const { return nobs; }

    const int getNX() const { return nX; }

    virtual const double getNids() const { return nobs; }
    virtual const int getMaxT() const { return 0; }
    virtual const int getMinT() const { return 0; }

    /// @brief Return the model type
    /// @return model type
    const ESASfaModelType getModelType() const { return modelType; }

    /// @brief Return column vector of dependent variable
    /// @return column vector of dependent variable
    const arma::dcolvec& getY() { return y; }
    const arma::dcolvec* getYPtr() { return &y; }

    /// @brief Return matrix of independent variables
    /// @return matrix of independent variables
    const arma::dmat getX() { return x; }
    const arma::dmat* getXPtr() { return &x; }

    /// @brief Return the range of the parameters for the X variables
    /// @return range of the parameters for the X variables
    virtual std::pair<int, int> getXRange(const arma::dcolvec& params) const
    {
        checkParamsShape(params);
        return std::make_pair(0, this->nX - 1);
    }

    /// @brief Return the parameters corresponding to the X variables
    /// @param params column vector of parameters
    /// @return parameters corresponding to the X variables
    virtual arma::dcolvec paramX(const arma::dcolvec& params) const
    {
        std::pair<int, int> xrange = getXRange(params);
        return params.rows(xrange.first, xrange.second);
    }

    /// @brief Return total number of variables (all components)
    /// @return total number of variables
    inline virtual unsigned int nParams() const { return this->nX; }

    /// @brief Return if the x matrix has an intercept term
    /// @return true if intercept term is present, false otherwise
    virtual bool xHasInterceptTerm() const
    {
        return this->matrixHasInterceptTerm(this->x);
    }

protected:

    arma::dcolvec y;
    arma::dmat x;
    const ESASfaModelType modelType;
    const unsigned int nX;
    int nobs;
    bool arraysContiguous;

    /// @brief Check parameter shape is within bounds
    /// @param params column vector of parameters
    /// @throw std::invalid_argument if params shape is invalid
    virtual void checkParamsShape(const arma::dcolvec& params) const
    {
        if (params.n_rows > nX) {
            throw std::invalid_argument("parameters exceeds the number of available elements in x");
        }
    }
    unsigned int findX() { return x.n_cols; }

    /// @brief Check if a matrix has an intercept term (at column position 0)
    /// @param x matrix to check
    /// @return true if intercept term is present, false otherwise
    bool matrixHasInterceptTerm(const arma::dmat& x) const
    {
        // check if the first column is all ones
        arma::dmat firstCol = x.col(0);
        bool allOnes = arma::all(arma::vectorise(firstCol) == 1.0);
        return allOnes;
    }
};

#endif // ESA_DATA_BASE_HPP