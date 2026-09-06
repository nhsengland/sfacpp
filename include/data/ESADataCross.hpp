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
 * @file ESADataCross.hpp
 * @brief ESADataCross class header file
 * @date 2025-02-01
 * @author Edmund Haacke
 */

#ifndef ESA_DATA_CROSS_HPP
#define ESA_DATA_CROSS_HPP

#include <utility>
#include <optional>
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
#include "data/ESADataBase.hpp"
#include "utils/enums.hpp"

class ESADataCross : public ESADataBase {

public:

    /// @brief Constructor
    /// @param y Column vector of the dependent variable
    /// @param x Matrix of the independent variables
    /// @param mu Optional matrix of the variables determining inefficiency mean for truncated norm dist
    /// @param zu Optional matrix of the variables determining inefficiency (half-, and trunc- normal)
    /// @param zv Optional matrix of the variables determining stochastic noise (half-, and trunc- normal)
    /// @param modelType From ESASfaModelType enumeration defining the model type in use
    ESADataCross(
        const arma::dcolvec* y,
        const arma::dmat* x,
        const arma::dmat* mu,
        const arma::dmat* zu,
        const arma::dmat* zv,
        const ESASfaModelType modelType,
        bool arraysContiguous = true
    );

    /// @brief Callable function to process the data
    /// @param cb callable function (y, x, mu, zu, zv)
    /// @return matrix of results
    arma::dmat data_callable(
        std::function<arma::dmat(
            const arma::dcolvec&,
            const arma::dmat&,
            const std::optional<arma::dmat>&,
            const std::optional<arma::dmat>&,
            const std::optional<arma::dmat>&
        )> cb
    ) const;

    /// @brief Callable function to process the data
    /// @param cb callable function (y, x, mu, zu, zv)
    /// @return double
    double data_callable_dbl(
        std::function<double(
            const arma::dcolvec&,
            const arma::dmat&,
            const std::optional<arma::dmat>&,
            const std::optional<arma::dmat>&,
            const std::optional<arma::dmat>&
        )> cb
    ) const;

    /**
     * @brief Callable function to process the data
     * @param cb callable function (y, x, mu, zu, zv)
     * @param ele1 pointer to matrix for element one
     * @param ele2 pointer to matrix for element two
     * @param shouldSumEle1 whether or not to sum element 1
     * @param shouldSumEle2 whether or not to sum element 2
     */
    void data_callable(
        std::function<void(
            const arma::dcolvec&,
            const arma::dmat&,
            const std::optional<arma::dmat>&,
            const std::optional<arma::dmat>&,
            const std::optional<arma::dmat>&,
            arma::dmat*,
            arma::dmat*
        )> cb,
        arma::dmat* ele1,
        arma::dmat* ele2
    ) const;

    /// @brief Return the range of the parameters for the X variables
    /// @return range of the parameters for the X variables
    std::pair<int, int> getXRange(const arma::dcolvec& params) const override;

    /// @brief Return the range of the parameters for the mu variables
    /// @return range of the parameters for the mu variables
    std::optional<std::pair<int, int>> getMuRange(const arma::dcolvec& params) const;

    /// @brief Return the range of the parameters for the Zu variables
    /// @return range of the parameters for the Zu variables
    std::optional<std::pair<int, int>> getZuRange(const arma::dcolvec& params) const;

    /// @brief Return the range of the parameters for the Zv variables
    /// @return range of the parameters for the Zv variables
    std::optional<std::pair<int, int>> getZvRange(const arma::dcolvec& params) const;

    /// @brief Return the parameters corresponding to the X variables
    /// @param params column vector of parameters
    /// @return parameters corresponding to the X variables
    arma::dcolvec paramX(const arma::dcolvec& params) const override;

    /// @brief Return the parameters corresponding to the mu variables
    /// @param params column vector of parameters
    /// @return parameters corresponding to the mu variables
    arma::dcolvec paramMu(const arma::dcolvec& params) const;

    /// @brief Return the parameters corresponding to the Zu variables
    /// @param params column vector of parameters
    /// @return parameters corresponding to the Zu variables
    arma::dcolvec paramZu(const arma::dcolvec& params) const;

    /// @brief Return the parameters corresponding to the Zv variables
    /// @param params column vector of parameters
    /// @return parameters corresponding to the Zv variables
    arma::dcolvec paramZv(const arma::dcolvec& params) const;

    /// @brief Return data for mu
    /// @return matrix of mu
    const std::optional<arma::dmat>& getMu() const { return mu; }

    /// @brief Return data for zu
    /// @return matrix of zu
    const std::optional<arma::dmat>& getZu() const { return zu; }

    /// @brief Return data for zv
    /// @return matrix of zv
    const std::optional<arma::dmat>& getZv() const { return zv; }

    /// @brief Return total number of parameters
    /// @return total number of parameters
    unsigned int nParams() const override;

private:
    std::optional<arma::dmat> mu;
    std::optional<arma::dmat> zu;
    std::optional<arma::dmat> zv;
    const unsigned int nMu;
    const unsigned int nZu;
    const unsigned int nZv;

    
    void checkParamsShape(const arma::dcolvec& params) const override;

    /// @brief find number of mu variables
    unsigned int findMu(){ return mu ? mu->n_cols : 0; };

    /// @brief find number of zu variables
    unsigned int findZu(){ return zu ? zu->n_cols : 0; };
    
    /// @brief find number of zv variables
    unsigned int findZv(){ return zv ? zv->n_cols : 0; };
};


#endif // ESA_DATA_CROSS_HPP