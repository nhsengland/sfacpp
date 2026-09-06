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
 * @file ESADataCross.cpp
 * @brief ESADataCross class implementation file
 * @date 2025-02-01
 * @author Edmund Haacke
 */

#include "data/ESADataCross.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"

/// Constructor
ESADataCross::ESADataCross(
    const arma::dcolvec* y,
    const arma::dmat* x,
    const arma::dmat* mu,
    const arma::dmat* zu,
    const arma::dmat* zv,
    const ESASfaModelType modelType,
    const bool arraysContiguous
) : ESADataBase(y, x, modelType, arraysContiguous),
    mu(esautils::makeZeroCopyStrictView(mu)),
    zu(esautils::makeZeroCopyStrictView(zu)),
    zv(esautils::makeZeroCopyStrictView(zv)),
    nMu(this->findMu()),
    nZu(this->findZu()),
    nZv(this->findZv())
{
    // check if number of rows in zu, zv align with y
    if (zu && zu->n_rows != nobs){
        throw std::invalid_argument("zu must have the same number of rows as y");
    }
    if (zv && zv->n_rows != nobs){
        throw std::invalid_argument("zv must have the same number of rows as y");
    }
    if (mu && mu->n_rows != nobs){
        throw std::invalid_argument("mu must have the same number of rows as y");
    }
}

/// Callable function to process the data - return matrix
arma::dmat ESADataCross::data_callable(
    std::function<arma::dmat(
        const arma::dcolvec&,
        const arma::dmat&,
        const std::optional<arma::dmat>&,
        const std::optional<arma::dmat>&,
        const std::optional<arma::dmat>&
    )> cb
) const
{
    return cb(y, x, mu, zu, zv);
}

/// Callable function to process the data - return double
double ESADataCross::data_callable_dbl(
    std::function<double(
        const arma::dcolvec&,
        const arma::dmat&,
        const std::optional<arma::dmat>&,
        const std::optional<arma::dmat>&,
        const std::optional<arma::dmat>&
    )> cb
) const
{
    return cb(y, x, mu, zu, zv);
}

////
void ESADataCross::data_callable(
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
) const
{
    arma::dmat e1, e2;
    cb(y, x, mu, zu, zv, &e1, &e2);
    if (ele1) *ele1 = e1;
    if (ele2) *ele2 = e2;
}

/// Check shape of params
void ESADataCross::checkParamsShape(const arma::dcolvec& params) const
{
    if (params.n_rows > (nX + nMu + nZu + nZv)) {
        throw std::invalid_argument("parameters exceeds the number of available elements in x, mu, zu, zv");
    }
}

/// Get range of the parameters for the X variables
std::pair<int, int> ESADataCross::getXRange(const arma::dcolvec& params) const
{
    checkParamsShape(params);
    return std::make_pair(0, nX - 1);
}

/// Get range of the parameters for the mu variables
std::optional<std::pair<int, int>> ESADataCross::getMuRange(const arma::dcolvec& params) const
{
    checkParamsShape(params);
    if (!mu) {
        return std::nullopt;
    }
    std::pair<int, int> r = std::make_pair(nX, nX + nMu - 1);
    return std::make_optional<std::pair<int, int>>(std::move(r));
}

/// Get range of the parameters for the zu variables
std::optional<std::pair<int, int>> ESADataCross::getZuRange(const arma::dcolvec& params) const
{
    checkParamsShape(params);
    if (!zu) {
        return std::nullopt;
    }
    std::pair<int, int> r = std::make_pair(nX + nMu, nX + nMu + nZu - 1);
    return std::make_optional<std::pair<int, int>>(std::move(r));
}

/// Get range of the parameters for the zv variables
std::optional<std::pair<int, int>> ESADataCross::getZvRange(const arma::dcolvec& params) const
{
    checkParamsShape(params);
    if (!zv) {
        return std::nullopt;
    }
    std::pair<int, int> r = std::make_pair(nX + nMu + nZu, nX + nMu + nZu + nZv - 1);
    return std::make_optional<std::pair<int, int>>(std::move(r));
}

/// Get parameters corresponding to the X variables
arma::dcolvec ESADataCross::paramX(const arma::dcolvec& params) const
{
    std::pair<int, int> r = getXRange(params);
    return params.rows(r.first, r.second);
}

/// Get parameters corresponding to the mu variables
arma::dcolvec ESADataCross::paramMu(const arma::dcolvec& params) const
{
    std::optional<std::pair<int, int>> r = getMuRange(params);
    if (!r) return arma::dcolvec();
    return params.rows(r->first, r->second);
}

/// Get parameters corresponding to the zu variables
arma::dcolvec ESADataCross::paramZu(const arma::dcolvec& params) const
{
    std::optional<std::pair<int, int>> r = getZuRange(params);
    if (!r) return arma::dcolvec();
    return params.rows(r->first, r->second);
}

/// Get parameters corresponding to the zv variables
arma::dcolvec ESADataCross::paramZv(const arma::dcolvec& params) const
{
    std::optional<std::pair<int, int>> r = getZvRange(params);
    if (!r) return arma::dcolvec();
    return params.rows(r->first, r->second);
}

unsigned int ESADataCross::nParams() const {
    return (nX + nMu + nZu + nZv);
}