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
 * @file sandwich.cpp
 * @author edmund haacke
 * @date 2025-12-14
 */

#include <map>
#include "utils/sandwich.hpp"
#include "utils/log/logs.hpp"

arma::dmat sandwich::clusteredVcov(const arma::dmat& vcov, const arma::dmat& jac, const arma::Col<int>& idVec)
{
    // check idVec is same length as jacobian
    if (idVec.n_rows != jac.n_rows) {

        ESALogger::logger()->warn("Could not compute clustered standard errors: mismatch in length between id vec ({}) and jacobian ({})", idVec.n_rows, jac.n_rows);
        return vcov;
    }
    std::map<int, arma::rowvec> clusterMap;
    for (size_t i = 0; i < jac.n_rows; i++) {
        int cid = idVec.at(i);
        if (clusterMap.find(cid) == clusterMap.end()) clusterMap[cid] = jac.row(i);
        else clusterMap[cid] += jac.row(i);
    }
    arma::mat U(clusterMap.size(), jac.n_cols);
    int g = 0;
    for (auto const& [key, val] : clusterMap) U.row(g++) = val;
    arma::dmat meat = U.t() * U;
    arma::mat sandwich = vcov * meat * vcov;
    double N = (double)jac.n_rows;
    double K = (double)jac.n_cols;
    double G = (double)clusterMap.size();
    double adj = (N > K && G > 1) ? ((N - 1.0) / (N - K)) * (G / (G - 1.0)) : 1.0;
    return sandwich * adj;
}