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
 * @file modelsummary.cpp
 * @author edmund haacke
 * @date 2025-12-14
 */

#include "utils/modelsummary.hpp"
#include "math/esatdist.hpp"


arma::mat modelsummary::getSummary(const arma::dcolvec& coefs, const arma::dmat& vcov, double confLevel, unsigned int nobs, unsigned int nparams)
{
    // degrees of freedom - number of observations less the number of parameters
    unsigned int df = nobs - nparams;
    // call other method
    return modelsummary::getSummary(coefs, vcov, confLevel, df);
}
arma::mat modelsummary::getSummary(const arma::dcolvec& coefs, const arma::dmat& vcov, double confLevel, unsigned int df)
{
    // calculate the standard errors
    arma::dmat se = arma::sqrt(arma::abs(vcov.diag()));
    // calculate t-statistics
    arma::dmat t = coefs / se;
    arma::dmat tabs = arma::abs(t);
    // calculate critical value - exclude lower tail
    double t1 = esatdist::qt((1.0 - confLevel) / 2.0, df, 0.0, false);
    // calculate p-values (exc. lower tail)
    arma::dmat p = esatdist::pt(tabs, df, 0.0, false) * 2.0;
    // calculate confident intervals
    arma::dmat ciLwr = coefs - (t1 * se);
    arma::dmat ciUpr = coefs + (t1 * se);
    // create output matrix
    arma::dmat res(coefs.n_rows, 6);
    // set column 1: parameter estimates
    res.col(0) = coefs;
    // set column 2: standard errors
    res.col(1) = se;
    // set column 3: tstats
    res.col(2) = t;
    // set column 4: p-values
    res.col(3) = p;
    // set column 5: lower bound of CI
    res.col(4) = ciLwr;
    // set column 6: upper bound of CI
    res.col(5) = ciUpr;
    return res;
}
