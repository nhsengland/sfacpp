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
 * 
 */

#include "optim/ESAOptimResult.hpp"
#include "math/esamath.hpp"
#include "math/esatdist.hpp"
#include "utils/esautils.hpp"

// variance-covariance matrix, based on the hessian
arma::dmat ESAOptimResultSuccess::getVcov(double eigentol) const
{
    arma::dmat h1 = arma::inv(-this->hess);
    // since we have been using the AVERAGE hessian, need to scale it back up
    // so divide h1 through N
    // h1 = h1 / this->N;
    return h1;
}

// Model summary
arma::dmat ESAOptimResultSuccess::modelSummary(double conflvl) const
{
    // calculate the standard errors
    arma::dmat vcov = this->getVcov();
    arma::dmat se = arma::sqrt(arma::abs(vcov.diag()));
    // calculate t-statistics
    arma::dmat t = this->x / se;
    arma::dmat tabs = arma::abs(t);
    // degrees of freedom - number of observations less the number of parameters
    unsigned int df = this->nobs - this->x.n_rows;
    // calculate critical value - exclude lower tail
    double t1 = esatdist::qt((1.0 - conflvl) / 2.0, df, 0.0, false);
    // calculate p-values (exc. lower tail)
    arma::dmat p = esatdist::pt(tabs, df, 0.0, false) * 2.0;
    // calculate confident intervals
    arma::dmat ciLwr = this->x - (t1 * se);
    arma::dmat ciUpr = this->x + (t1 * se);
    // create output matrix
    arma::dmat res(this->x.n_rows, 6);
    // set column 1: parameter estimates
    res.col(0) = this->x;
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