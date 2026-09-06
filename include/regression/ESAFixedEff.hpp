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

#ifndef ESA_FIXED_EFF_HPP
#define ESA_FIXED_EFF_HPP

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
#include "utils/esautils.hpp"

typedef struct FixedEffResult {
    arma::dmat params;
    arma::dmat residual;
    arma::dmat fixedEffect;
} FixedEffResult;

class ESAFixedEff {
public:
    // Constructor
    ESAFixedEff(
        const arma::dmat* y,
        const arma::dmat* X,
        const arma::Col<int>* idVec
    ) : y(esautils::makeZeroCopyStrictViewNoOpt(y)),
        X(esautils::makeZeroCopyStrictViewNoOpt(X)),
        idVec(esautils::makeZeroCopyStrictViewColNoOpt<int>(idVec))
    {

    }

    FixedEffResult fit() const
    {
        // initialization
        FixedEffResult result;
        // unique identifiers
        const arma::Col<int> uniqIds = arma::unique(this->idVec);
        const int N = uniqIds.n_rows;
        const int nobs = this->y.n_rows;
        // to store within transformation
        arma::dcolvec yWithin = this->y;
        arma::dmat XWithin = this->X;
        arma::dcolvec yMeans(nobs);
        // first, demean y, X by id
        for (int i = 0; i < N; i++) {
            int currId = uniqIds(i);
            // locate indicies matching this id
            arma::uvec inds = arma::find(this->idVec == currId);
            // calculate means for this group
            double y_bar_i = arma::mean(arma::mean(this->y.rows(inds)));
            arma::rowvec x_bar_i = arma::mean(this->X.rows(inds), 0);
            // subtract means (within transformation)
            yWithin.elem(inds) -= y_bar_i;
            // XWithin.rows(inds).each_col() -= x_bar_i;
            XWithin.rows(inds) -= arma::repmat(x_bar_i, inds.n_elem, 1);
            // store y means (for nobs)
            yMeans.elem(inds).fill(y_bar_i);
        }
        // estimate beta using OLS on demeaned data: y_within = X_within * beta
        arma::dcolvec betaFe = arma::pinv(XWithin) * yWithin;
        // recover overall intercept and residuals
        double overallIntercept = arma::mean(this->y) - arma::dot(arma::mean(this->X, 0), betaFe);
        // reconstruct total residuals
        arma::dcolvec totalResid = this->y - overallIntercept - this->X*betaFe;
        // reconstruct parameter vector
        betaFe(0) = overallIntercept;
        result.params = betaFe;
        result.residual = totalResid;
        return result;
    }

private:
    const arma::dcolvec y;
    const arma::dmat X;
    const arma::Col<int> idVec;


};

#endif // ESA_FIXED_EFF_HPP