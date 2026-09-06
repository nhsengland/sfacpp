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
 * @file modelsummary.hpp
 * @author edmund haacke
 * @date 2025-12-14
 */

#ifndef MODEL_SUMMARY_HPP
#define MODEL_SUMMARY_HPP

// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif

namespace modelsummary {

    arma::mat getSummary(const arma::dcolvec& coefs, const arma::dmat& vcov, double confLevel, unsigned int df);
    arma::mat getSummary(const arma::dcolvec& coefs, const arma::dmat& vcov, double confLevel, unsigned int nobs, unsigned int nparams);

}

#endif // MODEL_SUMMARY_HPP