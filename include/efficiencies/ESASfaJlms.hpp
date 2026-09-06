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

#ifndef ESA_SFA_JLMS_HPP
#define ESA_SFA_JLMS_HPP

#include <memory>
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

class ESASfaJlms {

public:
    /// Constructor
    ESASfaJlms(std::shared_ptr<ESADataBase>& dataObjPtr, const double s);

    /// @brief Calculate the inefficiency score (conditional on eps) based on Jondrow et al. (1982)
    /// @param par
    /// @return column vector of estimated inefficiency scores
    arma::dmat ineffPredJlms(const arma::dcolvec& par, const int nsim, const int haltonStart, const int seed);

    /// @brief Calculate the efficiency score (conditional on eps) based on Jondrow et al. (1982)
    /// @param par
    /// @return column vector of estimated efficiency scores
    arma::dmat effPredJlms(const arma::dcolvec& par, const int nsim, const int haltonStart, const int seed);

protected:
    // items any derived classes can use
    const std::shared_ptr<ESADataBase>& dataObjPtr;
    const double s;

    /// @brief Calculate the inefficiency score (conditional on eps) based on Jondrow et al. (1982) for TRE
    /// @param par
    /// @return column vector of inefficiency scores
    arma::dmat ineffPredJlmsTrePanelHalfNormal(
        const unsigned int idx,
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& zuit,
        const arma::dmat& zvit,
        const arma::dmat& zvi0,
        const int s,
        const int nsim,
        const arma::dmat& draws
    );
    arma::dmat ineffPredJlmsTrePanelTruncNormal(
        const unsigned int idx,
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& zuit,
        const arma::dmat& zvit,
        const arma::dmat& zvi0,
        const arma::dmat& muuit,
        const int s,
        const int nsim,
        const arma::dmat& draws
    );

    /// @brief  Calculate the efficiency score (conditional on eps) based on Jondrow et al. (1982) for TRE
    /// @param par 
    /// @return column vector of efficiency scores
    arma::dmat effPredJlmsTrePanelHalfNormal(
        const unsigned int idx,
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& zuit,
        const arma::dmat& zvit,
        const arma::dmat& zvi0,
        const int s,
        const int nsim,
        const arma::dmat& draws
    );
    arma::dmat effPredJlmsTrePanelTruncNormal(
        const unsigned int idx,
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& zuit,
        const arma::dmat& zvit,
        const arma::dmat& zvi0,
        const arma::dmat& muuit,
        const int s,
        const int nsim,
        const arma::dmat& draws
    );

    /// @brief Calculate the inefficiency score (conditional on eps) based on Jondrow et al. (1982) for TFE
    /// @param par 
    /// @return column vector of inefficiency scores
    arma::dmat ineffPredJlmsTfeHalfNormal(
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& zuit,
        const arma::dmat& zvit,
        const int s
    );
    arma::dmat ineffPredJlmsTfeTruncNormal(
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& zuit,
        const arma::dmat& zvit,
        const arma::dmat& muuit,
        const int s
    );

    /// @brief Calculate the efficiency score (conditional on eps) based on Jondrow et al. (1982) for TFE
    /// @param par 
    /// @return column vector of efficiency scores
    arma::dmat effPredJlmsTfeHalfNormal(
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& zuit,
        const arma::dmat& zvit,
        const int s
    );
    arma::dmat effPredJlmsTfeTruncNormal(
        const arma::dcolvec& par,
        const arma::dmat& y,
        const arma::dmat& x,
        const arma::dmat& zuit,
        const arma::dmat& zvit,
        const arma::dmat& muuit,
        const int s
    );
};

#endif // ESA_SFA_JLMS_HPP