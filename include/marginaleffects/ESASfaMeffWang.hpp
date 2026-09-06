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


// File: ESASfaMeffWang.hpp
// Author: Edmund Haacke
// Date: 2024-12-28
// Description:
//     Class to derive the non-monotonic marginal effects as stated by Wang (2002) on inefficiency dE(u_it) / dz[k].
//     Non-monotonicity means that each determinant can have both a positive and negative effect on the inefficiency,
//     with the sign of the effect depending on the values of z_it.
//     It is noted that these marginal effects are on unconditional E(u_it) rather than conditional E(u_it|eps_it).
//     Kumbhakar and Sun () argue that since the JLMS estimator is used for estimating inefficiency, which is
//     conditional on eps_it, the marginal effects should also be based on the same formula.

#ifndef ESA_SFA_MEFF_WANG_HPP
#define ESA_SFA_MEFF_WANG_HPP

#include <vector>
#include <memory>
#include <string>
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

#include "marginaleffects/ESASfaMeff.hpp"
#include "data/ESADataBase.hpp"
#include "model/ESASfaModelTerms.hpp"
#include "data/ESADataPanel.hpp"

class ESASfaMeffWang : public ESASfaMeff {

protected:

    template <typename TMu, typename TZu>
    arma::dmat calculateWangMeff(
        const std::optional<arma::dcolvec>& bmu,
        const std::optional<arma::dcolvec>& bzu,
        const std::optional<TMu>& muIn,
        const std::optional<TZu>& zuIn,
        const std::vector<std::string>& outColNames,
        const ESASfaModelTerms& modelTerms,
        const int nobs,
        const std::optional<ESASfaModelTermLocation> locMean,
        const ESASfaModelTermLocation locZu
    ) const;

    // arma::dmat calculateWangMeff(
    //     const std::optional<arma::dcolvec>& bZmuit,
    //     const std::optional<arma::dcolvec>& bZuit,
    //     const std::optional<arma::dmat>& zmuit,
    //     const std::optional<arma::dmat>& zuit,
    //     const std::vector<std::string>& outColNames,
    //     const ESASfaModelTerms& modelTerms,
    //     const int nobs
    // ) const;

public:
    /// Constructor
    ESASfaMeffWang(
        const std::shared_ptr<ESADataBase> dataObjPtr,
        const int s,
        const int nsim,
        const int seed
    ) : ESASfaMeff(dataObjPtr, s, nsim, seed){};

    /// @brief Calculate the marginal effects of the inefficiency on the determinant
    /// @param par The parameter vector
    /// @param modelTerms The model terms
    /// @return The marginal effects
    ESASfaMeffReturn marginalEffects(const arma::dcolvec& par, const ESASfaModelTerms& modelTerms) const override;

};

#endif // ESA_SFA_MEFF_WANG_HPP