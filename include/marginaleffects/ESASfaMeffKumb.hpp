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


// File: ESASfaMeffKumb.hpp
// Author: Edmund Haacke
// Date: 2024-12-28
// Description:
//     Class to derive the non-monotonic marginal effects as stated by Kumbhakar and Sun () on inefficiency dE(u_it|eps_it) / dz[k].

#ifndef ESA_SFA_MEFF_KUMB_HPP
#define ESA_SFA_MEFF_KUMB_HPP

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

#include "marginaleffects/ESASfaMeff.hpp"

class ESASfaMeffKumb : public ESASfaMeff {

public:
    /// Constructor
    ESASfaMeffKumb(
        const std::shared_ptr<ESADataPanel> dataObj,
        const int s,
        const int nsim,
        const int seed
    ) : ESASfaMeff(dataObjPtr, s, nsim, seed){};

    /// @brief Calculate the marginal effects of the inefficiency on the determinant
    ESASfaMeffReturn marginalEffects(const arma::dcolvec& par, const ESASfaModelTerms& modelTerms) const override;

};

#endif // ESA_SFA_MEFF_KUMB_HPP