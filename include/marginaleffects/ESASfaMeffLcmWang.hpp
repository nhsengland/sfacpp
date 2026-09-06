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

#ifndef ESA_SFA_MEFF_LCM_WANG_HPP
#define ESA_SFA_MEFF_LCM_WANG_HPP

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

#include "marginaleffects/ESASfaMeffWang.hpp"
#include "data/ESADataLCM.hpp"
#include "data/ESADataPanel.hpp"
#include "model/ESASfaModelTerms.hpp"
#include "utils/lcmutils.hpp"

struct ESASfaMeffLcmReturn {
    arma::dmat marginalEffectsWeighted;             // nObs x K
    std::vector<arma::dmat> marginalEffectsPerClass; // C matrices, each nObs x K
    std::vector<std::string> columnNames;
};

class ESASfaMeffLcmWang : public ESASfaMeffWang {

public:
    ESASfaMeffLcmWang(
        std::shared_ptr<ESADataPanelLCM> lcmDataPtr,
        std::shared_ptr<ESADataBase> helperDataPtr,
        const int s,
        const int nsim,
        const int seed
    );

    /**
     * @brief Calculate posterior-weighted marginal effects for LC-TRE model
     * @param par Full LC-TRE parameter vector
     * @param posteriors Pre-computed posteriors (nFirms x C)
     * @param modelTerms Model terms object
     * @return Struct with per-class and weighted marginal effects
     */
    std::unique_ptr<ESASfaMeffLcmReturn> lcmMarginalEffects(
        const arma::dcolvec& par,
        const arma::dmat& posteriors,
        const ESASfaModelTerms& modelTerms
    ) const;

private:
    std::shared_ptr<ESADataPanelLCM> lcmDataPtr;
};

#endif // ESA_SFA_MEFF_LCM_WANG_HPP
