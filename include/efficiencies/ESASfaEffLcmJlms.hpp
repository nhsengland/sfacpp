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

#ifndef ESA_SFA_EFF_LCM_JLMS_HPP
#define ESA_SFA_EFF_LCM_JLMS_HPP

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

#include <memory>
#include "data/ESADataLCM.hpp"
#include "data/ESADataPanel.hpp"
#include "efficiencies/ESASfaEffLcmTre.hpp"
#include "utils/lcmutils.hpp"

class ESASfaEffLcmJlms {
public:
    ESASfaEffLcmJlms(
        std::shared_ptr<ESADataPanelLCM> lcmDataPtr,
        std::shared_ptr<ESADataPanel> helperDataPtr,
        const double s
    );

    /**
     * @brief Compute per-class JLMS efficiency scores and posterior-weight them
     * @param par Full LC-TRE parameter vector
     * @param posteriors precomputed posterior class probabilities (nFirms x C)
     * @param nsim number of simulation draws for integrating out the random effect
     * @param haltonStart offset for Halton prime bases
     * @param seed random seed
     * @return Struct containing per-class and weighted efficiency scores
     */
    std::unique_ptr<ESASfaEffLcmScores> efficiencyScores(
        const arma::dcolvec& par,
        const arma::dmat& posteriors,
        const int nsim = 100,
        const int haltonStart = 0,
        const int seed = 7
    );

private:
    std::shared_ptr<ESADataPanelLCM> lcmDataPtr;
    std::shared_ptr<ESADataPanel> helperDataPtr;
    const double s;
};

#endif // ESA_SFA_EFF_LCM_JLMS_HPP
