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

#include "marginaleffects/ESASfaMeffLcmWang.hpp"
#include "utils/log/logs.hpp"
#include "utils/esautils.hpp"

ESASfaMeffLcmWang::ESASfaMeffLcmWang(
    std::shared_ptr<ESADataPanelLCM> lcmDataPtr,
    std::shared_ptr<ESADataBase> helperDataPtr,
    const int s,
    const int nsim,
    const int seed
) : ESASfaMeffWang(helperDataPtr, s, nsim, seed),
    lcmDataPtr(lcmDataPtr)
{}

std::unique_ptr<ESASfaMeffLcmReturn> ESASfaMeffLcmWang::lcmMarginalEffects(
    const arma::dcolvec& par,
    const arma::dmat& posteriors,
    const ESASfaModelTerms& modelTerms
) const
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    unsigned int nClasses = lcmData.getNClasses();
    int nObs = dataObj.getNobs();
    unsigned int nFirms = static_cast<unsigned int>(lcmData.getNids());
    // allocate vec to hold marginal fxt for each class
    std::vector<arma::dmat> meffPerClass(nClasses);
    std::vector<std::string> colNames;
    // iterate thru each latent class
    for (unsigned int c = 0; c < nClasses; c++) {
        arma::dcolvec par_c = lcmutils::buildClassParamVec(lcmData, par, c);
        ESASfaMeffReturn classMeff = this->marginalEffects(par_c, modelTerms);
        meffPerClass[c] = classMeff.marginalEffects;
        if (c == 0) {
            colNames = classMeff.columnNames;
        }
    }
    if (colNames.empty()) {
        return nullptr;
    }
    // expand firm-level posteriors to observation level
    arma::dmat posteriorsObs(nObs, nClasses, arma::fill::zeros);
    auto expandInner = [&posteriors, &nClasses](
        const unsigned int idx,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zui0,
        const auto& zvi0,
        arma::dmat* e1,
        arma::dmat* e2
    ) {
        int nT = y.get_ref().n_rows;
        *e1 = arma::repmat(posteriors.row(idx), nT, 1);
    };
    arma::dmat posteriorsExpanded;
    dataObj.panelCallable(expandInner, &posteriorsExpanded, nullptr, false, false);
    posteriorsObs = posteriorsExpanded;
    // posterior-weighted marginal effects
    int nK = colNames.size();
    arma::dmat weighted(nObs, nK, arma::fill::zeros);
    for (unsigned int c = 0; c < nClasses; c++) {
        for (int k = 0; k < nK; k++) {
            weighted.col(k) += meffPerClass[c].col(k) % posteriorsObs.col(c);
        }
    }
    // build return struct
    auto result = std::make_unique<ESASfaMeffLcmReturn>();
    result->marginalEffectsWeighted = weighted;
    result->marginalEffectsPerClass = meffPerClass;
    result->columnNames = colNames;
    return result;
}
