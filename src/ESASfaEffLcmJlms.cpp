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

#include "efficiencies/ESASfaEffLcmJlms.hpp"
#include "efficiencies/ESASfaJlms.hpp"
#include "utils/log/logs.hpp"

ESASfaEffLcmJlms::ESASfaEffLcmJlms(
    std::shared_ptr<ESADataPanelLCM> lcmDataPtr,
    std::shared_ptr<ESADataPanel> helperDataPtr,
    const double s
) : lcmDataPtr(lcmDataPtr), helperDataPtr(helperDataPtr), s(s) {}

std::unique_ptr<ESASfaEffLcmScores> ESASfaEffLcmJlms::efficiencyScores(
    const arma::dcolvec& par,
    const arma::dmat& posteriors,
    const int nsim,
    const int haltonStart,
    const int seed
)
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    ESADataPanel& dataObj = *helperDataPtr;
    unsigned int nClasses = lcmData.getNClasses();
    int nObs = dataObj.getNobs();
    unsigned int nFirms = static_cast<unsigned int>(lcmData.getNids());

    arma::dmat transientPerClass(nObs, nClasses, arma::fill::zeros);

    for (unsigned int c = 0; c < nClasses; c++) {
        arma::dcolvec par_c = lcmutils::buildClassParamVec(lcmData, par, c);

        std::shared_ptr<ESADataBase> helperBase = std::static_pointer_cast<ESADataBase>(helperDataPtr);
        ESASfaJlms jlmsCalc(helperBase, s);
        arma::dmat classEff = jlmsCalc.effPredJlms(par_c, nsim, haltonStart, seed);
        transientPerClass.col(c) = classEff;
    }

    // expand firm-level posteriors to observation level for weighting
    arma::dmat posteriorsObs(nObs, nClasses, arma::fill::zeros);
    auto expandInner = [&posteriors, &posteriorsObs, &nClasses](
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

    // posterior-weighted transient efficiency
    arma::dmat transientWeighted = arma::sum(transientPerClass % posteriorsObs, 1);

    auto result = std::make_unique<ESASfaEffLcmScores>();
    result->posteriors = posteriors;
    result->transientPerClass = transientPerClass;
    result->transientWeighted = transientWeighted;
    // JLMS for TRE does not produce persistent efficiency scores
    result->persistentPerClass = std::nullopt;
    result->persistentWeighted = std::nullopt;
    return result;
}
