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

#include "efficiencies/ESASfaEffLcmTre.hpp"
#include "efficiencies/ESASfaEffGtre.hpp"
#include "utils/log/logs.hpp"
#include "utils/esautils.hpp"

ESASfaEffLcmTre::ESASfaEffLcmTre(
    std::shared_ptr<ESADataPanelLCM> lcmDataPtr,
    std::shared_ptr<ESADataPanel> helperDataPtr,
    const double s
) : lcmDataPtr(lcmDataPtr), helperDataPtr(helperDataPtr), s(s) {}

std::unique_ptr<ESASfaEffLcmScores> ESASfaEffLcmTre::efficiencyScores(
    const arma::dcolvec& par,
    const arma::dmat& posteriors,
    const int nsim,
    const int haltonStart,
    const int seed,
    const bool threaded
)
{
    ESADataPanelLCM& lcmData = *lcmDataPtr;
    ESADataPanel& dataObj = *helperDataPtr;
    unsigned int nClasses = lcmData.getNClasses();
    int nObs = dataObj.getNobs();
    unsigned int nFirms = static_cast<unsigned int>(lcmData.getNids());
    // for each class, compute efficiency scores using the existing TRE logic
    arma::dmat transientPerClass(nObs, nClasses, arma::fill::zeros);
    arma::dmat persistentPerClass(nObs, nClasses, arma::fill::zeros);
    // iterate thru latent classes
    for (unsigned int c = 0; c < nClasses; c++) {
        arma::dcolvec par_c = lcmutils::buildClassParamVec(lcmData, par, c);
        // use existing TRE logic
        ESASfaEffGtre effCalc(
            std::static_pointer_cast<ESADataBase>(helperDataPtr), s
        );
        std::unique_ptr<ESASfaEffScores> classEff = effCalc.efficiencyScores(
            par_c, nsim, haltonStart, seed, threaded
        );
        // extract elements
        if (classEff && classEff->transient) {
            transientPerClass.col(c) = classEff->transient.value();
        }
        if (classEff && classEff->persistent) {
            persistentPerClass.col(c) = classEff->persistent.value();
        }
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
    // posterior-weighted efficiencies
    arma::dmat transientWeighted = arma::sum(transientPerClass % posteriorsObs, 1);
    arma::dmat persistentWeighted = arma::sum(persistentPerClass % posteriorsObs, 1);
    // build return struct
    auto result = std::make_unique<ESASfaEffLcmScores>();
    result->posteriors = posteriors;
    result->transientPerClass = transientPerClass;
    result->transientWeighted = transientWeighted;
    result->persistentPerClass = std::make_optional(persistentPerClass);
    result->persistentWeighted = std::make_optional(persistentWeighted);
    return result;
}
