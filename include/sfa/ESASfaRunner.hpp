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
 * @file ESASfaRunner.hpp
 * @author edmund haacke
 * @date 2025-12-14
 */

#ifndef ESA_SFA_RUNNER_HPP
#define ESA_SFA_RUNNER_HPP

#include <memory>
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
#include "data/ESADataBase.hpp"
#include "sfa/ESASfaBase.hpp"
#include "utils/enums.hpp"
#include "optim/ESAOptimResult.hpp"
#include "marginaleffects/ESASfaMeff.hpp"
#include "model/ESASfaModelTerms.hpp"
#include "sfa/HaltonSettings.hpp"
#include "efficiencies/effscores.hpp"
#include "efficiencies/ESASfaEffGtre.hpp"

class ESASfaRunner {

public:
    // Inputs
    const int prodCost;
    const std::string modelTypeStr;
    const std::string distStr;
    const int seed;
    const int nsim;
    const int printLevel;
    // Internal enums
    ESASfaModelType mT;
    ESASfaModelFamily mF;
    ESASfaModelDistribution mD;
    std::shared_ptr<ESADataBase> dataObjPtr;
    std::shared_ptr<ESADataBase> lcmDataObjPtr;  // ESADataPanelLCM for LC models
    std::shared_ptr<ESASfaBase> modelObjPtr;
    std::shared_ptr<ESASfaModelTerms> modelTerms;

    // constructor
    ESASfaRunner(
        const std::string& model,
        const std::string& dist,
        const int prodCost,
        const int seed,
        const int nsim,
        const int printLevel = 1
    );

    ESASfaRunner(
        const ESASfaModelType mT,
        const int prodCost,
        const int seed,
        const int nsim,
        const int printLevel = 1
    );

    void loadData(
        const arma::dcolvec* y,
        const arma::dmat* x,
        const arma::dmat* zmuit = nullptr,
        const arma::dmat* zuit = nullptr,
        const arma::dmat* zvit = nullptr,
        const arma::dmat* zui0 = nullptr,
        const arma::dmat* zvi0 = nullptr,
        const arma::Col<int>* idVec = nullptr,
        const arma::Col<int>* timeVec = nullptr
    );

    void loadDataLCM(
        const arma::dcolvec* y,
        const arma::dmat* x,
        const arma::dmat* seg,
        const arma::dmat* zmuit,
        const arma::dmat* zuit,
        const arma::dmat* zvit,
        const arma::dmat* zvi0,
        const arma::Col<int>* idVec,
        const arma::Col<int>* timeVec,
        const unsigned int nClasses
    );

    void setupModel(
        const HaltonSettings hsetting,
        const std::optional<std::vector<std::string>> termsX,
        const std::optional<std::vector<std::string>> termsZmuit,
        const std::optional<std::vector<std::string>> termsZuit,
        const std::optional<std::vector<std::string>> termsZvit,
        const std::optional<std::vector<std::string>> termsZui0,
        const std::optional<std::vector<std::string>> termsZvi0
    );

    /**
     * 
     */
    std::unique_ptr<ESAOptimResult> runOptimization(
        const std::optional<arma::dcolvec> startVals,
        HessianCalcMethod hessCalcMethod,
        const int hessianCalcNumApproxAccuracy
    );

    /**
     * 
     */
    arma::dmat buildSummary(const arma::dcolvec& coefs, const arma::dmat& vcov, const double confLevel, const int degreesFreedom);

    /**
     * 
     */
    void estimateMarginalEffects(
        const std::string& method,
        const arma::dcolvec& coefs,
        const bool esimateCI,
        const double confLvl,
        const int bootstrapReps,
        std::unique_ptr<ESASfaMeffReturn>& meffOut,
        std::unique_ptr<ESASfaMeffCIReturn>& meffCIOut
    );

    /**
     * 
     */
    std::unique_ptr<ESASfaEffScores> estimateEfficiencyScores(
        const arma::dcolvec& coefs,
        const int ghkSims = 1000,
        const int haltonStart = 2
    );
    
};

#endif // ESA_SFA_RUNNER_HPP