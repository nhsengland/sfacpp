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

// File: ESASfaMeff.cpp
// Author: Edmund Haacke
// Date: 2024-12-28
// Description:
//     Parent class for calculating marginal effects of the inefficiency on the determinants.

#include "marginaleffects/ESASfaMeff.hpp"
#include <mutex>
#include <iostream>
#include <optional>
#include "sfa/ESASfaBase.hpp"
#include "sfa/ESASfaGtreBad.hpp"
#include "sfa/ESASfaTfeGreene.hpp"
#include "sfa/ESASfaTreGreene.hpp"
#include "math/esandist.hpp"
#include "utils/esautils.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "utils/esaparallel.hpp"
#include "optim/ESAGlobalOptimParams.hpp"
#include "optim/optimparams.hpp"
#include "optim/esaoptimization.hpp"

/// Constructor
ESASfaMeff::ESASfaMeff(
    const std::shared_ptr<ESADataBase> dataObjPtr,
    const int s,
    const int nsim,
    const int seed
) : dataObjPtr(dataObjPtr), _s(s), _nsim(nsim), _seed(seed){
    // only support panel for the time being
    if (!dynamic_cast<ESADataBase*>(dataObjPtr.get())){
        throw std::invalid_argument("data object is not of type ESADataBase (panel)");
    }
};

ESASfaMeffReturn ESASfaMeff::marginalEffects(const arma::dcolvec& par, const ESASfaModelTerms& modelTerms) const 
{
   // construct an empty marginal effect return struct
   ESASfaMeffReturn out;
   return out;
}

/// Confidence intervals for 
std::unique_ptr<ESASfaMeffCIReturn> ESASfaMeff::bootstrappedCIs(
    const std::shared_ptr<ESASfaBase> mdlPtr,
    const arma::dcolvec& par,
    const ESASfaModelTerms& modelTerms,
    double confidenceLevel,
    const int bsReps,
    const bool useCustomStartVals
){
    // dereference pointer to data object
    ESADataBase& dataObj = (ESADataBase&)*dataObjPtr;
    // upper and lower values for CIs
    double lowerCL = (1.0 - confidenceLevel) / 2.0;
    double upperCL = 1.0 - lowerCL;
    // model family
    ESASfaModelFamily modelFamily = ESAEnums::getModelFamily(dataObj.getModelType());
    // vector of marginal effects
    std::vector<ESASfaMeffReturn> effects;
    // effects.reserve(bsReps);
    std::mutex mtx;
    int successCnt = 0;
    std::optional<arma::dcolvec> optStartVals = std::nullopt;
    if (useCustomStartVals) optStartVals = std::make_optional<arma::dcolvec>(par);
    // loop through the bootstrap samples using multi-threading; // async_par_for
    //esaparallel::thread_par_for(0, bsReps, [this, &effects, &par, &modelFamily, &mtx, &modelTerms, &successCnt, &optStartVals](unsigned i){
    // since we already use parallelerization for calculation of LL/gradient/hessian,
    // use a normal loop to iterate thru bootstrap replications.
    const int maximumRuns = bsReps + bsReps;
    int i = 0;
    // for (int i = 0; i < bsReps; i++){
    while (successCnt < bsReps && i < maximumRuns) {
        // generate the pseudo sample
        ESADataPanel pseudoSample = generatePseudoSample(par, this->_seed + i, this->_s);
        // fit the model to the pseudo sample
        std::unique_ptr<ESAOptimResultSuccess> result = fitBsModel(
            mdlPtr,
            std::make_shared<ESADataPanel>(pseudoSample),
            optStartVals,
            modelFamily,
            this->_s,
            this->_nsim,
            this->_seed
        );
        // check if the optimization was successful (e.g., result is not nullptr)
        if (result != nullptr){
            // get the parameter vector from the optimization result
            arma::dcolvec parBs = result->getX();
            // calculate the marginal effects
            ESASfaMeffReturn mf = marginalEffects(parBs, modelTerms);
            // vectors aren't thread safe, so need to lock it - acquire mutex
            // std::lock_guard<std::mutex> lock(mtx);
            effects.push_back(mf);
            successCnt++;
        }
        i++;
    }//, true);
    // check if it hit the maximum runs, and didnt hit the success numbers
    if (successCnt < bsReps) {
        ESALogger::logger()->error("Ran {} replications, only {} for a target of {} were successful :(", i, successCnt, bsReps);
        return nullptr;
    }
    ESALogger::logger()->info("Bootstrapped marginal effects: {} / {} successful", successCnt, bsReps);
    // iterate thru each of the variables
    // check that 0 index exists
    if (effects.size() == 0) return nullptr;
    // number of variables
    size_t numVars = effects[0].marginalEffects.n_cols;
    std::vector<std::string> meCnames = effects[0].columnNames;
    size_t numObs = dataObj.getNobs();
    // create matricies for both lower and upper bounds
    arma::dmat lowerCI(numObs, numVars);
    arma::dmat upperCI(numObs, numVars);
    // iterate thru each of the variables
    for (size_t i = 0; i < numVars; ++i){
        // for each record / observation, extract the ith column
        for (size_t j = 0; j < numObs; ++j){
            // setup empty vector to store all values (so can calculate percentiles)
            std::vector<double> colVals(effects.size());
            // iterate through each marginal effect
            for (size_t k = 0; k < effects.size(); ++k){
                colVals[k] = effects[k].marginalEffects(j, i);
            }
            // sort the values
            std::sort(colVals.begin(), colVals.end());
            // calculate the lower and upper bounds
            size_t lwrIdx = static_cast<size_t>(lowerCL * colVals.size());
            size_t uprIdx = static_cast<size_t>(upperCL * colVals.size());
            double lwrVal = colVals[lwrIdx];
            double uprVal = colVals[uprIdx];
            // assign to the relevant matrix
            lowerCI(j, i) = lwrVal;
            upperCI(j, i) = uprVal;
        } 
    }
    // create instance of ESASfaMeffCIReturn
    ESASfaMeffCIReturn out;
    out.lowerCI = lowerCI;
    out.upperCI = upperCI;
    out.columnNames = meCnames;
    return std::make_unique<ESASfaMeffCIReturn>(out);
}

std::unique_ptr<ESAOptimResultSuccess> ESASfaMeff::fitBsModel(
    const std::shared_ptr<ESASfaBase>& mdlPtr,
    const std::shared_ptr<ESADataPanel> pseudoSample,
    const std::optional<arma::dcolvec>&optStartVals,
    const ESASfaModelFamily modelFamily,
    const int prodCost,
    const int numSims,
    const int seed
)
{
    std::unique_ptr<ESAOptimResult> resultUnk;
    const unsigned int printLevel = 4;
    // optimization options from global singleton
    ESAGlobalOptimParams *globalOpts = ESAGlobalOptimParams::GetInstance();
    ModelSolver ms = globalOpts->mainModelSolver;
    ESAOptimParams op = globalOpts->mainOptimParams;
    HessianCalcMethod hm = globalOpts->hessianMethod;
    int hmAcc = globalOpts->hessianNumApproxAcc;
    bool threaded = globalOpts->optimThreaded;

    std::shared_ptr<ESASfaBase> modelObjPtr;
    if (modelFamily == ESASfaModelFamily::TFE) {
        ESASfaTfeGreene tfe(pseudoSample, prodCost);
        modelObjPtr = std::make_shared<ESASfaTfeGreene>(tfe);
    } else if (modelFamily == ESASfaModelFamily::TRE) {
        if (mdlPtr == nullptr) throw std::invalid_argument("mdlPtr in fitBsModel is null");
        ESASfaTreGreene tre(pseudoSample, prodCost, numSims, seed, mdlPtr->getHaltonDrawsPtr(), *mdlPtr->getHaltonSettings());
        modelObjPtr = std::make_shared<ESASfaTreGreene>(tre);
    } else if (modelFamily == ESASfaModelFamily::GTRE) {
        if (mdlPtr == nullptr) throw std::invalid_argument("mdlPtr in fitBsModel is null");
        ESASfaGtreBad gtre(pseudoSample, prodCost, numSims, seed, mdlPtr->getHaltonDrawsPtr(), *mdlPtr->getHaltonSettings());
        modelObjPtr = std::make_shared<ESASfaGtreBad>(gtre);
    } else {
        throw std::invalid_argument("Unexpected model family in bootstrap replications for marginal effects");
    }
    arma::dcolvec startVals;
    if (optStartVals) {
        startVals = optStartVals.value();
    } else {
        // generate new starting values
        startVals = modelObjPtr->startingValues();
    }
    try {
        resultUnk = esaoptimization::optimize(ms, startVals, modelObjPtr, op, printLevel, hm, hmAcc, threaded);
    } catch(std::exception& e) {
        return nullptr;
    }
    // check if successful
    if (resultUnk->getDidConverge()){
        // convergence occured
        ESAOptimResultSuccess& result = (ESAOptimResultSuccess&)(*resultUnk);
        return std::make_unique<ESAOptimResultSuccess>(result);
    }
    return nullptr;
}

/// Generate the pseudo sample based on the fitted parameter vector, return wrapped
/// in an ESASfaData object
ESADataPanel ESASfaMeff::generatePseudoSample(
    const arma::dcolvec& par,
    const int seed,
    const int prodCost
) const {
    // pointer to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    int nobs = dataObj.getNobs();
    ESASfaModelType modelType = dataObj.getModelType();
    ESASfaModelFamily modelFamily = ESAEnums::getModelFamily(modelType);
    // extract the parameters for the model
    arma::dcolvec b_x = dataObj.paramX(par);
    arma::dmat x_star = dataObj.getX() * b_x;
    // ------------------------------------------------------------------------------------
    // draw v_it* from N(0, sigma_vit^2)
    std::optional<arma::dcolvec> b_zvit_opt = dataObj.paramZvit(par);
    if (!b_zvit_opt) throw std::runtime_error("expected to retrieve 'b_zvit'");
    arma::dcolvec b_zvit = b_zvit_opt.value();
    // check data attribute exists too
    std::optional<arma::dmat> zvit_opt = dataObj.getZvit();
    if (!zvit_opt) throw std::runtime_error("expected to retrieve zvit");
    arma::dmat sigma2_vit = esautils::processSig2Term(b_zvit, zvit_opt.value());
    // draws from the normal distribution
    arma::dmat vit = esandist::ndraw<double>(
        arma::dmat(nobs, 1, arma::fill::zeros), sigma2_vit, seed
    );
    // ------------------------------------------------------------------------------------
    // draw u_it* from N+(0, sigma_uit^2)
    std::optional<arma::dcolvec> b_zuit_opt = dataObj.paramZuit(par);
    if (!b_zuit_opt) throw std::runtime_error("expected to retrieve 'b_zuit'");
    arma::dcolvec b_zuit = b_zuit_opt.value();
    // check data attribute exists too...
    std::optional<arma::dmat> zuit_opt = dataObj.getZuit();
    if (!zuit_opt) throw std::runtime_error("expected to retrieve zuit");
    arma::dmat sigma2_uit = esautils::processSig2Term(b_zuit, zuit_opt.value());
    arma::dmat mu = arma::dmat(nobs, 1, arma::fill::zeros);
    // calculate mu if the model is a truncated normal model
    if (ESAEnums::isTruncNormalModel(modelType)){
        // check options...
        if (!dataObj.paramZmuit(par)) throw std::runtime_error("missing 'paramZmuit'");
        if (!dataObj.getZmuit()) throw std::runtime_error("missing 'zmuit'");
        arma::dcolvec b_mu = dataObj.paramZmuit(par).value();
        arma::dmat sigma2_muuit = b_mu * dataObj.getZmuit().value();
    }
    arma::dmat uit = arma::abs(esandist::ndraw<double>(mu, sigma2_uit, seed));
    // ------------------------------------------------------------------------------------
    // draw v_i0* from N+(0, sigma_vi0^2) (for TRE) ??
    arma::dmat vi0 = arma::dmat(nobs, 1, arma::fill::zeros);
    if (modelFamily == ESASfaModelFamily::TRE){
        // check options exist
        if (!dataObj.paramZvi0(par)) throw std::runtime_error("missing 'paramZvi0'");
        if (!dataObj.getZvi0()) throw std::runtime_error("missing 'zvi0'");
        arma::dcolvec b_zvi0 = dataObj.paramZvi0(par).value();
        arma::dmat sigma2_vi0 = esautils::processSig2Term(b_zvi0, dataObj.getZvi0().value());
        // draw from the half normal distribution, N(0, sigma_vi0^2)
        vi0 = esandist::ndraw<double>(arma::dmat(nobs, 1, arma::fill::zeros), sigma2_vi0, seed);
    }
    // ------------------------------------------------------------------------------------
    // draw u_i0* from N(0, sigma_ui0^2) (for GTRE) ??
    arma::dmat ui0 = arma::dmat(nobs, 1, arma::fill::zeros);
    if (modelFamily == ESASfaModelFamily::GTRE){
        if (!dataObj.paramZui0(par)) throw std::runtime_error("missing 'paramZui0'");
        if (!dataObj.getZui0()) throw std::runtime_error("missing 'zui0'");
        arma::dcolvec b_zui0 = dataObj.paramZui0(par).value();
        arma::dmat sigma2_ui0 = esautils::processSig2Term(b_zui0, dataObj.getZui0().value());
        // draw from the half normal distribution, N+(0, sigma_ui0^2)
        ui0 = arma::abs(esandist::ndraw<double>(arma::dmat(nobs, 1, arma::fill::zeros), sigma2_ui0, seed));
    }
    // ------------------------------------------------------------------------------------
    // calculate y* = x + v_it* - u_it* + v_i0* - u_i0*
    arma::dcolvec y_star = x_star + vit + vi0 - prodCost * (uit + ui0);
    // create the data object
    if (modelFamily == ESASfaModelFamily::TFE){
        throw std::runtime_error("not implemented TFE call for Marginal Effects yet");
        if (ESAEnums::isTruncNormalModel(modelType)){
            // ESASfaData obj(y_star, dataObj.getX(), dataObj.getZuit(), dataObj.getZvit(), dataObj.getZvi0(), dataObj.getIdVec(), dataObj.getTimeVec(), dataObj.getZmuit(), dataObj.getModelType());
        } else {
            ESADataPanel obj(
                &y_star, // output
                dataObj.getXPtr(), // X
                dataObj.getIdVecPtr(), // firm identifiers
                dataObj.getTimeVecPtr(), // time identifiers
                dataObj.getModelType(), // model type
                nullptr, // mean of time-varying ineff (trunc norm)
                dataObj.getZuitPtr(), // determinants of time-varying ienff
                dataObj.getZvitPtr(), // determinants of stochastic noise
                nullptr, // determinants of time-invariant ineff
                nullptr // determinants of firm effects
            );
            return obj;
        }
    } else if (modelFamily == ESASfaModelFamily::TRE){
        if (ESAEnums::isTruncNormalModel(modelType)){
            ESADataPanel obj(
                &y_star, // output
                dataObj.getXPtr(), // X
                dataObj.getIdVecPtr(), // firm identifiers
                dataObj.getTimeVecPtr(), // time identifiers
                dataObj.getModelType(), // model type
                dataObj.getZmuitPtr(), // mean of time-varying ineff (trunc norm)
                dataObj.getZuitPtr(), // determinants of time-varying ienff
                dataObj.getZvitPtr(), // determinants of stochastic noise
                nullptr, // determinants of time-invariant ineff
                dataObj.getZvi0Ptr() // determinants of firm effects
            );
            // obj(y_star, dataObj.getX(), dataObj.getZuit(), dataObj.getZvit(), dataObj.getZvi0(), dataObj.getIdVec(), dataObj.getTimeVec(), dataObj.getZmuit(), dataObj.getModelType());
            return obj;
        } else {
            ESADataPanel obj(
                &y_star, // output
                dataObj.getXPtr(), // X
                dataObj.getIdVecPtr(), // firm identifiers
                dataObj.getTimeVecPtr(), // time identifiers
                dataObj.getModelType(), // model type
                nullptr, // mean of time-varying ineff (trunc norm)
                dataObj.getZuitPtr(), // determinants of time-varying ienff
                dataObj.getZvitPtr(), // determinants of stochastic noise
                nullptr, // determinants of time-invariant ineff
                dataObj.getZvi0Ptr() // determinants of firm effects
            );
            return obj;
        }
    } else if (modelFamily == ESASfaModelFamily::GTRE){
        ESADataPanel obj(
            &y_star, // output
            dataObj.getXPtr(), // X
            dataObj.getIdVecPtr(), // firm identifiers
            dataObj.getTimeVecPtr(), // time identifiers
            dataObj.getModelType(), // model type
            nullptr, // mean of time-varying ineff (trunc norm)
            dataObj.getZuitPtr(), // determinants of time-varying ienff
            dataObj.getZvitPtr(), // determinants of stochastic noise
            dataObj.getZui0Ptr(), // determinants of time-invariant ineff
            dataObj.getZvi0Ptr() // determinants of firm effects
        );
        return obj;
    }
    throw std::runtime_error("Model family not recognized");
}