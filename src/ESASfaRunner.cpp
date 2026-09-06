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
 * @file ESASfaRunner.cpp
 * @date 2025-12-17
 * @author edmund haacke
 */

#include <chrono>
#include "sfa/ESASfaRunner.hpp"
// ---- data ----
#include "data/ESADataCross.hpp"
#include "data/ESADataPanel.hpp"
#include "data/ESADataLCM.hpp"
#include "interface/interface_utils.hpp"
// ---- sfa models ----
#include "sfa/ESASfaCross.hpp"
#include "sfa/ESASfaTfeGreene.hpp"
#include "sfa/ESASfaTreGreene.hpp"
#include "sfa/ESASfaGtreBad.hpp"
#include "sfa/ESASfaLcTre.hpp"
#include "sfa/ESASfaLcmCross.hpp"
// ---- optimization ----
#include "optim/esaoptimization.hpp"
#include "optim/ESAGlobalOptimParams.hpp"
// ---- marginal effects ----
#include "marginaleffects/ESASfaMeffWang.hpp"
// ---- utilities ----
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "utils/modelsummary.hpp"

// python specific imports
#ifdef PYPACKAGE
#include <pybind11/pybind11.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#endif

using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// constructor 
ESASfaRunner::ESASfaRunner(
    const std::string& model,
    const std::string& dist,
    const int prodCost,
    const int seed,
    const int nsim,
    const int printLevel
) : prodCost(prodCost),
    seed(seed),
    nsim(nsim),
    printLevel(printLevel)
{
    // get the model type, distribution, and family
    ESASfaModelType mT = ESAEnums::getModelTypeForNameAndDist(model, dist);
    this->mT = mT;
    this->mF = ESAEnums::getModelFamily(mT);
    this->mD = ESAEnums::getDistribution(mT);
    // if on python, set the seed - in R, this has to be set from R
    #ifdef PYPACKAGE
    arma::arma_rng::set_seed(seed);
    #endif // PYPACKAGE
}

ESASfaRunner::ESASfaRunner(
    const ESASfaModelType mT,
    const int prodCost,
    const int seed,
    const int nsim,
    const int printLevel
) : prodCost(prodCost),
    seed(seed),
    nsim(nsim),
    printLevel(printLevel)
{
    this->mT = mT;
    this->mF = ESAEnums::getModelFamily(mT);
    this->mD = ESAEnums::getDistribution(mT);
    // if on python, set the seed - in R, this has to be set from R
    #ifdef PYPACKAGE
    arma::arma_rng::set_seed(seed);
    #endif // PYPACKAGE
}

// 
void ESASfaRunner::loadData(
    const arma::dcolvec* y,
    const arma::dmat* x,
    const arma::dmat* zmuit,
    const arma::dmat* zuit,
    const arma::dmat* zvit,
    const arma::dmat* zui0,
    const arma::dmat* zvi0,
    const arma::Col<int>* idVec,
    const arma::Col<int>* timeVec
)
{
    // check required arguments have a value
    if (mF == ESASfaModelFamily::TFE || mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE) {
        if (!idVec) throw std::invalid_argument("'idVec' must be provided for panel models");
        if (!timeVec) throw std::invalid_argument("'timeVec' must be provided for panel models");
    }
    if (mF == ESASfaModelFamily::TFE || mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE || mF == ESASfaModelFamily::CROSS) {
        if (!zuit) throw std::invalid_argument("'zuit' must be provided.");
        if (!zvit) throw std::invalid_argument("'zvit' must be provided.");
    }
    if (mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE) {
        if (!zvi0) throw std::invalid_argument("'zvi0' must be provided for TRE/GTRE.");
    }
    if (mF == ESASfaModelFamily::GTRE) {
        if (!zui0) throw std::invalid_argument("'zui0' must be provided for GTRE.");
    }
    if (mD == ESASfaModelDistribution::TNORM) {
        if (!zmuit) throw std::invalid_argument("'zmuit' must be provided for truncated normal.");
    }
    // empty matricies
    arma::dmat empty_dmat;
    arma::Col<int> empty_ivec;
    this->dataObjPtr = interface::createDataObject(y, x, zmuit, zuit, zvit, zui0, zvi0, idVec, timeVec, mT);
}

void ESASfaRunner::loadDataLCM(
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
)
{
    // initial checks
    if (mF != ESASfaModelFamily::LC_TRE && mF != ESASfaModelFamily::LC_X) {
        throw std::invalid_argument("loadDataLCM only supports LC_TRE and LC_X models");
    }
    if (!seg) throw std::invalid_argument("'seg' (concomitant variables) must be provided for LC models");
    if (!zuit) throw std::invalid_argument("'zuit' must be provided for LC models");
    if (!zvit) throw std::invalid_argument("'zvit' must be provided for LC models");
    if (nClasses < 1) throw std::invalid_argument("nClasses must be >= 1 for latent class models");

    if (mF == ESASfaModelFamily::LC_TRE) {
        // specific checks for the latent class tre model
        if (!idVec) throw std::invalid_argument("'idVec' must be provided for LC panel models");
        if (!timeVec) throw std::invalid_argument("'timeVec' must be provided for LC panel models");
        if (!zvi0) throw std::invalid_argument("'zvi0' must be provided for LC-TRE models");

        ESASfaModelType innerMT;
        if (mT == ESASfaModelType::LC_TRE_HNORM) {
            innerMT = ESASfaModelType::TRE_HNORM_ZUIT;
        } else if (mT == ESASfaModelType::LC_TRE_TNORM) {
            innerMT = ESASfaModelType::TRE_TNORM_ZUIT;
        } else {
            throw std::runtime_error("Unsupported LC-TRE model type in loadDataLCM");
        }
        // create a data panel view for the latent class structure
        this->lcmDataObjPtr = std::make_shared<ESADataPanelLCM>(
            seg, y, x, idVec, timeVec, mT,
            zmuit, zuit, zvit, zvi0, nClasses,
            nullptr, // transition (not supported for LC-TRE)
            true     // arraysContiguous
        );
        // since underlying TRE logic relies on the data panel class, need to use that
        // shouldn't affect memory majorly since they're just views 
        this->dataObjPtr = std::make_shared<ESADataPanel>(
            y, x, idVec, timeVec, innerMT,
            zmuit, zuit, zvit, nullptr, zvi0,
            true // arraysContiguous
        );
    } else {
        // LC_X (cross-sectional)
        this->lcmDataObjPtr = std::make_shared<ESADataPanelLCM>(
            seg, y, x, nullptr, nullptr, mT,
            zmuit, zuit, zvit, nullptr, nClasses,
            nullptr, true
        );
    }
}

void ESASfaRunner::setupModel(
    const HaltonSettings hsetting,
    const std::optional<std::vector<std::string>> termsX,
    const std::optional<std::vector<std::string>> termsZmuit,
    const std::optional<std::vector<std::string>> termsZuit,
    const std::optional<std::vector<std::string>> termsZvit,
    const std::optional<std::vector<std::string>> termsZui0,
    const std::optional<std::vector<std::string>> termsZvi0
)
{
    modelTerms = std::make_shared<ESASfaModelTerms>(termsX, termsZmuit, termsZuit, termsZvit, termsZui0, termsZvi0, mT);
    // ---- setup model objects ----
    if (mF == ESASfaModelFamily::TFE){
        ESASfaTfeGreene modelObj(dataObjPtr, prodCost);
        modelObjPtr = std::make_shared<ESASfaTfeGreene>(modelObj);
    } else if (mF == ESASfaModelFamily::TRE){
        ESASfaTreGreene modelObj(dataObjPtr, prodCost, nsim, seed, hsetting);
        modelObjPtr = std::make_shared<ESASfaTreGreene>(modelObj);
    } else if (mF == ESASfaModelFamily::GTRE){
        ESASfaGtreBad modelObj(dataObjPtr, prodCost, nsim, seed, hsetting);
        modelObjPtr = std::make_shared<ESASfaGtreBad>(modelObj);
    } else if (mF == ESASfaModelFamily::CROSS){
        ESASfaCross modelObj(dataObjPtr, prodCost);
        modelObjPtr = std::make_shared<ESASfaCross>(modelObj);
    } else if (mF == ESASfaModelFamily::LC_TRE) {
        auto lcmData = std::dynamic_pointer_cast<ESADataPanelLCM>(lcmDataObjPtr);
        auto helperData = std::dynamic_pointer_cast<ESADataPanel>(dataObjPtr);
        if (!lcmData || !helperData) {
            throw std::runtime_error("LC_TRE requires loadDataLCM to be called before setupModel");
        }
        modelObjPtr = std::make_shared<ESASfaLcTre>(lcmData, helperData, prodCost, nsim, seed, hsetting);
    } else if (mF == ESASfaModelFamily::LC_X) {
        auto lcmData = std::dynamic_pointer_cast<ESADataPanelLCM>(lcmDataObjPtr);
        if (!lcmData) {
            throw std::runtime_error("LC_X requires loadDataLCM to be called before setupModel");
        }
        modelObjPtr = std::make_shared<ESASfaLcmCross>(lcmData, prodCost);
    }
    if (modelObjPtr == nullptr){
        throw std::runtime_error("Model object is null - something went wrong somewhere.");
    }
}

std::unique_ptr<ESAOptimResult> ESASfaRunner::runOptimization(
    const std::optional<arma::dcolvec> startVals,
    HessianCalcMethod hessCalcMethod,
    const int hessianCalcNumApproxAccuracy
)
{
    // get the global optim parameters
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    ESAOptimParams mainOptimParams = globalOptimParams->mainOptimParams;
    ModelSolver mainModelSolver = globalOptimParams->mainModelSolver;
    // ---- starting values parsing ----

    arma::dcolvec theta0 = modelObjPtr->startingValues();    
    // if the user provided starting values, overwrite these with the default values provided
    if (startVals){
        // check the size of the starting values provided by the user
        if (startVals.value().n_rows > theta0.n_rows){
            ESALogger::logger()->warn("Starting values provided by the user are larger than the default starting values. Truncating the user provided values.");
        } else if (startVals.value().n_rows < theta0.n_rows){
            ESALogger::logger()->warn("Starting values provided by the user are smaller than requring. Padding user values default starting values.");
        }
        for (size_t i = 0; i < theta0.n_rows; i++){
            if (i < startVals.value().n_rows){
                theta0.at(i) = startVals.value().at(i);
            }
        }
    }
    if (this->printLevel > 0){
        ESALogger::logger()->info("The starting values for the model are: {}", theta0);
    }
    // ---- maximum likelihood estimation ----
    // run optimization
    // in python - silence the current logger - prevent issue with GIL
    #ifdef PYPACKAGE
    std::shared_ptr<spdlog::logger> current_logger = spdlog::default_logger();
    // Create a Ring Buffer Sink (Thread Safe) where 1024 is the max number of log messages to keep. 
    // Oldest messages are dropped if full.
    auto ring_sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(1024);
    auto buffer_logger = std::make_shared<spdlog::logger>("buffer_logger", ring_sink);
    // Set this as the active logger so C++ writes to memory, not Python
    spdlog::set_default_logger(buffer_logger);
    #endif // PYPACKAGE
    // timer now 
    auto tStart = std::chrono::high_resolution_clock::now();
    // run optimization
    std::unique_ptr<ESAOptimResult> resultUnk;
    {
        // in python we also have to release GIL
        #ifdef PYPACKAGE
        pybind11::gil_scoped_release release;
        #endif // PYPACKAGE
        resultUnk = esaoptimization::optimize(
            mainModelSolver, // which optimization algorithm to use
            theta0, // starting values
            modelObjPtr, // pointer to SFA class - sfa model to use
            mainOptimParams, // optimization parameters
            this->printLevel, // print level
            hessCalcMethod, // method to calculate hessian matrix
            hessianCalcNumApproxAccuracy, // if numerically approximate hessian, what accuracy level to use
            globalOptimParams->optimThreaded
        );
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    /// in python, reset the logger
    #ifdef PYPACKAGE
    esautils::log::flushRingBufferToTarget(current_logger);
    spdlog::set_default_logger(current_logger);
    spdlog::drop("buffer_logger");
    #endif // PYPACKAGE
    std::chrono::duration<double, std::milli> ms_diff = tEnd - tStart;
    if (this->printLevel > 0){ 
        ESALogger::logger()->info("Optimization took {:.4f} mins", ms_diff.count()/1000/60);
    }
    return resultUnk;
}

arma::dmat ESASfaRunner::buildSummary(
    const arma::dcolvec& coefs,
    const arma::dmat& vcov,
    const double confLevel,
    const int degreesFreedom
)
{
    return modelsummary::getSummary(coefs, vcov, confLevel, degreesFreedom);
}


void ESASfaRunner::estimateMarginalEffects(
    const std::string& method,
    const arma::dcolvec& coefs,
    const bool estimateCI,
    const double confLvl,
    const int bootstrapReps,
    std::unique_ptr<ESASfaMeffReturn>& meffOut,
    std::unique_ptr<ESASfaMeffCIReturn>& meffCIOut
)
{
    if (method == "wang2002") {
        // wang 2002 results
        std::shared_ptr<ESASfaMeffWang> meffObj = std::make_unique<ESASfaMeffWang>(ESASfaMeffWang(dataObjPtr, prodCost, nsim, seed));
        // calculate marginal effects
        ESASfaMeffReturn meff = meffObj->marginalEffects(coefs, *modelTerms);
        if (estimateCI) {
            // calculate bootstrapped confidence intervals
            meffCIOut = meffObj->bootstrappedCIs(
                this->modelObjPtr,
                coefs, // estimate params,
                *modelTerms,
                confLvl, // confidence level
                bootstrapReps, // bootstrap replications
                false // use default starting values
            );
        }
        meffOut = std::make_unique<ESASfaMeffReturn>(meff);
    }
}

std::unique_ptr<ESASfaEffScores> ESASfaRunner::estimateEfficiencyScores(
    const arma::dcolvec& coefs,
    const int ghkSims,
    const int haltonStart
)
{
    if (printLevel > 0){
        ESALogger::logger()->info("Estimating efficiency scores...");
    }
    auto tStart = std::chrono::high_resolution_clock::now();
    std::unique_ptr<ESASfaEffScores> effScoresPtr = nullptr;
    // tre/gtre only supported atm
    if (this->mF == ESASfaModelFamily::TRE || this->mF == ESASfaModelFamily::GTRE) {
        ESASfaEffGtre gtreEff(this->dataObjPtr, this->prodCost);
        ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
        // for python protection - release GIL, since this is potentially multithreaded
        // as before, unhook logger in python
        #ifdef PYPACKAGE
        // setup the ring buffer {thread safe}
        std::shared_ptr<spdlog::logger> currLogger = spdlog::default_logger();
        auto ringSink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(1024);
        auto bufferLogger = std::make_shared<spdlog::logger>("buffer_logger", ringSink);
        // set as default, so we write to memory, not Python
        spdlog::set_default_logger(bufferLogger);
        #endif // PYPACKAGE
        {
            #ifdef PYPACKAGE
            pybind11::gil_scoped_release release;
            #endif // PYPACKAGE
            effScoresPtr = gtreEff.efficiencyScores(
                coefs, ghkSims, haltonStart, seed, globalOptimParams->optimThreaded
            );
        }
        // in python, reset the logger
        #ifdef PYPACKAGE
        esautils::log::flushRingBufferToTarget(currLogger);
        spdlog::set_default_logger(currLogger);
        spdlog::drop("buffer_logger");
        #endif // PYPACKAGE
    } else {
        ESALogger::logger()->warn("not implemented BC1988 eff scores for TFE");
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_diff = tEnd - tStart;
    if (printLevel > 0) {
        ESALogger::logger()->info("Efficiency score calculation took {:.4f} mins", ms_diff.count()/1000/60);
    }
    return effScoresPtr;
}