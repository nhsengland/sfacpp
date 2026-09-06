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
 * @file pysfacpp.cpp
 * @author edmund haacke
 * @date 2025-12-08
 * @details pybind11 code
 */

#ifdef PYPACKAGE
#include <atomic>
#include <thread>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <spdlog/spdlog.h>
#include <mutex>
#include <armadillo>
#include <variant>
// ---- utilities ----
#include "utils/enums.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "utils/esautils.hpp"
#include "utils/statuskeys.hpp"
#include "utils/esaparallel.hpp"
#include "utils/ESASingletonStatuses.hpp"
#include "utils/modelsummary.hpp"
#include "utils/sandwich.hpp"
#include "utils/ThreadContext.hpp"
#include "marginaleffects/ESASfaMeff.hpp"
#include "marginaleffects/ESASfaMeffLcmWang.hpp"
#include "efficiencies/ESASfaEffLcmTre.hpp"
#include "efficiencies/ESASfaEffLcmJlms.hpp"
#include "sfa/ESASfaLcTre.hpp"
#include "sfa/ESASfaLcmCross.hpp"
#include "utils/memoryusage.hpp"
#include "sfa/HaltonSettings.hpp"
#include "efficiencies/effscores.hpp"
#include "pyinterface/numpy_conv.hpp"
#include "math/llratiotest.hpp"
#include "utils/globalstate.hpp"
#include "utils/SearchStartVals.hpp"

#ifdef WITHEIGEN
#ifdef RPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include "Eigen/Core"
#endif // RPACKAGE
#endif //WITHEIGEN

// ---- interface utilities ----
#include "pyinterface/pyinterface_utils.hpp"
#include "interface/interface_utils.hpp"
// ---- runner ----
#include "sfa/ESASfaRunner.hpp"
// ---- optimization (for parallel searches) ----
#include "optim/esaoptimization.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

// global flag for whether python logger is active
static std::atomic<bool> gPythonLoggerActive{true};

// ┌─────────────────────────────────────────────────────────────┐
// │            Searches (for checking convergence)                    │
// └─────────────────────────────────────────────────────────────┘
/**
 * @details Run some 
 * @param y_in
 * @param x_in
 * @param zmuit_in
 * @param zuit_in
 */
py::dict pysfacpp_searches(
    const py::array_t<double>& y_in,
    const py::array_t<double>& x_in,
    const std::optional<py::array_t<double>>& zmuit_in,
    const std::optional<py::array_t<double>>& zuit_in,
    const std::optional<py::array_t<double>>& zvit_in,
    const std::optional<py::array_t<double>>& zui0_in,
    const std::optional<py::array_t<double>>& zvi0_in,
    const std::optional<py::array_t<double>>& start_in,
    const std::optional<py::array_t<int>>& idVec_in,
    const std::optional<py::array_t<int>>& timeVec_in,
    const int reps = 500,
    const int maxRepIter = 100,
    const double slengthFrontier = 2.0,
    const double slengthSigmas = 0.8,
    const int maxStartValFindAttempt = 50,
    const int prodCost = 1,
    const std::string& model = "tre",
    const std::string& dist = "hnorm",
    const std::string& method = "tr",
    const std::string& methodLib = "dlib",
    const unsigned int nsim = 500,
    const int seed = 1234,
    const int printLevel = 0,
    const int nthreads = 20,
    const int haltonBase = 2,
    const int haltonBurnin = 1000,
    const int haltonUi0Base = 3,
    const bool scrambledHalton = true,
    const bool shuffledHalton = false,
    const bool shouldCopy = false,
    const int displayConsoleWidth = 120,
    const int displayDecimalPlaces = 4,
    const int maximumAttempts = 100000,
    const bool overrideStartValCheck = false
)
{
    // ---- python specific (zero-copy) conversions ----
    const arma::colvec y = numpy_arma_conv::pyToCol<double>(y_in, shouldCopy);
    const arma::dmat x = numpy_arma_conv::pyToArma<double>(x_in, shouldCopy);
    arma::dmat zmuit_, zuit_, zvit_, zui0_, zvi0_;
    arma::Col<int> idVec_, timeVec_;
    if (zmuit_in.has_value()) zmuit_ = numpy_arma_conv::pyToArma<double>(zmuit_in.value(), shouldCopy);
    if (zuit_in.has_value()) zuit_ = numpy_arma_conv::pyToArma<double>(zuit_in.value(), shouldCopy);
    if (zvit_in.has_value()) zvit_ = numpy_arma_conv::pyToArma<double>(zvit_in.value(), shouldCopy);
    if (zui0_in.has_value()) zui0_ = numpy_arma_conv::pyToArma<double>(zui0_in.value(), shouldCopy);
    if (zvi0_in.has_value()) zvi0_ = numpy_arma_conv::pyToArma<double>(zvi0_in.value(), shouldCopy);
    if (idVec_in.has_value()) idVec_ = numpy_arma_conv::pyToCol<int>(idVec_in.value(), shouldCopy);
    if (timeVec_in.has_value()) timeVec_ = numpy_arma_conv::pyToCol<int>(timeVec_in.value(), shouldCopy);
    // ---- setup & settings ----
    ESASfaModelType mT = ESAEnums::getModelTypeForNameAndDist(model, dist);
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    // set number of threads to user amount
    esaparallel::setThreadCount(nthreads);
    // reset TLS structs if needed
    esaparallel::modelChangeFlushUnneededTLS(mF);
    HessianCalcMethod hessCalcMethod = HessianCalcMethod::ANALYTICAL;
    // modify optimization parameters
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    ESAOptimParams mainOptimParams;
    mainOptimParams.maxit = maxRepIter;
    mainOptimParams.seed = seed;
    globalOptimParams->mainOptimParams = mainOptimParams;
    globalOptimParams->mainModelSolver = ESAEnums::getModelSolverForMethodAndLib(method, methodLib);
    globalOptimParams->hessianMethod = hessCalcMethod;
    globalOptimParams->hessianNumApproxAcc = 0;
    globalOptimParams->optimThreaded = (nthreads > 1);
    // halton settings
    HaltonSettings hsetting = interface::haltonSettingsForOpts(haltonBase, haltonBurnin, haltonUi0Base, scrambledHalton, shuffledHalton);
    // --- setup runner ----
    ESASfaRunner runner(model, dist, prodCost, seed, nsim, printLevel);
    // next, load the data
    runner.loadData(
        &y,
        &x,  
        zmuit_in.has_value() ? &zmuit_ : nullptr,
        zuit_in.has_value() ? &zuit_ : nullptr,
        zvit_in.has_value() ? &zvit_ : nullptr,
        zui0_in.has_value() ? &zui0_ : nullptr,
        zvi0_in.has_value() ? &zvi0_ : nullptr,
        idVec_in.has_value() ? &idVec_ : nullptr,
        timeVec_in.has_value() ? &timeVec_ : nullptr
    );
    // setup the model
    runner.setupModel(hsetting, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    arma::dcolvec theta0;
    int k = runner.dataObjPtr->nParams();
    if (start_in.has_value()) {
        theta0 = numpy_arma_conv::pyToCol(start_in.value(), true);
        if (theta0.n_rows != k) {
            throw std::invalid_argument(
                "User provided starting values, but of incorrect dimensions, got " +
                std::to_string(theta0.n_rows) + " but expected " + std::to_string(k)
            );
        }
    } else {
        theta0 = runner.modelObjPtr->startingValues();
    }
    // allocate matrix to store coefficients
    arma::dmat allCoefs(k, reps);
    // column vector to score loglikelihood scores
    arma::dcolvec lls(reps, arma::fill::value(std::numeric_limits<double>::quiet_NaN()));
    // highest log likelihood score found; index position of it
    double maxLL = -99999999999;
    int maxLLpos = -1;
    double totalMinsTaken = 0.0;
    int failedStart = 0, successfulConverge = 0;
    int i = 0;
    int cntr = -1;
    #ifdef PYPACKAGE
    pyinterface::SignalHandlerGuard guard;
    #endif // PYPACKAGE
    // iterate thru the number of repetitions
    // for (int i = 0; i < reps; i++) {
    while ((i < reps) && (cntr < maximumAttempts)) {
        cntr++;
        if (globalstate::InterruptRequested.load(std::memory_order_relaxed)) {
            pyinterface::throwPyInterruptIfNeeded();
        }
        // also check standard Python signals (handles race conditions)
        if (PyErr_CheckSignals() != 0) throw py::error_already_set();
        // start the timer
        auto tStart = std::chrono::high_resolution_clock::now();
        arma::dcolvec theta0Rep(theta0.n_rows);
        bool couldFindStart = esautils::findValidStartValues(
            runner.dataObjPtr,
            runner.modelObjPtr,
            theta0,
            theta0Rep,
            cntr,
            slengthFrontier,
            slengthSigmas,
            seed,
            (overrideStartValCheck ? 1 : maxStartValFindAttempt),
            (printLevel > 2)
        );
        // if couldnt find any good values, bin off this loop iteration
        if (!couldFindStart && !overrideStartValCheck) {
            ESALogger::logger()->info("Could not find appropriate starting values.");
            failedStart++;
            continue;
        }
        // run the optimization
        std::unique_ptr<ESAOptimResult> resultUnk = runner.runOptimization(theta0Rep, hessCalcMethod, 0);
        // check again for a user interrupt
        if (globalstate::InterruptRequested.load(std::memory_order_relaxed)) {
            pyinterface::throwPyInterruptIfNeeded();
        }
        if (PyErr_CheckSignals() != 0) throw py::error_already_set();
        // check whether converged or not
        if (resultUnk == nullptr){
            // fill with NaNs
            allCoefs.col(i).fill(std::numeric_limits<double>::quiet_NaN());
        } else if (resultUnk->getDidConverge()) {
            ESALogger::logger()->info("---- Trial Number {}/{} ----", i+1, reps);
            ESAOptimResultSuccess& optimRes = (ESAOptimResultSuccess&)*resultUnk;
            // insert the columns into the matrix of coefficients
            allCoefs.col(i) = optimRes.getX();
            // check whether the loglikelihood score is bigger than found previously
            double ll = optimRes.getLogLike();
            if (ll > maxLL) {
                maxLL = ll;
                maxLLpos = i;
            }
            lls(i) = ll;
            ESALogger::logger()->info("{}/{} converged with loglikelihood {:.12f} and parameter vector {}", i, reps, ll, optimRes.getX());
            successfulConverge++;
            i++;
        } else {
            // model failed to converge
            // might as well still fill in the coefficients
            ESAOptimResultFailed& optimRes = (ESAOptimResultFailed&)*resultUnk;
            allCoefs.col(i) = optimRes.getX();
        }
        // stop the timer
        auto tEnd = std::chrono::high_resolution_clock::now();
        // calculate the difference in milliseconds
        double min_diff = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count() / 1000.0 / 60.0;
        totalMinsTaken += min_diff;
        if ((i % 10 == 0) && (i > 0)) {
            // print elapsed time
            interface::printSearchesElapsedAndETA(ESALogger::logger(), totalMinsTaken, i, reps);
        }
    }
    ESALogger::logger()->info(
        "From {} trials, {} successfully converged. {} did not yield valid starting values. Try widening `slength`, and/or increasing attempts",
        reps, successfulConverge, failedStart
    );
    // print the searches (top 15)
    interface::printSearches(lls, allCoefs, 30, displayConsoleWidth, displayDecimalPlaces);
    // create the return dictionary for python
    py::dict out;
    out["ll_scores"] = numpy_arma_conv::colToPy<double>(lls);
    out["all_coefs"] = numpy_arma_conv::armaToPy<double>(allCoefs);
    out["maxLL"] = maxLL;
    out["maxLLpos"] = maxLLpos;
    if (maxLLpos >= 0) {
        arma::dcolvec bestCoef(allCoefs.col(maxLLpos));
        out["maxLLCoefVec"] = numpy_arma_conv::colToPy<double>(bestCoef);
    }
    // unhook the loggger
    ESALogger::logger()->sinks().clear();
    spdlog::set_default_logger(nullptr);
    return out;
}

// ┌─────────────────────────────────────────────────────────────┐
// │         LC-TRE Searches (random perturbations for LC model)       │
// └─────────────────────────────────────────────────────────────┘
py::dict pysfacpp_searches_lcm(
    const py::array_t<double>& y_in,
    const py::array_t<double>& x_in,
    const py::array_t<double>& seg_in,
    const std::optional<py::array_t<double>>& zmuit_in,
    const py::array_t<double>& zuit_in,
    const py::array_t<double>& zvit_in,
    const py::array_t<double>& zvi0_in,
    const py::array_t<int>& idVec_in,
    const py::array_t<int>& timeVec_in,
    const std::optional<py::array_t<double>>& start_in,
    const int nClasses = 2,
    const int reps = 500,
    const int maxRepIter = 100,
    const double slengthFrontier = 2.0,
    const double slengthSigmas = 0.8,
    const int maxStartValFindAttempt = 50,
    const int prodCost = 1,
    const std::string& dist = "hnorm",
    const std::string& method = "tr",
    const std::string& methodLib = "dlib",
    const unsigned int nsim = 500,
    const int seed = 1234,
    const int printLevel = 0,
    const int nthreads = 20,
    const int haltonBase = 2,
    const int haltonBurnin = 1000,
    const int haltonUi0Base = 3,
    const bool scrambledHalton = true,
    const bool shuffledHalton = false,
    const bool shouldCopy = false,
    const int displayConsoleWidth = 120,
    const int displayDecimalPlaces = 4,
    const int maximumAttempts = 100000,
    const bool overrideStartValCheck = false,
    const bool parallelSearches = true
)
{
    // ---- python specific (zero-copy) conversions ----
    const arma::colvec y = numpy_arma_conv::pyToCol<double>(y_in, shouldCopy);
    const arma::dmat x = numpy_arma_conv::pyToArma<double>(x_in, shouldCopy);
    const arma::dmat seg = numpy_arma_conv::pyToArma<double>(seg_in, shouldCopy);
    arma::dmat zmuit_, zuit_, zvi0_, zvit_;
    arma::Col<int> idVec_, timeVec_;
    if (zmuit_in.has_value()) zmuit_ = numpy_arma_conv::pyToArma<double>(zmuit_in.value(), shouldCopy);
    zuit_ = numpy_arma_conv::pyToArma<double>(zuit_in, shouldCopy);
    zvit_ = numpy_arma_conv::pyToArma<double>(zvit_in, shouldCopy);
    zvi0_ = numpy_arma_conv::pyToArma<double>(zvi0_in, shouldCopy);
    idVec_ = numpy_arma_conv::pyToCol<int>(idVec_in, shouldCopy);
    timeVec_ = numpy_arma_conv::pyToCol<int>(timeVec_in, shouldCopy);
    // ---- setup & settings ----
    const std::string model = "lctre";
    ESASfaModelType mT = ESAEnums::getModelTypeForNameAndDist(model, dist);
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    if (mF != ESASfaModelFamily::LC_TRE) {
        throw std::invalid_argument("pysfacpp_searches_lcm only supports LC-TRE models");
    }
    esaparallel::setThreadCount(nthreads);
    esaparallel::modelChangeFlushUnneededTLS(mF);
    HessianCalcMethod hessCalcMethod = ESAEnums::getHessianCalcMethod("bhhh");
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    ESAOptimParams mainOptimParams;
    mainOptimParams.maxit = maxRepIter;
    mainOptimParams.seed = seed;
    globalOptimParams->mainOptimParams = mainOptimParams;
    globalOptimParams->mainModelSolver = ESAEnums::getModelSolverForMethodAndLib(method, methodLib);
    globalOptimParams->hessianMethod = hessCalcMethod;
    globalOptimParams->hessianNumApproxAcc = 0;
    globalOptimParams->optimThreaded = (nthreads > 1);
    HaltonSettings hsetting = interface::haltonSettingsForOpts(haltonBase, haltonBurnin, haltonUi0Base, scrambledHalton, shuffledHalton);
    // ---- setup runner ----
    ESASfaRunner runner(model, dist, prodCost, seed, nsim, printLevel);
    runner.loadDataLCM(
        &y,
        &x,
        &seg,
        zmuit_in.has_value() ? &zmuit_ : nullptr,
        &zuit_,
        &zvit_,
        &zvi0_,
        &idVec_,
        &timeVec_,
        static_cast<unsigned int>(nClasses)
    );
    runner.setupModel(hsetting, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    arma::dcolvec theta0;
    int k = runner.lcmDataObjPtr->nParams();
    if (start_in.has_value()) {
        theta0 = numpy_arma_conv::pyToCol(start_in.value(), true);
        if (static_cast<int>(theta0.n_rows) != k) {
            throw std::invalid_argument(
                "User provided starting values, but of incorrect dimensions, got " +
                std::to_string(theta0.n_rows) + " but expected " + std::to_string(k)
            );
        }
    } else {
        theta0 = runner.modelObjPtr->startingValues();
    }
    // allocate storage
    arma::dmat allCoefs(k, reps);
    arma::dcolvec lls(reps, arma::fill::value(std::numeric_limits<double>::quiet_NaN()));
    double maxLL = -99999999999;
    int maxLLpos = -1;
    int failedStart = 0, successfulConverge = 0;
    // pointer to lcm data object
    auto lcmData = std::dynamic_pointer_cast<ESADataPanelLCM>(runner.lcmDataObjPtr);
    ModelSolver mainModelSolver = globalOptimParams->mainModelSolver;
    // get the signal handler guard for python
    #ifdef PYPACKAGE
    pyinterface::SignalHandlerGuard guard;
    #endif // PYPACKAGE
    // seperate logic depending if using parallel searches or sequential searches
    // note that the parallel searches only runs for the number of set searches,
    // whereas the while loop runs until there have been sufficient successful searches
    if (parallelSearches) {
        // ---- parallel search mode ----
        // un multiple searches concurrently (each single-threaded internally).
        // Set optimThreaded=false so model evaluations within each search don't
        // submit to the thread pool (avoiding nested submission).
        globalOptimParams->optimThreaded = false;
        // result struct for each parallel search 
        struct ParSearchResult {
            arma::dcolvec coefs;
            double ll = std::numeric_limits<double>::quiet_NaN();
            bool converged = false;
            bool validStart = false;
        };
        std::vector<ParSearchResult> searchResults(reps);
        // release python gil
        {
            pybind11::gil_scoped_release release;
            BS::thread_pool<>& pool = esaparallel::getOptimPool();
            // each search itieration
            BS::multi_future<void> futures = pool.submit_sequence(0, reps,
                [&](int idx) {
                    if (globalstate::InterruptRequested.load(std::memory_order_relaxed)) return;
                    // get starting values - seed will depend on the search iteration
                    arma::dcolvec theta0Rep(k);
                    bool couldFindStart = esautils::findValidStartValuesLCM(
                        lcmData,
                        runner.modelObjPtr,
                        theta0,
                        theta0Rep,
                        idx,
                        slengthFrontier, slengthSigmas,
                        seed,
                        (overrideStartValCheck ? 1 : maxStartValFindAttempt),
                        false
                    );
                    if (!couldFindStart && !overrideStartValCheck) {
                        searchResults[idx].validStart = false;
                        return;
                    }
                    searchResults[idx].validStart = true;
                    // directly call optimization instead of using runner - since
                    // dont hold gil can't print (and will cause crash if try access)
                    std::unique_ptr<ESAOptimResult> result = esaoptimization::optimize(
                        mainModelSolver,
                        theta0Rep,
                        runner.modelObjPtr,
                        mainOptimParams,
                        0, // printLevel
                        hessCalcMethod,
                        0,
                        false // threaded = false
                    );
                    // check convergence for this iteration
                    if (result && result->getDidConverge()) {
                        ESAOptimResultSuccess& success = static_cast<ESAOptimResultSuccess&>(*result);
                        searchResults[idx].coefs = success.getX();
                        searchResults[idx].ll = success.getLogLike();
                        searchResults[idx].converged = true;
                    } else if (result) {
                        ESAOptimResultFailed& failed = static_cast<ESAOptimResultFailed&>(*result);
                        searchResults[idx].coefs = failed.getX();
                    }
                }
            );
            futures.wait();
        }
        // check if interrupted during parallel searches
        pyinterface::throwPyInterruptIfNeeded();
        // restore the threaded mode that was switched off
        globalOptimParams->optimThreaded = (nthreads > 1);
        // collect results into the output matrices
        int outIdx = 0;
        for (int idx = 0; idx < reps; ++idx) {
            if (!searchResults[idx].validStart) {
                failedStart++;
                continue;
            }
            if (searchResults[idx].converged) {
                allCoefs.col(outIdx) = searchResults[idx].coefs;
                lls(outIdx) = searchResults[idx].ll;
                if (searchResults[idx].ll > maxLL) {
                    maxLL = searchResults[idx].ll;
                    maxLLpos = outIdx;
                }
                successfulConverge++;
                outIdx++;
            } else if (searchResults[idx].coefs.n_rows > 0) {
                allCoefs.col(outIdx) = searchResults[idx].coefs;
                outIdx++;
            }
        }
    } else {
        // ---- sequential search mode ----
        double totalMinsTaken = 0.0;
        int i = 0;
        int cntr = -1;
        while ((i < reps) && (cntr < maximumAttempts)) {
            cntr++;
            if (globalstate::InterruptRequested.load(std::memory_order_relaxed)) {
                pyinterface::throwPyInterruptIfNeeded();
            }
            if (PyErr_CheckSignals() != 0) throw py::error_already_set();
            auto tStart = std::chrono::high_resolution_clock::now();
            arma::dcolvec theta0Rep(theta0.n_rows);
            bool couldFindStart = esautils::findValidStartValuesLCM(
                lcmData, runner.modelObjPtr, theta0, theta0Rep,
                cntr, slengthFrontier, slengthSigmas, seed,
                (overrideStartValCheck ? 1 : maxStartValFindAttempt),
                (printLevel > 2)
            );
            if (!couldFindStart && !overrideStartValCheck) {
                ESALogger::logger()->info("Could not find appropriate starting values.");
                failedStart++;
                continue;
            }
            // run the optimization
            std::unique_ptr<ESAOptimResult> resultUnk = runner.runOptimization(theta0Rep, hessCalcMethod, 0);
            // check if a python interrupt was required, and throw if needed
            if (globalstate::InterruptRequested.load(std::memory_order_relaxed)) {
                pyinterface::throwPyInterruptIfNeeded();
            }
            if (PyErr_CheckSignals() != 0) throw py::error_already_set();
            if (resultUnk == nullptr) {
                allCoefs.col(i).fill(std::numeric_limits<double>::quiet_NaN());
            } else if (resultUnk->getDidConverge()) {
                if (printLevel > 0){
                    ESALogger::logger()->info("---- Trial Number {}/{} ----", i+1, reps);
                }
                ESAOptimResultSuccess& optimRes = (ESAOptimResultSuccess&)*resultUnk;
                allCoefs.col(i) = optimRes.getX();
                double ll = optimRes.getLogLike();
                if (ll > maxLL) {
                    maxLL = ll;
                    maxLLpos = i;
                }
                lls(i) = ll;
                if (printLevel > 0){
                    ESALogger::logger()->info("{}/{} converged with loglikelihood {:.12f}", i, reps, ll);
                }
                successfulConverge++;
                i++;
            } else {
                ESAOptimResultFailed& optimRes = (ESAOptimResultFailed&)*resultUnk;
                allCoefs.col(i) = optimRes.getX();
            }
            auto tEnd = std::chrono::high_resolution_clock::now();
            double min_diff = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count() / 1000.0 / 60.0;
            totalMinsTaken += min_diff;
            if ((i % 10 == 0) && (i > 0) && (printLevel > 0)) {
                interface::printSearchesElapsedAndETA(ESALogger::logger(), totalMinsTaken, i, reps);
            }
        }
    }
    // print summary
    if (printLevel > 0){
        ESALogger::logger()->info(
            "From {} trials, {} successfully converged. {} did not yield valid starting values.",
            reps, successfulConverge, failedStart
        );
        interface::printSearches(lls, allCoefs, 30, displayConsoleWidth, displayDecimalPlaces);
    }
    py::dict out;
    out["ll_scores"] = numpy_arma_conv::colToPy<double>(lls);
    out["all_coefs"] = numpy_arma_conv::armaToPy<double>(allCoefs);
    out["maxLL"] = maxLL;
    out["maxLLpos"] = maxLLpos;
    if (maxLLpos >= 0) {
        arma::dcolvec bestCoef(allCoefs.col(maxLLpos));
        out["maxLLCoefVec"] = numpy_arma_conv::colToPy<double>(bestCoef);
    }
    ESALogger::logger()->sinks().clear();
    spdlog::set_default_logger(nullptr);
    return out;
}

// ┌─────────────────────────────────────────────────────────────┐
// │                       Main Estimation Logic.                      │
// └─────────────────────────────────────────────────────────────┘

/**
 * @details The main function called from python
 * @param y_in 
 * @param x_in
 * @param zmuit_in
 * @param zuit_in
 * @param zvit_in
 * @param zui0_in
 * @param zvi0_in
 * @param start_in
 * @param idVec_in
 * @param timeVec_in
 * @param prodCost
 * @param model
 * @param dist
 */
py::dict pysfacpp_internal(
    py::array_t<double> y_in,
    py::array_t<double> x_in,
    std::optional<py::array_t<double>> zmuit_in,
    std::optional<py::array_t<double>> zuit_in,
    std::optional<py::array_t<double>> zvit_in,
    std::optional<py::array_t<double>> zui0_in,
    std::optional<py::array_t<double>> zvi0_in,
    std::optional<py::array_t<double>> start_in,
    std::optional<py::array_t<int>> idVec_in,
    std::optional<py::array_t<int>> timeVec_in,
    const int prodCost = 1,
    const std::string& model = "tre",
    const std::string& dist = "hnorm",
    const std::string& method = "tr",
    const std::string& methodLib = "dlib",
    const std::string& marginalEffect = "wang2002",
    const std::optional<py::dict> optimOpts = std::nullopt,
    std::optional<std::vector<std::string>> termsX_in = std::nullopt,
    std::optional<std::vector<std::string>> termsZmuit_in = std::nullopt,
    std::optional<std::vector<std::string>> termsZuit_in = std::nullopt,
    std::optional<std::vector<std::string>> termsZvit_in = std::nullopt,
    std::optional<std::vector<std::string>> termsZui0_in = std::nullopt,
    std::optional<std::vector<std::string>> termsZvi0_in = std::nullopt,
    const unsigned int nsim = 100,
    const std::string& hessianCalc = "analytical",
    const unsigned int hessianCalcNumApproxAccuracy = 3,
    const int seed = 1234,
    const double confidenceLevel = 0.95,
    const bool estimateMarginalEffects = false,
    const bool estimateMargEffCI = false,
    const unsigned int marginalEffectBootstrapReps = 500,
    const unsigned int printLevel = 2,
    const bool clusteredSE = true,
    const int nthreads = 20,
    const bool calculateEfficiencyScores = false,
    const int ghkSimReps = 2000,
    const int haltonBase = 2,
    const int haltonBurnin = 1000,
    const int haltonUi0Base = 3,
    const bool scrambledHalton = true,
    const bool shuffledHalton = false,
    const bool shouldCopy = false,
    const int displayDecimalPlaces = 5,
    const int displayConsoleWidth = 120,
    const std::optional<std::string>& idColName = std::nullopt,
    const std::optional<std::string>& timeColName = std::nullopt
)
{
    // set the default logger to be the one attached from python
    auto bridge = spdlog::get("python_bridge");
    if (bridge) {
        spdlog::set_default_logger(bridge);
    } else {
        // Fallback or warning to prevent crash
        std::cout << "Could not attach to a logger :(" << std::endl;
        try 
        {
            auto logger = spdlog::basic_logger_mt("basic_logger", "logs/sfacpp.txt");
            spdlog::set_default_logger(logger);
        }
        catch (const spdlog::spdlog_ex &ex)
        {
            std::cout << "Log init failed " << ex.what() << std::endl;
            throw std::runtime_error("No logging available");
        }
    }
    ESALogger::logger()->info("current memory usage (RSS) @ entry is {}", memoryusage::get_memory_usage_mb());
    // ---- python specific conversions ----
    // zero-copy conversion
    // copy=false lets Armadillo point to python memory - BUT - python inputs must be F-Contiguous (column major)
    const arma::colvec y = numpy_arma_conv::pyToCol<double>(y_in, shouldCopy);
    const arma::dmat x = numpy_arma_conv::pyToArma<double>(x_in, shouldCopy);
    arma::dmat zmuit_, zuit_, zvit_, zui0_, zvi0_;
    arma::Col<int> idVec_, timeVec_;
    if (zmuit_in.has_value()) zmuit_ = numpy_arma_conv::pyToArma<double>(zmuit_in.value(), shouldCopy);
    if (zuit_in.has_value()) zuit_ = numpy_arma_conv::pyToArma<double>(zuit_in.value(), shouldCopy);
    if (zvit_in.has_value()) zvit_ = numpy_arma_conv::pyToArma<double>(zvit_in.value(), shouldCopy);
    if (zui0_in.has_value()) zui0_ = numpy_arma_conv::pyToArma<double>(zui0_in.value(), shouldCopy);
    if (zvi0_in.has_value()) zvi0_ = numpy_arma_conv::pyToArma<double>(zvi0_in.value(), shouldCopy);
    if (idVec_in.has_value()) idVec_ = numpy_arma_conv::pyToCol<int>(idVec_in.value(), shouldCopy);
    if (timeVec_in.has_value()) timeVec_ = numpy_arma_conv::pyToCol<int>(timeVec_in.value(), shouldCopy);
    // can copy the starting vector its only smol
    std::optional<arma::colvec> startVals_ = std::nullopt;
    if (start_in) startVals_ = std::make_optional(numpy_arma_conv::pyToArma<double>(start_in.value()));
    // ---- initial load ----
    // status message on first load
    ESASingletonStatuses* stati = ESASingletonStatuses::GetInstance();
    if (!stati->getStatus(ESAStatusKeys::kHasShownInitialError)){
        ESALogger::logger()->warn("This package is experimental, and therefore may produce incorrect results. Use at your own risk.");
        ESALogger::logger()->warn("NHS England and the authors bear no responsibility or liability for the use of, and any consequences from, this package.");
        ESALogger::logger()->warn( "Alternative packages include 'sfpanel' in Stata, 'npsf', 'sfaR', 'frontier' in R.");
        ESALogger::logger()->warn("Please report any issues to the package maintainer.");
        stati->setStatus(ESAStatusKeys::kHasShownInitialError, true);
    }
    ESASfaModelType mT = ESAEnums::getModelTypeForNameAndDist(model, dist);
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    // ---- setup for threading ----
    // set number of threads to user amount
    esaparallel::setThreadCount(nthreads);
    // reset TLS structs if needed
    esaparallel::modelChangeFlushUnneededTLS(mF);
    // ---- general settings ----
    // whether or not to cluster the standard errors - don't do it in cross-sectional
    bool shouldClusterSE = (mF == ESASfaModelFamily::CROSS) ? false : clusteredSE;
    // model solver for main model
    ModelSolver mainModelSolver = ESAEnums::getModelSolverForMethodAndLib(method, methodLib);
    // hessian method 
    HessianCalcMethod hessCalcMethod = ESAEnums::getHessianCalcMethod(hessianCalc);
    // default optim params
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    ESAOptimParams mainOptimParams, startValOptimParams;
    // modify optimization parameters
    pyinterface::setupOptimParams(mainOptimParams, optimOpts, seed);
    globalOptimParams->mainOptimParams = mainOptimParams;
    globalOptimParams->mainModelSolver = mainModelSolver;
    globalOptimParams->hessianMethod = hessCalcMethod;
    globalOptimParams->hessianNumApproxAcc = hessianCalcNumApproxAccuracy;
    globalOptimParams->optimThreaded = (nthreads > 1);
    // ---- halton settings ----
    HaltonSettings hsetting = interface::haltonSettingsForOpts(haltonBase, haltonBurnin, haltonUi0Base, scrambledHalton, shuffledHalton);
    // ---- setup runner ----
    ESASfaRunner runner(model, dist, prodCost, seed, nsim, printLevel);
    // next, load the data
    runner.loadData(
        &y,
        &x,  
        zmuit_in.has_value() ? &zmuit_ : nullptr,
        zuit_in.has_value() ? &zuit_ : nullptr,
        zvit_in.has_value() ? &zvit_ : nullptr,
        zui0_in.has_value() ? &zui0_ : nullptr,
        zvi0_in.has_value() ? &zvi0_ : nullptr,
        idVec_in.has_value() ? &idVec_ : nullptr,
        timeVec_in.has_value() ? &timeVec_ : nullptr
    );
    // setup the model, its terms
    runner.setupModel(hsetting, termsX_in, termsZmuit_in, termsZuit_in, termsZvit_in, termsZui0_in, termsZvi0_in);
    if (printLevel > 0){
        ESALogger::logger()->info("current memory usage (RSS) after setup is {}", memoryusage::get_memory_usage_mb());
    }
    // run optimization
    std::unique_ptr<ESAOptimResult> resultUnk = runner.runOptimization(
        startVals_,
        hessCalcMethod,
        hessianCalcNumApproxAccuracy
    );
    if (printLevel > 0){
        ESALogger::logger()->info("current memory usage (RSS) post optimization is {}", memoryusage::get_memory_usage_mb());
    }
    py::dict res;
    // check the result
    if (resultUnk == nullptr) {
        ESALogger::logger()->error("Unexpected internal error, nothing was returned from optimization");
    } else if (resultUnk->getDidConverge()) {
        // succesfully converged
        // instance of ESAOptimResultSuccess
        ESAOptimResultSuccess& optimRes = (ESAOptimResultSuccess&)*resultUnk;
        // for all the elements, these are rvalues when accessed from the success object - create
        // an l-value for them
        arma::dcolvec coefs = optimRes.getX();
        std::vector<std::string> allTerms = runner.modelTerms->allTerms();
        arma::dmat vcov = optimRes.getVcov();
        double llscore = optimRes.getLogLike();
        // calculated the average hessian and gradient - multiply by that factor to get original hess/grad
        arma::dmat hess = (optimRes.getHessian() * optimRes.getN());
        arma::dmat grad = (optimRes.getGradient() * optimRes.getN());
        // jacobian matrix - gradient at each obs
        arma::dmat jac = optimRes.getGradientIndividual();
        // degrees of freedom 
        int dof = optimRes.degreesFreedom();
        // nobs
        int nobs = optimRes.getNobs();
        int nparam = runner.dataObjPtr->nParams();
        int maxT = runner.dataObjPtr->getMaxT();
        int minT = runner.dataObjPtr->getMinT();
        double gnorm = optimRes.getGnorm();
        // ---- clustered se if desired ----
        arma::dmat vcovSummary;
        if (shouldClusterSE) {
            arma::Col<int> empty_ivec;
            vcovSummary = sandwich::clusteredVcov(vcov, jac, idVec_in.has_value() ? idVec_ : empty_ivec);
        } else {
            vcovSummary = vcov;
        }
        ESASigmaParams sigParams = runner.modelObjPtr->getSigmaParams(coefs);
        double sigma_uit = sigParams.s_uit, sigma_vit = sigParams.s_vit;
        double sigma_ui0 = sigParams.s_ui0, sigma_vi0 = sigParams.s_vi0;
        double lambda = sigParams.lambda, lambda_0 = sigParams.lambda_0;
        double BigLambda = sigParams.BigLambda; 
        // model summary
        arma::dmat msummary = runner.buildSummary(coefs, vcovSummary, confidenceLevel, dof);
        if (calculateEfficiencyScores) {
            ESALogger::logger()->info("current memory usage (RSS) before efficiency scores is {}", memoryusage::get_memory_usage_mb());
            // calculate efficiency scores
            std::unique_ptr<ESASfaEffScores> effs = runner.estimateEfficiencyScores(
                coefs,
                ghkSimReps, // number of ghk simulations
                0 // start position for prime numbers, used for halton bases
            );
            if (effs != nullptr) {
                // tre/gtre
                // extract the matricies
                if ((*effs).transient) {
                    arma::dmat effTransient = (*effs).transient.value();
                    // add sanity check to matrix
                    if (effTransient.n_elem > 0) {
                        res["efficiencyTransient"] = numpy_arma_conv::armaToPy<double>(effTransient);
                    } else {
                        ESALogger::logger()->error("Transient efficiency matrix is empty");
                    }
                }
                if ((*effs).persistent) {
                    arma::dmat effPersistent = (*effs).persistent.value();
                    if (effPersistent.n_elem > 0) {
                        res["efficiencyPersistent"] = numpy_arma_conv::armaToPy<double>(effPersistent);
                    } else {
                        ESALogger::logger()->error("Persistent efficiency matrix is empty");
                    }
                }
            } else {
                ESALogger::logger()->warn("No efficiency scores were calculated");
            }
            ESALogger::logger()->info("current memory usage (RSS) after efficiency scores is {}", memoryusage::get_memory_usage_mb());
        }
        // ---- print the output ----
        if (printLevel > 0) {
            interface::printModelOutput(
                runner.dataObjPtr, // ptr to data object
                msummary, // model summary
                allTerms, // model terms
                sigParams, // sigma parameters
                nsim,
                llscore,
                optimRes.getGnorm(),
                shouldClusterSE,
                hsetting,
                confidenceLevel, // confint,
                displayDecimalPlaces, // decimal places
                displayConsoleWidth, // console width
                idColName,
                timeColName
            );
        }
        // ---- fill python return dictionary
        res["par"] = numpy_arma_conv::colToPy<double>(coefs);
        res["vars"] = allTerms;
        res["logLikelihood"] = llscore;
        res["degreesFreedom"] = dof;
        res["nobs"] = nobs;
        res["maxT"] = maxT;
        res["minT"] = minT;
        res["modelSummary"] = numpy_arma_conv::armaToPy<double>(msummary);
        res["clusteredSE"] = shouldClusterSE;
        res["vcov"] = numpy_arma_conv::armaToPy<double>(vcovSummary);
        res["hessian"] = numpy_arma_conv::armaToPy<double>(hess);
        res["gradient"] = numpy_arma_conv::armaToPy<double>(grad);
        res["jacobian"] = numpy_arma_conv::armaToPy<double>(jac);
        res["nparam"] = nparam;
        res["gnorm"] = gnorm;
        // n firms
        if (mF != ESASfaModelFamily::CROSS) {
            int nfirms = optimRes.getN();
            res["nfirm"] = nfirms;
        }
        // fill the sigma parameters
        py::dict sigmas;
        sigmas["sigma_uit"] = sigma_uit;
        sigmas["sigma_vit"] = sigma_vit;
        sigmas["sigma_vi0"] = sigma_vi0;
        sigmas["sigma_ui0"] = sigma_ui0;
        sigmas["lambda"] = lambda;
        sigmas["lambda_0"] = lambda_0;
        sigmas["BigLambda"] = BigLambda;
        res["sigmas"] = sigmas;
        // ---- calculate marginal effects if desired ----
        if (estimateMarginalEffects){
            ESALogger::logger()->info("current memory usage (RSS) before marginal effects is {}", memoryusage::get_memory_usage_mb());
            std::unique_ptr<ESASfaMeffReturn> meff = nullptr;
            std::unique_ptr<ESASfaMeffCIReturn> meffCIs = nullptr;
            runner.estimateMarginalEffects(
                marginalEffect, // method
                coefs, // parameter vector/ coefficients
                estimateMargEffCI, // whether or not to estimate confidence intervals
                confidenceLevel, // conf level
                marginalEffectBootstrapReps,
                meff,
                meffCIs
            );
            // check result
            if (meff == nullptr) {
                ESALogger::logger()->error("Calculation of marginal effects failed");
            } else {
                arma::dmat meffs = meff->marginalEffects;
                std::vector<std::string> meffCnames = meff->columnNames;
                // add to results dictionary
                res["marginalEffects"] = numpy_arma_conv::armaToPy<double>(meffs);
                res["marginalEffectsNames"] = meffCnames;
            }
            if (estimateMargEffCI && meffCIs == nullptr) {
                ESALogger::logger()->error("Calculation of confidence intervals for marginal effects failed");
            } else if (estimateMargEffCI) {
                arma::dmat meffLower = meffCIs->lowerCI;
                arma::dmat meffUpper = meffCIs->upperCI;
                std::vector<std::string> meffCols = meffCIs->columnNames;
                res["marginalEffectsLwrCI"] = numpy_arma_conv::armaToPy<double>(meffLower);
                res["maringalEffectsUprCI"] = numpy_arma_conv::armaToPy<double>(meffUpper);
                res["marginalEffectsCICols"] = meffCols;
            }
            ESALogger::logger()->info("current memory usage (RSS) after marginal effects is {}", memoryusage::get_memory_usage_mb());
            meff.reset();
            meffCIs.reset();
        }
    } else {
        // failed to converge
        // instance of ESAOptimResultFailed
        ESAOptimResultFailed& optimRes = (ESAOptimResultFailed&)*resultUnk;
        arma::dcolvec failCoef = optimRes.getX();
        std::vector<std::string> failTerms = runner.modelTerms->allTerms();
        py::dict res;
        res["par"] = numpy_arma_conv::colToPy<double>(failCoef);
        res["vars"] = failTerms;
        ESALogger::logger()->warn("Optimization failed. Only returning parameters");
    }
    resultUnk.reset();
    runner.modelObjPtr.reset();
    runner.dataObjPtr.reset();
    // unhook logger
    ESALogger::logger()->info("current memory usage (RSS) on exit is {}", memoryusage::get_memory_usage_mb());
    ESALogger::logger()->sinks().clear();
    spdlog::set_default_logger(nullptr);
    return res;
}

// ┌─────────────────────────────────────────────────────────────┐
// │                       LCM interface                               │
// └─────────────────────────────────────────────────────────────┘

py::dict pysfacpp_internal_lcm(
    py::array_t<double> y_in,
    py::array_t<double> x_in,
    py::array_t<double> seg_in,
    std::optional<py::array_t<double>> zmuit_in,
    py::array_t<double> zuit_in,
    py::array_t<double> zvit_in,
    py::array_t<double> zvi0_in,
    py::array_t<int> idVec_in,
    py::array_t<int> timeVec_in,
    const int nClasses = 2,
    std::optional<py::array_t<double>> start_in = std::nullopt,
    const int prodCost = 1,
    const std::string& model = "lctre",
    const std::string& dist = "hnorm",
    const std::string& method = "tr",
    const std::string& methodLib = "dlib",
    const std::optional<py::dict> optimOpts = std::nullopt,
    std::optional<std::vector<std::string>> termsX_in = std::nullopt,
    std::optional<std::vector<std::string>> termsZmuit_in = std::nullopt,
    std::optional<std::vector<std::string>> termsZuit_in = std::nullopt,
    std::optional<std::vector<std::string>> termsZvit_in = std::nullopt,
    std::optional<std::vector<std::string>> termsZvi0_in = std::nullopt,
    std::optional<std::vector<std::string>> termsSeg_in = std::nullopt,
    const unsigned int nsim = 100,
    const std::string& hessianCalc = "analytical",
    const int seed = 1234,
    const double confidenceLevel = 0.95,
    const unsigned int printLevel = 2,
    const bool clusteredSE = true,
    const int nthreads = 20,
    const int haltonBase = 2,
    const int haltonBurnin = 1000,
    const int haltonUi0Base = 3,
    const bool scrambledHalton = true,
    const bool shuffledHalton = false,
    const bool shouldCopy = false,
    const int displayDecimalPlaces = 5,
    const int displayConsoleWidth = 120,
    const std::optional<std::string>& idColName = std::nullopt,
    const std::optional<std::string>& timeColName = std::nullopt,
    const std::string& efficiencyMethod = "colombi"
)
{
    // setup logging functionality
    auto bridge = spdlog::get("python_bridge");
    if (bridge) {
        spdlog::set_default_logger(bridge);
    } else {
        try {
            auto logger = spdlog::basic_logger_mt("basic_logger_lcm", "logs/sfacpp_lcm.txt");
            spdlog::set_default_logger(logger);
        } catch (const spdlog::spdlog_ex& ex) {
            throw std::runtime_error("No logging available for LCM");
        }
    }
    if (printLevel > 3) {
        ESALogger::logger()->info("current memory usage (RSS) @ entry is {}", memoryusage::get_memory_usage_mb());
    }
    // ---- conversions ----
    const arma::colvec y = numpy_arma_conv::pyToCol<double>(y_in, shouldCopy);
    const arma::dmat x = numpy_arma_conv::pyToArma<double>(x_in, shouldCopy);
    const arma::dmat seg = numpy_arma_conv::pyToArma<double>(seg_in, shouldCopy);
    arma::dmat zmuit_, zuit_, zvit_, zvi0_;
    if (zmuit_in.has_value()) zmuit_ = numpy_arma_conv::pyToArma<double>(zmuit_in.value(), shouldCopy);
    zuit_ = numpy_arma_conv::pyToArma<double>(zuit_in, shouldCopy);
    zvit_ = numpy_arma_conv::pyToArma<double>(zvit_in, shouldCopy);
    zvi0_ = numpy_arma_conv::pyToArma<double>(zvi0_in, shouldCopy);
    arma::Col<int> idVec_ = numpy_arma_conv::pyToCol<int>(idVec_in, shouldCopy);
    arma::Col<int> timeVec_ = numpy_arma_conv::pyToCol<int>(timeVec_in, shouldCopy);
    std::optional<arma::colvec> startVals_ = std::nullopt;
    if (start_in) startVals_ = std::make_optional(numpy_arma_conv::pyToArma<double>(start_in.value()));
    // ---- model setup ----
    ESASfaModelType mT = ESAEnums::getModelTypeForNameAndDist(model, dist);
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    if (mF != ESASfaModelFamily::LC_TRE) {
        throw std::invalid_argument("pysfacpp_internal_lcm only supports LC-TRE models (model='lctre')");
    }
    esaparallel::setThreadCount(nthreads);
    esaparallel::modelChangeFlushUnneededTLS(mF);
    ModelSolver mainModelSolver = ESAEnums::getModelSolverForMethodAndLib(method, methodLib);
    HessianCalcMethod hessCalcMethod = ESAEnums::getHessianCalcMethod(hessianCalc);
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    ESAOptimParams mainOptimParams;
    pyinterface::setupOptimParams(mainOptimParams, optimOpts, seed);
    globalOptimParams->mainOptimParams = mainOptimParams;
    globalOptimParams->mainModelSolver = mainModelSolver;
    globalOptimParams->hessianMethod = hessCalcMethod;
    globalOptimParams->hessianNumApproxAcc = 0;
    globalOptimParams->optimThreaded = (nthreads > 1);
    HaltonSettings hsetting = interface::haltonSettingsForOpts(haltonBase, haltonBurnin, haltonUi0Base, scrambledHalton, shuffledHalton);
    // ---- runner ----
    ESASfaRunner runner(model, dist, prodCost, seed, nsim, printLevel);
    runner.loadDataLCM(
        &y,
        &x,
        &seg,
        zmuit_in.has_value() ? &zmuit_ : nullptr,
        &zuit_,
        &zvit_,
        &zvi0_,
        &idVec_,
        &timeVec_,
        static_cast<unsigned int>(nClasses)
    );
    runner.setupModel(hsetting, termsX_in, termsZmuit_in, termsZuit_in, termsZvit_in, std::nullopt, termsZvi0_in);
    if (printLevel > 3) {
        ESALogger::logger()->info("current memory usage (RSS) after setup is {}", memoryusage::get_memory_usage_mb());
    }
    // ---- optimization ----
    std::unique_ptr<ESAOptimResult> resultUnk = runner.runOptimization(startVals_, hessCalcMethod, 0);
    if (printLevel > 3) {
        ESALogger::logger()->info("current memory usage (RSS) post optimization is {}", memoryusage::get_memory_usage_mb());
    }
    py::dict res;
    if (resultUnk == nullptr) {
        ESALogger::logger()->error("Unexpected internal error, nothing was returned from optimization");
    } else if (resultUnk->getDidConverge()) {
        ESAOptimResultSuccess& optimRes = (ESAOptimResultSuccess&)*resultUnk;
        arma::dcolvec coefs = optimRes.getX();
        arma::dmat vcov = optimRes.getVcov();
        double llscore = optimRes.getLogLike();
        arma::dmat hess = (optimRes.getHessian() * optimRes.getN());
        arma::dmat grad = (optimRes.getGradient() * optimRes.getN());
        arma::dmat jac = optimRes.getGradientIndividual();
        int nobs = optimRes.getNobs();
        double gnorm = optimRes.getGnorm();
        int nfirms = optimRes.getN();
        // clustered SE — LCM gradHess returns firm-level jac (one row per firm),
        // so build a trivial id vec [0..nfirms-1] rather than passing the obs-level idVec_
        arma::dmat vcovSummary;
        if (clusteredSE) {
            arma::Col<int> firmIdVec = arma::regspace<arma::Col<int>>(0, nfirms - 1);
            vcovSummary = sandwich::clusteredVcov(vcov, jac, firmIdVec);
        } else {
            vcovSummary = vcov;
        }
        res["par"] = numpy_arma_conv::colToPy<double>(coefs);
        res["logLikelihood"] = llscore;
        res["nobs"] = nobs;
        res["nparam"] = static_cast<int>(coefs.n_rows);
        res["nClasses"] = nClasses;
        res["nfirm"] = nfirms;
        res["vcov"] = numpy_arma_conv::armaToPy<double>(vcovSummary);
        res["hessian"] = numpy_arma_conv::armaToPy<double>(hess);
        res["gradient"] = numpy_arma_conv::armaToPy<double>(grad);
        res["jacobian"] = numpy_arma_conv::armaToPy<double>(jac);
        res["gnorm"] = gnorm;
        res["clusteredSE"] = clusteredSE;
        res["converged"] = true;
        // ---- post-estimation: posteriors, efficiency scores, marginal effects ----
        auto lcmDataTyped = std::dynamic_pointer_cast<ESADataPanelLCM>(runner.lcmDataObjPtr);
        auto helperDataTyped = std::dynamic_pointer_cast<ESADataPanel>(runner.dataObjPtr);
        ESASfaLcTre* lcmModelPtr = dynamic_cast<ESASfaLcTre*>(runner.modelObjPtr.get());
        // 
        if (lcmDataTyped && helperDataTyped && lcmModelPtr) {
            // calculate posterior class probabilities
            arma::dmat posteriors = lcmModelPtr->computePosteriors(coefs);
            res["posteriors"] = numpy_arma_conv::armaToPy<double>(posteriors);
            // calculate efficiency scores
            try {
                std::unique_ptr<ESASfaEffLcmScores> effScores;
                if (efficiencyMethod == "jlms") {
                    ESASfaEffLcmJlms effCalc(lcmDataTyped, helperDataTyped, prodCost);
                    effScores = effCalc.efficiencyScores(
                        coefs, posteriors, nsim, 0, seed
                    );
                } else {
                    ESASfaEffLcmTre effCalc(lcmDataTyped, helperDataTyped, prodCost);
                    effScores = effCalc.efficiencyScores(
                        coefs, posteriors, nsim * 20, 0, seed, (nthreads > 1)
                    );
                }
                if (effScores) {
                    res["efficiencyTransient"] = numpy_arma_conv::armaToPy<double>(effScores->transientWeighted);
                    res["efficiencyTransientPerClass"] = numpy_arma_conv::armaToPy<double>(effScores->transientPerClass);
                    if (effScores->persistentWeighted) {
                        res["efficiencyPersistent"] = numpy_arma_conv::armaToPy<double>(effScores->persistentWeighted.value());
                        res["efficiencyPersistentPerClass"] = numpy_arma_conv::armaToPy<double>(effScores->persistentPerClass.value());
                    }
                }
            } catch (const std::exception& e) {
                ESALogger::logger()->warn("Efficiency score computation failed: {}", e.what());
            }
            // calculate marginal effects
            if (runner.modelTerms) {
                try {
                    ESASfaMeffLcmWang meffCalc(lcmDataTyped, runner.dataObjPtr, prodCost, nsim, seed);
                    std::unique_ptr<ESASfaMeffLcmReturn> meffs = meffCalc.lcmMarginalEffects(
                        coefs, posteriors, *runner.modelTerms
                    );
                    if (meffs) {
                        res["marginalEffects"] = numpy_arma_conv::armaToPy<double>(meffs->marginalEffectsWeighted);
                        res["marginalEffectsNames"] = meffs->columnNames;
                        // per-class marginal effects as a list
                        py::list meffPerClassList;
                        for (unsigned int c = 0; c < nClasses; c++) {
                            meffPerClassList.append(numpy_arma_conv::armaToPy<double>(meffs->marginalEffectsPerClass[c]));
                        }
                        res["marginalEffectsPerClass"] = meffPerClassList;
                    }
                } catch (const std::exception& e) {
                    ESALogger::logger()->warn("Marginal effects computation failed: {}", e.what());
                }
            }
            // term labelling for each latent class + segmenting variables
            if (runner.modelTerms) {
                std::vector<std::string> allTerms;
                // segmentation terms
                unsigned int nSeg = lcmDataTyped->getNSeg();
                std::optional<std::vector<std::string>> segTermNames = termsSeg_in;
                for (unsigned int c = 0; c < static_cast<unsigned int>(nClasses) - 1; c++) {
                    for (unsigned int k = 0; k < nSeg; k++) {
                        std::string tName = segTermNames.has_value() ? segTermNames.value()[k] : ("Seg_" + std::to_string(k));
                        allTerms.push_back("alpha_" + std::to_string(c) + ":" + tName);
                    }
                }
                // per-class frontier terms
                std::vector<std::string> baseTerms = runner.modelTerms->allTerms();
                for (unsigned int c = 0; c < static_cast<unsigned int>(nClasses); c++) {
                    for (const auto& t : baseTerms) {
                        allTerms.push_back("Class_" + std::to_string(c) + ":" + t);
                    }
                }
                res["vars"] = allTerms;
            }
            // calculate sigma parameters per class
            py::list sigmasPerClass;
            for (unsigned int c = 0; c < static_cast<unsigned int>(nClasses); c++) {
                arma::dcolvec par_c = lcmutils::buildClassParamVec(*lcmDataTyped, coefs, c);
                ESASigmaParams sp = lcmModelPtr->ESASfaTreGreene::getSigmaParams(par_c);
                py::dict classSigmas;
                classSigmas["sigma_uit"] = sp.s_uit;
                classSigmas["sigma_vit"] = sp.s_vit;
                classSigmas["sigma_vi0"] = sp.s_vi0;
                classSigmas["lambda"] = sp.lambda;
                sigmasPerClass.append(classSigmas);
            }
            res["sigmasPerClass"] = sigmasPerClass;
        }
    } else {
        ESAOptimResultFailed& optimRes = (ESAOptimResultFailed&)*resultUnk;
        arma::dcolvec failCoef = optimRes.getX();
        res["par"] = numpy_arma_conv::colToPy<double>(failCoef);
        res["converged"] = false;
        ESALogger::logger()->warn("LCM optimization failed. Only returning parameters");
    }
    resultUnk.reset();
    runner.modelObjPtr.reset();
    runner.dataObjPtr.reset();
    runner.lcmDataObjPtr.reset();
    if (printLevel > 3){
        ESALogger::logger()->info("current memory usage (RSS) on exit is {}", memoryusage::get_memory_usage_mb());
    }
    ESALogger::logger()->sinks().clear();
    spdlog::set_default_logger(nullptr);
    return res;
}

// ┌─────────────────────────────────────────────────────────────┐
// │                 LCM Cross-Sectional interface                      │
// └─────────────────────────────────────────────────────────────┘

py::dict pysfacpp_internal_lcm_cross(
    py::array_t<double> y_in,
    py::array_t<double> x_in,
    py::array_t<double> seg_in,
    py::array_t<double> zuit_in,
    py::array_t<double> zvit_in,
    const int nClasses = 2,
    std::optional<py::array_t<double>> start_in = std::nullopt,
    const int prodCost = 1,
    const std::string& dist = "hnorm",
    const std::string& method = "tr",
    const std::string& methodLib = "dlib",
    const std::optional<py::dict> optimOpts = std::nullopt,
    std::optional<std::vector<std::string>> termsX_in = std::nullopt,
    std::optional<std::vector<std::string>> termsZuit_in = std::nullopt,
    std::optional<std::vector<std::string>> termsZvit_in = std::nullopt,
    std::optional<std::vector<std::string>> termsSeg_in = std::nullopt,
    const std::string& hessianCalc = "analytical",
    const int seed = 1234,
    const double confidenceLevel = 0.95,
    const unsigned int printLevel = 2,
    const int nthreads = 20,
    const bool shouldCopy = false,
    const int displayDecimalPlaces = 5,
    const int displayConsoleWidth = 120
)
{
    auto bridge = spdlog::get("python_bridge");
    if (bridge) {
        spdlog::set_default_logger(bridge);
    } else {
        try {
            auto logger = spdlog::basic_logger_mt("basic_logger_lcm_cross", "logs/sfacpp_lcm_cross.txt");
            spdlog::set_default_logger(logger);
        } catch (const spdlog::spdlog_ex& ex) {
            throw std::runtime_error("No logging available for LCM Cross");
        }
    }
    if (printLevel > 3){
        ESALogger::logger()->info("current memory usage (RSS) @ entry is {}", memoryusage::get_memory_usage_mb());
    }
    // ---- conversions ----
    const arma::colvec y = numpy_arma_conv::pyToCol<double>(y_in, shouldCopy);
    const arma::dmat x = numpy_arma_conv::pyToArma<double>(x_in, shouldCopy);
    const arma::dmat seg = numpy_arma_conv::pyToArma<double>(seg_in, shouldCopy);
    arma::dmat zuit_ = numpy_arma_conv::pyToArma<double>(zuit_in, shouldCopy);
    arma::dmat zvit_ = numpy_arma_conv::pyToArma<double>(zvit_in, shouldCopy);
    std::optional<arma::colvec> startVals_ = std::nullopt;
    if (start_in) startVals_ = std::make_optional(numpy_arma_conv::pyToArma<double>(start_in.value()));
    // ---- model setup ----
    std::string model = "lccross";
    ESASfaModelType mT = ESAEnums::getModelTypeForNameAndDist(model, dist);
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    esaparallel::setThreadCount(nthreads);
    esaparallel::modelChangeFlushUnneededTLS(mF);
    ModelSolver mainModelSolver = ESAEnums::getModelSolverForMethodAndLib(method, methodLib);
    HessianCalcMethod hessCalcMethod = ESAEnums::getHessianCalcMethod(hessianCalc);
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    ESAOptimParams mainOptimParams;
    pyinterface::setupOptimParams(mainOptimParams, optimOpts, seed);
    globalOptimParams->mainOptimParams = mainOptimParams;
    globalOptimParams->mainModelSolver = mainModelSolver;
    globalOptimParams->hessianMethod = hessCalcMethod;
    globalOptimParams->hessianNumApproxAcc = 0;
    globalOptimParams->optimThreaded = (nthreads > 1);
    HaltonSettings hsetting;
    // ---- runner ----
    ESASfaRunner runner(model, dist, prodCost, seed, 0, printLevel);
    runner.loadDataLCM(
        &y, &x, &seg,
        nullptr,  // zmuit
        &zuit_, &zvit_,
        nullptr,  // zvi0
        nullptr,  // idVec
        nullptr,  // timeVec
        static_cast<unsigned int>(nClasses)
    );
    runner.setupModel(hsetting, termsX_in, std::nullopt, termsZuit_in, termsZvit_in, std::nullopt, std::nullopt);
    if (printLevel > 3) {
        ESALogger::logger()->info("current memory usage (RSS) after setup is {}", memoryusage::get_memory_usage_mb());
    }
    // ---- optimization ----
    std::unique_ptr<ESAOptimResult> resultUnk = runner.runOptimization(startVals_, hessCalcMethod, 0);
    if (printLevel > 3) {
        ESALogger::logger()->info("current memory usage (RSS) post optimization is {}", memoryusage::get_memory_usage_mb());
    }
    py::dict res;
    if (resultUnk == nullptr) {
        ESALogger::logger()->error("Unexpected internal error, nothing was returned from optimization");
    } else if (resultUnk->getDidConverge()) {
        ESAOptimResultSuccess& optimRes = (ESAOptimResultSuccess&)*resultUnk;
        arma::dcolvec coefs = optimRes.getX();
        arma::dmat vcov = optimRes.getVcov();
        double llscore = optimRes.getLogLike();
        arma::dmat hess = (optimRes.getHessian() * optimRes.getN());
        arma::dmat grad = (optimRes.getGradient() * optimRes.getN());
        arma::dmat jac = optimRes.getGradientIndividual();
        int nobs = optimRes.getNobs();
        double gnorm = optimRes.getGnorm();

        res["par"] = numpy_arma_conv::colToPy<double>(coefs);
        res["logLikelihood"] = llscore;
        res["nobs"] = nobs;
        res["nparam"] = static_cast<int>(coefs.n_rows);
        res["nClasses"] = nClasses;
        res["vcov"] = numpy_arma_conv::armaToPy<double>(vcov);
        res["hessian"] = numpy_arma_conv::armaToPy<double>(hess);
        res["gradient"] = numpy_arma_conv::armaToPy<double>(grad);
        res["jacobian"] = numpy_arma_conv::armaToPy<double>(jac);
        res["gnorm"] = gnorm;
        res["converged"] = true;

        // ---- Posterior class probabilities ----
        auto lcmDataTyped = std::dynamic_pointer_cast<ESADataPanelLCM>(runner.lcmDataObjPtr);
        if (lcmDataTyped) {
            unsigned int nC = lcmDataTyped->getNClasses();
            arma::dcolvec segParams = lcmDataTyped->paramSeg(coefs);
            const arma::dmat& segMat = lcmDataTyped->getSeg();
            arma::dmat posteriors(nobs, nC);
            for (int i = 0; i < nobs; i++) {
                arma::rowvec z_i = segMat.row(i);
                arma::dcolvec pi_i = lcmutils::computeClassProbs(z_i, segParams, nC);
                // Compute per-class log-likelihoods
                arma::dcolvec logLik_c(nC);
                for (unsigned int c = 0; c < nC; c++) {
                    arma::dcolvec b_x = lcmDataTyped->paramX(coefs, c);
                    auto b_zuit_opt = lcmDataTyped->paramZuit(coefs, c);
                    auto b_zvit_opt = lcmDataTyped->paramZvit(coefs, c);
                    arma::dcolvec b_zuit = b_zuit_opt.value();
                    arma::dcolvec b_zvit = b_zvit_opt.value();
                    auto zuitOpt = lcmDataTyped->getZuit();
                    auto zvitOpt = lcmDataTyped->getZvit();
                    double xb = arma::as_scalar(lcmDataTyped->getX().row(i) * b_x);
                    double eps = lcmDataTyped->getY()(i) - xb;
                    double sigma2u = std::exp(arma::as_scalar(zuitOpt.value().row(i) * b_zuit));
                    double sigma2v = std::exp(arma::as_scalar(zvitOpt.value().row(i) * b_zvit));
                    double dens = ESASfaLcmCross::densityHalfNormal(eps, sigma2u, sigma2v, prodCost);
                    logLik_c(c) = std::log(std::max(dens, 1e-300));
                }
                arma::dcolvec logPi_i = arma::log(pi_i);
                arma::dcolvec tau_i = lcmutils::computePosteriors(logPi_i, logLik_c);
                posteriors.row(i) = tau_i.t();
            }
            res["posteriors"] = numpy_arma_conv::armaToPy<double>(posteriors);
        }

        // ---- Term labelling ----
        if (runner.modelTerms && lcmDataTyped) {
            std::vector<std::string> allTerms;
            unsigned int nSeg = lcmDataTyped->getNSeg();
            std::optional<std::vector<std::string>> segTermNames = termsSeg_in;
            for (unsigned int c = 0; c < static_cast<unsigned int>(nClasses) - 1; c++) {
                for (unsigned int k = 0; k < nSeg; k++) {
                    std::string tName = segTermNames.has_value() ? segTermNames.value()[k] : ("Seg_" + std::to_string(k));
                    allTerms.push_back("alpha_" + std::to_string(c) + ":" + tName);
                }
            }
            std::vector<std::string> baseTerms = runner.modelTerms->allTerms();
            for (unsigned int c = 0; c < static_cast<unsigned int>(nClasses); c++) {
                for (const auto& t : baseTerms) {
                    allTerms.push_back("Class_" + std::to_string(c) + ":" + t);
                }
            }
            res["vars"] = allTerms;
        }
    } else {
        ESAOptimResultFailed& optimRes = (ESAOptimResultFailed&)*resultUnk;
        arma::dcolvec failCoef = optimRes.getX();
        res["par"] = numpy_arma_conv::colToPy<double>(failCoef);
        res["converged"] = false;
        ESALogger::logger()->warn("LCM Cross optimization failed. Only returning parameters");
    }
    resultUnk.reset();
    runner.modelObjPtr.reset();
    runner.lcmDataObjPtr.reset();
    if (printLevel > 3){
        ESALogger::logger()->info("current memory usage (RSS) on exit is {}", memoryusage::get_memory_usage_mb());
    }
    ESALogger::logger()->sinks().clear();
    spdlog::set_default_logger(nullptr);
    return res;
}

// ┌─────────────────────────────────────────────────────────────┐
// │                       Python <> spdlog logging                    │
// └─────────────────────────────────────────────────────────────┘
using PythonLogCallback = std::function<void(std::string category, std::string msg, int level)>;

// Custom spdlog sink
// This sink receives log messages from spdlog and forwards them to a Python callback
template<typename Mutex>
class python_sink : public spdlog::sinks::base_sink<Mutex> {
public:
    // The callback takes the message string and the log level (int)
    explicit python_sink(PythonLogCallback callback) 
        : callback_(std::move(callback)) {}

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        // if logger deactived/python finalizing, dont acquire GIL
        if (!gPythonLoggerActive || !Py_IsInitialized()) return;
        // Acquire the GIL before calling into Python
        py::gil_scoped_acquire acquire;
        // Extract the log message. 
        // msg.payload is the raw message content (without spdlog timestamps/formatting).
        std::string payload(msg.payload.data(), msg.payload.size());
        // extract logger name
        std::string logger_name(msg.logger_name.data(), msg.logger_name.size());
        // msg.level is an enum; cast it to int for easier handling in Python
        callback_(logger_name, payload, static_cast<int>(msg.level));
    }

    void flush_() override {
        // Python logging flushes automatically usually, but you could implement force flush here
    }

private:
    PythonLogCallback callback_;
};

// helper function to register this sink
void registerPythonLogger(PythonLogCallback callback) {
    // Create the sink, protected by a mutex
    auto sink = std::make_shared<python_sink<std::mutex>>(callback);
    // Create a new logger with this sink
    auto logger = std::make_shared<spdlog::logger>("python_bridge", sink);
    // set this as the global default logger
    spdlog::set_default_logger(logger);
    // log everything and let Python filter by level
    spdlog::set_level(spdlog::level::trace);
}

void teardownPythonLogger() {
    gPythonLoggerActive = false;
    spdlog::set_default_logger(nullptr);
    spdlog::shutdown();
}


// ┌─────────────────────────────────────────────────────────────┐
// │                  Model summary print interface                    │
// └─────────────────────────────────────────────────────────────┘

void callPrintModelTable(
    const std::string& label,
    const py::array_t<double>& modelSummaryIn,
    const std::vector<std::string>& modelSummaryTerms,
    const std::optional<std::map<std::string, double>>& extraParams,
    const double confInt,
    const int decimalPlaces,
    const int consoleWidth
)
{
    arma::dmat modelSummary = numpy_arma_conv::pyToArma<double>(modelSummaryIn, false);
    if (modelSummary.n_cols != 6) {
        throw std::invalid_argument(
            "Model summary expected 6 columns, found " + std::to_string(modelSummary.n_cols)
        );
    }
    if (modelSummary.n_rows != modelSummaryTerms.size()) {
        throw std::invalid_argument(
            "Row count (" + std::to_string(modelSummary.n_rows) +
            ") != term count (" + std::to_string(modelSummaryTerms.size()) + ")"
        );
    }
    std::map<std::string, double> ep = extraParams.value_or(std::map<std::string, double>{});
    interface::printModelTable(label, modelSummary, modelSummaryTerms, ep, confInt, decimalPlaces, consoleWidth);
    ESALogger::logger()->sinks().clear();
    spdlog::set_default_logger(nullptr);
}

void callPrintLcmOutput(
    const int nobs,
    const int nids,
    const int maxT,
    const int minT,
    const py::array_t<double>& modelSummaryIn,
    const std::vector<std::string>& modelSummaryTerms,
    const std::vector<std::map<std::string, double>>& sigmasPerClass,
    const int nClasses,
    const int nsim,
    const double llscore,
    const double gnorm,
    const bool clusteredSE,
    const int haltonBase,
    const int haltonBurnin,
    const int haltonUi0Base,
    const bool scrambledHalton,
    const bool shuffledHalton,
    const bool useGhq,
    const double confInt,
    const int decimalPlaces,
    const int consoleWidth,
    const std::optional<std::string>& idColName = std::nullopt,
    const std::optional<std::string>& timeColName = std::nullopt
)
{
    arma::dmat modelSummary = numpy_arma_conv::pyToArma<double>(modelSummaryIn, false);
    if (modelSummary.n_cols != 6) {
        throw std::invalid_argument(
            "Model summary expected 6 columns, found " + std::to_string(modelSummary.n_cols)
        );
    }
    if (modelSummary.n_rows != modelSummaryTerms.size()) {
        throw std::invalid_argument(
            "Row count (" + std::to_string(modelSummary.n_rows) +
            ") != term count (" + std::to_string(modelSummaryTerms.size()) + ")"
        );
    }
    HaltonSettings hsetting = interface::haltonSettingsForOpts(
        haltonBase, haltonBurnin, haltonUi0Base, scrambledHalton, shuffledHalton
    );
    interface::printLcmOutput(
        nobs, nids, maxT, minT,
        modelSummary, modelSummaryTerms,
        sigmasPerClass, nClasses,
        nsim, llscore, gnorm, clusteredSE,
        hsetting, useGhq, confInt, decimalPlaces, consoleWidth,
        idColName, timeColName
    );
}

void callPrintModelSummary(
    const std::string& model,
    const std::string& dist,
    const int nobs,
    const int nids,
    const int maxT,
    const int minT,
    const py::array_t<double>& modelSummaryIn,
    const std::vector<std::string>& modelSummaryTerms,
    // to reconstruct ESASigmaParams
    const std::optional<double> s_uit,
    const std::optional<double> s_vit,
    const std::optional<double> s_vi0,
    const std::optional<double> lambda,
    const std::optional<double> s_ui0,
    const std::optional<double> lambda_0,
    const std::optional<double> BigLambda,
    // 
    const int nsim,
    const double llscore,
    const double gnorm,
    const bool clusteredSE,
    // to reconstruct the halton setting struct
    const int haltonBase,
    const int haltonBurnin,
    const int haltonUi0Base,
    const bool scrambledHalton,
    const bool shuffledHalton,
    // 
    const double confInt,
    const int decimalPlaces,
    const int consoleWidth,
    const std::optional<std::string>& idColName = std::nullopt,
    const std::optional<std::string>& timeColName = std::nullopt
)
{
    // reconstruct the model type
    ESASfaModelType mT = ESAEnums::getModelTypeForNameAndDist(model, dist);
    if (mT == ESASfaModelType::MODEL_UNKNOWN) {
        throw std::invalid_argument("Not expecting an unknown model");
    }
    // wrap the model summary into an armadillo matrix
    arma::dmat modelSummary = numpy_arma_conv::pyToArma<double>(modelSummaryIn, false);
    // expecting 6 columns, if dont have that, then throw exception
    if (modelSummary.n_cols != 6) {
        throw std::invalid_argument(
            "Model summary is expected to have 6 columns, found " + std::to_string(modelSummary.n_cols)
        );
    }
    // check that the number of rows match
    if (modelSummary.n_rows != modelSummaryTerms.size()) {
        throw std::invalid_argument(
            "Number of rows in model summary (" + std::to_string(modelSummary.n_rows) +
            ") does not match number of variable names (" + std::to_string(modelSummaryTerms.size())
        );
    }
    // reconstruct the sigma struct
    ESASigmaParams sigParams = interface::reconstructSigmaParams(
        s_uit, s_vit, s_vi0, lambda, s_ui0, lambda_0, BigLambda
    );
    // reconstruct the halton setting struct
    HaltonSettings hsetting = interface::haltonSettingsForOpts(
        haltonBase, haltonBurnin, haltonUi0Base, scrambledHalton, shuffledHalton
    );
    // call function to print the model summary
    interface::printModelOutput(
        mT, // model type
        nobs, // number of observations
        nids, // number of firm identifiers
        maxT, // maximum number of time periods
        minT, // minimum number of time periods
        modelSummary, // coefficients, std err, pval etc
        modelSummaryTerms, // variable names
        sigParams, // sigma parameters
        nsim, // number of simulations//halton draws
        llscore, // loglikelihood score 
        gnorm, // gradient norm
        clusteredSE, // whether or not clustered SEs were estimated
        hsetting, // halton settings
        confInt, // confidence level (for CIs in model summary)
        decimalPlaces, // number of decimal places to use
        consoleWidth, // width of the console
        idColName,
        timeColName
    );
    // unhook the loggger
    ESALogger::logger()->sinks().clear();
    spdlog::set_default_logger(nullptr);
}

// ┌─────────────────────────────────────────────────────────────┐
// │                       Pybind11 Module                             │
// └─────────────────────────────────────────────────────────────┘
PYBIND11_MODULE(pysfacpp_internal, m) {

    m.doc() = R"pbdoc(
        Documentation
        -------------

        .. currentmodule:: pysfacpp_internal
        .. autosummary::
            :toctree: _generate

            add
    )pbdoc";

    m.def("_register_logger", &registerPythonLogger, "Redirect spdlog to Python logging");
    m.def("_teardown_logger", &teardownPythonLogger, "Disable python logging bridge");

    m.def(
        "_get_available_threads",
        [](){ return std::thread::hardware_concurrency(); },
        "available number of system threads"
    );

    m.def(
        "_flush_tls",
        [](){ 
            try {
                esaparallel::getOptimPool().reset();
            } catch(...) {}
        },
        "Flush thread local storage by resetting threads"
    );
    m.def("memusage", [](){ return memoryusage::get_memory_usage_mb(); });

    m.def(
        "_pysfacpp_internal", 
        &pysfacpp_internal, 
        py::return_value_policy::take_ownership,
        R"pbdoc(
            Python interface to internal sfacpp functions
        )pbdoc"
    );

    m.def(
        "_pysfacpp_searches",
        &pysfacpp_searches,
        py::return_value_policy::take_ownership,
        R"pbdoc(
            Python interface to internal sfacpp searching function
        )pbdoc"
    );

    m.def(
        "_pysfacpp_searches_lcm",
        &pysfacpp_searches_lcm,
        py::return_value_policy::take_ownership,
        R"pbdoc(
            Python interface to LC-TRE searching function (random perturbations of starting values).
        )pbdoc"
    );

    m.def(
        "_pysfacpp_internal_lcm",
        &pysfacpp_internal_lcm,
        py::return_value_policy::take_ownership,
        R"pbdoc(
            Python interface to internal LCM (Latent Class Model) sfacpp functions.
            Uses PSO + Trust Region optimization by default.
        )pbdoc"
    );

    m.def(
        "_pysfacpp_internal_lcm_cross",
        &pysfacpp_internal_lcm_cross,
        py::return_value_policy::take_ownership,
        R"pbdoc(
            Python interface to cross-sectional LCM (Latent Class Model) sfacpp functions.
        )pbdoc"
    );

    m.def(
        "_lrtest", [](const double ll0, const double ll1, const int nparam0, const int nparam1, const double dp = 5) {
            postestimation::LogLikeRatioTest test = postestimation::logLikelihoodRatioTest(ll0, ll1, nparam0, nparam1);
            interface::printLrTest(test, dp);
            py::dict res;
            res["statistic"] = test.stat;
            res["p_value"] = test.pval;
            res["dof"] = test.df;
            return res;
        },
        R"pbdoc(

        )pbdoc"
    );

    m.def(
        "_print_model_summary",
        &callPrintModelSummary,
        R"pbdoc(
            Print the model summary.
        )pbdoc"
    );

    m.def(
        "_print_model_table",
        &callPrintModelTable,
        R"pbdoc(
            Print only the regression table (no header block) for a parameter subset.
            Used by LC-TRE display to print one table per latent class.
        )pbdoc"
    );

    m.def(
        "_print_lcm_output",
        &callPrintLcmOutput,
        R"pbdoc(
            Print the full LC-TRE output: header block, segmentation params, then one
            table per latent class with per-class sigma extras in a single unified box.
        )pbdoc"
    );

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

#endif // PYPACKAGE