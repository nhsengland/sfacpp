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
 * @file sfacpp.cpp
 * @brief C++ functions for the SFA package
 * @date 2025-02-05
 * @author edmund haacke
 */

#include <memory>
#include <variant>
#include <string>
#include <mutex>
#include <iostream>
#include <exception>
#include <algorithm>
#include <random>
#include <chrono>

#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#endif

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
#include "efficiencies/ESASfaJlms.hpp"
#include "efficiencies/ESASfaEffGtre.hpp"
#include "utils/memoryusage.hpp"
#include "optim/ESAGlobalOptimParams.hpp"
// ---- interface utilities ----
#include "rinterface/rinterface_utils.hpp"
#include "interface/interface_utils.hpp"
// ---- runner ----
#include "sfa/ESASfaRunner.hpp"
#include "math/llratiotest.hpp"
#include "utils/SearchStartVals.hpp"

#ifdef WITHEIGEN
#ifdef RPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include "Eigen/Core"
#endif // RPACKAGE
#endif //WITHEIGEN

#if defined(RPACKAGE)

// should be automatically called when R loads the library
// extern "C" void R_init_sfacpp(DllInfo* info)
// {
//     initializeThreadContext();
//     // register routines (appaz needed)
//     R_registerRoutines(info, NULL, NULL, NULL, NULL);
//     R_useDynamicSymbols(info, FALSE);
// }
// // should be automatically called when R unloads the library
// extern "C" void R_unload_sfacpp(DllInfo* info)
// {
//     // clean up thread local storage key
//     teardownThreadContext();
// }

namespace {
    std::unique_ptr<arma::mat> makeView(double* mainPtr, int nRows, int startCol, int nCols){
        if (nCols <= 0) return nullptr;
        // calculate pointer offset - R, and armadillo are column major
        // pointer moves by nRows * columns skipped
        double* subPtr = mainPtr + (static_cast<size_t>(startCol) * static_cast<size_t>(nRows));
        // matrix wrapper (dont copy)
        return std::make_unique<arma::mat>(subPtr, nRows, nCols, false, true);
    }
}

// [[Rcpp::export]]
Rcpp::List sfacpp_internal(
    const Rcpp::NumericVector& y_,
    const Rcpp::NumericMatrix& allData_,
    const Rcpp::IntegerVector& colCounts_,
    Rcpp::Nullable<Rcpp::IntegerVector> idVec_ = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> timeVec_ = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> startVals_ = R_NilValue,
    const int prodCost = 1,
    const std::string& model = "tre",
    const std::string& dist = "hnorm",
    const std::string& method = "tr",
    const std::string& methodLib = "dlib",
    const std::string& marginalEffect = "wang2002",
    Rcpp::Nullable<Rcpp::List> optimOpts = R_NilValue,
    Rcpp::Nullable<std::vector<std::string>> termsX_ = R_NilValue,
    Rcpp::Nullable<std::vector<std::string>> termsZmuit_ = R_NilValue,
    Rcpp::Nullable<std::vector<std::string>> termsZuit_ = R_NilValue,
    Rcpp::Nullable<std::vector<std::string>> termsZvit_ = R_NilValue,
    Rcpp::Nullable<std::vector<std::string>> termsZui0_ = R_NilValue,
    Rcpp::Nullable<std::vector<std::string>> termsZvi0_ = R_NilValue,
    const unsigned int nsim = 100,
    const std::string& hessianCalc = "analytical",
    const unsigned int hessianCalcNumApproxAccuracy = 3,
    const int seed = 1234,
    const double confidenceLevel = 0.95,
    const bool estimateMarginalEffects = false,
    const bool estimateMargEffCI = false,
    const unsigned int marginalEffectBootstrapReps = 500,
    const unsigned int printLevel = 2,
    const bool clusteredSE = false,
    const int haltonBase = 2,
    const int haltonBurnin = 1000,
    const int haltonUi0Base = 3,
    const bool scrambledHalton = true,
    const bool shuffledHalton = false,
    const int nthreads = 20,
    const bool calculateEfficiencyScores = false,
    const int ghkSimReps = 2000,
    const int displayDecimalPlaces = 5,
    const int displayConsoleWidth = 120,
    Rcpp::Nullable<std::string> idColName = R_NilValue,
    Rcpp::Nullable<std::string> timeColName = R_NilValue
){
    // setup the default logger
    ESASingletonStatuses* stati = ESASingletonStatuses::GetInstance();
    if (!stati->getStatus(ESAStatusKeys::kHasShownInitialError)){
        ESALogger::logger()->warn("This package is experimental, and therefore may produce incorrect results. Use at your own risk.");
        ESALogger::logger()->warn("NHS England and the authors bear no responsibility or liability for the use of, and any consequences from, this package.");
        ESALogger::logger()->warn( "Alternative packages include 'sfpanel' in Stata, 'npsf', 'sfaR', 'frontier' in R.");
        ESALogger::logger()->warn("Please report any issues to the package maintaner.");
        stati->setStatus(ESAStatusKeys::kHasShownInitialError, true);
    }
    ESASfaModelType mT = ESAEnums::getModelTypeForNameAndDist(model, dist);
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    // hessian 
    HessianCalcMethod hessCalcMethod = ESAEnums::getHessianCalcMethod(hessianCalc);
    // ---- process data ----
    // R stores matrices in Column-Major, which matches Armadillo.
    // const_cast needed- Rcpp's begin() returns const
    const arma::vec y(const_cast<double*>(y_.begin()), y_.size(), false, true);
    // unpack counts
    if (colCounts_.size() < 6) Rcpp::stop("Expected at least 6 elements in colCounts_");
    int nX = colCounts_[0];
    int nZmuit = colCounts_[1];
    int nZuit = colCounts_[2];
    int nZvit = colCounts_[3];
    int nZui0 = colCounts_[4];
    int nZvi0 = colCounts_[5];
    // points to main matrix
    int nrows = allData_.nrow();
    double* ptrData = const_cast<double*>(allData_.begin());
    int currCol = 0;
    auto ptrX = makeView(ptrData, nrows, currCol, nX);
    currCol += nX;
    if (!ptrX) Rcpp::stop("model must have x variables!");
    // zmuit
    auto ptrZmuit = makeView(ptrData, nrows, currCol, nZmuit);
    currCol += nZmuit;
    // zuit
    auto ptrZuit = makeView(ptrData, nrows, currCol, nZuit);
    currCol += nZuit;
    // zvit
    auto ptrZvit = makeView(ptrData, nrows, currCol, nZvit);
    currCol += nZvit;
    // Zui0
    auto ptrZui0 = makeView(ptrData, nrows, currCol, nZui0);
    currCol += nZui0;
    // Zvi0
    auto ptrZvi0 = makeView(ptrData, nrows, currCol, nZvi0);
    // check for time & id vectors
    arma::Col<int> idVec, timeVec;
    bool hasIds = idVec_.isNotNull();
    if (hasIds) {
        Rcpp::IntegerVector _tmp = Rcpp::as<Rcpp::IntegerVector>(idVec_);
        idVec = arma::Col<int>(const_cast<int*>(_tmp.begin()), _tmp.size(), false, true);
    }
    bool hasTime = timeVec_.isNotNull();
    if (hasTime) {
        Rcpp::IntegerVector _tmp = Rcpp::as<Rcpp::IntegerVector>(timeVec_);
        timeVec = arma::Col<int>(const_cast<int*>(_tmp.begin()), _tmp.size(), false, true);
    }
    // copy starting values (since theyre small)
    std::optional<arma::dcolvec> startVals = std::nullopt;
    if (startVals_.isNotNull()) {
        startVals = std::make_optional<arma::dcolvec>(Rcpp::as<arma::dcolvec>(startVals_));
    }
    // ---- setup for threading ----
    esaparallel::setThreadCount(nthreads);
    esaparallel::modelChangeFlushUnneededTLS(mF);
    // ---- general settings ----
    bool shouldClusterSE = (mF == ESASfaModelFamily::CROSS) ? false : clusteredSE;
    ModelSolver mainModelSolver = ESAEnums::getModelSolverForMethodAndLib(method, methodLib);
    // set to global singleton class
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    // set default optim params
    ESAOptimParams mainOptimParams;
    // modify optimization parameters
    rinterface::setupOptimParams(mainOptimParams, optimOpts, seed);
    globalOptimParams->mainOptimParams = mainOptimParams;
    globalOptimParams->mainModelSolver = mainModelSolver;
    globalOptimParams->hessianMethod = hessCalcMethod;
    globalOptimParams->hessianNumApproxAcc = hessianCalcNumApproxAccuracy;
    globalOptimParams->optimThreaded = (nthreads > 1);
    // ---- halton settings ----
    // settings for halton draw
    HaltonSettings hsetting = interface::haltonSettingsForOpts(haltonBase, haltonBurnin, haltonUi0Base, scrambledHalton, shuffledHalton);
    // ---- setup runner ----
    ESASfaRunner runner(model, dist, prodCost, seed, nsim, printLevel);
    // load the data
    runner.loadData(
        &y,
        ptrX.get(),
        nZmuit > 0 ? ptrZmuit.get() : nullptr,
        nZuit > 0 ? ptrZuit.get() : nullptr,
        nZvit > 0 ? ptrZvit.get() : nullptr,
        nZui0 > 0 ? ptrZui0.get() : nullptr,
        nZvi0 > 0 ? ptrZvi0.get() : nullptr,
        hasIds == true ? &idVec : nullptr,
        hasTime == true ? &timeVec : nullptr
    );
    // setup the model, its terms
    std::optional<std::vector<std::string>> termsX = std::nullopt, termsZmuit = std::nullopt, termsZuit = std::nullopt, termsZvit = std::nullopt, termsZui0 = std::nullopt, termsZvi0 = std::nullopt;
    if (termsX_.isNotNull()){
        termsX = Rcpp::as<std::vector<std::string>>(termsX_);
    }
    if (termsZmuit_.isNotNull()){
        termsZmuit = Rcpp::as<std::vector<std::string>>(termsZmuit_);
    }
    if (termsZuit_.isNotNull()){
        termsZuit = Rcpp::as<std::vector<std::string>>(termsZuit_);
    }
    if (termsZvit_.isNotNull()){
        termsZvit = Rcpp::as<std::vector<std::string>>(termsZvit_);
    }
    if (termsZui0_.isNotNull()){
        termsZui0 = Rcpp::as<std::vector<std::string>>(termsZui0_);
    }
    if (termsZvi0_.isNotNull()){
        termsZvi0 = Rcpp::as<std::vector<std::string>>(termsZvi0_);
    }
    runner.setupModel(hsetting, termsX, termsZmuit, termsZuit, termsZvit, termsZui0, termsZvi0);
    ESALogger::logger()->info("current memory usage (RSS) after setup is {}", memoryusage::get_memory_usage_mb());
    // run optimization
    std::unique_ptr<ESAOptimResult> resultUnk = runner.runOptimization(
        startVals,
        hessCalcMethod,
        hessianCalcNumApproxAccuracy
    );
    ESALogger::logger()->info("current memory usage (RSS) post optimization is {}", memoryusage::get_memory_usage_mb());
    // list to store elements for R
    Rcpp::List res;
    // check the result
    if (resultUnk == nullptr) {
        ESALogger::logger()->error("Unexpected internal error, nothing was returned from optimization");
    } else if (resultUnk->getDidConverge()) {
        // succesfully converged
        // instance of ESAOptimResultSuccess
        ESAOptimResultSuccess& optimRes = (ESAOptimResultSuccess&)*resultUnk;
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
        // ---- clustered SE if defined ----
        arma::dmat vcovSummary;
        if (shouldClusterSE) {
            arma::Col<int> empty_ivec;
            vcovSummary = sandwich::clusteredVcov(vcov, jac, hasIds == true ? idVec : empty_ivec);
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
        // whether or not to calculate efficiency scores
        if (calculateEfficiencyScores) {
            ESALogger::logger()->info("current memory usage (RSS) before efficiency scores is {}", memoryusage::get_memory_usage_mb());
            std::unique_ptr<ESASfaEffScores> effs = runner.estimateEfficiencyScores(
                coefs,
                ghkSimReps, // number of ghk simulations
                0 // start position for prime numbers, used for halton bases
            );
            if (effs) {
                if ((*effs).transient) {
                    arma::dmat effTransient = (*effs).transient.value();
                    // add sanity check
                    if (effTransient.n_elem > 0) {
                        res["efficiencyTransient"] = effTransient;
                    } else {
                        ESALogger::logger()->error("Transient efficiency matrix is empty");
                    }
                }
                if ((*effs).persistent) {
                    arma::dmat effPersistent = (*effs).persistent.value();
                    if (effPersistent.n_elem > 0) {
                        res["efficiencyPersistent"] = effPersistent;
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
            std::optional<std::string> idColUse = std::nullopt, timeColUse = std::nullopt;
            if (idColName.isNotNull()) idColUse = std::make_optional(Rcpp::as<std::string>(idColName));
            if (timeColName.isNotNull()) timeColUse = std::make_optional(Rcpp::as<std::string>(timeColName));

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
                idColUse,
                timeColUse
            );
        }
        // ---- fill R return list ----
        res["par"] = coefs;
        res["vars"] = allTerms;
        res["logLikelihood"] = llscore;
        res["degreesFreedom"] = dof;
        res["nobs"] = nobs;
        res["maxT"] = maxT;
        res["minT"] = minT;
        res["modelSummary"] = msummary;
        res["vcov"] = vcovSummary;
        res["hessian"] = hess;
        res["gradient"] = grad;
        res["jacobian"] = jac;
        res["nparam"] = nparam;
        res["gnorm"] = gnorm;
        res["clusteredSE"] = shouldClusterSE;
        // n firms
        if (mF != ESASfaModelFamily::CROSS) {
            int nfirms = optimRes.getN();
            res["nfirm"] = nfirms;
        }
        // extract all the parameters
        Rcpp::NumericVector v = Rcpp::NumericVector::create(
            Rcpp::Named("sigma_uit", sigma_uit),
            Rcpp::Named("sigma_vit", sigma_vit),
            Rcpp::Named("sigma_vi0", sigma_vi0),
            Rcpp::Named("sigma_ui0", sigma_ui0),
            Rcpp::Named("lambda", lambda),
            Rcpp::Named("lambda_0", lambda_0),
            Rcpp::Named("BigLambda", BigLambda)
        );
        // assign numeric vector to result
        res["sigmas"] = v;
        // ---- calculate marginal effects if desired ----
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
                res["marginalEffects"] = meffs;
                res["marginalEffectsNames"] = meffCnames;
            }
            if (estimateMargEffCI && meffCIs == nullptr) {
                ESALogger::logger()->error("Calculation of confidence intervals for marginal effects failed");
            } else if (estimateMargEffCI) {
                arma::dmat meffLower = meffCIs->lowerCI;
                arma::dmat meffUpper = meffCIs->upperCI;
                std::vector<std::string> meffCols = meffCIs->columnNames;
                res["marginalEffectsLwrCI"] = meffLower;
                res["marginalEffectsUprCI"] = meffUpper;
                res["marginalEffectsCICols"] = meffCols;
            }
            ESALogger::logger()->info("current memory usage (RSS) after marginal effects is {}", memoryusage::get_memory_usage_mb());
            meff.reset();
            meffCIs.reset();
    } else {
        // failed to converge
        // instance of ESAOptimResultFailed
        ESAOptimResultFailed& optimRes = (ESAOptimResultFailed&)*resultUnk;
        res["par"] = optimRes.getX();
        res["vars"] = runner.modelTerms->allTerms();
        ESALogger::logger()->warn("Optimization failed. Only returning parameters");
    }
    // set to sfacpp class
    res.attr("class") = "sfacpp";
    return res;
}

// [[Rcpp::export]]
Rcpp::List sfacpp_internal_searches(
    const Rcpp::NumericVector& y_,
    const Rcpp::NumericMatrix& allData_,
    const Rcpp::IntegerVector& colCounts_,
    Rcpp::Nullable<Rcpp::IntegerVector> idVec_ = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> timeVec_ = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> startVals_ = R_NilValue,
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
    const unsigned int nsim = 100,
    const int seed = 1234,
    const unsigned int printLevel = 0,
    const int nthreads = 20,
    const int haltonBase = 2,
    const int haltonBurnin = 1000,
    const int haltonUi0Base = 3,
    const bool scrambledHalton = true,
    const bool shuffledHalton = false,
    const int displayConsoleWidth = 120,
    const int displayDecimalPlaces = 4
)
{
    ESASfaModelType mT = ESAEnums::getModelTypeForNameAndDist(model, dist);
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    // hessian 
    HessianCalcMethod hessCalcMethod = HessianCalcMethod::ANALYTICAL;
    // ---- process data ----
    // R stores matrices in Column-Major, which matches Armadillo.
    // const_cast needed- Rcpp's begin() returns const
    const arma::vec y(const_cast<double*>(y_.begin()), y_.size(), false, true);
    // unpack counts
    if (colCounts_.size() < 6) Rcpp::stop("Expected at least 6 elements in colCounts_");
    int nX = colCounts_[0];
    int nZmuit = colCounts_[1];
    int nZuit = colCounts_[2];
    int nZvit = colCounts_[3];
    int nZui0 = colCounts_[4];
    int nZvi0 = colCounts_[5];
    // points to main matrix
    int nrows = allData_.nrow();
    double* ptrData = const_cast<double*>(allData_.begin());
    int currCol = 0;
    auto ptrX = makeView(ptrData, nrows, currCol, nX);
    currCol += nX;
    if (!ptrX) Rcpp::stop("model must have x variables!");
    // zmuit
    auto ptrZmuit = makeView(ptrData, nrows, currCol, nZmuit);
    currCol += nZmuit;
    // zuit
    auto ptrZuit = makeView(ptrData, nrows, currCol, nZuit);
    currCol += nZuit;
    // zvit
    auto ptrZvit = makeView(ptrData, nrows, currCol, nZvit);
    currCol += nZvit;
    // Zui0
    auto ptrZui0 = makeView(ptrData, nrows, currCol, nZui0);
    currCol += nZui0;
    // Zvi0
    auto ptrZvi0 = makeView(ptrData, nrows, currCol, nZvi0);
    // check for time & id vectors
    arma::Col<int> idVec, timeVec;
    bool hasIds = idVec_.isNotNull();
    if (hasIds) {
        Rcpp::IntegerVector _tmp = Rcpp::as<Rcpp::IntegerVector>(idVec_);
        idVec = arma::Col<int>(const_cast<int*>(_tmp.begin()), _tmp.size(), false, true);
    }
    bool hasTime = timeVec_.isNotNull();
    if (hasTime) {
        Rcpp::IntegerVector _tmp = Rcpp::as<Rcpp::IntegerVector>(timeVec_);
        timeVec = arma::Col<int>(const_cast<int*>(_tmp.begin()), _tmp.size(), false, true);
    }
    // copy starting values (since theyre small)
    std::optional<arma::dcolvec> startVals = std::nullopt;
    if (startVals_.isNotNull()) {
        startVals = std::make_optional<arma::dcolvec>(Rcpp::as<arma::dcolvec>(startVals_));
    }
    // ---- setup for threading ----
    esaparallel::setThreadCount(nthreads);
    esaparallel::modelChangeFlushUnneededTLS(mF);
    // ---- general settings ----
    ModelSolver mainModelSolver = ESAEnums::getModelSolverForMethodAndLib(method, methodLib);
    // set to global singleton class
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    // modify optimization parameters
    ESAOptimParams mainOptimParams;
    mainOptimParams.maxit = maxRepIter;
    mainOptimParams.seed = seed;
    globalOptimParams->mainOptimParams = mainOptimParams;
    globalOptimParams->mainModelSolver = mainModelSolver;
    globalOptimParams->hessianMethod = hessCalcMethod;
    globalOptimParams->hessianNumApproxAcc = 0;
    globalOptimParams->optimThreaded = (nthreads > 1);
    // ---- halton settings ----
    // settings for halton draw
    HaltonSettings hsetting = interface::haltonSettingsForOpts(haltonBase, haltonBurnin, haltonUi0Base, scrambledHalton, shuffledHalton);
    // ---- setup runner ----
    ESASfaRunner runner(model, dist, prodCost, seed, nsim, printLevel);
    // load the data
    runner.loadData(
        &y,
        ptrX.get(),
        nZmuit > 0 ? ptrZmuit.get() : nullptr,
        nZuit > 0 ? ptrZuit.get() : nullptr,
        nZvit > 0 ? ptrZvit.get() : nullptr,
        nZui0 > 0 ? ptrZui0.get() : nullptr,
        nZvi0 > 0 ? ptrZvi0.get() : nullptr,
        hasIds == true ? &idVec : nullptr,
        hasTime == true ? &timeVec : nullptr
    );
    // setup the model, its terms
    std::optional<std::vector<std::string>> termsX = std::nullopt, termsZmuit = std::nullopt, termsZuit = std::nullopt, termsZvit = std::nullopt, termsZui0 = std::nullopt, termsZvi0 = std::nullopt;
    runner.setupModel(hsetting, termsX, termsZmuit, termsZuit, termsZvit, termsZui0, termsZvi0);
    // 
    int k = runner.dataObjPtr->nParams();
    arma::dcolvec theta0;
    if (startVals.has_value()){
        // be a bit more strict about how big the vector passed into R is
        if (startVals.value().n_rows != k) {
            throw std::invalid_argument(
                "User provided starting arguments, but of incorrect dimensions, got " + 
                std::to_string(startVals.value().n_rows) + " but expected " + std::to_string(k)
            );
        }
        theta0 = startVals.value();
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
    // iterate thru the number of repetitions
    for (int i = 0; i < reps; i++) {
        ESALogger::logger()->info("---- Trial Number {}/{} ----", i+1, reps);
        // start the timer
        auto tStart = std::chrono::high_resolution_clock::now();
        // create pertubated starting values
        arma::dcolvec theta0Rep(theta0.n_rows);
        bool couldFindStart = esautils::findValidStartValues(
            runner.dataObjPtr,
            runner.modelObjPtr,
            theta0,
            theta0Rep,
            i,
            slengthFrontier,
            slengthSigmas,
            seed,
            maxStartValFindAttempt,
            (printLevel > 2)
        );
        // if couldnt find any good values, bin off this loop iteration
        if (!couldFindStart) {
            ESALogger::logger()->info("Could not find appropriate starting values.");
            continue;
        }
        // run the optimization
        std::unique_ptr<ESAOptimResult> resultUnk = runner.runOptimization(theta0Rep, hessCalcMethod, 0);
        // check again for a user interrupt
        if (globalstate::InterruptRequested.load(std::memory_order_relaxed)) {
            Rcpp::checkUserInterrupt();
        }
        // check whether converged or not
        if (resultUnk == nullptr){
            // fill with NaNs
            allCoefs.col(i).fill(std::numeric_limits<double>::quiet_NaN());
        } else if (resultUnk->getDidConverge()) {
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
        } else {
            // model failed to converge
            // might as well still fill in the coefficients
            ESAOptimResultFailed& optimRes = (ESAOptimResultFailed&)*resultUnk;
            allCoefs.col(i) = optimRes.getX();
        }
        // stop the timer
        auto tEnd = std::chrono::high_resolution_clock::now();
        // calculate the difference in mins
        double min_diff = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count() / 1000.0 / 60.0;
        totalMinsTaken += min_diff;
        if ((i % 10 == 0) && i > 0) {
            // print elapsed time
            interface::printSearchesElapsedAndETA(ESALogger::logger(), totalMinsTaken, i, reps);
        }
    }
    // print searches (top 15)
    interface::printSearches(lls, allCoefs, 15, displayConsoleWidth, displayDecimalPlaces);
    Rcpp::List out;
    out["ll_scores"] = lls;
    out["all_coefs"] = allCoefs;
    out["maxLL"] = maxLL;
    out["maxLLpos"] = maxLLpos + 1; // remember R is 1-indexed
    if (maxLLpos >= 0) {
        arma::dcolvec bestCoef(allCoefs.col(maxLLpos));
        out["maxLLCoefVec"] = bestCoef;
    }
    return out;
}

// [[Rcpp::export]]
Rcpp::List sfacpp_internal_lcm(
    const Rcpp::NumericVector& y_,
    const Rcpp::NumericMatrix& allData_,
    const Rcpp::NumericMatrix& segData_,
    const Rcpp::IntegerVector& colCounts_,
    const Rcpp::IntegerVector& idVec_,
    const Rcpp::IntegerVector& timeVec_,
    const int nClasses = 2,
    Rcpp::Nullable<Rcpp::NumericVector> startVals_ = R_NilValue,
    const int prodCost = 1,
    const std::string& model = "lctre",
    const std::string& dist = "hnorm",
    const std::string& method = "pso_tr",
    const std::string& methodLib = "dlib",
    Rcpp::Nullable<Rcpp::List> optimOpts = R_NilValue,
    Rcpp::Nullable<std::vector<std::string>> termsX_ = R_NilValue,
    Rcpp::Nullable<std::vector<std::string>> termsZmuit_ = R_NilValue,
    Rcpp::Nullable<std::vector<std::string>> termsZuit_ = R_NilValue,
    Rcpp::Nullable<std::vector<std::string>> termsZvit_ = R_NilValue,
    Rcpp::Nullable<std::vector<std::string>> termsZvi0_ = R_NilValue,
    Rcpp::Nullable<std::vector<std::string>> termsSeg_ = R_NilValue,
    const unsigned int nsim = 100,
    const std::string& hessianCalc = "bhhh",
    const int seed = 1234,
    const double confidenceLevel = 0.95,
    const unsigned int printLevel = 2,
    const bool clusteredSE = true,
    const int haltonBase = 2,
    const int haltonBurnin = 1000,
    const int haltonUi0Base = 3,
    const bool scrambledHalton = true,
    const bool shuffledHalton = false,
    const int nthreads = 20,
    const int displayDecimalPlaces = 5,
    const int displayConsoleWidth = 120,
    Rcpp::Nullable<std::string> idColName = R_NilValue,
    Rcpp::Nullable<std::string> timeColName = R_NilValue
){
    ESASfaModelType mT = ESAEnums::getModelTypeForNameAndDist(model, dist);
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    if (mF != ESASfaModelFamily::LC_TRE) {
        Rcpp::stop("sfacpp_internal_lcm only supports LC-TRE models (model='lctre')");
    }
    HessianCalcMethod hessCalcMethod = ESAEnums::getHessianCalcMethod(hessianCalc);
    // ---- process data ----
    const arma::vec y(const_cast<double*>(y_.begin()), y_.size(), false, true);
    // unpack column counts: nX, nZmuit, nZuit, nZvit, nZvi0
    if (colCounts_.size() < 5) Rcpp::stop("Expected at least 5 elements in colCounts_ (nX, nZmuit, nZuit, nZvit, nZvi0)");
    int nX = colCounts_[0];
    int nZmuit = colCounts_[1];
    int nZuit = colCounts_[2];
    int nZvit = colCounts_[3];
    int nZvi0 = colCounts_[4];
    // allData matrix views
    int nrows = allData_.nrow();
    double* ptrData = const_cast<double*>(allData_.begin());
    int currCol = 0;
    auto ptrX = makeView(ptrData, nrows, currCol, nX);
    currCol += nX;
    if (!ptrX) Rcpp::stop("model must have x variables!");
    auto ptrZmuit = makeView(ptrData, nrows, currCol, nZmuit);
    currCol += nZmuit;
    auto ptrZuit = makeView(ptrData, nrows, currCol, nZuit);
    currCol += nZuit;
    auto ptrZvit = makeView(ptrData, nrows, currCol, nZvit);
    currCol += nZvit;
    auto ptrZvi0 = makeView(ptrData, nrows, currCol, nZvi0);
    // segmentation matrix
    const arma::dmat segMat(const_cast<double*>(segData_.begin()), segData_.nrow(), segData_.ncol(), false, true);
    // id and time vectors (required for LC panel models)
    arma::Col<int> idVec(const_cast<int*>(idVec_.begin()), idVec_.size(), false, true);
    arma::Col<int> timeVec(const_cast<int*>(timeVec_.begin()), timeVec_.size(), false, true);
    // starting values
    std::optional<arma::dcolvec> startVals = std::nullopt;
    if (startVals_.isNotNull()) {
        startVals = std::make_optional<arma::dcolvec>(Rcpp::as<arma::dcolvec>(startVals_));
    }
    // ---- threading ----
    esaparallel::setThreadCount(nthreads);
    esaparallel::modelChangeFlushUnneededTLS(mF);
    // ---- optimization settings ----
    ModelSolver mainModelSolver = ESAEnums::getModelSolverForMethodAndLib(method, methodLib);
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    ESAOptimParams mainOptimParams;
    rinterface::setupOptimParams(mainOptimParams, optimOpts, seed);
    globalOptimParams->mainOptimParams = mainOptimParams;
    globalOptimParams->mainModelSolver = mainModelSolver;
    globalOptimParams->hessianMethod = hessCalcMethod;
    globalOptimParams->hessianNumApproxAcc = 0;
    globalOptimParams->optimThreaded = (nthreads > 1);
    // ---- halton settings ----
    HaltonSettings hsetting = interface::haltonSettingsForOpts(haltonBase, haltonBurnin, haltonUi0Base, scrambledHalton, shuffledHalton);
    // ---- setup runner ----
    ESASfaRunner runner(model, dist, prodCost, seed, nsim, printLevel);
    runner.loadDataLCM(
        &y,
        ptrX.get(),
        &segMat,
        nZmuit > 0 ? ptrZmuit.get() : nullptr,
        nZuit > 0 ? ptrZuit.get() : nullptr,
        nZvit > 0 ? ptrZvit.get() : nullptr,
        nZvi0 > 0 ? ptrZvi0.get() : nullptr,
        &idVec,
        &timeVec,
        static_cast<unsigned int>(nClasses)
    );
    // setup model terms
    std::optional<std::vector<std::string>> termsX = std::nullopt, termsZmuit = std::nullopt,
        termsZuit = std::nullopt, termsZvit = std::nullopt, termsZvi0 = std::nullopt;
    if (termsX_.isNotNull()) termsX = Rcpp::as<std::vector<std::string>>(termsX_);
    if (termsZmuit_.isNotNull()) termsZmuit = Rcpp::as<std::vector<std::string>>(termsZmuit_);
    if (termsZuit_.isNotNull()) termsZuit = Rcpp::as<std::vector<std::string>>(termsZuit_);
    if (termsZvit_.isNotNull()) termsZvit = Rcpp::as<std::vector<std::string>>(termsZvit_);
    if (termsZvi0_.isNotNull()) termsZvi0 = Rcpp::as<std::vector<std::string>>(termsZvi0_);
    runner.setupModel(hsetting, termsX, termsZmuit, termsZuit, termsZvit, std::nullopt, termsZvi0);
    ESALogger::logger()->info("current memory usage (RSS) after setup is {}", memoryusage::get_memory_usage_mb());
    // ---- run optimization ----
    std::unique_ptr<ESAOptimResult> resultUnk = runner.runOptimization(
        startVals, hessCalcMethod, 0
    );
    ESALogger::logger()->info("current memory usage (RSS) post optimization is {}", memoryusage::get_memory_usage_mb());
    // ---- build result ----
    Rcpp::List res;
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
        int nparam = runner.lcmDataObjPtr ?
            static_cast<int>(runner.lcmDataObjPtr->nParams())
            : 0;
        double gnorm = optimRes.getGnorm();
        int nfirms = optimRes.getN();
        // clustered SE
        arma::dmat vcovSummary;
        if (clusteredSE) {
            vcovSummary = sandwich::clusteredVcov(vcov, jac, idVec);
        } else {
            vcovSummary = vcov;
        }
        // fill results
        res["par"] = coefs;
        res["logLikelihood"] = llscore;
        res["nobs"] = nobs;
        res["nparam"] = static_cast<int>(coefs.n_rows);
        res["nClasses"] = nClasses;
        res["nfirm"] = nfirms;
        res["vcov"] = vcovSummary;
        res["hessian"] = hess;
        res["gradient"] = grad;
        res["jacobian"] = jac;
        res["gnorm"] = gnorm;
        res["clusteredSE"] = clusteredSE;
        res["converged"] = true;
    } else {
        ESAOptimResultFailed& optimRes = (ESAOptimResultFailed&)*resultUnk;
        res["par"] = optimRes.getX();
        res["converged"] = false;
        ESALogger::logger()->warn("Optimization failed. Only returning parameters");
    }
    res.attr("class") = "sfacpp_lcm";
    return res;
}

// [[Rcpp::export]]
Rcpp::List sfacpp_internal_lrtest(const double ll0, const double ll1, const int nparam0, const int nparam1, const double dp = 5)
{
    Rcpp::List res;
    postestimation::LogLikeRatioTest test = postestimation::logLikelihoodRatioTest(ll0, ll1, nparam0, nparam1);
    interface::printLrTest(test, dp);
    res["statistic"] = test.stat;
    res["p_value"] = test.pval;
    res["dof"] = test.df;
    return res;
}

// [[Rcpp::export]]
void sfacpp_internal_call_print_model_summary(
    const std::string& model,
    const std::string& dist,
    const int nobs,
    const int nids,
    const int maxT,
    const int minT, 
    const Rcpp::NumericMatrix& modelSummaryIn,
    const std::vector<std::string>& modelSummaryTerms,
    Rcpp::Nullable<Rcpp::List> sigmaParams,
    const int nsim,
    const double llscore,
    const double gnorm,
    const bool clusteredSE,
    const int haltonBase,
    const int haltonBurnin,
    const int haltonUi0Base,
    const bool scrambledHalton,
    const bool shuffledHalton,
    const double confInt,
    const int decimalPlaces,
    const int consoleWidth,
    Rcpp::Nullable<std::string> idColName = R_NilValue,
    Rcpp::Nullable<std::string> timeColName = R_NilValue
)
{
    // reconstruct the model type
    ESASfaModelType mT = ESAEnums::getModelTypeForNameAndDist(model, dist);
    if (mT == ESASfaModelType::MODEL_UNKNOWN) {
        throw std::invalid_argument("Not expecting an unknown model");
    }
    // create the armadillo matrix from the 
    const arma::dmat modelSummary(
        const_cast<double*>(modelSummaryIn.begin()),
        modelSummaryIn.nrow(),
        modelSummaryIn.ncol(),
        false,
        true
    );
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

    std::optional<double> s_uit = std::nullopt, s_vit = std::nullopt, s_vi0 = std::nullopt, lambda = std::nullopt;
    std::optional<double> s_ui0 = std::nullopt, lambda_0 = std::nullopt, BigLambda = std::nullopt;
    if (sigmaParams.isNotNull()) {
        Rcpp::List sp = Rcpp::as<Rcpp::List>(sigmaParams);
        if (sp.containsElementNamed("sigma_uit")) s_uit = std::make_optional(sp["sigma_uit"]);
        if (sp.containsElementNamed("sigma_vit")) s_vit = std::make_optional(sp["sigma_vit"]);
        if (sp.containsElementNamed("sigma_vi0")) s_vi0 = std::make_optional(sp["sigma_vi0"]);
        if (sp.containsElementNamed("sigma_ui0")) s_ui0 = std::make_optional(sp["sigma_ui0"]);
        if (sp.containsElementNamed("lambda")) lambda = std::make_optional(sp["lambda"]);
        if (sp.containsElementNamed("lambda_0")) lambda_0 = std::make_optional(sp["lambda_0"]);
        if (sp.containsElementNamed("BigLambda")) BigLambda = std::make_optional(sp["BigLambda"]);
    }
    // reconstruct the sigma struct
    ESASigmaParams sigParams = interface::reconstructSigmaParams(
        s_uit, s_vit, s_vi0, lambda, s_ui0, lambda_0, BigLambda
    );
    // reconstruct the halton setting struct
    HaltonSettings hsetting = interface::haltonSettingsForOpts(
        haltonBase, haltonBurnin, haltonUi0Base, scrambledHalton, shuffledHalton
    );

    std::optional<std::string> idColUse = std::nullopt, timeColUse = std::nullopt;
    if (idColName.isNotNull()) idColUse = std::make_optional(Rcpp::as<std::string>(idColName));
    if (timeColName.isNotNull()) timeColUse = std::make_optional(Rcpp::as<std::string>(timeColName));
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
        idColUse,
        timeColUse
    );
}

#endif 
