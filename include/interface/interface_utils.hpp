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

#ifndef INTERFACE_UTILS_HPP
#define INTERFACE_UTILS_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <tuple>
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
#include <spdlog/spdlog.h>
#include "data/ESADataBase.hpp"
#include "utils/enums.hpp"
#include "sfa/HaltonSettings.hpp"
#include "sfa/ESASigmaParams.hpp"
#include "math/llratiotest.hpp"

namespace interface {
    /**
     * @brief Instansiate an instance of ESADataBase (or it's subclasses), depending on the model type
     * @param y pointer to column vector of the response
     * @param x pointer to matrix containing the independent variables (frontier)
     * @param zmuit pointer to the matrix containing the determinants of the mean of inefficiency (truncated normal only)
     * @param zuit pointer to the matrix containing the determinants of inefficiency (time-varying)
     * @param zvit pointer to the matrix containing the determinants of time-varying stochastic noise
     * @param zui0 pointer to the matrix containing the determinants of the time-invariant inefficiency (GTRE only)
     * @param zvi0 pointer to the matrix containing the determinants of the time-invariant firm effects (TRE & GTRE only)
     * @param idVec pointer to the column vector identifying firms (all panel models)
     * @param timeVec pointer to the column vector identifying time periods (all panel data models)
     * @param mT The model type
     * @return shared_ptr to the instansiated data object
     */
    std::shared_ptr<ESADataBase> createDataObject(
        const arma::dcolvec* y,
        const arma::dmat* x,
        const arma::dmat* zmuit,
        const arma::dmat* zuit,
        const arma::dmat* zvit,
        const arma::dmat* zui0,
        const arma::dmat* zvi0,
        const arma::Col<int>* idVec,
        const arma::Col<int>* timeVec,
        const ESASfaModelType& mT
    );
    
    /**
     * @brief Reduced-form function to print the model summary/model output
     * @param mT The model type
     * @param modelSummary The armadillo matrix (nParam x 6) containing estimate, std err, pval & CIs etc
     * @param modelSummaryTerms A vector of strings representing the parameter names, nParam in length
     * @param sigParams An instance of ESASigmaParams, containing calculated sigma/lambda values
     * @param nsim The number of simulations//halton draws - GTRE/TRE only
     * @param llscore The log-likelihood score of the model
     * @param gnorm The norm of the gradient
     * @param clusteredSE Whether or not clustered standard errors were calculated for the variance-covariance matrix
     * @param hsetting An instance of HaltonSettings, containing parameters use for conducting halton draws
     * @param confInt The confidence level used to estimate the confidence intervals in modelSummary
     * @param decimalPlaces The number of decimal places to display
     * @param consoleWidth The width of the console
     * @param idColName Optionally, the name of the column identifying firms
     * @param timeColName Optionally, the name of the column identifying time dimension
     */
    void printModelOutput(
        const std::shared_ptr<ESADataBase>& dataObjPtr,
        const arma::dmat& modelSummary,
        const std::vector<std::string>& modelSummaryTerms,
        const ESASigmaParams& sigParams,
        const int nsim,
        const double llscore,
        const double gnorm,
        const bool clusteredSE,
        const HaltonSettings& hsetting,
        const double confInt,
        const int decimalPlaces,
        const int consoleWidth,
        const std::optional<std::string>& idColName,
        const std::optional<std::string>& timeColName
    );

    /**
     * @brief Main function to print the model summary/model output
     * @param mT The model type
     * @param nobs The number of observations
     * @param nids The number of ids (e.g., firms), for panel data only
     * @param maxT The maximum number of panels for a firm, for panel data only
     * @param minT The minimum number of panels for a firm, for panel data only
     * @param modelSummary The armadillo matrix (nParam x 6) containing estimate, std err, pval & CIs etc
     * @param modelSummaryTerms A vector of strings representing the parameter names, nParam in length
     * @param sigParams An instance of ESASigmaParams, containing calculated sigma/lambda values
     * @param nsim The number of simulations//halton draws - GTRE/TRE only
     * @param llscore The log-likelihood score of the model
     * @param gnorm The norm of the gradient
     * @param clusteredSE Whether or not clustered standard errors were calculated for the variance-covariance matrix
     * @param hsetting An instance of HaltonSettings, containing parameters use for conducting halton draws
     * @param confInt The confidence level used to estimate the confidence intervals in modelSummary
     * @param decimalPlaces The number of decimal places to display
     * @param consoleWidth The width of the console
     * @param idColName Optionally, the name of the column identifying firms
     * @param timeColName Optionally, the name of the column identifying time dimension
     */
    void printModelOutput(
        const ESASfaModelType mT,
        const int nobs,
        const int nids,
        const int maxT,
        const int minT,
        const arma::dmat& modelSummary,
        const std::vector<std::string>& modelSummaryTerms,
        const ESASigmaParams& sigParams,
        const int nsim,
        const double llscore,
        const double gnorm,
        const bool clusteredSE,
        const HaltonSettings& hsetting,
        const double confInt,
        const int decimalPlaces,
        const int consoleWidth,
        const std::optional<std::string>& idColName,
        const std::optional<std::string>& timeColName
    );

    /**
     * @brief Print the full LC-TRE output: header block, segmentation params table,
     *        then one table per latent class with per-class sigma extras.
     *
     * Terms are expected to use the naming convention:
     *   - Segmentation params: "alpha_C:name"  (no "Class_" prefix)
     *   - Per-class params:    "Class_C:name"  (C is the zero-based class index)
     *
     * Within each class table the "Class_C:" prefix is stripped so that the standard
     * Zuit_/Zvit_/Zui0_/Zvi0_ divider logic fires automatically.
     *
     * @param nobs             Total number of observations
     * @param nids             Number of groups / firms
     * @param maxT             Maximum number of time periods for any group
     * @param minT             Minimum number of time periods for any group
     * @param modelSummary     nParam x 6 matrix: estimate, se, z, p, ci_lo, ci_hi
     * @param modelSummaryTerms Parameter names (nParam in length)
     * @param sigmasPerClass   Per-class sigma/lambda name->value maps (indexed by class)
     * @param nClasses         Number of latent classes
     * @param nsim             Number of simulation draws / quadrature points used
     * @param llscore          Log-likelihood value
     * @param gnorm            Gradient norm at convergence
     * @param clusteredSE      Whether clustered SEs were used
     * @param hsetting         Halton draw settings (ignored when useGhq is true)
     * @param useGhq           True when using Gauss-Hermite quadrature (em_ghq), false for Halton
     * @param confInt          Confidence level for CI columns (e.g. 0.95)
     * @param decimalPlaces    Number of decimal places to display
     * @param consoleWidth     Console width for layout
     * @param idColName        Optional name of the id/group column
     * @param timeColName      Optional name of the time column
     */
    void printLcmOutput(
        const int nobs,
        const int nids,
        const int maxT,
        const int minT,
        const arma::dmat& modelSummary,
        const std::vector<std::string>& modelSummaryTerms,
        const std::vector<std::map<std::string, double>>& sigmasPerClass,
        const int nClasses,
        const int nsim,
        const double llscore,
        const double gnorm,
        const bool clusteredSE,
        const HaltonSettings& hsetting,
        const bool useGhq,
        const double confInt,
        const int decimalPlaces,
        const int consoleWidth,
        const std::optional<std::string>& idColName,
        const std::optional<std::string>& timeColName
    );

    /**
     * @brief Print only the regression table (no header block) for a subset of parameters.
     * Used by LC-TRE display to print one table per class after a shared header.
     * @param label A label printed above the table (e.g. "Class 0")
     * @param modelSummary nParam x 6 matrix: estimate, se, z, p, ci_lo, ci_hi
     * @param modelSummaryTerms Parameter names (nParam in length); Zuit_/Zvit_/Zui0_/Zvi0_ prefixes trigger dividers
     * @param extraParams Optional name->value map printed below the table (e.g. sigmas)
     * @param confInt Confidence level used for CI columns
     * @param decimalPlaces Number of decimal places
     * @param consoleWidth Width of the console
     */
    void printModelTable(
        const std::string& label,
        const arma::dmat& modelSummary,
        const std::vector<std::string>& modelSummaryTerms,
        const std::map<std::string, double>& extraParams,
        const double confInt,
        const int decimalPlaces,
        const int consoleWidth
    );

    /**
     * @brief Print a log-likelihood ratio test to the raw log
     * @param lrTest an instance of LogLikeRatioTest
     * @param dp The number of decimal places to display
     */
    void printLrTest(const postestimation::LogLikeRatioTest& lrTest, const int dp);

    /**
     * @brief Construct a HaltonSettings struct from some options
     * @param haltonBase The halton base to use for vi0 draw
     * @param haltonBurnin How many elements to drop at the start
     * @param haltionUi0Base The (prime) halton base to use for ui0 draw
     * @param scrambledHalton Whether or not to scramble the halton sequence
     * @param shuffledHalton Whether or not to shuffle the halton sequence
     * @return HaltonSettings struct
     */
    HaltonSettings haltonSettingsForOpts(
        const int haltonBase = 2,
        const int haltonBurnin = 1000,
        const int haltonUi0Base = 3,
        const bool scrambledHalton = true,
        const bool shuffledHalton = true
    );

    /**
     * @brief Reconstruct a ESASigmaParams struct
     * @param s_uit The sigma_uit value
     * @param s_vit The sigma_vit value
     * @param s_vi0 The sigma_vi0 value
     * @param lambda The lambda value
     * @param s_ui0 The sigma_ui0 value
     * @param lambda_0 The lambda_0 value
     * @param BigLambda The BigLambda value
     * @return ESASigmaParams struct
     */
    ESASigmaParams reconstructSigmaParams(
        const std::optional<double> s_uit,
        const std::optional<double> s_vit,
        const std::optional<double> s_vi0,
        const std::optional<double> lambda,
        const std::optional<double> s_ui0,
        const std::optional<double> lambda_0,
        const std::optional<double> BigLambda
    );

    /**
     * @brief Given a column vector of loglikehood scores, filter for valid, and sort descendingly
     * @param llScores A column vector of loglikelihood scores
     * @return tuple containing the (filtered) sorted loglikelihood scores, and how they map to the original index
     */
    std::tuple<arma::dcolvec, arma::uvec> sortMapLogLikeSearches(const arma::dcolvec& llScores);

    /**
     * @brief Print a table summary of the top searches, to the raw log
     * @param logLikeScores A column vector of log likelihood scores
     * @param params A matrix (nparam x nsearches) of estimated parameters
     * @param topN Number of top rows to print (at most)
     * @param consoleWidth The width of the console
     * @param decimalPlaces The number of decimal places to use
     */
    void printSearches(
        const arma::dcolvec& logLikeScores,
        const arma::dmat& params,
        const int topN = 15,
        const int consoleWidth = 120,
        const int decimalPlaces = 3
    );

    /**
     * @brief Print the elapsed time for searches, and how estimated to go
     * @param lgr the logger to print to
     * @param totalMinsTaken Total duration taken, in minutes
     * @param i the current loop iteration
     * @param reps the total number of repetitions
     */
    void printSearchesElapsedAndETA(
        const std::shared_ptr<spdlog::logger>& lgr,
        const double& totalMinsTaken,
        const int& i,
        const int& reps
    );
}

#endif // INTERFACE_UTILS_HPP