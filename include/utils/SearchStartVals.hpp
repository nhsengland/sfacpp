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
 * @file SearchStartVals.hpp
 * @author edmund haacke
 * @date 2025-12-28
 * @details 
 */

#ifndef SEARCH_START_VALS_HPP
#define SEARCH_START_VALS_HPP

#include <memory>
#include <random>
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
#include "data/ESADataBase.hpp"
#include "data/ESADataPanel.hpp"
#include "data/ESADataLCM.hpp"
#include "sfa/ESASfaBase.hpp"
#include "sfa/ESASfaTreGreene.hpp"
#include "sfa/ESASfaGtreBad.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"


namespace esautils {

    /**
     * @brief helper function to check whether parameterized sigma term is in bounds
     * @tparam subview or dmat
     * @param data The matrix / view of a matrix holding the raw data
     * @param coef The coefficients to multiply data by
     * @param clampLwr the lower allowable bound (exp(-10) = 0.00004539992976)
     * @param clampUpr the upper allowable bound (exp(10) = 22,026.4657948067)
     * @return whether or not in bounds (true if it is)
     */
    template <typename T>
    inline bool isSigma2InBounds(
        const std::optional<T>& data,
        const arma::dcolvec& coef,
        const double clampLwr = -10.0,
        const double clampUpr = 10.0
    )
    {
        // multiplication
        if (!data.has_value()) return true;
        arma::dcolvec s = data.value() * coef;
        double max = s.max();
        double min = s.min();
        return (max < clampUpr && min > clampLwr);
    }

    /**
     * @brief helper function to find some valid starting values
     * @param dataObjPtr shared pointer to data object
     * @param sfaObjPtr shared pointer to model object
     */
    inline bool findValidStartValues(
        const std::shared_ptr<ESADataBase> dataObjPtr,
        const std::shared_ptr<ESASfaBase> sfaObjPtr,
        const arma::dcolvec& baseStart,
        arma::dcolvec& startOut,
        const int currentIter,
        const double slengthFrontier = 2.0,
        const double slengthSigmas = 0.8,
        const int baseSeed = 1234,
        const int maxAttempts = 100,
        const bool shouldLog = false
    ) {
        // check data is panel [don't support anything else]
        if (!dynamic_cast<ESADataPanel*>(dataObjPtr.get())) {
            throw std::invalid_argument("Only panel data structure is supported");
        }
        // derefence pointer to underlying dataobject
        ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
        // 
        bool validStart = false;
        int attempts = 0;

        double slengthSigmasCurr = slengthSigmas;
        double slengthFrontierCurr = slengthFrontier;
        const double clamp_lwr = -10.0;
        const double clamp_upr = 10.0;

        double slengthSigmaUit = slengthSigmasCurr;
        double slengthSigmaVit = slengthSigmasCurr;
        double slengthSigmaUi0 = slengthSigmasCurr;
        double slengthSigmaVi0 = slengthSigmasCurr;

        while (!validStart && attempts < maxAttempts) {
            attempts++;
            // generate an initial perturbation
            // use C++ random number generator for cross compatability between R & Python
            std::mt19937 engine(baseSeed + currentIter + attempts * 999);
            std::uniform_real_distribution<double> dist(-0.5, 0.5);
            arma::dcolvec perturb(baseStart.n_rows);
            perturb.imbue([&]() { return(dist(engine)); });
            // find there the frontier terms are, and multiply those by slengthFrontier
            perturb.head(dataObj.getNX()) *= slengthFrontierCurr;
            // perturb.tail((perturb.n_rows - dataObj.getNX())) *= slengthSigmasCurr;
            // perturb each sigma component with their own scaling factor
            std::optional<std::pair<int, int>> zuitPos = dataObj.getZuitRange(baseStart);
            std::optional<std::pair<int, int>> zvitPos = dataObj.getZvitRange(baseStart);
            std::optional<std::pair<int, int>> zui0Pos = dataObj.getZui0Range(baseStart);
            std::optional<std::pair<int, int>> zvi0Pos = dataObj.getZvi0Range(baseStart);
            if (zuitPos) perturb.rows(zuitPos.value().first, zuitPos.value().second) *= slengthSigmaUit;
            if (zvitPos) perturb.rows(zvitPos.value().first, zvitPos.value().second) *= slengthSigmaVit;
            if (zui0Pos) perturb.rows(zui0Pos.value().first, zui0Pos.value().second) *= slengthSigmaUi0;
            if (zvi0Pos) perturb.rows(zvi0Pos.value().first, zvi0Pos.value().second) *= slengthSigmaVi0;
            arma::dcolvec thetaTry = baseStart + perturb;
            startOut = thetaTry;
            // for the variances; depending on whether they are parameterised, or not
            // ensure that they can't go into Inf / Zero
            bool sigmaUitFailed = false, sigmaVitFailed = false, sigmaUi0Failed = false, sigmaVi0Failed = false;
            bool parameterisedSigmaFailed = dataObj.dataCallableStatus(
                [
                    &dataObj, &thetaTry, &clamp_lwr, &clamp_upr,
                    &zuitPos, &zvitPos, &zui0Pos, &zvi0Pos,
                    &sigmaUitFailed, &sigmaVitFailed, &sigmaUi0Failed, &sigmaVi0Failed
                ](
                const auto& y,
                const auto& x,
                const auto& zmuit,
                const auto& zuit,
                const auto& zvit,
                const auto& zui0,
                const auto& zvi0
            ) {
                // go thru each of the elements for zuit, zvit, zui0, zvi0
                if (dataObj.getNZuit() == 1 && zuitPos.has_value()) {
                    // zuit exists, but isn't parameterised - clamp it
                    thetaTry.rows(zuitPos.value().first, zuitPos.value().second).clamp(clamp_lwr, clamp_upr);
                } else if (dataObj.getNZuit() > 1 && zuitPos.has_value()) {
                    // zuit exists, and it is parameterised. check if lower and max values are ok
                    if (!isSigma2InBounds(zuit, thetaTry.rows(zuitPos.value().first, zuitPos.value().second), clamp_lwr, clamp_upr)) {
                        // it failed - the sigma2uit exploded/vanished
                        sigmaUitFailed = true;
                    }
                }
                // again for zvit
                if (dataObj.getNZvit() == 1 && zvitPos.has_value()) {
                    // zvit exists, but isn't parameterised, clamp
                    thetaTry.rows(zvitPos.value().first, zvitPos.value().second).clamp(clamp_lwr, clamp_upr);
                } else if (dataObj.getNZvit() > 1 && zvitPos.has_value()) {
                    if (!isSigma2InBounds(zvit, thetaTry.rows(zvitPos.value().first, zvitPos.value().second), clamp_lwr, clamp_upr)) {
                        // failed, sigma2vit exploded/vanished
                        sigmaVitFailed = true;
                    }
                }
                // again for zui0
                if (dataObj.getNZui0() == 1 && zui0Pos.has_value()) {
                    // zui0 exists, but isn't parameterised.
                    thetaTry.rows(zui0Pos.value().first, zui0Pos.value().second).clamp(clamp_lwr, clamp_upr);
                } else if (dataObj.getNZui0() > 1 && zui0Pos.has_value()) {
                    if (!isSigma2InBounds(zui0, thetaTry.rows(zui0Pos.value().first, zui0Pos.value().second), clamp_lwr, clamp_upr)) {
                        sigmaUi0Failed = true;
                    }
                }
                // again for zvi0
                if (dataObj.getNZvi0() == 1 && zvi0Pos.has_value()) {
                    thetaTry.rows(zvi0Pos.value().first, zvi0Pos.value().second).clamp(clamp_lwr, clamp_upr);
                } else if (dataObj.getNZvi0() > 1 && zvi0Pos.has_value()) {
                    if (!isSigma2InBounds(zvi0, thetaTry.rows(zvi0Pos.value().first, zvi0Pos.value().second), clamp_lwr, clamp_upr)) {
                        sigmaVi0Failed = true;
                    }
                }
                // all of them passed
                return (sigmaUitFailed || sigmaVitFailed || sigmaUi0Failed || sigmaVi0Failed);
            });
            if (shouldLog) {
                ESALogger::logger()->info(
                    "sigma2uit: {}, sigma2vit: {}, sigma2ui0: {}, sigma2vi0: {}",
                    sigmaUitFailed, sigmaVitFailed, sigmaUi0Failed, sigmaVi0Failed
                );
            }
            // if it passed, then check if the loglikelihood score is finite
            if (!parameterisedSigmaFailed) {
                if (std::isfinite(sfaObjPtr->operator()(thetaTry))) {
                    // 
                    startOut = thetaTry;
                    validStart = true;
                    break;
                }
            }
            // try to shrink the search width for both 
            slengthFrontierCurr *= 0.8;
            slengthSigmasCurr *= 0.8;
            // if (slengthSigmasCurr < 1e-5) {
            //     slengthFrontierCurr = slengthFrontier;
            //     slengthSigmasCurr = slengthSigmas;
            // }
            // only adjust the sigmas if they were shown to generate an invalid out of bounds number; 
            if (sigmaUitFailed) slengthSigmaUit = (slengthSigmaUit < 1e-5) ? slengthSigmas : (slengthSigmaUit * 0.8);
            if (sigmaVitFailed) slengthSigmaVit = (slengthSigmaVit < 1e-5) ? slengthSigmas : (slengthSigmaVit * 0.8);
            if (sigmaUi0Failed) slengthSigmaUi0 = (slengthSigmaUi0 < 1e-5) ? slengthSigmas : (slengthSigmaUi0 * 0.8);
            if (sigmaVi0Failed) slengthSigmaVi0 = (slengthSigmaVi0 < 1e-5) ? slengthSigmas : (slengthSigmaVi0 * 0.8);
            
        }
        if (shouldLog) {
            ESALogger::logger()->info("Found valid starting values: {} Took {} attempts", validStart, attempts);
            if (validStart) {
                ESALogger::logger()->info("{}", startOut);
            }
        }
        return validStart;
    }

    /**
     * @brief find valid starting values for LC-TRE models
     * @details perturbs seg/transition params uniformly, and per-class frontier/sigma
     *          params with separate scaling, then accepts if LL is finite.
     */
    inline bool findValidStartValuesLCM(
        const std::shared_ptr<ESADataPanelLCM> lcmDataObjPtr,
        const std::shared_ptr<ESASfaBase> sfaObjPtr,
        const arma::dcolvec& baseStart,
        arma::dcolvec& startOut,
        const int currentIter,
        const double slengthFrontier = 2.0,
        const double slengthSigmas = 0.8,
        const int baseSeed = 1234,
        const int maxAttempts = 100,
        const bool shouldLog = false
    ) {
        const int nClasses = static_cast<int>(lcmDataObjPtr->getNClasses());
        const int nX = lcmDataObjPtr->getNX();
        const int nZmuit = lcmDataObjPtr->getNZmuit();
        const int nZuit = lcmDataObjPtr->getNZuit();
        const int nZvit = lcmDataObjPtr->getNZvit();
        const int nZvi0 = lcmDataObjPtr->getNZvi0();
        const int nSegTotal = lcmDataObjPtr->getTotalSegParams();
        const int nTransTotal = lcmDataObjPtr->getTotalTransitionParams();
        const int classBase = nSegTotal + nTransTotal;
        const int perClass = nX + nZmuit + nZuit + nZvit + nZvi0;
        const int nFrontier = nX + nZmuit; // frontier + mean-ineff terms
        const int nSigmas = nZuit + nZvit + nZvi0; // log-sigma terms

        double slLengthFrontierCurr = slengthFrontier;
        double slLengthSigmasCurr = slengthSigmas;

        bool validStart = false;
        int attempts = 0;
        while (!validStart && attempts < maxAttempts) {
            attempts++;
            std::mt19937 engine(baseSeed + currentIter + attempts * 999);
            std::uniform_real_distribution<double> dist(-0.5, 0.5);
            arma::dcolvec perturb(baseStart.n_rows);
            perturb.imbue([&]() { return dist(engine); });
            // seg/transition params: uniform perturbation
            if (nSegTotal > 0)   perturb.rows(0, nSegTotal - 1) *= slLengthFrontierCurr;
            if (nTransTotal > 0) perturb.rows(nSegTotal, classBase - 1) *= slLengthFrontierCurr;
            // per-class params: frontier scaled by slLengthFrontierCurr, sigmas by slLengthSigmasCurr
            for (int c = 0; c < nClasses; ++c) {
                int base = classBase + c * perClass;
                if (nFrontier > 0)
                    perturb.rows(base, base + nFrontier - 1) *= slLengthFrontierCurr;
                if (nSigmas > 0)
                    perturb.rows(base + nFrontier, base + perClass - 1) *= slLengthSigmasCurr;
            }
            arma::dcolvec thetaTry = baseStart + perturb;
            startOut = thetaTry;
            if (std::isfinite(sfaObjPtr->operator()(thetaTry))) {
                validStart = true;
            } else {
                slLengthFrontierCurr *= 0.8;
                slLengthSigmasCurr   *= 0.8;
            }
        }
        if (shouldLog) {
            ESALogger::logger()->info("LCM: Found valid starting values: {} Took {} attempts", validStart, attempts);
        }
        return validStart;
    }
}


#endif // SEARCH_START_VALS_HPP