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
 * @file ESADataPanel.hpp
 * @brief Class to hold panel data - updated version
 * @date 2025-02-22
 * @author Edmund Haacke
 */

#ifndef ESA_DATA_PANEL_HPP
#define ESA_DATA_PANEL_HPP

#include <atomic>
#include <utility>
#include <variant>
#include <vector>
#include <optional>
#include <stdexcept>
#include <BS_thread_pool.hpp>
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
#include "utils/enums.hpp"
#include "data/ESADataBase.hpp"
#include "utils/esaparallel.hpp"
#include "utils/esautils.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "utils/globalstate.hpp"
#include "utils/interrupts.hpp"

#ifdef PYPACKAGE
#include "pyinterface/pyinterface_utils.hpp"
#endif //PYPACKAGE

using FirmIdsVar = std::variant<std::vector<arma::uvec>, std::vector<std::pair<arma::uword, arma::uword>>>;
using FirmIdsRange = std::vector<std::pair<arma::uword, arma::uword>>;
using FirmIdsInds = std::vector<arma::uvec>;

class ESADataPanel : public ESADataBase {

public:

    // ---- constructors ----
    /// @brief default constructor
    ESADataPanel();

    /// @brief constructor
    /// @param y Column vector of the dependent variable (output)
    /// @param x Matrix of independent variables (in the production function)
    /// @param idVec Unsigned integer column vector of firm identifiers
    /// @param timeVec Unsigned integer column vector of time identifiers
    /// @param modelType Enumeration from ESASfaModelType representing the model type
    /// @param zmuit Optional matrix of determinants of mean of time-varying inefficiency (trunc-normal only)
    /// @param zuit Optional matrix of determinants of time-invariant inefficiency component
    /// @param zvit Optional matrix of determinants of time-varying inefficiency component
    /// @param zui0 Optional matrix of determinants of stochastic noise component
    /// @param zvi0 Optional matrix of determinants of firm effects component
    ESADataPanel(
        const arma::dcolvec* y,
        const arma::dmat* x,
        const arma::Col<int>* idVec,
        const arma::Col<int>* timeVec,
        const ESASfaModelType modelType,
        const arma::dmat* zmuit,
        const arma::dmat* zuit,
        const arma::dmat* zvit,
        const arma::dmat* zui0,
        const arma::dmat* zvi0,
        const bool arraysContiguous = true
    );

    // ---- getter methods for attributes ----
    const ESASfaModelType getModelType() const { return modelType; }
    const arma::dcolvec& getY() const { return y; }
    const arma::dmat& getX() const { return x; }
    const double getNids() const override { return nids; }
    const std::optional<arma::dmat>& getZmuit() { return zmuit; }
    const std::optional<arma::dmat>& getZuit() { return zuit; }
    const std::optional<arma::dmat>& getZvit() { return zvit; }
    const std::optional<arma::dmat>& getZui0() { return zui0; }
    const std::optional<arma::dmat>& getZvi0() { return zvi0; }
    const arma::dmat* getZmuitPtr() { return zmuit.has_value() ? &zmuit.value() : nullptr; }
    const arma::dmat* getZuitPtr() { return zuit.has_value() ? &zuit.value() : nullptr; }
    const arma::dmat* getZvitPtr() { return zvit.has_value() ? &zvit.value() : nullptr; }
    const arma::dmat* getZui0Ptr() { return zui0.has_value() ? &zui0.value() : nullptr; }
    const arma::dmat* getZvi0Ptr() { return zvi0.has_value() ? &zvi0.value() : nullptr; }
    const arma::Col<int>* getIdVecPtr() { return &idVec; }
    const arma::Col<int>* getTimeVecPtr() { return &timeVec; }
    const arma::Col<int>& getIdVec() const { return idVec; }
    const arma::Col<int>& getTimeVec() const { return timeVec; }
    const int getMaxT() const override { return maxT; }
    const int getMinT() const override { return minT; }
    const int getNZmuit() const { return nZmuit; }
    const int getNZuit() const { return nZuit; }
    const int getNZvit() const { return nZvit; }
    const int getNZvi0() const { return nZvi0; }
    const int getNZui0() const { return nZui0; }

    // ---- getter methods for parameter positions ----
    std::pair<int, int> getXRange(const arma::dcolvec& params) const override;
    std::optional<std::pair<int, int>> getZmuitRange(const arma::dcolvec& params) const;
    std::optional<std::pair<int, int>> getZuitRange(const arma::dcolvec& params) const;
    std::optional<std::pair<int, int>> getZvitRange(const arma::dcolvec& params) const;
    std::optional<std::pair<int, int>> getZui0Range(const arma::dcolvec& params) const;
    std::optional<std::pair<int, int>> getZvi0Range(const arma::dcolvec& params) const;
    
    // ---- getter methods for coefficients given a parameter vector ----
    arma::dcolvec paramX(const arma::dcolvec& params) const override;
    std::optional<arma::dcolvec> paramZmuit(const arma::dcolvec& params) const;
    std::optional<arma::dcolvec> paramZuit(const arma::dcolvec& params) const;
    std::optional<arma::dcolvec> paramZvit(const arma::dcolvec& params) const;
    std::optional<arma::dcolvec> paramZui0(const arma::dcolvec& params) const;
    std::optional<arma::dcolvec> paramZvi0(const arma::dcolvec& params) const;

    unsigned int nParams() const override; 

    // ---- callable functions for the data ----
    /// @brief Callable function to process each individual panel (e.g., for every ID)
    /// @param cb callback sends across: index, y, x, zmuit, zuit, zvit, zui0, zvi0
    /// @return matrix of results, having executed the callback
    template <typename Func>
    arma::dmat panelCallable(Func&& cb, const bool shouldSum = false) const
    {
        std::vector<arma::dmat> out(this->nids);
        for (int i = 0; i < this->nids; i++) {
            arma::dmat panRes;
            // contiguous memory {subviews}
            if (this->arraysContiguous) {
                std::pair<arma::uword, arma::uword> startEnd = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(this->firmIdRows)[i];
                arma::uword s = startEnd.first;
                arma::uword e = startEnd.second;
                // create views {zero allocation}
                const auto y_sub = this->y.rows(s, e);
                const auto x_sub = this->x.rows(s, e);
                auto zmuit_sub = makeView(this->zmuit, s, e);
                auto zuit_sub = makeView(this->zuit, s, e);
                auto zvit_sub = makeView(this->zvit, s, e);
                auto zui0_sub = makeView(this->zui0, s, e);
                auto zvi0_sub = makeView(this->zvi0, s, e);
                panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
            } else {
                // non-contiguous - copy
                const arma::uvec inds = std::get<std::vector<arma::uvec>>(this->firmIdRows)[i];
                arma::dmat y_sub = this->y.rows(inds);
                arma::dmat x_sub = this->x.rows(inds);
                auto zmuit_sub = makeCopy(this->zmuit, inds);
                auto zuit_sub = makeCopy(this->zuit, inds);
                auto zvit_sub = makeCopy(this->zvit, inds);
                auto zui0_sub = makeCopy(this->zui0, inds);
                auto zvi0_sub = makeCopy(this->zvi0, inds);
                panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
            }
            out[i] = std::move(panRes);
        }
        if (!shouldSum) return esautils::stackMatricies<double>(out, true);
        return esautils::sumMatricies<double>(out);
    }
    
    template <typename Func>
    arma::dmat panelCallable(const arma::Col<int>& subsetIdents, Func&& cb, const bool shouldSum = false) const
    {
        std::vector<arma::dmat> out(this->nids);
        FirmIdsVar subsetRows = getFirmIdRowsForSubset(subsetIdents);
        for (int i = 0; i < this->nids; i++) {
            arma::dmat panRes;
            // contiguous memory {subviews}
            if (this->arraysContiguous) {
                std::pair<arma::uword, arma::uword> startEnd = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(subsetRows)[i];
                arma::uword s = startEnd.first;
                arma::uword e = startEnd.second;
                // create views {zero allocation}
                const auto y_sub = this->y.rows(s, e);
                const auto x_sub = this->x.rows(s, e);
                auto zmuit_sub = makeView(this->zmuit, s, e);
                auto zuit_sub = makeView(this->zuit, s, e);
                auto zvit_sub = makeView(this->zvit, s, e);
                auto zui0_sub = makeView(this->zui0, s, e);
                auto zvi0_sub = makeView(this->zvi0, s, e);
                panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
            } else {
                // non-contiguous - copy
                const arma::uvec inds = std::get<std::vector<arma::uvec>>(subsetRows)[i];
                arma::dmat y_sub = this->y.rows(inds);
                arma::dmat x_sub = this->x.rows(inds);
                auto zmuit_sub = makeCopy(this->zmuit, inds);
                auto zuit_sub = makeCopy(this->zuit, inds);
                auto zvit_sub = makeCopy(this->zvit, inds);
                auto zui0_sub = makeCopy(this->zui0, inds);
                auto zvi0_sub = makeCopy(this->zvi0, inds);
                panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
            }
            out[i] = std::move(panRes);
        }
        if (!shouldSum) return esautils::stackMatricies<double>(out, true);
        return esautils::sumMatricies<double>(out);
    }

    template <typename Func>
    void panelCallable(
        Func&& cb,
        arma::dmat* ele1,
        arma::dmat* ele2,
        const bool shouldSumEle1 = false,
        const bool shouldSumEle2 = false
    ) const
    {
        std::vector<arma::dmat> vec_ele1(this->nids), vec_ele2(this->nids);
        for (int i = 0; i < this->nids; i++) {
            if (this->arraysContiguous) {
                std::pair<arma::uword, arma::uword> startEnd = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(this->firmIdRows)[i];
                arma::uword s = startEnd.first;
                arma::uword e = startEnd.second;
                // create views {zero allocation}
                const auto y_sub = this->y.rows(s, e);
                const auto x_sub = this->x.rows(s, e);
                auto zmuit_sub = makeView(this->zmuit, s, e);
                auto zuit_sub = makeView(this->zuit, s, e);
                auto zvit_sub = makeView(this->zvit, s, e);
                auto zui0_sub = makeView(this->zui0, s, e);
                auto zvi0_sub = makeView(this->zvi0, s, e);
                arma::dmat e1, e2;
                cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub, &e1, &e2);
                vec_ele1[i] = std::move(e1);
                vec_ele2[i] = std::move(e2);
            } else {
                // non-contiguous - copy
                const arma::uvec inds = std::get<std::vector<arma::uvec>>(this->firmIdRows)[i];
                arma::dmat y_sub = this->y.rows(inds);
                arma::dmat x_sub = this->x.rows(inds);
                auto zmuit_sub = makeCopy(this->zmuit, inds);
                auto zuit_sub = makeCopy(this->zuit, inds);
                auto zvit_sub = makeCopy(this->zvit, inds);
                auto zui0_sub = makeCopy(this->zui0, inds);
                auto zvi0_sub = makeCopy(this->zvi0, inds);
                arma::dmat e1, e2;
                cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub, &e1, &e2);
                vec_ele1[i] = std::move(e1);
                vec_ele2[i] = std::move(e2);
            }
        }
        if (shouldSumEle1 && ele1) *ele1 = esautils::sumMatricies<double>(vec_ele1);
        if (!shouldSumEle1 && ele1) *ele1 = esautils::stackMatricies<double>(vec_ele1, true);
        if (shouldSumEle2 && ele2) *ele2 = esautils::sumMatricies<double>(vec_ele2);
        if (!shouldSumEle2 && ele2) *ele2 = esautils::stackMatricies<double>(vec_ele2, true);
    }

    template <typename Func>
    void panelCallable(
        const arma::Col<int>& subsetIdents,
        Func&& cb,
        arma::dmat* ele1,
        arma::dmat* ele2,
        const bool shouldSumEle1 = false,
        const bool shouldSumEle2 = false
    ) const
    {
        std::vector<arma::dmat> vec_ele1(this->nids), vec_ele2(this->nids);
        FirmIdsVar subsetRows = getFirmIdRowsForSubset(subsetIdents);
        for (int i = 0; i < this->nids; i++) {
            if (this->arraysContiguous) {
                std::pair<arma::uword, arma::uword> startEnd = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(subsetRows)[i];
                arma::uword s = startEnd.first;
                arma::uword e = startEnd.second;
                // create views {zero allocation}
                const auto y_sub = this->y.rows(s, e);
                const auto x_sub = this->x.rows(s, e);
                auto zmuit_sub = makeView(this->zmuit, s, e);
                auto zuit_sub = makeView(this->zuit, s, e);
                auto zvit_sub = makeView(this->zvit, s, e);
                auto zui0_sub = makeView(this->zui0, s, e);
                auto zvi0_sub = makeView(this->zvi0, s, e);
                arma::dmat e1, e2;
                cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub, &e1, &e2);
                vec_ele1[i] = std::move(e1);
                vec_ele2[i] = std::move(e2);
            } else {
                // non-contiguous - copy
                const arma::uvec inds = std::get<std::vector<arma::uvec>>(subsetRows)[i];
                arma::dmat y_sub = this->y.rows(inds);
                arma::dmat x_sub = this->x.rows(inds);
                auto zmuit_sub = makeCopy(this->zmuit, inds);
                auto zuit_sub = makeCopy(this->zuit, inds);
                auto zvit_sub = makeCopy(this->zvit, inds);
                auto zui0_sub = makeCopy(this->zui0, inds);
                auto zvi0_sub = makeCopy(this->zvi0, inds);
                arma::dmat e1, e2;
                cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub, &e1, &e2);
                vec_ele1[i] = std::move(e1);
                vec_ele2[i] = std::move(e2);
            }
        }
        if (shouldSumEle1 && ele1) *ele1 = esautils::sumMatricies<double>(vec_ele1);
        if (!shouldSumEle1 && ele1) *ele1 = esautils::stackMatricies<double>(vec_ele1, true);
        if (shouldSumEle2 && ele2) *ele2 = esautils::sumMatricies<double>(vec_ele2);
        if (!shouldSumEle2 && ele2) *ele2 = esautils::stackMatricies<double>(vec_ele2, true);
    }

    /// @brief Callable function to process each individual panel using threading [e.g., for each ID]
    /// @param cb callback sends across: index, y, x, zmuit, zuit, zvit, zui0, zvi0
    /// @return matrix of results, having executed the callback
    template <typename Func>
    arma::dmat panelCallableThreaded(Func&& cb, const bool shouldSum = false) const
    {
        BS::thread_pool<>& pool = esaparallel::getOptimPool();
        std::vector<arma::dmat> out(this->nids);
        // instansiate struct which holds information whether there was an exception
        esaparallel::ParallelTaskContext ctx;
        // for python, also install signal guard
        #ifdef PYPACKAGE
        pyinterface::SignalHandlerGuard guard;
        #endif // PYPACKAGE
        // define the progress counter
        std::atomic<size_t> progress{0};
        // define the loop task - which iterate thru each Id
        BS::multi_future<void> loop_future = pool.submit_loop(
            0,
            this->nids,
            [this, &out, &cb, &ctx, &progress](const std::size_t i)
            {
                // check if another thread crashed - if so, bin off this one too
                // memory_order_relaxed means no sync/ordering constraints on reads/writes
                if (ctx.panic.load(std::memory_order_relaxed)) return;
                // in python, check for user interrupt
                #ifdef PYPACKAGE
                if (globalstate::InterruptRequested.load(std::memory_order_relaxed)) return;
                #endif // PYPACKAGE
                arma::dmat panRes;
                // wrap entire code in a try catch block
                try {
                    // contiguous memory {subviews}
                    if (this->arraysContiguous) {
                        std::pair<arma::uword, arma::uword> startEnd = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(this->firmIdRows)[i];
                        arma::uword s = startEnd.first;
                        arma::uword e = startEnd.second;
                        // create views {zero allocation}
                        const auto y_sub = this->y.rows(s, e);
                        const auto x_sub = this->x.rows(s, e);
                        auto zmuit_sub = makeView(this->zmuit, s, e);
                        auto zuit_sub = makeView(this->zuit, s, e);
                        auto zvit_sub = makeView(this->zvit, s, e);
                        auto zui0_sub = makeView(this->zui0, s, e);
                        auto zvi0_sub = makeView(this->zvi0, s, e);
                        
                        panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
                    } else {
                        // non-contiguous - copy
                        const arma::uvec inds = std::get<std::vector<arma::uvec>>(this->firmIdRows)[i];
                        arma::dmat y_sub = this->y.rows(inds);
                        arma::dmat x_sub = this->x.rows(inds);
                        auto zmuit_sub = makeCopy(this->zmuit, inds);
                        auto zuit_sub = makeCopy(this->zuit, inds);
                        auto zvit_sub = makeCopy(this->zvit, inds);
                        auto zui0_sub = makeCopy(this->zui0, inds);
                        auto zvi0_sub = makeCopy(this->zvi0, inds);
                        panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
                    }
                    out[i] = std::move(panRes);
                }
                catch (const std::exception& e) {
                    // catch & signal
                    ctx.signalError(e);
                }
                catch (...) {
                    ctx.signalError(std::runtime_error("Unknown error in 'panelCallableThreaded'"));
                }
                // increment the progress ccounter
                progress.fetch_add(1, std::memory_order_relaxed);
            }
        );
        // loop_future.wait();
        esautils::waitForInterrupt(loop_future, progress, this->nids, ctx);
        // check whether we errored out
        if (ctx.panic.load()) {
            throw std::runtime_error("Parallel execution failed with " + ctx.errMsg);
        }
        // for python throw interrupt if needed
        #ifdef PYPACKAGE
        pyinterface::throwPyInterruptIfNeeded();
        #endif // PYPACKAGE
        if (!shouldSum) return esautils::stackMatricies<double>(out, true);
        return esautils::sumMatricies<double>(out);
    }

    template <typename Func>
    arma::dmat panelCallableThreaded(const arma::Col<int>& subsetIdents, Func&& cb, const bool shouldSum = false) const
    {
        BS::thread_pool<>& pool = esaparallel::getOptimPool();
        std::vector<arma::dmat> out(this->nids);
        FirmIdsVar subsetRows = getFirmIdRowsForSubset(subsetIdents);
        BS::multi_future<void> loop_future = pool.submit_loop(
            0,
            this->nids,
            [this, &out, &cb, &subsetRows](const std::size_t i)
            {
                arma::dmat panRes;
                // contiguous memory {subviews}
                if (this->arraysContiguous) {
                    std::pair<arma::uword, arma::uword> startEnd = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(subsetRows)[i];
                    arma::uword s = startEnd.first;
                    arma::uword e = startEnd.second;
                    // create views {zero allocation}
                    const auto y_sub = this->y.rows(s, e);
                    const auto x_sub = this->x.rows(s, e);
                    auto zmuit_sub = makeView(this->zmuit, s, e);
                    auto zuit_sub = makeView(this->zuit, s, e);
                    auto zvit_sub = makeView(this->zvit, s, e);
                    auto zui0_sub = makeView(this->zui0, s, e);
                    auto zvi0_sub = makeView(this->zvi0, s, e);
                    panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
                } else {
                    // non-contiguous - copy
                    const arma::uvec inds = std::get<std::vector<arma::uvec>>(subsetRows)[i];
                    arma::dmat y_sub = this->y.rows(inds);
                    arma::dmat x_sub = this->x.rows(inds);
                    auto zmuit_sub = makeCopy(this->zmuit, inds);
                    auto zuit_sub = makeCopy(this->zuit, inds);
                    auto zvit_sub = makeCopy(this->zvit, inds);
                    auto zui0_sub = makeCopy(this->zui0, inds);
                    auto zvi0_sub = makeCopy(this->zvi0, inds);
                    panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
                }
                out[i] = std::move(panRes);
            }
        );
        loop_future.wait();
        if (!shouldSum) return esautils::stackMatricies<double>(out, true);
        return esautils::sumMatricies<double>(out);
    }

    template <typename Func>
    void panelCallableThreaded(
        Func&& cb,
        arma::dmat* ele1,
        arma::dmat* ele2,
        const bool shouldSumEle1 = false,
        const bool shouldSumEle2 = false
    ) const
    {
        BS::thread_pool<>& pool = esaparallel::getOptimPool();
        std::vector<arma::dmat> vec_ele1(this->nids), vec_ele2(this->nids);
        // instansiate the struct which holds information on whether there was an exception
        esaparallel::ParallelTaskContext ctx;
        // for python - install signal guard
        #ifdef PYPACKAGE
        pyinterface::SignalHandlerGuard guard;
        #endif // PYPACKAGE
        // define the progress counter
        std::atomic<size_t> progress{0};
        // define the loop task - which iterates thru each panel ID
        BS::multi_future<void> loop_future = pool.submit_loop(
            0,
            this->nids,
            [this, &vec_ele1, &vec_ele2, &cb, &ctx, &progress](const std::size_t i)
            {
                // check if another thread crashed - if so - bin this one off
                if (ctx.panic.load(std::memory_order_relaxed)) return;
                // in python, check for user interrupt
                #ifdef PYPACKAGE
                if (globalstate::InterruptRequested.load(std::memory_order_relaxed)) return;
                #endif // PYPACKAGE
                // wrap entire code in a try catch block
                try {
                    // contiguous memory {subviews}
                    if (this->arraysContiguous) {
                        std::pair<arma::uword, arma::uword> startEnd = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(this->firmIdRows)[i];
                        arma::uword s = startEnd.first;
                        arma::uword e = startEnd.second;
                        // create views {zero allocation}
                        const auto y_sub = this->y.rows(s, e);
                        const auto x_sub = this->x.rows(s, e);
                        auto zmuit_sub = makeView(this->zmuit, s, e);
                        auto zuit_sub = makeView(this->zuit, s, e);
                        auto zvit_sub = makeView(this->zvit, s, e);
                        auto zui0_sub = makeView(this->zui0, s, e);
                        auto zvi0_sub = makeView(this->zvi0, s, e);
                        arma::dmat e1, e2;
                        cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub, &e1, &e2);
                        vec_ele1[i] = std::move(e1);
                        vec_ele2[i] = std::move(e2);
                    } else {
                        // non-contiguous - copy
                        const arma::uvec inds = std::get<std::vector<arma::uvec>>(this->firmIdRows)[i];
                        arma::dmat y_sub = this->y.rows(inds);
                        arma::dmat x_sub = this->x.rows(inds);
                        auto zmuit_sub = makeCopy(this->zmuit, inds);
                        auto zuit_sub = makeCopy(this->zuit, inds);
                        auto zvit_sub = makeCopy(this->zvit, inds);
                        auto zui0_sub = makeCopy(this->zui0, inds);
                        auto zvi0_sub = makeCopy(this->zvi0, inds);
                        arma::dmat e1, e2;
                        cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub, &e1, &e2);
                        vec_ele1[i] = std::move(e1);
                        vec_ele2[i] = std::move(e2);
                    }
                }
                catch (const std::exception& e) {
                    // catch & signal exception
                    ctx.signalError(e);
                }
                catch (...) {
                    ctx.signalError(std::runtime_error("Unknown error in 'panelCallableThreaded'"));
                }
                // increment the progress counter
                progress.fetch_add(1, std::memory_order_relaxed);
            }
        );
        // loop_future.wait();
        esautils::waitForInterrupt(loop_future, progress, this->nids, ctx);
        // check whether caught any exceptions
        if (ctx.panic.load()) {
            throw std::runtime_error("Parallel execution failed with " + ctx.errMsg);
        }
        // check for python interrupt
        #ifdef PYPACKAGE
        pyinterface::throwPyInterruptIfNeeded();
        #endif // PYPACKAGE
        if (shouldSumEle1 && ele1) *ele1 = esautils::sumMatricies<double>(vec_ele1);
        if (!shouldSumEle1 && ele1) *ele1 = esautils::stackMatricies<double>(vec_ele1, true);
        if (shouldSumEle2 && ele2) *ele2 = esautils::sumMatricies<double>(vec_ele2);
        if (!shouldSumEle2 && ele2) *ele2 = esautils::stackMatricies<double>(vec_ele2, true);
    }
    
    template <typename Func>
    void panelCallableThreaded(
        const arma::Col<int>& subsetIdents,
        Func&& cb,
        arma::dmat* ele1,
        arma::dmat* ele2,
        const bool shouldSumEle1 = false,
        const bool shouldSumEle2 = false
    ) const
    {
        BS::thread_pool<>& pool = esaparallel::getOptimPool();
        std::vector<arma::dmat> vec_ele1(this->nids), vec_ele2(this->nids);
        FirmIdsVar subsetRows = getFirmIdRowsForSubset(subsetIdents);
        BS::multi_future<void> loop_future = pool.submit_loop(
            0,
            this->nids,
            [this, &vec_ele1, &vec_ele2, &cb, &subsetRows](const std::size_t i)
            {
                // contiguous memory {subviews}
                if (this->arraysContiguous) {
                    std::pair<arma::uword, arma::uword> startEnd = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(subsetRows)[i];
                    arma::uword s = startEnd.first;
                    arma::uword e = startEnd.second;
                    // create views {zero allocation}
                    const auto y_sub = this->y.rows(s, e);
                    const auto x_sub = this->x.rows(s, e);
                    auto zmuit_sub = makeView(this->zmuit, s, e);
                    auto zuit_sub = makeView(this->zuit, s, e);
                    auto zvit_sub = makeView(this->zvit, s, e);
                    auto zui0_sub = makeView(this->zui0, s, e);
                    auto zvi0_sub = makeView(this->zvi0, s, e);
                    arma::dmat e1, e2;
                    cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub, &e1, &e2);
                    vec_ele1[i] = std::move(e1);
                    vec_ele2[i] = std::move(e2);
                } else {
                    // non-contiguous - copy
                    const arma::uvec inds = std::get<std::vector<arma::uvec>>(subsetRows)[i];
                    arma::dmat y_sub = this->y.rows(inds);
                    arma::dmat x_sub = this->x.rows(inds);
                    auto zmuit_sub = makeCopy(this->zmuit, inds);
                    auto zuit_sub = makeCopy(this->zuit, inds);
                    auto zvit_sub = makeCopy(this->zvit, inds);
                    auto zui0_sub = makeCopy(this->zui0, inds);
                    auto zvi0_sub = makeCopy(this->zvi0, inds);
                    arma::dmat e1, e2;
                    cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub, &e1, &e2);
                    vec_ele1[i] = std::move(e1);
                    vec_ele2[i] = std::move(e2);
                }
            }
        );
        loop_future.wait();
        if (shouldSumEle1 && ele1) *ele1 = esautils::sumMatricies<double>(vec_ele1);
        if (!shouldSumEle1 && ele1) *ele1 = esautils::stackMatricies<double>(vec_ele1, true);
        if (shouldSumEle2 && ele2) *ele2 = esautils::sumMatricies<double>(vec_ele2);
        if (!shouldSumEle2 && ele2) *ele2 = esautils::stackMatricies<double>(vec_ele2, true);
    }

    /// @brief Callable function to process each individual panel (e.g., for every ID) and sum the result
    /// @param cb callback sends across: index, y, x, zmuit, zuit, zvit, zui0, zvi0
    /// @return single value of the result, having executed the callback on each panel, and totaled the result
    template <typename Func>
    double panelCallableSum(Func&& cb) const
    {
        double total = 0.0;
        for (int i = 0; i < this->nids; i++) {
            double panRes = 0.0;
            // contiguous memory {subviews}
            if (this->arraysContiguous) {
                std::pair<arma::uword, arma::uword> startEnd = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(this->firmIdRows)[i];
                arma::uword s = startEnd.first;
                arma::uword e = startEnd.second;
                // create views {zero allocation}
                const auto y_sub = this->y.rows(s, e);
                const auto x_sub = this->x.rows(s, e);
                auto zmuit_sub = makeView(this->zmuit, s, e);
                auto zuit_sub = makeView(this->zuit, s, e);
                auto zvit_sub = makeView(this->zvit, s, e);
                auto zui0_sub = makeView(this->zui0, s, e);
                auto zvi0_sub = makeView(this->zvi0, s, e);
                panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
            } else {
                // non-contiguous - copy
                const arma::uvec inds = std::get<std::vector<arma::uvec>>(this->firmIdRows)[i];
                arma::dmat y_sub = this->y.rows(inds);
                arma::dmat x_sub = this->x.rows(inds);
                auto zmuit_sub = makeCopy(this->zmuit, inds);
                auto zuit_sub = makeCopy(this->zuit, inds);
                auto zvit_sub = makeCopy(this->zvit, inds);
                auto zui0_sub = makeCopy(this->zui0, inds);
                auto zvi0_sub = makeCopy(this->zvi0, inds);
                panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
            }
            total += panRes;
        }
        return total;
    }

    template <typename Func>
    double panelCallableSum(const arma::Col<int>& subsetIdents, Func&& cb) const
    {
        double total = 0.0;
        FirmIdsVar subsetRows = getFirmIdRowsForSubset(subsetIdents);
        for (int i = 0; i < this->nids; i++) {
            double panRes = 0.0;
            // contiguous memory {subviews}
            if (this->arraysContiguous) {
                std::pair<arma::uword, arma::uword> startEnd = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(subsetRows)[i];
                arma::uword s = startEnd.first;
                arma::uword e = startEnd.second;
                // create views {zero allocation}
                const auto y_sub = this->y.rows(s, e);
                const auto x_sub = this->x.rows(s, e);
                auto zmuit_sub = makeView(this->zmuit, s, e);
                auto zuit_sub = makeView(this->zuit, s, e);
                auto zvit_sub = makeView(this->zvit, s, e);
                auto zui0_sub = makeView(this->zui0, s, e);
                auto zvi0_sub = makeView(this->zvi0, s, e);
                panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
            } else {
                // non-contiguous - copy
                const arma::uvec inds = std::get<std::vector<arma::uvec>>(subsetRows)[i];
                arma::dmat y_sub = this->y.rows(inds);
                arma::dmat x_sub = this->x.rows(inds);
                auto zmuit_sub = makeCopy(this->zmuit, inds);
                auto zuit_sub = makeCopy(this->zuit, inds);
                auto zvit_sub = makeCopy(this->zvit, inds);
                auto zui0_sub = makeCopy(this->zui0, inds);
                auto zvi0_sub = makeCopy(this->zvi0, inds);
                panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
            }
            total += panRes;
        }
        return total;
    }

    template <typename Func>
    double panelCallableSumThreaded(Func&& cb) const
    {
        BS::thread_pool<>& pool = esaparallel::getOptimPool();
        // final total
        arma::dcolvec result(this->nids);
        // instansiate struct which holds information about whether or not exception occured
        esaparallel::ParallelTaskContext ctx;
        // for python, install signal guard
    #ifdef PYPACKAGE
        pyinterface::SignalHandlerGuard guard;
    #endif
        // progress counter
        std::atomic<size_t> progress{0};
        // define the loop task - which iterates thru each ID
        BS::multi_future<void> loop_future = pool.submit_loop(
            0,
            this->nids,
            [this, &result, &cb, &ctx, &progress](const std::size_t i)
            {
                // check if another thread crashed - if so, bin off
                if (ctx.panic.load(std::memory_order_relaxed)) return;
                // in python, check for a user interrupt
            #ifdef PYPACKAGE
                if (globalstate::InterruptRequested.load(std::memory_order_relaxed)) return;
            #endif // PYPACKAGE
                try {
                    // ordered identifiers
                    double panRes = 0.0;
                    if (this->arraysContiguous) {
                        std::pair<arma::uword, arma::uword> startEnd = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(this->firmIdRows)[i];
                        arma::uword s = startEnd.first;
                        arma::uword e = startEnd.second;
                        // create views {zero allocation}
                        const auto y_sub = this->y.rows(s, e);
                        const auto x_sub = this->x.rows(s, e);
                        auto zmuit_sub = makeView(this->zmuit, s, e);
                        auto zuit_sub = makeView(this->zuit, s, e);
                        auto zvit_sub = makeView(this->zvit, s, e);
                        auto zui0_sub = makeView(this->zui0, s, e);
                        auto zvi0_sub = makeView(this->zvi0, s, e);
                        panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
                    } else {
                        // non-contiguous - copy
                        const arma::uvec inds = std::get<std::vector<arma::uvec>>(this->firmIdRows)[i];
                        arma::dmat y_sub = this->y.rows(inds);
                        arma::dmat x_sub = this->x.rows(inds);
                        auto zmuit_sub = makeCopy(this->zmuit, inds);
                        auto zuit_sub = makeCopy(this->zuit, inds);
                        auto zvit_sub = makeCopy(this->zvit, inds);
                        auto zui0_sub = makeCopy(this->zui0, inds);
                        auto zvi0_sub = makeCopy(this->zvi0, inds);
                        panRes = cb(i, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zui0_sub, zvi0_sub);
                    }
                    // increment the total
                    result(i) = panRes;
                }
                catch (const std::exception& e) {
                    // catch & sigma
                    ctx.signalError(e);
                }
                catch (...) {
                    ctx.signalError(std::runtime_error("Unknown error in 'panelCallableSumThreaded'"));
                }
                // increment the progress counter
                progress.fetch_add(1, std::memory_order_relaxed);
            }
        );
        // check for interrupts [in R]
        esautils::waitForInterrupt(loop_future, progress, this->nids, ctx);
        // check whether errored out
        if (ctx.panic.load()) {
            throw std::runtime_error("Parallel executiion failed with " + ctx.errMsg);
        }
        // for python, throw interrupt if needed
    #ifdef PYPACKAGE
        pyinterface::throwPyInterruptIfNeeded();
    #endif // PYPACKAGE
        // sum up the column vector
        return arma::accu(result);
    }

    template <typename Func>
    void dataCallableRowMeans(Func&& cb, arma::dmat* ele1, arma::dmat* ele2) const {
        int nids = this->getNids();
        arma::dmat yM(nids, 1), xM(nids, this->nX);
        std::optional<arma::dmat> zmuitM = std::nullopt, zuitM = std::nullopt, zvitM = std::nullopt, zui0M = std::nullopt, zvi0M = std::nullopt;
        if (this->zmuit) zmuitM = std::make_optional<arma::dmat>(arma::dmat(nids, this->nZmuit));
        if (this->zuit) zuitM = std::make_optional<arma::dmat>(arma::dmat(nids, this->nZuit));
        if (this->zvit) zvitM = std::make_optional<arma::dmat>(arma::dmat(nids, this->nZvit));
        if (this->zui0) zui0M = std::make_optional<arma::dmat>(arma::dmat(nids, this->nZui0));
        if (this->zvi0) zvi0M = std::make_optional<arma::dmat>(arma::dmat(nids, this->nZvi0));
        for (int i = 0; i < nids; i++) {
            if (this->arraysContiguous) {
                std::pair<arma::uword, arma::uword> startEnd = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(this->firmIdRows)[i];
                arma::uword s = startEnd.first, e = startEnd.second;
                yM.row(i) = arma::mean(this->y.rows(s, e));
                xM.row(i) = arma::mean(this->x.rows(s, e));
                if (this->zmuit) zmuitM.value().row(i) = arma::mean(this->zmuit.value().rows(s, e));
                if (this->zuit) zuitM.value().row(i) = arma::mean(this->zuit.value().rows(s, e));
                if (this->zvit) zvitM.value().row(i) = arma::mean(this->zvit.value().rows(s, e));
                if (this->zui0) zui0M.value().row(i) = arma::mean(this->zui0.value().rows(s, e));
                if (this->zvi0) zvi0M.value().row(i) = arma::mean(this->zvi0.value().rows(s, e));
            } else {
                const arma::uvec inds = std::get<std::vector<arma::uvec>>(this->firmIdRows)[i];
                yM.row(i) = arma::mean(this->y.rows(inds));
                xM.row(i) = arma::mean(this->x.rows(inds));
                if (this->zmuit) zmuitM.value().row(i) = arma::mean(this->zmuit.value().rows(inds));
                if (this->zuit) zuitM.value().row(i) = arma::mean(this->zuit.value().rows(inds));
                if (this->zvit) zvitM.value().row(i) = arma::mean(this->zvit.value().rows(inds));
                if (this->zui0) zui0M.value().row(i) = arma::mean(this->zui0.value().rows(inds));
                if (this->zvi0) zvi0M.value().row(i) = arma::mean(this->zvi0.value().rows(inds));
            }
        }
        cb(yM, xM, zmuitM, zuitM, zvitM, zui0M, zvi0M, ele1, ele2);
    }

    /// @brief Callable function to process the entire data
    /// @param cb callback sends across: index, y, x, zmuit, zuit, zvit, zui0, zvi0
    /// @return Matrix of the result, having executed on the entire data at once
    template <typename Func>
    arma::dmat dataCallable(Func&& cb) const{
        return cb(this->y, this->x, this->zmuit, this->zuit, this->zvit, this->zui0, this->zvi0);
    }

    template <typename Func>
    void dataCallable(Func&& cb, arma::dmat* ele1, arma::dmat* ele2) const
    {
        cb(this->y, this->x, this->zmuit, this->zuit, this->zvit, this->zui0, this->zvi0, ele1, ele2);
    }

    template <typename Func>
    void dataCallable(
        Func&& cb,
        arma::dmat* ele1,
        arma::dmat* ele2,
        arma::dmat* ele3
    ) const {
        cb(this->y, this->x, this->zmuit, this->zuit, this->zvit, this->zui0, this->zvi0, ele1, ele2, ele3);
    }

    /// @brief Callable function to process the entire data
    /// @param cb callback sends across: index, y, x, zmuit, zuit, zvit, zui0, zvi0
    /// @return single value of the result having executed on the entire data at once
    template <typename Func>
    double dataCallableSum(Func&& cb) const
    {
        return cb(this->y, this->x, this->zmuit, this->zuit, this->zvit, this->zui0, this->zvi0);
    }

    template <typename Func>
    bool dataCallableStatus(Func&& cb) const {
        return cb(this->y, this->x, this->zmuit, this->zuit, this->zvit, this->zui0, this->zvi0);
    }


private:

    const arma::Col<int> idVec;
    const arma::Col<int> timeVec;
    const std::optional<arma::dmat> zmuit;
    const std::optional<arma::dmat> zuit;
    const std::optional<arma::dmat> zvit;
    const std::optional<arma::dmat> zui0;
    const std::optional<arma::dmat> zvi0;
    const unsigned int nZmuit;
    const unsigned int nZuit;
    const unsigned int nZvit;
    const unsigned int nZui0;
    const unsigned int nZvi0;
    int nids;
    unsigned int npanels;
    bool balanced;
    const arma::Col<int> uniqueIds;
    FirmIdsVar firmIdRows;
    int maxT;
    int minT;

    void checkParamsShape(const arma::dcolvec& params) const override;

    /**
     * @details Check if a time-invariant (per firm) column is truly that
     * @param tol the tolerance 
     */
    void checkTimeInvariance(const double tol = 1e-8) const;

    unsigned int findZmuit();
    unsigned int findZuit();
    unsigned int findZvit();
    unsigned int findZui0();
    unsigned int findZvi0();

    std::optional<arma::dmat> validateZmuit(const std::optional<arma::dmat>& m, const ESASfaModelType mT) const;
    std::optional<arma::dmat> validateZuit(const std::optional<arma::dmat>& m, const ESASfaModelType mT) const;
    std::optional<arma::dmat> validateZvit(const std::optional<arma::dmat>& m, const ESASfaModelType mT) const;
    std::optional<arma::dmat> validateZui0(const std::optional<arma::dmat>& m, const ESASfaModelType mT) const;
    std::optional<arma::dmat> validateZvi0(const std::optional<arma::dmat>& m, const ESASfaModelType mT) const;

    // Helper for optional views without copying
    template <typename T>
    std::optional<arma::subview<double>> makeView(const std::optional<T>& m, arma::uword start, arma::uword end) const;
    // Helper for non-contiguous
    template <typename T>
    std::optional<arma::dmat> makeCopy(const std::optional<T>& m, const arma::uvec& inds) const;
    // Helper to return variant for subset
    FirmIdsVar getFirmIdRowsForSubset(const arma::Col<int>& subsetIdents) const;
    // find maxT
    void calculateMaxT(const FirmIdsVar& precalc, bool isContig);
};

#endif // ESA_DATA_PANEL_HPP