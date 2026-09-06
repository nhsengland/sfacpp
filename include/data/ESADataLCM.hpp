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

#ifndef ESA_DATA_PANEL_LCM
#define ESA_DATA_PANEL_LCM

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

class ESADataPanelLCM : public ESADataBase {

public:
    // ---- constructors ----
    /// @brief default constructor
    ESADataPanelLCM();

    /**
     * @brief main constructor
     * @param seg Matrix of concomitant variables to determine (initial) latent class membership
     * @param y Column vector of the dependent variable (output)
     * @param x Matrix of independent variables (int the production function)
     * @param idVec Unsigned integer column vector of firm identifiers (optional for X-sectional)
     * @param timeVec Unsigned integer column vector of time identifiers (optional for X-sectional)
     * @param zmuit Optional matrix of determinants of mean of time-varying inefficiency (trunc-normal only)
     * @param zuit Optional matrix of determinants of time-varying inefficiency component
     * @param zvit Optional matrix of determinants of the stochastic noise component
     * @param zvi0 Optional matrix of determinants of the firm effects component
     * @param nClasses The number of latent classes
     * @param transition Matrix of time-varying concomitant variables affecting transition probability
     */
    ESADataPanelLCM(
        const arma::dmat* seg,
        const arma::dcolvec* y,
        const arma::dmat* x,
        const arma::Col<int>* idVec,
        const arma::Col<int>* timeVec,
        const ESASfaModelType modelType,
        const arma::dmat* zmuit,
        const arma::dmat* zuit,
        const arma::dmat* zvit,
        const arma::dmat* zvi0,
        const unsigned int nClasses,
        const arma::dmat* transition = nullptr,
        const bool arraysContiguous = true
    );

    // ---- getter methods for attributes ----
    const ESASfaModelType getModelType() const { return modelType; }
    const arma::dmat& getSeg() const { return seg; }
    const arma::dcolvec& getY() const { return this->y; }
    const arma::dmat& getX() const { return this->x; }
    const double getNids() const override { return nids; }
    const std::optional<arma::dmat>& getZmuit() { return zmuit; }
    const std::optional<arma::dmat>& getZuit() { return zuit; }
    const std::optional<arma::dmat>& getZvit() { return zvit; }
    const std::optional<arma::dmat>& getZvi0() { return zvi0; }
    const std::optional<arma::dmat>& getTransition() { return transition; }
    const arma::dmat* getZmuitPtr() { return zmuit.has_value() ? &zmuit.value() : nullptr; }
    const arma::dmat* getZuitPtr() { return zuit.has_value() ? &zuit.value() : nullptr; }
    const arma::dmat* getZvitPtr() { return zvit.has_value() ? &zvit.value() : nullptr; }
    const arma::dmat* getZvi0Ptr() { return zvi0.has_value() ? &zvi0.value() : nullptr; }
    const arma::dmat* getTransitionPtr() { return transition.has_value() ? &transition.value() : nullptr; }
    const arma::Col<int>* getIdVecPtr() { return idVec.has_value() ? &idVec.value() : nullptr; }
    const arma::Col<int>* getTimeVecPtr() { return timeVec.has_value() ? &timeVec.value() : nullptr; }
    const std::optional<arma::Col<int>>& getIdVec() const { return idVec; }
    const std::optional<arma::Col<int>>& getTimeVec() const { return timeVec; }
    const int getMaxT() const override { return maxT; }
    const int getMinT() const override { return minT; }
    const int getNZmuit() const { return nZmuit; }
    const int getNZuit() const { return nZuit; }
    const int getNZvit() const { return nZvit; }
    const int getNZvi0() const { return nZvi0; }
    const int getNSeg() const { return nSeg; }
    const int getNClasses() const { return nLc; }
    const int getNTransition() const { return nTransition; }
    int getTotalSegParams() const { return (nLc > 1) ? (nLc - 1) * nSeg : 0; }
    int getTotalTransitionParams() const { return nLc * (nLc - 1) * nTransition; }
    int getClassParamsBase() const { return getTotalSegParams() + getTotalTransitionParams(); }

    /**
     * @brief override parameter vector checking function
     * @param params the column vector of parameters to check
     * @throw invalid_argument if incorrect number of rows
     */
    void checkParamsShape(const arma::dcolvec& params) const override
    {
        if (params.n_cols != 1) {
            throw std::invalid_argument("params must be a column vector");
        }
        int expected = getTotalSegParams() + getTotalTransitionParams() + (
            this->nLc * (this->nX + this->nZmuit + this->nZuit + this->nZvit + this->nZvi0)
        );
        if (params.n_rows != expected) {
            throw std::invalid_argument("params must have the correct number of rows");
        }
    }

    /**
     * @brief getter methods for parameter positions (per latent class)
     * @details the structure is as follows:
     *  1. segmentation variables
     *  2. transition variables
     *  3. per latent class:
     *      - x-vars
     *      - zmuit
     *      - zuit
     *      - zvit
     *      - zvi0
     */
    /**
     * @brief Get parameter positions for segmentation variables
     * @param params column vector of parameters
     * @return pair of index positions
     */
    std::pair<int, int> getSegRange(const arma::dcolvec& params) const
    {
        checkParamsShape(params);
        int nSegTotal = getTotalSegParams();
        if (nSegTotal == 0) return std::make_pair(0, -1);
        return std::make_pair(0, nSegTotal - 1);
    }
    /**
     * @brief Get parameter position for transition variables, if existing
     * @param params column vector of parameters
     * @return optional pair of index positions
     */
    std::optional<std::pair<int, int>> getTransitionRange(const arma::dcolvec& params) const
    {
        checkParamsShape(params);
        int nTransTotal = getTotalTransitionParams();
        if (nTransTotal == 0) return std::nullopt;
        int nSegTotal = getTotalSegParams();
        return std::make_optional<std::pair<int, int>>(nSegTotal, nSegTotal + nTransTotal - 1);
    }
    /**
     * @brief Get parameter position for frontier for a given latent class
     * @param params column vector of parameters
     * @param lc the latent class to extract for
     * @return pair of index positions
     */
    std::pair<int, int> getXRange(const arma::dcolvec& params, const unsigned int lc) const
    {
        checkParamsShape(params);
        if (lc > nLc) throw std::invalid_argument("'lc' is beyond existing number of latent classes");
        int base = getClassParamsBase();
        int totalLC = nX + nZmuit + nZuit + nZvit + nZvi0;
        int lhs = base + (lc * totalLC) + 0;
        int rhs = base + (lc * totalLC) + nX - 1;
        return std::make_pair(lhs, rhs);
    }
    /**
     * @brief Get parameter position for truncated mean for a given latent class
     * @param params column vector of parameters
     * @param lc the latent class to extract for
     * @return an optional pair of index positions
     */
    std::optional<std::pair<int, int>> getZmuitRange(const arma::dcolvec& params, const unsigned int lc) const
    {
        checkParamsShape(params);
        if (lc > nLc) throw std::invalid_argument("'lc' is beyond existing number of latent classes");
        if (nZmuit == 0) return std::nullopt;
        int base = getClassParamsBase();
        int totalLC = nX + nZmuit + nZuit + nZvit + nZvi0;
        int lhs = base + (lc * totalLC) + nX;
        int rhs = base + (lc * totalLC) + nX + nZmuit - 1;
        return std::make_optional<std::pair<int, int>>(lhs, rhs);
    }
    /**
     * @brief Get parameter position for determinants of time-varying inefficiency for a given latent class
     * @param params column vector of parameters
     * @param lc the latent class to extract for
     * @return an optional pair of index positions
     */
    std::optional<std::pair<int, int>> getZuitRange(const arma::dcolvec& params, const unsigned int lc) const
    {
        checkParamsShape(params);
        if (lc > nLc) throw std::invalid_argument("'lc' is beyond existing number of latent classes");
        if (nZuit == 0) return std::nullopt;
        int base = getClassParamsBase();
        int totalLC = nX + nZmuit + nZuit + nZvit + nZvi0;
        int lhs = base + (lc * totalLC) + nX + nZmuit;
        int rhs = base + (lc * totalLC) + nX + nZmuit + nZuit - 1;
        return std::make_optional<std::pair<int, int>>(lhs, rhs);
    }
    /**
     * @brief Get parameter position for determinants of stochastic noise for a given latent class
     * @param params column vector of parameters
     * @param lc the latent class to extract for
     * @return an optional pair of index positions
     */
    std::optional<std::pair<int, int>> getZvitRange(const arma::dcolvec& params, const unsigned int lc) const
    {
        checkParamsShape(params);
        if (lc > nLc) throw std::invalid_argument("'lc' is beyond existing number of latent classes");
        if (nZvit == 0) return std::nullopt;
        int base = getClassParamsBase();
        int totalLC = nX + nZmuit + nZuit + nZvit + nZvi0;
        int lhs = base + (lc * totalLC) + nX + nZmuit + nZuit;
        int rhs = base + (lc * totalLC) + nX + nZmuit + nZuit + nZvit - 1;
        return std::make_optional<std::pair<int, int>>(lhs, rhs);
    }
    /**
     * @brief Get parameter positions for determinants of latent firm effects
     * @param params column vector of parameters
     * @param lc the latent class to extract for
     * @return an optional pair of index positions
     */
    std::optional<std::pair<int, int>> getZvi0Range(const arma::dcolvec& params, const unsigned int lc) const
    {
        checkParamsShape(params);
        if (lc > nLc) throw std::invalid_argument("'lc' is beyond existing number of latent classes");
        if (nZvi0 == 0) return std::nullopt;
        int base = getClassParamsBase();
        int totalLC = nX + nZmuit + nZuit + nZvit + nZvi0;
        int lhs = base + (lc * totalLC) + nX + nZmuit + nZuit + nZvit;
        int rhs = base + (lc * totalLC) + nX + nZmuit + nZuit + nZvit + nZvi0 - 1;
        return std::make_optional<std::pair<int, int>>(lhs, rhs);
    }

    // ---- getter methods for coefficients given a parameter vector ----
    /**
     * @brief extract the segmentation parameters from a parameter vector
     * @param params column vector of parameters
     * @return column vector
     */
    arma::dcolvec paramSeg(const arma::dcolvec& params) const
    {
        std::pair<int, int> rr = this->getSegRange(params);
        if (rr.second < rr.first) return arma::dcolvec();
        arma::dcolvec r = params.rows(rr.first, rr.second);
        return r;
    }
    /**
     * @brief extract the transition parameters from a parameter vector, if they exist
     * @param params column vector of parameters
     * @return optional column vector
     */
    std::optional<arma::dcolvec> paramTransition(const arma::dcolvec& params) const
    {
        std::optional<std::pair<int, int>> rr = this->getTransitionRange(params);
        if (!rr) return std::nullopt;
        arma::dcolvec r = params.rows(rr->first, rr->second);
        return std::make_optional<arma::dcolvec>(std::move(r));
    }
    /**
     * @brief extract the frontier parameters for a given latent class from a parameter vector
     * @param params column vector of parameters
     * @param lc the latent class
     * @return column vector
     */
    arma::dcolvec paramX(const arma::dcolvec& params, const unsigned int lc) const
    {
        std::pair<int, int> rr = this->getXRange(params, lc);
        arma::dcolvec r = params.rows(rr.first, rr.second);
        return r;
    }
    /**
     * @brief extract the truncated mean parameters
     * @param params column vector of parameters
     * @param lc the latent class
     * @return optional column vector
     */
    std::optional<arma::dcolvec> paramZmuit(const arma::dcolvec& params, const unsigned int lc) const
    {
        std::optional<std::pair<int, int>> rr = this->getZmuitRange(params, lc);
        if (!rr) return std::nullopt;
        arma::dcolvec r = params.rows(rr->first, rr->second);
        return std::optional<arma::dcolvec>(std::move(r));
    }
    /**
     * @brief extract the determinants of time-varying inefficiency for a given latent class
     * @param params column vector of parameters
     * @param lc the latent class
     * @return optional column vector
     */
    std::optional<arma::dcolvec> paramZuit(const arma::dcolvec& params, const unsigned int lc) const
    {
        std::optional<std::pair<int, int>> rr = this->getZuitRange(params, lc);
        if (!rr) return std::nullopt;
        arma::dcolvec r = params.rows(rr->first, rr->second);
        return std::optional<arma::dcolvec>(std::move(r));
    }
    /**
     * @brief extract the determinants of stochastic noise for a given latent class
     * @param params column vector of parameters
     * @param lc the latent class
     * @return optional column vector
     */
    std::optional<arma::dcolvec> paramZvit(const arma::dcolvec& params, const unsigned int lc) const
    {
        std::optional<std::pair<int, int>> rr = this->getZvitRange(params, lc);
        if (!rr) return std::nullopt;
        arma::dcolvec r = params.rows(rr->first, rr->second);
        return std::optional<arma::dcolvec>(std::move(r));
    }
    /**
     * @brief extract the determinants of latent firm heterogeneity, for a given latent class
     * @param params column vector of parameters
     * @param lc the latent class
     * @return optional column vector
     */
    std::optional<arma::dcolvec> paramZvi0(const arma::dcolvec& params, const unsigned int lc) const
    {
        std::optional<std::pair<int, int>> rr = this->getZvi0Range(params, lc);
        if (!rr) return std::nullopt;
        arma::dcolvec r = params.rows(rr->first, rr->second);
        return std::optional<arma::dcolvec>(std::move(r));
    }
    
    /**
     * @brief total number of parameters including
     */
    unsigned int nParams() const override {
        return getTotalSegParams() + getTotalTransitionParams() +
               this->nLc * (this->nX + this->nZmuit + this->nZuit + this->nZvit + this->nZvi0);
    }
    unsigned int nParamsLc() const { return this->nLc; }

    // ---- callable functions for the data ----
    /**
     * @brief callable function to process each individual panel (e.g., for every ID)
     * @details where not time-series, then the whole matrix is returned
     * @param cb callback sends across: index, seg, y, x, zmuit, zuit, zvit, zvi0, transition
     * @return matrix of results, having executed the callback
     */
    template <typename Func>
    void dataCallable(
        Func&& cb,
        arma::dmat* ele1 = nullptr,
        arma::dmat* ele2 = nullptr,
        const bool shouldSumEle1 = false,
        const bool shouldSumEle2 = false,
        const bool threaded = false
    ) const
    {
        // general setup
        bool hasIds = this->idVec.has_value();
        int numIds = hasIds ? this->nids : 1;
        std::vector<arma::dmat> vec_ele1(numIds), vec_ele2(numIds);
        // progress counter - only needed for the multithreaded
        std::atomic<size_t> progress{0};

        esaparallel::ParallelTaskContext ctx;

        // Lambda wrapper for threaded/non-threaded environments
        auto runFunc = [this, &vec_ele1, &vec_ele2, &cb, &progress, &ctx](const std::size_t i) {
            if (ctx.panic.load(std::memory_order_relaxed)) return;
            arma::dmat panRes;
            try {
                // For simplicity in non-threaded version, just call callback with subviews
                if (this->isPanel && this->idVec.has_value() && this->firmIdRows.has_value()) {
                    // has panel data
                    if (this->arraysContiguous) {
                        auto& ranges = std::get<std::vector<std::pair<arma::uword, arma::uword>>>(*this->firmIdRows);
                        if (i < ranges.size()) {
                            arma::uword s = ranges[i].first, e = ranges[i].second;
                            const auto seg_sub = this->seg.rows(s, e);
                            const auto y_sub = this->y.rows(s, e);
                            const auto x_sub = this->x.rows(s, e);
                            auto zmuit_sub = makeView(this->zmuit, s, e);
                            auto zuit_sub = makeView(this->zuit, s, e);
                            auto zvit_sub = makeView(this->zvit, s, e);
                            auto zvi0_sub = makeView(this->zvi0, s, e);
                            auto transition_sub = makeView(this->transition, s, e);
                            panRes = cb(i, seg_sub, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zvi0_sub, transition_sub);
                        }
                    } else {
                        auto& inds_vec = std::get<std::vector<arma::uvec>>(*this->firmIdRows);
                        if (i < inds_vec.size()) {
                            const arma::uvec& inds = inds_vec[i];
                            arma::dmat seg_sub = this->seg.rows(inds);
                            arma::dmat y_sub = this->y.rows(inds);
                            arma::dmat x_sub = this->x.rows(inds);
                            auto zmuit_sub = makeCopy(this->zmuit, inds);
                            auto zuit_sub = makeCopy(this->zuit, inds);
                            auto zvit_sub = makeCopy(this->zvit, inds);
                            auto zvi0_sub = makeCopy(this->zvi0, inds);
                            auto transition_sub = makeCopy(this->transition, inds);
                            panRes = cb(i, seg_sub, y_sub, x_sub, zmuit_sub, zuit_sub, zvit_sub, zvi0_sub, transition_sub);
                        }
                    }
                } else {
                    // cross-sectional
                    auto zmuit_sub = makeView(this->zmuit, 0, this->seg.n_rows - 1);
                    auto zuit_sub = makeView(this->zuit, 0, this->seg.n_rows - 1);
                    auto zvit_sub = makeView(this->zvit, 0, this->seg.n_rows - 1);
                    auto zvi0_sub = makeView(this->zvi0, 0, this->seg.n_rows - 1);
                    auto transition_sub = makeView(this->transition, 0, this->seg.n_rows - 1);
                    panRes = cb(i, this->seg, this->y, this->x, zmuit_sub, zuit_sub, zvit_sub, zvi0_sub, transition_sub);
                }
                vec_ele1[i] = std::move(panRes);
            } catch (const std::exception& e) {
                ctx.signalError(e);
            } catch (...) {
                ctx.signalError(std::runtime_error("Unknown error in ESADataLCM::dataCallable"));
            }
            progress.fetch_add(1, std::memory_order_relaxed);
        };

        int numIds_int = numIds;
        if (threaded) {
            BS::thread_pool<>& pool = esaparallel::getOptimPool();
            BS::multi_future<void> loop_future = pool.submit_loop(0, numIds_int, runFunc);
            esautils::waitForInterrupt(loop_future, progress, numIds, ctx);
        } else {
            for (int i = 0; i < numIds_int; i++) runFunc(i);
        }
        if (ctx.panic.load()) {
            throw std::runtime_error("Parallel execution failed: " + ctx.errMsg);
        }
        // sum over panels
        if (shouldSumEle1 && ele1) *ele1 = esautils::sumMatricies<double>(vec_ele1);
        if (!shouldSumEle1 && ele1) *ele1 = esautils::stackMatricies<double>(vec_ele1, true);
        if (shouldSumEle2 && ele2) *ele2 = esautils::sumMatricies<double>(vec_ele2);
        if (!shouldSumEle2 && ele2) *ele2 = esautils::stackMatricies<double>(vec_ele2, true);
    }
    
private:

    const arma::dmat seg;
    const std::optional<arma::Col<int>> idVec;
    const std::optional<arma::Col<int>> timeVec;
    const std::optional<arma::dmat> zmuit;
    const std::optional<arma::dmat> zuit;
    const std::optional<arma::dmat> zvit;
    const std::optional<arma::dmat> zvi0;
    const std::optional<arma::dmat> transition;

    const unsigned int nSeg;
    const unsigned int nZmuit;
    const unsigned int nZuit;
    const unsigned int nZvit;
    const unsigned int nZvi0;
    const unsigned int nLc;
    const unsigned int nTransition;
    bool isPanel;
    unsigned int nids;
    unsigned int npanels;
    bool balanced;
    const std::optional<arma::Col<int>> uniqueIds;
    int maxT;
    int minT;
    std::optional<FirmIdsVar> firmIdRows;
    bool isMarkov;

    /**
     * @details check if a time-invariant column is truely that
     * @param tol the tolerance
     */
    void checkTimeInvariance(const double tol = 1e-8) const;

    /**
     * @details find the number of columns for a matrix ptr
     * @return number of columns, 0 if not populated
     */
    unsigned int findNCols(const arma::dmat* m) const {
        if (m) return m->n_cols;
        return 0;
    }
    unsigned int findSeg() const {
        return seg.n_cols;
    }
    unsigned int findZmuit() const {
        return zmuit.has_value() ? zmuit.value().n_cols : 0;
    }
    unsigned int findZuit() const {
        return zuit.has_value() ? zuit.value().n_cols : 0;
    }
    unsigned int findZvit() const {
        return zvit.has_value() ? zvit.value().n_cols : 0;
    }
    unsigned int findZvi0() const {
        return zvi0.has_value() ? zvi0.value().n_cols : 0;
    }

    /**
     * @details validate the inputs each of the data types
     */
    std::optional<arma::dmat> validateZmuit(const std::optional<arma::dmat>& m, const ESASfaModelType mT) const;
    std::optional<arma::dmat> validateZuit(const std::optional<arma::dmat>& m, const ESASfaModelType mT) const;
    std::optional<arma::dmat> validateZvit(const std::optional<arma::dmat>& m, const ESASfaModelType mT) const;
    std::optional<arma::dmat> validateZvi0(const std::optional<arma::dmat>& m, const ESASfaModelType mT) const;

    /**
     * @details helper function for optional views without copying
     */
    template <typename T>
    std::optional<arma::subview<double>> makeView(const std::optional<T>& m, arma::uword start, arma::uword end) const {
        if (!m.has_value()) return std::nullopt;
        return std::make_optional(m.value().rows(start, end));
    }

    /**
     * @details helper function when data is not continuous - make a copy
     */
    template <typename T>
    std::optional<arma::dmat> makeCopy(const std::optional<T>& m, const arma::uvec& inds) const {
        if (!m.has_value()) return std::nullopt;
        return std::make_optional(m.value().rows(inds));
    }

    /**
     * @details calculate maxT
     */
    void calculateMaxT(const FirmIdsVar& precalc, bool isContig);
};

#endif // ESA_DATA_PANEL_LCM