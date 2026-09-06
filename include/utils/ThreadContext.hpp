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

#ifndef THREAD_STATE_HPP
#define THREAD_STATE_HPP

#include <memory>
#include "thread_cache/esamathTC.hpp"
#include "thread_cache/ESASfaEffGtreTC.hpp"
#include "thread_cache/ESASfaGtreBadTC.hpp"
#include "thread_cache/ESASfaTreBaseTC.hpp"
#include "thread_cache/ESASfaTreGreeneTC.hpp"
#include "thread_cache/ESASfaLcmBaseTC.hpp"

struct ThreadContext {

    // store pointers (instead of actual struct, since we might not need them all)
    // ---- esamathTC -----
    std::unique_ptr<thread_cache_math::WSGhkEstim> wsGhkEstim;
    // ---- ESASfaTreBaseTC ----
    std::unique_ptr<thread_cache::WSInternalAnalyticJacHess> treBaseJacHess;
    std::unique_ptr<thread_cache::WSWeightFromDensForPanel> treBaseWeightDens;
    // ---- ESASfaTreGreene ----
    std::unique_ptr<thread_cache::WSPanelDensityHalfNormal> treDensHNorm;
    std::unique_ptr<thread_cache::WSGradHessPanel> treGradHessPanel;
    std::unique_ptr<thread_cache::WSOperatorInner> treOpInner;
    // ---- ESASfaEffGtre ----
    std::unique_ptr<thread_cache_eff_gtre::WSGtrePanelEfficiency> gtreEffPanel;
    std::unique_ptr<thread_cache_eff_gtre::WSTrePanelEfficiency> treEffPanel;
    // ---- ESASfaGtreBad ----
    std::unique_ptr<thread_cache_gtre::WSDensityHalfNormal> gtreDensPanel;
    std::unique_ptr<thread_cache_gtre::WSGradHessInner> gtreGradHessPanel;
    // ---- ESASfaLcmBase ----
    std::unique_ptr<thread_cache_lcm::WSLcmInternalAnalyticJacHess> lcmGradPan;
    std::unique_ptr<thread_cache_lcm::WSLcmInternalAnalyticJacHessTN> lcmGradPanTN;
    


    ThreadContext() {

    }

    // called when thread dies
    ~ ThreadContext() {
        // unique_ptr is autocleaned
    }

};

ThreadContext* getContext();
// system calls - from pybind module entrypoint
void initializeThreadContext();
void teardownThreadContext();
void freeCurrentThreadContext();

#endif // THREAD_STATE_HPP