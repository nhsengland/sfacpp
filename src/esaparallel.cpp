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
 * @file esaparallel.cpp
 * @brief Implementation file for parallel processing
 */

#include <thread>
#include <stdexcept>
#include <optional>
#include <cmath>
#include "utils/esaparallel.hpp"
#include "utils/log/logs.hpp"
#include "utils/statuskeys.hpp"
#include "utils/ESASingletonStatuses.hpp"
#include "utils/ThreadContext.hpp"

BS::thread_pool<>& esaparallel::getOptimPool()
{
    // static - ensures only initialized the first time
    int able = std::thread::hardware_concurrency();
    static BS::thread_pool threadPool(floor(able * 0.8));
    ESASingletonStatuses* stati = ESASingletonStatuses::GetInstance();
    if (!stati->getStatus(ESAStatusKeys::kHasShownInitialThreadCount)){
        stati->setStatus(ESAStatusKeys::kHasShownInitialThreadCount, true);
    }
    return threadPool;
}

void esaparallel::modelChangeFlushUnneededTLS(const ESASfaModelFamily newModelFamily)
{
    // check if the global key for model family has been set - if not - first time running (since that is static)
    ESASingletonStatuses* stati = ESASingletonStatuses::GetInstance();
    int mF = ESAEnums::modelFamilyToInt(newModelFamily);
    if (stati->getStatusInt(ESAStatusKeys::kMostRecentModelFamily)){
        // there is an existing key 
        std::optional<int> optOld = stati->getStatusInt(ESAStatusKeys::kMostRecentModelFamily);
        if (!optOld) throw std::runtime_error("Did not find anything for key");
        int old = optOld.value();
        // if its the same as the new model family, don't need to do anything, return
        if (mF == old) return;
        // different 
        // try to convert to a modelFamily
        ESASfaModelFamily mFOld = ESAEnums::intToModelFamily(old);
        // depending on the new family, can destroy some of the old TLS structs to get some memory back
        // get the thread context 
        ThreadContext* ctx = getContext();
        if (mFOld == ESASfaModelFamily::TRE) {
            // the old model was TRE
            ctx->treDensHNorm.reset();
            ctx->treGradHessPanel.reset();
            ctx->treOpInner.reset();
        } else if (mFOld == ESASfaModelFamily::GTRE) {
            // old model was GTRE
        }
        // if the new family is neither TRE or GTRE, can also release the struct related to jac/hess since both use the same
        if (newModelFamily != ESASfaModelFamily::TRE && newModelFamily != ESASfaModelFamily::GTRE) {
            ctx->treBaseJacHess.reset();
        }
    }
    stati->setStatusInt(ESAStatusKeys::kMostRecentModelFamily, mF);
}

void esaparallel::setThreadCount(const int nthreads)
{
    if (nthreads < 1) throw std::invalid_argument("'nthreads' must be at least 1");
    // get the static threadpool
    BS::thread_pool<>& pool = getOptimPool();
    int tc = pool.get_thread_count();
    int able = std::thread::hardware_concurrency();
    bool shouldChangeNthreads = false;
    // first, check if we were already at max threadcount
    if (tc == able && nthreads > able){
        ESALogger::logger()->warn("Already at system limit of {}, {} is not possible. Keeping existing pool.", able, nthreads);
    } else if (nthreads != tc){
        // otherwise, can change threads
        shouldChangeNthreads = true;
    }
    // if the proposed number of threads is the same, then do nothing
    if (shouldChangeNthreads) {
        // check that the new number of threads of 
        int accUse = (nthreads > able) ? able : nthreads;
        // set the number of threads - let thread_local approach means c++ will handle destruction
        // of memory related to the unique_ptrs per thread
        pool.reset(accUse);
        if (nthreads > able) {
            ESALogger::logger()->warn("Targetted {} threads, but system only allows {}. Using {} instead.", nthreads, able, pool.get_thread_count());
        } else {
            ESALogger::logger()->info("Using {} threads", pool.get_thread_count());
        }
    }
}

/// via https://stackoverflow.com/a/61137359
void esaparallel::thread_par_for(unsigned start, unsigned end, std::function<void(unsigned i)> func, bool par){
    auto int_fn = [&func](unsigned int_start, unsigned seg_size){
        for (unsigned j = int_start; j < int_start + seg_size; j++){
            func(j);
        }
    };
    if (!par){
        return int_fn(start, end);
    }
    // number of threads
    unsigned nb_threads_hint = std::thread::hardware_concurrency();
    unsigned nb_threads = nb_threads_hint == 0 ? 8 : nb_threads_hint;
    // segments
    unsigned total_length = end - start;
    unsigned seg = total_length/nb_threads;
    unsigned last_seg = seg + total_length % nb_threads;
    // create threads
    auto threads_vec = std::vector<std::thread>();
    threads_vec.reserve(nb_threads);
    for (int k = 0; k < (nb_threads - 1); k++){
        unsigned current_start = seg*k;
        threads_vec.emplace_back(std::thread(int_fn, current_start, seg));
    }
    {
        unsigned current_start = seg*(nb_threads - 1);
        threads_vec.emplace_back(std::thread(int_fn, current_start, last_seg));
    }
    for (auto& th : threads_vec){
        th.join();
    }
}

/// via https://stackoverflow.com/a/61137359
void esaparallel::async_par_for(unsigned start, unsigned end, std::function<void(unsigned i)> func, bool par){
    auto int_fn = [&func](unsigned int_start, unsigned seg_size){
        for (unsigned j = int_start; j < (int_start + seg_size); j++){
            func(j);
        }
    };
    if (!par){
        return int_fn(start, end);
    }
    // number of threads
    unsigned nb_threads_hint = std::thread::hardware_concurrency();
    unsigned nb_threads = nb_threads_hint == 0 ? 8 : nb_threads_hint;
    // segments
    unsigned total_length = end - start;
    unsigned seg = total_length/nb_threads;
    unsigned last_seg = seg + total_length % nb_threads;
    // create threads
    auto futures_vec = std::vector<std::future<void>>();
    futures_vec.reserve(nb_threads);
    for (int k = 0; k < (nb_threads - 1); k++){
        unsigned current_start = seg*k;
        futures_vec.emplace_back(std::async(int_fn, current_start, seg));
    }
    {
        unsigned current_start = seg*(nb_threads - 1);
        futures_vec.emplace_back(std::async(std::launch::async, int_fn, current_start, last_seg));
    }
    for (auto& th : futures_vec){
        th.get();
    }
}