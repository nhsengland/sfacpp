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
 * @file interrupts.hpp
 * @author edmund haacke
 * @date 2025-12-26
 * @details Handle interrupts from R (pass thru for other platforms)
 */

#ifndef ESA_UTILS_INTERRUPTS_HPP
#define ESA_UTILS_INTERRUPTS_HPP

#include <atomic>
#include <thread>
#include <chrono>
#include "utils/esaparallel.hpp"

#ifdef RPACKAGE
#include <Rcpp.h>
#endif

namespace esautils {

    /**
     * @brief Wait for future to complete. On R, it also polls for user interrupts
     * @tparam FutureType - type of the BS::thread_pool future
     * @param future The future object returned by submit_loop
     * @param reference to atomic counter tracking completed tasks
     * @param total The total number of tasks to complete
     * @param ctx The context obj used to signal panic/stop to worker threads
     */
    template <typename FutureType>
    inline void waitForInterrupt(FutureType& future, std::atomic<size_t>& progress, size_t total, esaparallel::ParallelTaskContext& ctx)
    {
    #ifdef RPACKAGE
        // R-specific implementation - pool for interrupts whilst waiting
        while (progress.load(std::memory_order_relaxed) < total) {
            // check if already workers already paniced - then exit the loop
            if (ctx.panic.load(std::memory_order_relaxed)) {
                break;
            }
            // check for R interrupt
            try {
                Rcpp::checkUserInterrupt();
            } catch (...) {
                // there was an interrupt - signal workers to panic
                ctx.panic.store(true, std::memory_order_relaxed);
                // wait for workers to panic
                future.wait();
                // re-throw exception
                throw;
            }
            // sleep for 50ms
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        // final wait 
        future.wait();
    #else
        // non-R environment - just block main thread, using standard bs threadpool wait
        future.wait();
    #endif //
    }

}

#endif // ESA_UTILS_INTERRUPTS_HPP