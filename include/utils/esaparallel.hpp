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
 * @file esaparallel.hpp
 * @brief Header file for parallel processing
 * 
 */

#ifndef ESA_PARALLEL_HPP
#define ESA_PARALLEL_HPP

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <functional>
#include <future>
#include <BS_thread_pool.hpp>
#include "utils/enums.hpp"

namespace esaparallel {

    struct ParallelTaskContext {
        std::atomic<bool> panic{false};
        std::mutex errorMutex;
        std::string errMsg;

        void signalError(const std::exception& e) {
            bool expected = false;
            if (panic.compare_exchange_strong(expected, true)) {
                // write the error message
                std::lock_guard<std::mutex> lock(errorMutex);
                errMsg = e.what();
            }
        }
    };

    void thread_par_for(unsigned start, unsigned end, std::function<void(unsigned i)> func, bool par = true);

    void async_par_for(unsigned start, unsigned end, std::function<void(unsigned i)> func, bool par = true);

    BS::thread_pool<>& getOptimPool();

    /**
     * @brief 
     * @details Release memory allocated to thread-local storage where model family changes
     * @param newModelFamily
     */
    void modelChangeFlushUnneededTLS(const ESASfaModelFamily newModelFamily);

    /**
     * @brief set bs thread_pool number of threads
     * @details Set the number of threads
     * @param nthreads the desired number of threads (capped to system available)
     */
    void setThreadCount(const int nthreads);
}

#endif // ESA_PARALLEL_HPP