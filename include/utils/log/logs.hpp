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
 * @file logs.hpp
 * @brief Header file for logging functions
 */

#ifndef LOGS_HPP
#define LOGS_HPP

#include <iostream>
#include <fstream>
#include <memory>
#include <optional>

#if defined(RPACKAGE)
// [[Rcpp::depends(RcppSpdlog)]]
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ringbuffer_sink.h>
// #elif defined(LOCAL_TEST_BUILD)
#else
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#endif

namespace esautils::log {
    void flushRingBufferToTarget(std::shared_ptr<spdlog::logger> target);
}

class ESALogger {


public:

    ESALogger(
        bool toConsole = true,
        bool toFile = true
    );

    static std::shared_ptr<spdlog::logger> logger();
};

#endif // LOGS_HPP