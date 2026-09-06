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
 * @file logs.cpp
 * 
 */

#include <string>
#include "utils/log/logs.hpp"

#if defined(RPACKAGE)
// [[Rcpp::depends(RcppSpdlog)]]
#include <RcppSpdlog>
#else
#include <spdlog/spdlog.h>
#endif


void esautils::log::flushRingBufferToTarget(std::shared_ptr<spdlog::logger> target)
{
    // get current default logger, which holds the ring buffer sink
    auto currLogger = spdlog::default_logger();
    // access sinks vector
    if (currLogger->sinks().empty()) return;
    // iterate thru the sinks, and try to cast to ringbuffer_sink_mt
    for (auto& s : currLogger->sinks()) {
        auto ringSink = std::dynamic_pointer_cast<spdlog::sinks::ringbuffer_sink_mt>(s);
        if (ringSink) {
            // cast was successful - twas a ring buffer; can retrieve the messages
            auto bufMsgs = ringSink->last_raw();
            // iterate, and log to new target
            for (const auto& msg : bufMsgs) {
                // convert stringview to string
                std::string payload(msg.payload.data(), msg.payload.size());
                // relog
                target->log(msg.level, payload);
            }
        }
    }
}

ESALogger::ESALogger(
    bool toConsole,
    bool toFile
){

}

std::shared_ptr<spdlog::logger> ESALogger::logger() {
#ifdef RSINKLOG
    std::string log_name = "fromR";
    auto sp = spdlog::get(log_name);
    if (sp == nullptr){
        sp = spdlog::r_sink_mt(log_name);
    }
    sp.set_level(spdlog::level::trace);
    sp.set_pattern("[%H:%M:%S | td %t] [%n] [%^-%L-%$]  %v");
    spdlog::set_default_logger(sp);
    spdlog::flush_on(spdlog::level::trace);
    return spdlog::get(this->log_name.value());
#elif defined(PYPACKAGE)
    return spdlog::default_logger();
#else
    std::string log_name = "sfacpp";
    if (spdlog::get(log_name) == nullptr){
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::trace);
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("sfacpp.log", true);
        file_sink->set_level(spdlog::level::trace);
        spdlog::logger logger(log_name, {console_sink, file_sink});
        logger.set_level(spdlog::level::trace);
        // %z [thread %t] 
        logger.set_pattern("[%H:%M:%S | td %t] [%n] [%^-%L-%$]  %v");
        spdlog::set_default_logger(std::make_shared<spdlog::logger>(logger));
        spdlog::flush_on(spdlog::level::trace);
    }
    return spdlog::get(log_name);
#endif
}
