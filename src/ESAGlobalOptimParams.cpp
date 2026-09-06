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
 * @file ESAGlobalOptimParams.cpp
 * @brief Singleton class to hold global optimization parameters
 * @date 2025-04-14
 * @author Edmund Haacke
 */

#include "optim/ESAGlobalOptimParams.hpp"

ESAGlobalOptimParams* ESAGlobalOptimParams::pinstance_{nullptr};
std::mutex ESAGlobalOptimParams::mutex_;
ESAGlobalOptimParams* ESAGlobalOptimParams::GetInstance()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (pinstance_ == nullptr){
        pinstance_ = new ESAGlobalOptimParams();
    }
    return pinstance_;
}
