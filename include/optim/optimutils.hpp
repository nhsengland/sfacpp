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

#ifndef OPTIM_UTILS_HPP
#define OPTIM_UTILS_HPP

#include <map>
#include <string>
#include <variant>

#ifdef WITHOPTIMLIB
#define OPTIM_USE_OPENMP
#define OPTIM_ENABLE_ARMA_WRAPPERS
#include "optim.hpp"
#endif // WITHOPTIMLIB

#include "optim/optimparams.hpp"

namespace optim_utils {

    ESAOptimParams optimParamsForMap(const std::map<std::string, std::variant<std::string, double, unsigned int, int>>& m);

    // ---- optim lib utilities ----
    #ifdef WITHOPTIMLIB

    optim::algo_settings_t optimSettingsForParams(const ESAOptimParams& p);

    #endif // WITHOPTIMLIB

}

#endif // OPTIM_UTILS_HPP