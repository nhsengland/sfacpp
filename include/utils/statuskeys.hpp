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
 * @file statuskeys.hpp
 * 
 */

#ifndef STATUS_KEYS_HPP
#define STATUS_KEYS_HPP

#include <string>

namespace ESAStatusKeys {

    const std::string kHasShownInitialError = "has_shown_initial_error";
    const std::string kHasShownTRENoAnalyticalGradTNorm = "has_shown_tre_no_analytical_grad_tnorm";
    const std::string kHasShownTFENoAnalyticalGradTNorm = "has_shown_tfe_no_analytical_grad_tnorm";
    const std::string kHasShownInitialThreadCount = "has_shown_thread_count";
    const std::string kMostRecentModelFamily = "more_recent_model_family";
}

#endif // STATUS_KEYS_HPP