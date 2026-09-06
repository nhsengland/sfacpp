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
 * @file logsobj.hpp
 */

#ifndef LOGS_OBJ_HPP
#define LOGS_OBJ_HPP

#include <type_traits>
// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif
// --- end armadillo ---
// ---- spdlogger ----
#if defined(RPACKAGE)
// [[Rcpp::depends(RcppSpdlog)]]
#include <spdlog/fmt/bundled/base.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/fmt/ostr.h>
// #elif defined(LOCAL_TEST_BUILD)
#else
#include <spdlog/fmt/bundled/base.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/fmt/ostr.h>
#endif
// ---- import classes ----
#include "marginaleffects/ESASfaMeff.hpp"

// template <>
// struct fmt::formatter<ESASfaMeffReturn> {
//     constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
//         return ctx.begin();
//     }
//     auto format(const ESASfaMeffReturn& o, format_context& ctx) const -> format_context::iterator {
//         auto out = fmt::format_to(ctx.out(), )
//     }
// };


#endif // LOGS_OBJ_HPP