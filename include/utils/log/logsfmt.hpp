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
 * @file logsfmt.hpp
 */

#ifndef LOGS_FMT_HPP
#define LOGS_FMT_HPP

// [[Rcpp::depends(BH)]]
#include <boost/optional.hpp>
#include <optional>
#include <type_traits>
#include <vector>
#include <string>

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

#if defined(RPACKAGE)
// [[Rcpp::depends(RcppSpdlog)]]
#include <spdlog/fmt/bundled/base.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/fmt/ostr.h>
#elif defined(LOCAL_TEST_BUILD)
#include <spdlog/fmt/bundled/base.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/fmt/ostr.h>
#endif

// only relevant for cppnumsolvers
#if defined(WITHCPPNUMSOLVERS)
#include "cppoptlib/solver/solver.h"
#endif

#ifdef WITHDLIB
#include <dlib/matrix.h>
#endif // WITHDLIB

#ifdef WITHEIGEN
#ifdef RPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include "Eigen/Core"
#endif // RPACKAGE
#endif //WITHEIGEN


#ifdef WITHDLIB
template <typename T>
struct fmt::formatter<dlib::matrix<T, 1, 0>>{
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const dlib::matrix<T, 1, 0>& m, format_context& ctx) const -> format_context::iterator {
        // auto out = fmt::format_to(ctx.out(), "dlib::matrix<{},1,0> [{}]\t", typeid(T).name(), m.size());
        auto out = fmt::format_to(ctx.out(), "[{}] ", m.size());
        for (int i = 0; i < m.size(); i++){
            out = fmt::format_to(out, fmt::runtime("{:+.5f} "), m(i));
        }
        return out;
    }
};
template struct fmt::formatter<dlib::matrix<double, 1, 0>>;
template struct fmt::formatter<dlib::matrix<float, 1, 0>>;
template struct fmt::formatter<dlib::matrix<int, 1, 0>>;
template struct fmt::formatter<dlib::matrix<unsigned int, 1, 0>>;
#endif // WITHDLIB

#ifdef WITHDLIB
template <typename T>
struct fmt::formatter<dlib::matrix<T, 0, 1>>{
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const dlib::matrix<T, 0, 1>& m, format_context& ctx) const -> format_context::iterator {
        // auto out = fmt::format_to(ctx.out(), "dlib::matrix<{},0,1> [{}]\t", typeid(T).name(), m.size());
        auto out = fmt::format_to(ctx.out(), "[{}] ", m.size());
        for (int i = 0; i < m.size(); i++){
            out = fmt::format_to(out, fmt::runtime("{:+.5f} "), m(i));
        }
        return out;
    }
};
template struct fmt::formatter<dlib::matrix<double, 0, 1>>;
template struct fmt::formatter<dlib::matrix<float, 0, 1>>;
template struct fmt::formatter<dlib::matrix<int, 0, 1>>;
template struct fmt::formatter<dlib::matrix<unsigned int, 0, 1>>;
#endif // WITHDLIb

#ifdef WITHDLIB
template <typename T>
struct fmt::formatter<dlib::matrix<T>>{
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const dlib::matrix<T>& m, format_context& ctx) const -> format_context::iterator {
        // transpose the matrix if it is a column vector
        // auto out = fmt::format_to(ctx.out(), "dlib::matrix<{}> {} [{} x {}]\n", typeid(T).name(), (m.nc() == 1 ? "(transposed)" : ""), m.nr(), m.nc());
        auto out = fmt::format_to(ctx.out(), "[{}x{}] ", m.nr(), m.nc());
        for (int i = 0; i < m.nr(); i++){
            for (int j = 0; j < m.nc(); j++){
                out = fmt::format_to(out, fmt::runtime("{:+.5f} "), m(i, j));
            }
            if (m.nc() > 1) out = fmt::format_to(out, "\n");
        }
        return out;
    }
};
template struct fmt::formatter<dlib::matrix<double>>;
template struct fmt::formatter<dlib::matrix<float>>;
template struct fmt::formatter<dlib::matrix<int>>;
#endif // WITHDLIB


#ifdef WITHDLIB
template <typename T>
struct fmt::formatter<boost::optional<dlib::matrix<T>>>{
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const boost::optional<dlib::matrix<T>>& m, format_context& ctx) const -> format_context::iterator {
        if (m){
            return fmt::format_to(ctx.out(), fmt::runtime("{}"), m.get());
        }
        return fmt::format_to(ctx.out(), "boost::none");
    }
};
template struct fmt::formatter<boost::optional<dlib::matrix<double>>>;
template struct fmt::formatter<boost::optional<dlib::matrix<float>>>;
template struct fmt::formatter<boost::optional<dlib::matrix<int>>>;
#endif // WITHDLIB

#ifdef WITHDLIB
template <typename T>
struct fmt::formatter<boost::optional<dlib::matrix<T, 1, 0>>>{
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const boost::optional<dlib::matrix<T, 1, 0>>& m, format_context& ctx) const -> format_context::iterator {
        if (m){
            return fmt::format_to(ctx.out(), fmt::runtime("{}"), m.get());
        }
        return fmt::format_to(ctx.out(), "boost::none");
    }
};
template struct fmt::formatter<boost::optional<dlib::matrix<double, 1, 0>>>;
template struct fmt::formatter<boost::optional<dlib::matrix<float, 1, 0>>>;
template struct fmt::formatter<boost::optional<dlib::matrix<int, 1, 0>>>;
#endif // WITHDLIB

#ifdef WITHDLIB
template <typename T>
struct fmt::formatter<boost::optional<dlib::matrix<T, 0, 1>>>{
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const boost::optional<dlib::matrix<T, 0, 1>>& m, format_context& ctx) const -> format_context::iterator {
        if (m){
            return fmt::format_to(ctx.out(), fmt::runtime("{}"), m.get());
        }
        return fmt::format_to(ctx.out(), "boost::none");
    }
};
template struct fmt::formatter<boost::optional<dlib::matrix<double, 0, 1>>>;
template struct fmt::formatter<boost::optional<dlib::matrix<float, 0, 1>>>;
template struct fmt::formatter<boost::optional<dlib::matrix<int, 0, 1>>>;
#endif // WITHDLIB

#ifdef WITHEIGEN
template <typename T>
struct fmt::formatter<Eigen::Matrix<T, -1, 1>>{
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const Eigen::Matrix<T, -1, 1>& m, format_context& ctx) const -> format_context::iterator {
        auto out = fmt::format_to(ctx.out(), "Eigen::Matrix<{}> {} [{} x {}]\n", typeid(T).name(), (m.cols() == 1 ? "(transposed)" : ""), m.rows(), m.cols());
        for (int i = 0; i < m.rows(); i++){
            for (int j = 0; j < m.cols(); j++){
                out = fmt::format_to(out, fmt::runtime("{:+.5f} "), m(i, j));
            }
            if (m.cols() > 1) out = fmt::format_to(out, "\n");
        }
        return out;
    }
};
template struct fmt::formatter<Eigen::Matrix<double, -1, 1>>;

#endif // WITHEIGEN

#ifdef WITHCPPNUMSOLVERS
template <>
struct fmt::formatter<cppoptlib::solver::Status>{
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    
    auto format(const cppoptlib::solver::Status& s, format_context& ctx) const -> format_context::iterator {
        switch (s) {
            case cppoptlib::solver::Status::NotStarted:
                return fmt::format_to(ctx.out(), "Solver not started.");
                break;
            case cppoptlib::solver::Status::Continue:
                return fmt::format_to(ctx.out(), "Convergence criteria not reached.");
                break;
            case cppoptlib::solver::Status::IterationLimit:
                return fmt::format_to(ctx.out(), "Iteration limit reached.");
                break;
            case cppoptlib::solver::Status::XDeltaViolation:
                return fmt::format_to(ctx.out(), "Change in parameter vector too small.");
                break;
            case cppoptlib::solver::Status::FDeltaViolation:
                return fmt::format_to(ctx.out(), "Change in cost function value too small.");
                break;
            case cppoptlib::solver::Status::GradientNormViolation:
                return fmt::format_to(ctx.out(), "Gradient vector norm too small.");
                break;
            case cppoptlib::solver::Status::HessianConditionViolation:
                return fmt::format_to(ctx.out(), "Condition of Hessian / Covariance matrix too large.");
                break;
            case cppoptlib::solver::Status::Finished:
                return fmt::format_to(ctx.out(), "Finished.");
                break;
        }
        return fmt::format_to(ctx.out(), "unknown");
    }
};
// template struct fmt::formatter<cppoptlib::solver::Status>;
#endif // WITHCPPNUMSOLVERS


template <typename T>
struct fmt::formatter<arma::Col<T>>{
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const arma::Col<T>& m, format_context& ctx) const -> format_context::iterator {
        // auto out = fmt::format_to(ctx.out(), "arma::Col<{}> [{}]  ", typeid(T).name(), m.n_rows);
        auto out = fmt::format_to(ctx.out(), "[{}] ", m.n_rows);
        for (size_t i = 0; i < m.n_rows; i++) {
            if constexpr (
                std::is_same<T, int>::value || std::is_same<T, unsigned int>::value || std::is_same<T, arma::uword>::value
            ) {
                // out = fmt::format_to(out, fmt::runtime("{:} "), m(i));
                out = fmt::format_to(out, "{:} ", m(i));
            } else {
                // out = fmt::format_to(out, fmt::runtime("{:+.5f} "), m(i));
                out = fmt::format_to(out, "{:.5f} ", m(i));
            }
        }
        return out;
    }
};
template struct fmt::formatter<arma::Col<double>>;
template struct fmt::formatter<arma::Col<float>>;
template struct fmt::formatter<arma::Col<int>>;
template struct fmt::formatter<arma::Col<unsigned int>>;
// template struct fmt::formatter<arma::Col<arma::uword>>;

template <typename T>
struct fmt::formatter<arma::Row<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const arma::Row<T>& m, format_context& ctx) const -> format_context::iterator {
        // auto out = fmt::format_to(ctx.out(), "arma::Row<{}> [{}]  ", typeid(T).name(), m.n_cols);
        auto out = fmt::format_to(ctx.out(), "[{}] ", m.n_cols);
        for (size_t c = 0; c < m.n_cols; c++) {
            if constexpr (
                std::is_same<T, int>::value || std::is_same<T, unsigned int>::value
            ) {
                // out = fmt::format_to(out, fmt::runtime("{:} "), m(c));
                out = fmt::format_to(out, "{:} ", m(c));
            } else {
                // out = fmt::format_to(out, fmt::runtime("{:+.5f} "), m(c));
                out = fmt::format_to(out, "{:.5f} ", m(c));
            }
        }
        return out;
    }
};
template struct fmt::formatter<arma::Row<double>>;
template struct fmt::formatter<arma::Row<float>>;
template struct fmt::formatter<arma::Row<int>>;
template struct fmt::formatter<arma::Row<unsigned int>>;


template <typename T>
struct fmt::formatter<arma::Mat<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const arma::Mat<T>& m, format_context& ctx) const -> format_context::iterator {
        // transpose if a column vector
        // auto out = fmt::format_to(ctx.out(), "arma::Mat<{}> {}[{} x {}]", typeid(T).name(), (m.n_cols == 1 ? "(transposed)" : ""), m.n_rows, m.n_cols);
        auto out = fmt::format_to(ctx.out(), "[{}x{}] ", m.n_rows, m.n_cols);
        if (m.n_cols > 1 && m.n_rows > 1) out = fmt::format_to(out, "\n");
        else out = fmt::format_to(out, "  ");
        for (size_t r = 0; r < m.n_rows; r++) {
            for (size_t c = 0; c < m.n_cols; c++) {
                if constexpr (
                    std::is_same<T, int>::value || std::is_same<T, unsigned int>::value
                ) {
                    out = fmt::format_to(out, "{:} ", m(r, c));
                } else {
                    out = fmt::format_to(out, "{:.5f} ", m(r, c));
                }
            }
            if (m.n_cols > 1 && m.n_rows > 1) out = fmt::format_to(out, "\n");
        }
        return out;
    }
};
template struct fmt::formatter<arma::Mat<double>>;
template struct fmt::formatter<arma::Mat<float>>;
template struct fmt::formatter<arma::Mat<int>>;
template struct fmt::formatter<arma::Mat<unsigned int>>;

template <typename T>
struct fmt::formatter<std::optional<arma::Mat<T>>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const std::optional<arma::Mat<T>>& m, format_context& ctx) const -> format_context::iterator {
        if (m){
            // return fmt::format_to(ctx.out(), fmt::runtime("{}"), m.value());
            return fmt::format_to(ctx.out(), "{}", m.value());
        }
        return fmt::format_to(ctx.out(), "std::nullopt");
    }
};
template struct fmt::formatter<std::optional<arma::Mat<double>>>;
template struct fmt::formatter<std::optional<arma::Mat<float>>>;
template struct fmt::formatter<std::optional<arma::Mat<int>>>;
template struct fmt::formatter<std::optional<arma::Mat<unsigned int>>>;

template <typename T>
struct fmt::formatter<std::optional<arma::Col<T>>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const std::optional<arma::Col<T>>& m, format_context& ctx) const -> format_context::iterator {
        // if (m) return fmt::format_to(ctx.out(), fmt::runtime("{}"), m.value());
        if (m) return fmt::format_to(ctx.out(), "{}", m.value());
        return fmt::format_to(ctx.out(), "std::nullopt");
    }
};
template struct fmt::formatter<std::optional<arma::Col<double>>>;
template struct fmt::formatter<std::optional<arma::Col<float>>>;
template struct fmt::formatter<std::optional<arma::Col<int>>>;
template struct fmt::formatter<std::optional<arma::Col<unsigned int>>>;

template <typename T>
struct fmt::formatter<std::optional<arma::Row<T>>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const std::optional<arma::Row<T>>& m, format_context& ctx) const -> format_context::iterator {
        // if (m) return fmt::format_to(ctx.out(), fmt::runtime("{}"), m.value());
        if (m) return fmt::format_to(ctx.out(), "{}", m.value());
        return fmt::format_to(ctx.out(), "std::nullopt");
    }
};
template struct fmt::formatter<std::optional<arma::Row<double>>>;
template struct fmt::formatter<std::optional<arma::Row<float>>>;
template struct fmt::formatter<std::optional<arma::Row<int>>>;
template struct fmt::formatter<std::optional<arma::Row<unsigned int>>>;

/// ---- vectors ----
template <>
struct fmt::formatter<std::vector<std::string>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const std::vector<std::string>& v, format_context& ctx) const -> format_context::iterator {
        auto out = fmt::format_to(ctx.out(), "std::vector<std::string> [{}] \t [", v.size());
        for (const auto& x : v) {
            // out = fmt::format_to(out, fmt::runtime("{}, "), x);
            out = fmt::format_to(out, "{}, ", x);
        }
        out = fmt::format_to(out, "]");
        return out;
    }
};

template <typename T>
struct fmt::formatter<std::vector<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.begin();
    }
    auto format(const std::vector<T>& v, format_context& ctx) const -> format_context::iterator {
        auto out = fmt::format_to(ctx.out(), "std::vector<{}> [{}] \t [", typeid(T).name(), v.size());
        for (const auto& x : v) {
            if constexpr (
                std::is_same<T, int>::value || std::is_same<T, unsigned int>::value
            ) {
                // out = fmt::format_to(out, fmt::runtime("{:}, "), x);
                out = fmt::format_to(out, "{:}, ", x);
            } else {
                // out = fmt::format_to(out, fmt::runtime("{:+.5f}, "), x);
                out = fmt::format_to(out, "{:+.5f}, ", x);
            }
        }
        out = fmt::format_to(out, "]");
        return out;
    }
};

#endif // LOGS_FMT_HPP