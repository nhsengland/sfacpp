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
 * @file bfgs.hpp
 * @date 2025-12-29
 * @details combine dlib bfgs with more thuente line search
 */

#ifndef ESA_SFA_BFGS_HPP
#define ESA_SFA_BFGS_HPP

#include <string>
#include <memory>
#include <spdlog/spdlog.h>
#ifdef WITHDLIB
#include <dlib/matrix.h>
#endif //WITHDLIB

#include "optim/MoreThuente.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"

namespace optim {

#ifdef WITHDLIB

    /**
     * @brief Helper function to construct lbfgs/bfgs search strategy
     */
    template <typename T>
    inline std::unique_ptr<T> createStrategy() {
        return std::unique_ptr<T>(new T());
    }
    
    /**
     * @brief Helper function to construct L-BFGS with memory size argument
     */
    template <>
    inline std::unique_ptr<dlib::lbfgs_search_strategy> createStrategy<dlib::lbfgs_search_strategy>()
    {
        return std::unique_ptr<dlib::lbfgs_search_strategy>(new dlib::lbfgs_search_strategy(10));
    }

    /**
     * @brief solver for both bfgs and lbfgs method
     * @tparam fn
     * @tparam fn_der
     * @tparam strat
     */
    template <typename fn, typename fn_der, typename strat>
    inline double BfgsLbfgsSolve(
        fn& f,
        fn_der& der,
        dlib::matrix<double, 0, 1>& x,
        int& finalIter,
        const int printLevel = 1,
        int maxit = 500,
        double gnormTol = 1e-6
    )
    {
        std::unique_ptr<strat> strategy = createStrategy<strat>();
        // function evaluated at param
        double fval = f(x);
        // gradient evaluated at param
        dlib::matrix<double, 0, 1> g = der(x);
        // store direction vector
        dlib::matrix<double, 0, 1> direction;
        int iter = 0;
        // run iterations
        while (iter < maxit) {
            // check convergence based on gradient norm
            if (dlib::max(dlib::abs(g)) < gnormTol) {
                // converged
                break;
            }
            // get search direction - based on BFGS approximation of the hessian
            direction = strategy->get_next_direction(x, fval, g);
            // line-search
            // double alpha = 1.0;
            double alpha = (iter == 0) ? 0.1 : 1.0; // Soft start
            double fold = fval;
            // buffer for next gradient
            dlib::matrix<double, 0, 1> gNext;
            // update alpha to optimal step size
            fval = linesearch::StrongWolfeLineSearch(f, der, x, direction, fval, g, alpha, gNext);
            // update position; if linesearch failed (alpha = 0), need to reset strategy
            if (alpha == 0.0) {
                // check that the gradient is finite (aka not inf/nan) [since it can't be used for steepest descent]
                if (!dlib::is_finite(g)) {
                    // unrecoverable
                    ESALogger::logger()->error("Gradient is not finite during reset strategy.");
                    break;
                }
                // reset bfgs memory
                strategy = createStrategy<strat>();
                // steepest descent
                direction = -g;
                // retry
                alpha = 1.0;
                double c1 = 1e-4;
                double gdotd = dlib::dot(g, direction);
                // backtracking
                int backit = 0;
                bool backSuccess = false;
                while (backit < 20) {
                    dlib::matrix<double, 0, 1> xTry = x + alpha * direction;
                    // check that the step didn't create NaN params
                    if (!dlib::is_finite(xTry)) {
                        // if it did - reduce the step size
                        alpha *= 0.5;
                        backit++;
                        continue;
                    }
                    // evaluate function at these parameters
                    double ftry = f(xTry);
                    // ensure that the log-likelihood is finite (e.g., not NaN/Inf) 
                    if (std::isfinite(ftry) && ftry < fold + c1 * alpha * gdotd) {
                        fval = ftry;
                        backSuccess = true;
                        break;
                    }
                    alpha *= 0.5;
                    backit++;
                }
                if (!backSuccess) {
                    ESALogger::logger()->error("Backtracking failed.");
                    break;
                }
            } else {
                // success 
                // update param with step
                x = x + alpha * direction;
                g = std::move(gNext);
            }
            
            // check for stagation
            if (std::abs(fval - fold) < 1e-10 && dlib::max(dlib::abs(g)) > gnormTol) {
                break;
            }
            // if in python, connect to the logger
            #ifdef PYPACKAGE
            // if in python, we need to reconnect to the python logger, since we logged to a ring
            // buffer to prevent potential GIL issues.
            // check if there were any buffered messages, if so, flush to python
            esautils::log::flushRingBufferToTarget(spdlog::get("python_bridge"));
            // set the default logger back to the python bridge, so the following are printed
            spdlog::set_default_logger(spdlog::get("python_bridge"));
            spdlog::drop("buffer_logger");
            #endif // PYPACKAGE
            // print
            if (printLevel >= 1) {
                ESALogger::logger()->info("Iteration {:>5} (hessian is quasi-newton):\t\t log-likelihood = {:>13.6f}", iter, fval);
            }
            if (printLevel >= 2) ESALogger::logger()->info("• θ: {}", x);
            if (printLevel >= 3) ESALogger::logger()->info("• ∇: {}", g);
            if (printLevel >= 3) ESALogger::logger()->info("────────────────────────────────────────");
            #ifdef PYPACKAGE
            // dont think there is away to flush the ringsink, so create a new one
            auto ringSink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(1024);
            auto bufferLogger = std::make_shared<spdlog::logger>("buffer_logger", ringSink);
            spdlog::set_default_logger(bufferLogger);
            #endif // PYPACKAGE
            // increment the iteration count
            iter++;
        }
        finalIter = iter;
        return fval;
    }

    template <typename fn, typename fn_der>
    inline double LbfgsSolve(
        fn& f,
        fn_der& der,
        dlib::matrix<double, 0, 1>& x,
        int& finalIter,
        const int printLevel = 1,
        const int maxit = 500,
        const double gnormTol = 1e-6
    )
    {
        return BfgsLbfgsSolve<fn, fn_der, dlib::lbfgs_search_strategy>(f, der, x, finalIter, printLevel, maxit, gnormTol);
    }

    template <typename fn, typename fn_der>
    inline double BfgsSolve(
        fn& f,
        fn_der& der,
        dlib::matrix<double, 0, 1>& x,
        int& finalIter,
        const int printLevel = 1,
        const int maxit = 500,
        const double gnormTol = 1e-6
    )
    {
        return BfgsLbfgsSolve<fn, fn_der, dlib::bfgs_search_strategy>(f, der, x, finalIter, printLevel, maxit, gnormTol);
    }

#endif // WITHDLIB

}

#endif // ESA_SFA_BFGS_HPP