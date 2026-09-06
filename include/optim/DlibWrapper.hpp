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
 * @brief Wrapper class for dlib optimization algorithms
 * @author edmund haacke
 * @date 2025-11-08
 */

#ifndef DLIB_WRAPPER_HPP
#define DLIB_WRAPPER_HPP

#include <memory>
#include <string>
#include "sfa/ESASfaBase.hpp"
#include "utils/enums.hpp"
#include "utils/dlib2arma.h"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "utils/excepts.hpp"
#include "utils/globalstate.hpp"

#ifdef WITHDLIB
#include <dlib/matrix.h>
#endif //WITHDLIB

#ifdef PYPACKAGE
#include <spdlog/spdlog.h>
#endif

#if defined(WITHDLIB)

class DlibWrapper
{
public:
    // define type used to represent column vectors
    typedef dlib::matrix<double, 0, 1> column_vector;
    // define type used to represent hessian matrix
    typedef dlib::matrix<double> general_matrix;

    /**
     * @brief Constructor for class
     * @param f shared pointer to instance of SFA class
     * @param meth enumeration what approach to take to calculate hessian
     * @param analyticalGrad whether or not to use analytical gradient
     * @param numApproxAcc integer between 0-3 denoting how accurate to use num approx (if set in 'meth')
     * @param printLevel print level
     * @param threaded boolean whether or not to use threading when calculating grad & hess
     */
    DlibWrapper(
        std::shared_ptr<ESASfaBase> f,
        const HessianCalcMethod meth,
        const unsigned int printLevel = 1,
        const bool analyticalGrad = true,
        const unsigned int numApproxAcc = 0,
        const bool threaded = true,
        const int maxit = 100,
        const bool incrementIter = true
    ) : _f(f),
        _meth(meth),
        _analyticalGrad(analyticalGrad),
        _acc(numApproxAcc),
        _printlvl(printLevel),
        _threaded(threaded),
        _maxit(maxit),
        incrementCounter(incrementIter)
    {
        iter = std::make_unique<int>(0);
        currLL = std::make_unique<double>(0.0);
    }

    /**
     * @brief Calculate value of objective function - log likelihood score
     * @param x parameter vector to calculate log likelihood score at
     * @return double of the ll score
     */
    double operator()(const column_vector& x) const
    {
        if (globalstate::InterruptRequested.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Interrupted");
        }
        // check if hit the maximum number of iterations
        if (*iter >= _maxit) {
            throw std::runtime_error("Maximum number of iterations " + std::to_string(_maxit) + " reached.");
        }
        // wrap dlib column vector in armadillo column vector (no copy) // force a copy
        arma::Col<double> par = wrap_dlib_colvec_to_arma<double>(x, true);
        // calculate likelihood function
        double ll = _f->operator()(par);
        if (incrementCounter) *iter = *iter + 1;
        *currLL = ll;
        return ll;
    }
    
    /**
     * @brief Calculate only the derivated at a given parameter vector (required for BFGS/L-BFGS)
     * @param x parameter vector
     * @return column_vector gradient
     */
    column_vector derivative(const column_vector& x) const
    {
        // write dlib column vector into armadillo (no copy)
        arma::Col<double> par = wrap_dlib_colvec_to_arma(x, true);
        arma::dmat g1;
        // still call grad hess, but dont calculate the hessian; raise error if the gradient/hessian
        // are not finite
        _f->gradHess(par, true, &g1, nullptr);
        column_vector d = copy_arma_colvec_to_dlib(g1.t());
        return d;
    }

    column_vector derivated_negated(const column_vector& x) const
    {
        arma::Col<double> par = wrap_dlib_colvec_to_arma(x, true);
        arma::dmat g1;
        // still call grad hess, but dont calculate the hessian (again, because this is for 
        // bfgs, raise error)
        _f->gradHess(par, true, &g1, nullptr);
        return copy_and_negate_colvec_to_dlib(g1.t());
    }

    /**
     * @brief Calculate derivative and hessian at given parameter vector
     * @param x parameter vector to calculate deriv & hessian at
     * @param d column vector of gradient to return
     * @param h general matrix of hessian to return
     */
    void get_derivative_and_hessian(const column_vector& x, column_vector& d, general_matrix& h) const
    {
        // wrap dlib column vector in armadillo column vector (no copy) // force a copy
        arma::Col<double> par = wrap_dlib_colvec_to_arma<double>(x, true);
        // calculate gradient and hessian
        arma::dmat g1, h1;
        _f->gradHess(par, false, &g1, &h1);
        // ---- logging ----
        // depending on print level, print stuff to the log
        #ifdef PYPACKAGE
        if (_printlvl >= 1) {
            // if in python, we need to reconnect to the python logger, since we logged to a ring
            // buffer to prevent potential GIL issues.
            // check if there were any buffered messages, if so, flush to python
            esautils::log::flushRingBufferToTarget(spdlog::get("python_bridge"));
            // set the default logger back to the python bridge, so the following are printed
            spdlog::set_default_logger(spdlog::get("python_bridge"));
            spdlog::drop("buffer_logger");
        }
        #endif // PYPACKAGE

        if (_printlvl >= 1) {
            ESALogger::logger()->info("Iteration {:>5} (hessian is analytical):\t\t log-likelihood = {:>13.6f}", *iter.get(), *currLL.get());
        }
        if (_printlvl >= 2) ESALogger::logger()->info("• θ: {}", par);
        if (_printlvl >= 3) ESALogger::logger()->info("• ∇: {}", g1);
        if (_printlvl >= 7) ESALogger::logger()->info("\n• ∇2: {}", h1);
        if (_printlvl >= 3) ESALogger::logger()->info("────────────────────────────────────────");
        // check whether the hessian and gradient are finite
        if (!g1.is_finite()) {
            throw esaexcepts::GradientNotFinite("Gradient is not finite");
        }
        if (!h1.is_finite()) {
            throw esaexcepts::HessianNotFinite("Hessian is not finite");
        }
        // ---- return gradient & hessian ----
        // wrap the armadillo matricies back in dlib (no copy)
        d = wrap_arma_colvec_to_dlib(g1.t());
        h = wrap_arma_mat_to_dlib(h1);
        // once finished, go back to the ring buffer
        #ifdef PYPACKAGE
        if (_printlvl >= 1) {
            // dont think there is away to flush the ringsink, so create a new one
            auto ringSink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(1024);
            auto bufferLogger = std::make_shared<spdlog::logger>("buffer_logger", ringSink);
            spdlog::set_default_logger(bufferLogger);
        }
        #endif // PYPACKAGE
    }

    /**
     * @brief Helper to retrieve the current iteration count
     */
    int getNumIters() const
    {
        return *iter.get();
    }

    /**
     * @brief helper to reset count between steps (if desired)
     */
    void resetIters() {*iter = 0; }

    void setIterToValue(int val) {
        *iter = val;
    }

    void enableIncrementIter() {
        incrementCounter = true;
    }

    void disableIncrementIter() {
        incrementCounter = false;
    }

    int getPrintLevel() { return _printlvl; }

    std::shared_ptr<ESASfaBase> getModel() const { return _f; }

private:
    std::shared_ptr<ESASfaBase> _f;
    HessianCalcMethod _meth;
    bool _analyticalGrad;
    unsigned int _acc;
    unsigned int _printlvl;
    bool _threaded;
    int _maxit;
    std::unique_ptr<int> iter;
    std::unique_ptr<double> currLL;
    bool incrementCounter;
};

#endif // WITHDLIB

#endif // DLIB_WRAPPER_HPP