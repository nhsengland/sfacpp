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

#ifndef PYINTERFACE_UTILS_HPP
#define PYINTERFACE_UTILS_HPP

#ifdef PYPACKAGE

#include <optional>
#include <armadillo>
#include <csignal>
#include <pybind11/pybind11.h>
#include "utils/enums.hpp"
#include "optim/optimparams.hpp"
#include "optim/ESAGlobalOptimParams.hpp"
#include "utils/globalstate.hpp"

namespace py = pybind11;

namespace pyinterface {

    /**
     * Custom signal handler
     */

    inline void signalHandler(int signum) {
        if (signum == SIGINT) {
            globalstate::InterruptRequested.store(true, std::memory_order_relaxed);
        }
    }

    /**
     * RAII guard to swap & restore handlers
     */
    struct SignalHandlerGuard {
        typedef void(*handler_t)(int);
        handler_t oldHandler;

        SignalHandlerGuard() {
            globalstate::InterruptRequested.store(false, std::memory_order_relaxed);
            oldHandler = std::signal(SIGINT, signalHandler);
        }

        ~SignalHandlerGuard() {
            // restore python handler on exit
            std::signal(SIGINT, oldHandler);
        }
    };

    inline void throwPyInterruptIfNeeded() {
        if (globalstate::InterruptRequested.load(std::memory_order_relaxed)) {
            // acquire GIL
            py::gil_scoped_acquire acquire;
            // set keyboard interrupt exception
            // PyErr_SetNone(PyExc_KeyboardInterrupt);
            PyErr_SetString(PyExc_KeyboardInterrupt, "Execution interrupted by user.");
            throw py::error_already_set();
        }
    }
    
    /**
     * @brief Retrieve element by key if it exists
     * @tparam expected type of element to extract by key
     * @param l pybind11 python list
     * @param k a string of the key
     * @return optional holding value if it exists
     */
    template <typename T>
    std::optional<T> elementFromPyDict(py::dict& l, const std::string& k);

    /**
     * @brief setup optim params
     * @param optimParams instance of ESAOptimParams to load settings into
     * @param optimOpts optional pybind11 dictionary of options
     * @param seed the seed to use
     */
    void setupOptimParams(
        ESAOptimParams& optimParams,
        std::optional<py::dict> optimOpts,
        const int seed
    );

    /**
     * @brief process data matrices provided as arguments
     */
    void processDataMatricies(
        const arma::dcolvec& y,
        const arma::dmat& x,
        arma::dmat& zmuit,
        arma::dmat& zuit,
        arma::dmat& zvit,
        arma::dmat& zui0,
        arma::dmat& zvi0,
        arma::Col<int>& idVec,
        arma::Col<int>& timeVec,
        const ESASfaModelType& mT,
        const arma::colvec& y_,
        const arma::mat& x_,
        std::optional<arma::mat> zmuit_ = std::nullopt,
        std::optional<arma::mat> zuit_ = std::nullopt,
        std::optional<arma::mat> zvit_ = std::nullopt,
        std::optional<arma::mat> zui0_ = std::nullopt,
        std::optional<arma::mat> zvi0_ = std::nullopt,
        std::optional<arma::mat> startVals_ = std::nullopt,
        std::optional<arma::Col<int>> idVec_ = std::nullopt,
        std::optional<arma::Col<int>> timeVec_ = std::nullopt
    );
}

#endif //PYPACKAGE

#endif // PYINTERFACE_UTILS_HPP