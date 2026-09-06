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
 * 
 */

#ifndef ESA_OPTIMIZATION_HPP
#define ESA_OPTIMIZATION_HPP

#include <memory>
#include <optional>

// Armadillo
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

// cppoptlib
#ifdef WITHCPPNUMSOLVERS
#include "optim/CppOptLibWrapper.hpp"
#include "cppoptlib/function.h"
#endif

// optimlib
#ifdef WITHOPTIMLIB
#define OPTIM_USE_OPENMP
#define OPTIM_ENABLE_ARMA_WRAPPERS
#include "optim.hpp"
#endif

#ifdef WITHDLIB
#include "dlib/matrix.h"
#include "dlib/optimization.h"
#endif

#include "sfa/ESASfaBase.hpp"
#include "optim/ESAOptimResult.hpp"
#include "optim/optimparams.hpp"
#include "optim/ESAStepSizeCallback.hpp"
#include "utils/enums.hpp"

namespace esaoptimization
{
    // ---- overall functions ----
    std::unique_ptr<ESAOptimResult> optimize(
        const ModelSolver s,
        const arma::dcolvec& par,
        std::shared_ptr<ESASfaBase> f,
        const ESAOptimParams& optimParams,
        const unsigned int printLevel = 0,
        const HessianCalcMethod hessianMethod = HessianCalcMethod::NUM_APPROX,
        const unsigned int hessianNumApproxAccuracy = 3,
        const bool threaded = true
    );

    // ---- ensmallen optimization functions ----
    #ifdef WITHENSMALLEN

    template <class ObjectiveFunction, class OptimAlgo, class ObjFuncCallback = ESAStepNullCb>
    std::unique_ptr<ESAOptimResult> optimEnsmallen(
        const arma::dcolvec& par,
        std::shared_ptr<ESASfaBase> f,
        const ESAOptimParams& optimParams,
        const unsigned int printLevel = 0,
        const HessianCalcMethod hessianMethod = HessianCalcMethod::NUM_APPROX,
        const unsigned int hessianNumApproxAccuracy = 3,
        const bool threaded = true
    );

    std::unique_ptr<ESAOptimResult> optimEnsmallen(
        const ModelSolver s,
        const arma::dcolvec& par,
        std::shared_ptr<ESASfaBase> f,
        const ESAOptimParams& optimParams,
        const unsigned int printLevel = 0,
        const HessianCalcMethod hessianMethod = HessianCalcMethod::NUM_APPROX,
        const unsigned int hessianNumApproxAccuracy = 3,
        const bool threaded = true
    );
    
    #endif // WITHENSMALLEN

    // ---- cppoptlib optimization functions ----
    #if defined(WITHCPPNUMSOLVERS)

    bool convergedOptimCppOptLib(const cppoptlib::solver::Status& s);

    template <class ObjectiveFunction, class OptimAlgo>
    std::unique_ptr<ESAOptimResult> optimCppOptLib(
        const arma::dcolvec& par,
        std::shared_ptr<ESASfaBase> f,
        const ESAOptimParams& optimParams,
        const unsigned int printLevel = 0,
        const HessianCalcMethod hessianMethod = HessianCalcMethod::NUM_APPROX,
        const unsigned int hessianNumApproxAccuracy = 3,
        const bool threaded = true
    );

    std::unique_ptr<ESAOptimResult> optimCppOptLib(
        const ModelSolver s,
        const arma::dcolvec& par,
        std::shared_ptr<ESASfaBase> f,
        const ESAOptimParams& optimParams,
        const unsigned int printLevel = 0,
        const HessianCalcMethod hessianMethod = HessianCalcMethod::NUM_APPROX,
        const unsigned int hessianNumApproxAccuracy = 3,
        const bool threaded = true
    );

    #endif

    // ---- optimization using optimlib ----
    #if defined(WITHOPTIMLIB)

    std::unique_ptr<ESAOptimResult> optimOptimLib(
        const ModelSolver s,
        const arma::dcolvec& par,
        std::shared_ptr<ESASfaBase> f,
        const ESAOptimParams& optimParams,
        const unsigned int printLevel = 0,
        const HessianCalcMethod hessianMethod = HessianCalcMethod::NUM_APPROX,
        const unsigned int hessianNumApproxAccuracy = 3,
        const bool threaded = true
    );

    #endif

    // ---- optimization using dlib ----
    #if defined(WITHDLIB)

    std::unique_ptr<ESAOptimResult> optimDlib(
        const ModelSolver s,
        const arma::dcolvec& par,
        std::shared_ptr<ESASfaBase> f,
        const ESAOptimParams& optimParams,
        const unsigned int printLevel = 0,
        const HessianCalcMethod hessianMethod = HessianCalcMethod::BHHH_APPROX_ANALYTICAL_GRAD,
        const unsigned int hessianNumApproxAccuracy = 3,
        const bool threaded = true
    );

    #endif

    // ---- EM with Gauss-Hermite Quadrature (LC-TRE models) ----
    std::unique_ptr<ESAOptimResult> optimEM(
        const arma::dcolvec& par,
        std::shared_ptr<ESASfaBase> f,
        const ESAOptimParams& optimParams,
        const unsigned int printLevel = 0,
        const HessianCalcMethod hessCalcMethod = HessianCalcMethod::BHHH_APPROX_ANALYTICAL_GRAD
    );
}

#endif // ESA_OPTIMIZATION_HPP