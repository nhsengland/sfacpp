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


#include "optim/esaoptimization.hpp"
#include <chrono>
#include <type_traits>

// imports - relevant only for CppNumSolvers
#if defined(WITHCPPNUMSOLVERS)
// wrapper class
#include "optim/CppOptLibWrapper.hpp"
// from cppoptlib itself
#include "cppoptlib/solver/newton_descent.h"
#include "cppoptlib/solver/bfgs.h"
#include "cppoptlib/solver/conjugated_gradient_descent.h"
#include "cppoptlib/solver/gradient_descent.h"
#include "cppoptlib/solver/lbfgs.h"
#include "cppoptlib/solver/lbfgsb.h"
#include "cppoptlib/solver/nelder_mead.h"
#include "cppoptlib/solver/solver.h"
#include "cppoptlib/solver/progress.h"
#endif

// eigen imports
#ifdef WITHEIGEN
#ifdef RPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include "Eigen/Core"
#endif // RPACKAGE
#endif //WITHEIGEN
// dlib imports
#ifdef WITHDLIB
#include "optim/DlibWrapper.hpp"
#include "utils/dlib2arma.h"
#include "optim/pso.hpp"
#include "optim/ESAGlobalOptimParams.hpp"
#include "utils/enums.hpp"
#endif // WITHDLIB
// ensmallen
#ifdef WITHENSMALLEN
#include "sfa/ESASfaEnsmallenWrapper.hpp"
#endif // WITHENSMALLEN
// other imports
#include "optim/optimutils.hpp"
#include "data/ESADataPanel.hpp"
#include "utils/esautils.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "math/esamath.hpp"
#include "optim/bfgs.hpp"
// EM optimization
#include "sfa/ESASfaLcTre.hpp"
#include "sfa/ESASfaLcTreEM.hpp"


using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// ---- Main optimization function ----

std::unique_ptr<ESAOptimResult> esaoptimization::optimize(
    const ModelSolver s,
    const arma::dcolvec& par,
    std::shared_ptr<ESASfaBase> f,
    const ESAOptimParams& optimParams,
    const unsigned int printLevel,
    const HessianCalcMethod hessianMethod,
    const unsigned int hessianNumApproxAccuracy,
    const bool threaded
)
{
    // 
    ModelSolverLib lib = ESAEnums::libForModelSolver(s);
    std::unique_ptr<ESAOptimResult> resultUnk;

    if (lib == ModelSolverLib::CPPOPTLIB) {
        // check for cppoptlib compiler flag
        #ifdef WITHCPPNUMSOLVERS
        resultUnk = esaoptimization::optimCppOptLib(
            s, par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded
        );
        #else
        throw std::runtime_error("Missing compiler flag for CppOptLib");
        #endif //WITHCPPNUMSOLVERS
    } else if (lib == ModelSolverLib::OPTIMLIB) {
        // check for optimlib compiler flag
        #ifdef WITHOPTIMLIB
        resultUnk = esaoptimization::optimOptimLib(
            s, par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded
        );
        #else
        throw std::runtime_error("Missing compiler flag for OptimLib");
        #endif // WITHOPTIMLIB
    } else if (lib == ModelSolverLib::ENS) {
        // cheeck for ensmallen compiler flag
        #ifdef WITHENSMALLEN
        resultUnk = esaoptimization::optimEnsmallen(
            s, par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded
        );
        #else
        throw std::runtime_error("Missing compiler flag for Ensmallen");
        #endif // WITHENSMALLEN
    } else if (lib == ModelSolverLib::DLIB) {
        // dlib - check for dlib
        #ifdef WITHDLIB
        resultUnk = esaoptimization::optimDlib(
            s, par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded
        );
        #else
        throw std::runtime_error("Missing compiler flag for Dlib");
        #endif // WITHDLIB
    } else if (lib == ModelSolverLib::EM_LIB) {
        resultUnk = esaoptimization::optimEM(par, f, optimParams, printLevel, hessianMethod);
    } else {
        // throw error
        throw std::invalid_argument("Unsupported library");
    }
    return resultUnk;
}

// ---- Ensmallen Optimization ----
#ifdef WITHENSMALLEN

template <class ObjectiveFunction, class OptimAlgo, class ObjFuncCallback>
std::unique_ptr<ESAOptimResult> esaoptimization::optimEnsmallen(
    const arma::dcolvec& par,
    std::shared_ptr<ESASfaBase> f,
    const ESAOptimParams& optimParams,
    const unsigned int printLevel,
    const HessianCalcMethod hessianMethod,
    const unsigned int hessianNumApproxAccuracy,
    const bool threaded
)
{
    arma::dcolvec theta0(par);

    ObjectiveFunction objFun(f, true);
    // OptimAlgo optimizer;
    std::unique_ptr<OptimAlgo> optim;

    if constexpr (std::is_same<OptimAlgo, ens::GradientDescent>::value) {
        /* 
        Gradient Descent
            stepSize, maxIterations, tolerance
        */
        OptimAlgo optimizer(optimParams.step_size, optimParams.maxit, optimParams.grad_err_tol);
        optim = std::make_unique<OptimAlgo>(optimizer);
    } else if constexpr (std::is_same<OptimAlgo, ens::AdaBelief>::value) {
        /*
        Adabelief
            stepSize - step size for each iteration
            batchSize - number of points to process in a single step
            beta1 - exponential decay rate for 1st moment estimates
            beta2 - exponential decay rate for 2nd moment estimates
            epsilon - small constant for numerical stability
            maxIterations - max number of iterations allowed
            tolerance - maximum absolute tolerance to terminate algorithm
            shuffle - if true, the function order is shuffled, otherwise each fn visited in linear order
            resetPolicy - whether to reset parameters before every Optimize call (otherwise vals retained)
            exactObjective - calculate exact objective at end of optim
        */
        OptimAlgo optimizer(
            optimParams.step_size, // stepSize
            optimParams.batch_size, // batchSize
            optimParams.adam_beta_1, // beta1
            optimParams.adam_beta_2, // beta2
            optimParams.init_mean_sq_grad_param, // eps
            optimParams.maxit, // maxIterations
            optimParams.rel_objfn_change_err_tol, // tolerance
            optimParams.diff_seperable_shuffle, // shuffle
            optimParams.diff_seperable_reset_policy, // resetPolicy
            optimParams.diff_seperable_exact_obj // exactObjective
        );
        optim = std::make_unique<OptimAlgo>(optimizer);
    } else if constexpr (std::is_same<OptimAlgo, ens::Adam>::value) {
        /*
        Adam
            stepSize - step size for each iteration
            batchSize - number of points to process in a single step
            beta1 - exponential decay rate for first moment estimates
            beta2 - exponential decay rate for weighted infinity norm estimates
            eps - value used to initialize the mean squared gradient param
            maxIterations - max its
            tolerance - maximum absolute tolerance to terminate algorithm
            shuffle
            resetPolicy
            exactObjective - whether to calculate exact obj at end of the optimization
        */
        OptimAlgo optimizer(
            optimParams.step_size, // stepSize
            optimParams.batch_size, // batchSize
            optimParams.adam_beta_1, // beta1
            optimParams.adam_beta_2, // beta2
            optimParams.init_mean_sq_grad_param, // eps
            optimParams.maxit, // maxIterations
            optimParams.rel_objfn_change_err_tol, // tolerance
            optimParams.diff_seperable_shuffle, // shuffle
            optimParams.diff_seperable_reset_policy, // resetPolicy
            optimParams.diff_seperable_exact_obj // exactObjective
        );
        optim = std::make_unique<OptimAlgo>(optimizer);
    } else if constexpr (std::is_same<OptimAlgo, ens::SGDR<>>::value) {
        /* 
        Stochastic Gradient Descent with Restarts (SGDR)
            epochRestart - initial epoch where decay is applied
            multFactor - batch size multiplication factor
            batchSize - size of each mini-batch
            stepSize - step size for each iteration
            maxIterations - maximum number of iterations
            tolerance - maximum absolute tolerance to terminate algorithm
            shuffle - whether mini-batch order should be shuffled
            updatePolicy - instansiated update policy used to adjust the given parameters
            resetPolicy - if true, params reset before every Optimize call
            exactObjective - calculate exact objective
        */
        ens::MomentumUpdate upd(optimParams.momentum_decay);
        OptimAlgo optimizer(
            optimParams.sgd_epoch_restart, // epochRestart
            optimParams.sgd_batch_mult_factor, // multFactor
            optimParams.batch_size, // batchSize
            optimParams.step_size, // stepSize
            optimParams.maxit, // maxIterations
            optimParams.rel_objfn_change_err_tol, // tolerance
            optimParams.diff_seperable_shuffle, // shuffle
            upd, // instansiated updated policy
            optimParams.diff_seperable_reset_policy, // resetPolicy
            optimParams.diff_seperable_exact_obj // exactObjective
        );
        optim = std::make_unique<OptimAlgo>(optimizer);
    } else if constexpr (
        std::is_same<OptimAlgo, ens::SA<>>::value ||
        std::is_same<OptimAlgo, ens::SA<ens::ExponentialSchedule>>::value
    ) {
        /*
        Simulated Annealing
            coolingSchedule - instantiated cooling schedule
            maxIterations - maximum number of iterations allowed
            initT - initial temperature
            initMoves - number of initial iterations without changing temperature
            moveCtrlSweep - sweeps per feedback move control
            tolerance - tolerance to consider system frozen
            maxToleranceSweep - maximum sweeps below tolerance to consider system frozen
            maxMoveCoef - maximum move size
            initMoveSize - initial move size
            gain - proportional control in feedback move control
        */
        OptimAlgo optimizer(
            ens::ExponentialSchedule(), // coolingSchedule
            optimParams.maxit, // maxIterations
            optimParams.sa_init_temp, // initT
            optimParams.sa_init_moves, // initMoves
            optimParams.sa_move_ctrl_sweep, // moveCtrlSweep
            optimParams.sa_tol_frozen, // tolerance
            optimParams.sa_max_tol_sweep, // maxToleranceSweep
            optimParams.sa_max_move_coef, // maxMoveCoef
            optimParams.sa_init_move_coef, // initMoveCoef
            optimParams.sa_gain // gain
        );
        optim = std::make_unique<OptimAlgo>(optimizer);
    }
    if (!optim) throw std::runtime_error("something went wrong :(");
    if constexpr (
        std::is_same<ObjFuncCallback, ESAStepSizeDifferentiableCb>::value ||
        std::is_same<ObjFuncCallback, ESAStepSizeSeperableDifferentiableCb>::value
    ) {
        ObjFuncCallback ssc(f, true, printLevel, threaded);
        bool success = optim->Optimize(objFun, theta0, ssc);
        ESALogger::logger()->trace("ensmallen success? {}", success);
    } else {
        bool success = optim->Optimize(objFun, theta0);
        ESALogger::logger()->trace("ensmallen success? {}", success);
    }
    
    ESALogger::logger()->trace("post theta0 {}", theta0);
    double finalLL = f->operator()(theta0);
    ESALogger::logger()->trace("final ll: {}", finalLL);
    // arma::dmat g, h;
    // f->gradHess(thetaOut, 1e-8, true, hessianMethod, hessianNumApproxAccuracy, true, &g, &h);
    // double ll = f->operator()(thetaOut);
    // // 
    // unsigned int nobs = f->getDataObj()->getNobs();
    // arma::vec gTotal = -esamath::colSum(g).t() / nobs;
    // ESAOptimResultSuccess success(thetaOut, ll, g, gTotal, h);
    // return std::make_unique<ESAOptimResultSuccess>(success);
    ESAOptimResultFailed fOut(theta0);
    return std::make_unique<ESAOptimResultFailed>(fOut);
}

// explicit template instanstiasation
// >> differentiable algorithms <<
// algorithm: Gradient descent
template std::unique_ptr<ESAOptimResult> esaoptimization::optimEnsmallen<
    ESASfaEnsmallenDifferentiable, ens::GradientDescent, ESAStepNullCb
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);
template std::unique_ptr<ESAOptimResult> esaoptimization::optimEnsmallen<
    ESASfaEnsmallenDifferentiable, ens::GradientDescent, ESAStepSizeDifferentiableCb
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);


// >> differentiable seperable algorithms <<
// algorithm: Adam
template std::unique_ptr<ESAOptimResult> esaoptimization::optimEnsmallen<
    ESASfaEnsmallenDifferentiableSeperable, ens::Adam, ESAStepNullCb
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);
template std::unique_ptr<ESAOptimResult> esaoptimization::optimEnsmallen<
    ESASfaEnsmallenDifferentiableSeperable, ens::Adam, ESAStepSizeSeperableDifferentiableCb
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);
// algorithm: Adabelief
template std::unique_ptr<ESAOptimResult> esaoptimization::optimEnsmallen<
    ESASfaEnsmallenDifferentiableSeperable, ens::AdaBelief, ESAStepNullCb
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);
template std::unique_ptr<ESAOptimResult> esaoptimization::optimEnsmallen<
    ESASfaEnsmallenDifferentiableSeperable, ens::AdaBelief, ESAStepSizeSeperableDifferentiableCb
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);
// algorithm: SGDR 
template std::unique_ptr<ESAOptimResult> esaoptimization::optimEnsmallen<
    ESASfaEnsmallenDifferentiableSeperable, ens::SGDR<>, ESAStepNullCb
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);
template std::unique_ptr<ESAOptimResult> esaoptimization::optimEnsmallen<
    ESASfaEnsmallenDifferentiableSeperable, ens::SGDR<>, ESAStepSizeSeperableDifferentiableCb
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);

// >> arbitary algorithms <<
// algorithm: simulated annealing (SA)
// note ens::SA<> is equivalent to ens::SA<ens::ExponentialSchedule>
template std::unique_ptr<ESAOptimResult> esaoptimization::optimEnsmallen<
    ESASfaEnsmallenArbitary, ens::SA<ens::ExponentialSchedule>, ESAStepNullCb
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);



std::unique_ptr<ESAOptimResult> esaoptimization::optimEnsmallen(
    const ModelSolver s,
    const arma::dcolvec& par,
    std::shared_ptr<ESASfaBase> f,
    const ESAOptimParams& optimParams,
    const unsigned int printLevel,
    const HessianCalcMethod hessianMethod,
    const unsigned int hessianNumApproxAccuracy,
    const bool threaded
)
{
    std::unique_ptr<ESAOptimResult> resultUnk;
    try {
        if (s == ModelSolver::ENS_ADABELIEF) {
            resultUnk = esaoptimization::optimEnsmallen<
                ESASfaEnsmallenDifferentiableSeperable, ens::AdaBelief, ESAStepSizeSeperableDifferentiableCb
            >(par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded);
        } else if (s == ModelSolver::ENS_ADAM) {
            resultUnk = esaoptimization::optimEnsmallen<
                ESASfaEnsmallenDifferentiableSeperable, ens::Adam, ESAStepSizeSeperableDifferentiableCb
            >(par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded);
        } else if (s == ModelSolver::ENS_SGDR) {
            resultUnk = esaoptimization::optimEnsmallen<
                ESASfaEnsmallenDifferentiableSeperable, ens::SGDR<>, ESAStepSizeSeperableDifferentiableCb
            >(par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded);
        } else if (s == ModelSolver::ENS_GD) {
            resultUnk = esaoptimization::optimEnsmallen<
                ESASfaEnsmallenDifferentiable, ens::GradientDescent, ESAStepSizeDifferentiableCb
            >(par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded);
        } else if (s == ModelSolver::ENS_SA) {
            resultUnk = esaoptimization::optimEnsmallen<
                ESASfaEnsmallenArbitary, ens::SA<ens::ExponentialSchedule>, ESAStepNullCb
            >(par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded);
        }
    }
    catch (const std::exception& e)
    {
        // catch any exceptions, return a failed result
        ESALogger::logger()->error("Error in optimization: {}", e.what());
        ESAOptimResultFailed outF(par);
        resultUnk = std::make_unique<ESAOptimResultFailed>(outF);
    }
    return resultUnk;
}

#endif // WITHENSMALLEN

// functions for CppNumSolvers // cppoptlib
#if defined(WITHCPPNUMSOLVERS)

bool esaoptimization::convergedOptimCppOptLib(const cppoptlib::solver::Status& s)
{
    return (
        (s == cppoptlib::solver::Status::GradientNormViolation) ||
        (s == cppoptlib::solver::Status::Finished) ||
        (s == cppoptlib::solver::Status::XDeltaViolation) ||
        (s == cppoptlib::solver::Status::FDeltaViolation)
    );
}

template <class ObjectiveFunction, class OptimAlgo>
std::unique_ptr<ESAOptimResult> esaoptimization::optimCppOptLib(
    const arma::dcolvec& par,
    std::shared_ptr<ESASfaBase> f,
    const ESAOptimParams& optimParams,
    const unsigned int printLevel,
    const HessianCalcMethod hessianMethod,
    const unsigned int hessianNumApproxAccuracy,
    const bool threaded
)
{
    // instantasiate wrapper class that bridges SFA with cppoptlib
    ObjectiveFunction objFunc(f, hessianMethod, hessianNumApproxAccuracy);
    // custom stopping settings for all solvers
    auto stop = esaoptimwrappers::CustomStoppingSolverProgress<ObjectiveFunction, cppoptlib::function::FunctionState<double>>(optimParams);
    if (printLevel > 0) {
        ESALogger::logger()->info("CppOptLib: starting minimization, npar={}", par.n_elem);
    }
    Eigen::VectorXd x_init = esautils::armaToEigenVec(par);
    const auto initState = cppoptlib::function::FunctionState(x_init);
    OptimAlgo solver(stop);
    const auto [solutionState, solverProgress] = solver.Minimize(objFunc, initState);
    arma::dcolvec thetaOut = esautils::eigenToArmaVec<double>(solutionState.x);
    // check for convergence
    if (convergedOptimCppOptLib(solverProgress.status)){
        // calculate hessian and gradient
        arma::dmat g, h;
        f->gradHess(thetaOut, false, &g, &h);
        double ll = f->operator()(thetaOut);
        //
        unsigned int nobs = f->getDataObj()->getNobs();
        arma::vec gTotal = -g.t();
        h = -h;
        int N = f->getN();
        double gnorm = std::sqrt(arma::accu(arma::pow(gTotal, 2.0)));
        if (printLevel > 0) {
            ESALogger::logger()->info(
                "CppOptLib: converged (status={}), ll={:.6f}, gnorm={:.6e}, npar={}",
                static_cast<int>(solverProgress.status), ll, gnorm, thetaOut.n_elem
            );
        }
        ESAOptimResultSuccess success(f->getModelType(), thetaOut, ll, g, gTotal, h, N, nobs);
        return std::make_unique<ESAOptimResultSuccess>(success);
    }
    if (printLevel > 0) {
        ESALogger::logger()->warn(
            "CppOptLib: did not converge (status={})",
            static_cast<int>(solverProgress.status)
        );
    }
    ESAOptimResultFailed failed(thetaOut);
    return std::make_unique<ESAOptimResultFailed>(failed);
}
// explicit template instantisation
// Newton descent solver
template std::unique_ptr<ESAOptimResult> esaoptimization::optimCppOptLib<
    esaoptimwrappers::CppOptLibWrapper,
    cppoptlib::solver::NewtonDescent<esaoptimwrappers::CppOptLibWrapper>
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);
// Bfgs solver
template std::unique_ptr<ESAOptimResult> esaoptimization::optimCppOptLib<
    esaoptimwrappers::CppOptLibWrapper,
    cppoptlib::solver::Bfgs<esaoptimwrappers::CppOptLibWrapper>
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);
// Conjugated gradient descent
template std::unique_ptr<ESAOptimResult> esaoptimization::optimCppOptLib<
    esaoptimwrappers::CppOptLibWrapper,
    cppoptlib::solver::ConjugatedGradientDescent<esaoptimwrappers::CppOptLibWrapper>
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);
// Gradient descent
template std::unique_ptr<ESAOptimResult> esaoptimization::optimCppOptLib<
    esaoptimwrappers::CppOptLibWrapper,
    cppoptlib::solver::GradientDescent<esaoptimwrappers::CppOptLibWrapper>
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);
// Lbfgs
template std::unique_ptr<ESAOptimResult> esaoptimization::optimCppOptLib<
    esaoptimwrappers::CppOptLibWrapper,
    cppoptlib::solver::Lbfgs<esaoptimwrappers::CppOptLibWrapper>
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);
// Lbfgsb
template std::unique_ptr<ESAOptimResult> esaoptimization::optimCppOptLib<
    esaoptimwrappers::CppOptLibWrapper,
    cppoptlib::solver::Lbfgsb<esaoptimwrappers::CppOptLibWrapper>
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);
// Nelder mead
template std::unique_ptr<ESAOptimResult> esaoptimization::optimCppOptLib<
    esaoptimwrappers::CppOptLibWrapper,
    cppoptlib::solver::NelderMead<esaoptimwrappers::CppOptLibWrapper>
>(const arma::dcolvec&, std::shared_ptr<ESASfaBase>, const ESAOptimParams&, const unsigned int, const HessianCalcMethod, const unsigned int, const bool);

std::unique_ptr<ESAOptimResult> esaoptimization::optimCppOptLib(
    ModelSolver s,
    const arma::dcolvec& par,
    std::shared_ptr<ESASfaBase> f,
    const ESAOptimParams& optimParams,
    const unsigned int printLevel,
    const HessianCalcMethod hessianMethod,
    const unsigned int hessianNumApproxAccuracy,
    const bool threaded
)
{
    std::unique_ptr<ESAOptimResult> resultUnk;
    if (printLevel > 0) {
        ESALogger::logger()->info(
            "Starting CppOptLib optimization: solver={}, npar={}",
            ESAEnums::strForModelSolver(s), par.n_elem
        );
    }
    try {
        if (s == ModelSolver::CPPOPTLIB_BFGS) {
            resultUnk = esaoptimization::optimCppOptLib<
                esaoptimwrappers::CppOptLibWrapper,
                cppoptlib::solver::Bfgs<esaoptimwrappers::CppOptLibWrapper>
            >(par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded);
        } else if (s == ModelSolver::CPPOPTLIB_CONJUGATED_GD) {
            resultUnk = esaoptimization::optimCppOptLib<
                esaoptimwrappers::CppOptLibWrapper,
                cppoptlib::solver::ConjugatedGradientDescent<esaoptimwrappers::CppOptLibWrapper>
            >(par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded);
        } else if (s == ModelSolver::CPPOPTLIB_GD) {
            resultUnk = esaoptimization::optimCppOptLib<
                esaoptimwrappers::CppOptLibWrapper,
                cppoptlib::solver::GradientDescent<esaoptimwrappers::CppOptLibWrapper>
            >(par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded);
        } else if (s == ModelSolver::CPPOPTLIB_LBFGS) {
            resultUnk = esaoptimization::optimCppOptLib<
                esaoptimwrappers::CppOptLibWrapper,
                cppoptlib::solver::Lbfgs<esaoptimwrappers::CppOptLibWrapper>
            >(par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded);
        } else if (s == ModelSolver::CPPOPTLIB_LBFGSB) {
            resultUnk = esaoptimization::optimCppOptLib<
                esaoptimwrappers::CppOptLibWrapper,
                cppoptlib::solver::Lbfgsb<esaoptimwrappers::CppOptLibWrapper>
            >(par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded);
        } else if (s == ModelSolver::CPPOPTLIB_NELDER_MEAD) {
            resultUnk = esaoptimization::optimCppOptLib<
                esaoptimwrappers::CppOptLibWrapper,
                cppoptlib::solver::NelderMead<esaoptimwrappers::CppOptLibWrapper>
            >(par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded);
        } else if (s == ModelSolver::CPPOPTLIB_NEWTON) {
            resultUnk = esaoptimization::optimCppOptLib<
                esaoptimwrappers::CppOptLibWrapper,
                cppoptlib::solver::NewtonDescent<esaoptimwrappers::CppOptLibWrapper>
            >(par, f, optimParams, printLevel, hessianMethod, hessianNumApproxAccuracy, threaded);
        }
    } catch (std::exception& e) {
        // catch any exceptions, return a failed result
        ESALogger::logger()->error("Error in optimization: {}", e.what());
        ESAOptimResultFailed outF(par);
        return std::make_unique<ESAOptimResultFailed>(outF);
    }
    return resultUnk;
}

#endif

/// ---- OptimLib optimization functions ----
#if defined(WITHOPTIMLIB)

std::unique_ptr<ESAOptimResult> esaoptimization::optimOptimLib(
    const ModelSolver s,
    const arma::dcolvec& par,
    std::shared_ptr<ESASfaBase> f,
    const ESAOptimParams& optimParams,
    const unsigned int printLevel,
    const HessianCalcMethod hessianMethod,
    const unsigned int hessianNumApproxAccuracy,
    const bool threaded
)
{
    arma::dcolvec theta0(par);
    bool status = false;
    // get optimization settings from user preferences
    optim::algo_settings_t settings = optim_utils::optimSettingsForParams(optimParams);
    // set print level
    settings.print_level = printLevel;
    if (printLevel > 0) {
        ESALogger::logger()->info(
            "Starting OptimLib optimization: solver={}, npar={}",
            ESAEnums::strForModelSolver(s), par.n_elem
        );
    }

    //ESADataPanel& dataObj = (ESADataPanel&)*f->dataObjPtr();
    /**
     * REMEMBER: MINIMISATION! NEED TO NEGATE GRADIENT AND HESSIAN
     */
    auto wrapperFn = [
        &f,
        &hessianMethod,
        &hessianNumApproxAccuracy,
        &threaded
    ](const arma::dcolvec& x, arma::dcolvec* gOut, arma::dmat* hOut, void* optData){
        // calculate log likelihood
        double ll = f->operator()(x);
        if (gOut && hOut) {
            arma::dmat g, h;
            // call gradHess method
            f->gradHess(x, false, &g, &h);
            if (hOut) *hOut = -h;
            // optim lib wants as k x 1 column vector
            if (gOut) *gOut = -g.t();
        } else if (gOut && !hOut){
            // only calculate gradient, ignore hessian
            arma::dmat g = f->gradient(x, 1e-8, true);
            if (gOut) *gOut = -g.t();
        }
        ESALogger::logger()->trace("optim wrapper fn {}", ll);
        return -ll;
    };
    if (s == ModelSolver::OPTIMLIB_BFGS) {
        status = optim::bfgs(
            theta0,
            [&wrapperFn](const arma::dcolvec& x, arma::dcolvec* gradOut, void* optData) {
                return wrapperFn(x, gradOut, nullptr, optData);
            },
            nullptr, // opt_data - additional data passed to user provided function
            settings
        );
    } else if (s == ModelSolver::OPTIMLIB_LBFGS) {
        status = optim::lbfgs(
            theta0,
            [&wrapperFn](const arma::dcolvec& x, arma::dcolvec* gradOut, void* optData) {
                return wrapperFn(x, gradOut, nullptr, optData);
            },
            nullptr, // opt_data
            settings
        );
    } else if (s == ModelSolver::OPTIMLIB_CG) {
        status = optim::cg(
            theta0,
            [&wrapperFn](const arma::dcolvec& x, arma::dcolvec* gradOut, void* optData) {
                return wrapperFn(x, gradOut, nullptr, optData);
            },
            nullptr, // opt_data
            settings
        );
    } else if (s == ModelSolver::OPTIMLIB_GD) {
        status = optim::cg(
            theta0,
            [&wrapperFn](const arma::dcolvec& x, arma::dcolvec* gradOut, void* optData) {
                return wrapperFn(x, gradOut, nullptr, optData);
            },
            nullptr, // opt_data
            settings
        );
    } else if (s == ModelSolver::OPTIMLIB_NEWTON) {
        status = optim::newton(
            theta0,
            [&](const arma::dcolvec& x, arma::dcolvec* gradOut, arma::dmat* hessOut, void* optData) {
                return wrapperFn(x, gradOut, hessOut, optData);
            },
            nullptr,
            settings
        );
    } else {
        throw std::invalid_argument("Invalid optim algo " + ESAEnums::strForModelSolver(s));
    }
    if (printLevel > 0) {
        ESALogger::logger()->info("OptimLib: solver exited with status={}", status);
    }
    ESALogger::logger()->trace("OptimLib optimization status {}", status);
    std::unique_ptr<ESAOptimResult> resultUnk;
    // immediately exit and return failed object
    if (!status) {
        if (printLevel > 0) {
            ESALogger::logger()->warn("OptimLib: optimization reported failure, returning failed result");
        }
        ESAOptimResultFailed fOut(theta0);
        resultUnk = std::make_unique<ESAOptimResultFailed>(fOut);
    }
    // recalculate gradient & hessian
    arma::dmat g, h;
    f->gradHess(theta0, false, &g, &h);
    double ll = f->operator()(theta0);
    arma::vec gTotal = -g.t();
    // double check the gradient norm - just because it exited doesnt mean it converged
    double gnorm = std::sqrt(arma::accu(arma::pow(gTotal, 2.0)));
    bool gnormBelowTol = gnorm < optimParams.grad_err_tol_check;
    ESALogger::logger()->trace("gnorm {} is below tol? {}", gnorm, gnormBelowTol);
    if (printLevel > 0) {
        ESALogger::logger()->info(
            "OptimLib: ll={:.6f}, gnorm={:.6e}, below_tol={}",
            ll, gnorm, gnormBelowTol
        );
    }
    if (gnormBelowTol) {
        arma::dmat jac = -f->jacobian(theta0);
        int N = f->getN();
        int nobs = f->getDataObj()->getNobs();
        if (printLevel > 0) {
            ESALogger::logger()->info(
                "OptimLib: converged, ll={:.6f}, gnorm={:.6e}, N={}, nobs={}",
                ll, gnorm, N, nobs
            );
        }
        ESAOptimResultSuccess success(f->getModelType(), theta0, ll, jac, gTotal, h, N, nobs);
        return std::make_unique<ESAOptimResultSuccess>(success);
    } else {
        if (printLevel > 0) {
            ESALogger::logger()->warn(
                "OptimLib: gnorm={:.6e} exceeds tolerance={:.6e}, returning failed result",
                gnorm, optimParams.grad_err_tol_check
            );
        }
        ESAOptimResultFailed fOut(theta0);
        resultUnk = std::make_unique<ESAOptimResultFailed>(fOut);
    }
    return resultUnk;
}

#endif

#if defined(WITHDLIB)


double dlib_trusted_region(
    DlibWrapper& fn,
    DlibWrapper::column_vector& p,
    const ESAOptimParams& optimParams
)
{
    double out = std::numeric_limits<double>::quiet_NaN();
    auto stopStrategy = dlib::gradient_norm_stop_strategy(
        optimParams.grad_err_tol
    );
    out = dlib::find_max_trust_region(stopStrategy, fn, p, optimParams.tr_radius);
    return out;
}


double dlib_hybrid_maximization(
    DlibWrapper& fn,
    DlibWrapper::column_vector& params,
    const ESAOptimParams& optimParams,
    const ModelSolver step1 = ModelSolver::DLIB_BFGS,
    const ModelSolver step2 = ModelSolver::DLIB_TR
)
{
    double out = std::numeric_limits<double>::quiet_NaN();
    // first step - use a quasi-netwon method
    // helpers
    auto obj_fn = [&](const DlibWrapper::column_vector& x) {
        // wrap this in a try, so if we get infinite values/invalid values for calculating
        // the cdf/pdf, then line search will search for the next legit value
        try {
            return -fn(x);
        } catch(...)
        {
            // dont actually care about the exception, just return infinit
            return std::numeric_limits<double>::infinity();
        }
    };
    // force negation to be evaluated into a concrete matrix with ->
    auto der_fn = [&](const DlibWrapper::column_vector& x) -> DlibWrapper::column_vector {
        // same logic here, wrap in a try catch
        try {
            return fn.derivated_negated(x);
        } catch(...)
        {
            // return a matrix of infinites
            DlibWrapper::column_vector vec(x.nr());
            dlib::set_all_elements(vec, std::numeric_limits<double>::infinity());
            return vec;
        }
    }; 
    // for step 1, disable the wrapper from incrementing, and get the solver to pass them back
    fn.disableIncrementIter();
    int step1Iter = 0;
    if (step1 == ModelSolver::DLIB_LBFGS) {
        // use LBFGS - keep last 10 corrections
        optim::LbfgsSolve(
            obj_fn,
            der_fn,
            params,
            step1Iter,
            fn.getPrintLevel(),
            optimParams.maxit,
            optimParams.step1_grad_tol
        );
    } else if (step1 == ModelSolver::DLIB_BFGS) {
        // BFGS
        optim::BfgsSolve(
            obj_fn,
            der_fn,
            params,
            step1Iter,
            fn.getPrintLevel(),
            optimParams.maxit,
            optimParams.step1_grad_tol
        );
    } else {
        throw std::invalid_argument("step 1 solver must be bfgs/lbfgs");
    }
    // enable the incrementing again, and update with the number of iterations we did
    fn.enableIncrementIter();
    fn.setIterToValue(step1Iter);
    // step 2: full newton method with the full hessian OR trusted region
    if (step2 == ModelSolver::DLIB_TR) {
        auto stopStrategy = dlib::gradient_norm_stop_strategy(optimParams.grad_err_tol);
        out = dlib::find_max_trust_region(stopStrategy, fn, params, optimParams.tr_radius);
    } else {
        throw std::invalid_argument("step 2 solver must be trusted region");
    }
    return out;
}

// Run PSO (minimization) around a starting point, return best position found.
// stage: "focused" uses pso_init_range; "broad" uses pso_broad_range.
// The fn wrapper evaluates the negated log-likelihood so PSO minimizes.
static arma::vec dlib_pso_stage(
    DlibWrapper& fn,
    const arma::dcolvec& startPar,
    const ESAOptimParams& optimParams,
    bool broad,
    const unsigned int printLevel
)
{
    int dims = static_cast<int>(startPar.n_elem);
    double halfRange = broad ? optimParams.pso_broad_range : optimParams.pso_init_range;
    int maxIter = broad ? optimParams.pso_broad_max_iter : optimParams.pso_max_iter;

    // Disable threaded likelihood evaluation during PSO: each particle calls the model
    // operator sequentially, so submitting to the same BS::thread_pool from within its
    // evaluation would cause a nested submit deadlock/mutex error.
    ESAGlobalOptimParams* globalOpts = ESAGlobalOptimParams::GetInstance();
    bool prevThreaded = globalOpts->optimThreaded;
    globalOpts->optimThreaded = false;

    pso::SwarmOpt swarmOpt;
    swarmOpt.phi1 = optimParams.pso_phi1;
    swarmOpt.phi2 = optimParams.pso_phi2;
    swarmOpt.stagnation_patience = optimParams.pso_stagnation_patience;
    swarmOpt.stagnation_tol = optimParams.pso_stagnation_tol;
    swarmOpt.vmax_fraction = optimParams.pso_vmax_fraction;

    auto topo = ESAEnums::psoTopologyForStr(optimParams.pso_topology);
    std::unique_ptr<pso::TopologyBase> topology;
    // instansiate a subclass of TopologyBase representing the Topology to use
    if (topo == ESAPsoTopology::RING) {
        topology = std::make_unique<pso::TopologyRing>();
    } else if (topo == ESAPsoTopology::VONNEUMANN) {
        topology = std::make_unique<pso::TopologyVonNeumann>(optimParams.pso_num_particles);
    } else if (topo == ESAPsoTopology::DYNAMIC) {
        topology = std::make_unique<pso::TopologyDynamic>(
            optimParams.pso_num_particles,
            3,
            optimParams.pso_max_iter / std::max(1, optimParams.pso_num_particles / 5)
        );
    } else {
        topology = std::make_unique<pso::TopologyGlobal>();
    }

    // PSO minimizes: negated log-likelihood. Call the model directly (bypassing
    // DlibWrapper) so the lambda is thread-safe: no shared mutable state is written,
    // and each worker thread uses its own TLS scratch via getContext().
    std::shared_ptr<ESASfaBase> model = fn.getModel();
    auto objFn = [model](const arma::vec& x) -> double {
        arma::dcolvec par(x);
        try {
            return -(model->operator()(par));
        } catch (...) {
            return std::numeric_limits<double>::max();
        }
    };
    fn.disableIncrementIter();
    // construct swarm with a broad box, then re-center particles
    pso::Swarm swarm(
        optimParams.pso_num_particles,
        dims,
        -optimParams.pso_broad_range,
        optimParams.pso_broad_range,
        std::move(topology),
        objFn,
        swarmOpt
    );
    swarm.reinitCentered(startPar, halfRange);
    // Evaluate particles in parallel across the thread pool. optimThreaded is
    // false (set above), so each particle's likelihood uses panelCallableSum
    // internally — no nested pool submission.
    swarm.runParallel(maxIter, esaparallel::getOptimPool());
    globalOpts->optimThreaded = prevThreaded;
    fn.enableIncrementIter();
    fn.setIterToValue(maxIter);

    if (printLevel > 0) {
        ESALogger::logger()->info(
            "PSO {} stage complete: best negLL = {:.6f}", broad ? "broad" : "focused", swarm.globalBestVal()
        );
    }
    return swarm.globalBestPos();
}

std::unique_ptr<ESAOptimResult> esaoptimization::optimDlib(
    const ModelSolver s,
    const arma::dcolvec& par,
    std::shared_ptr<ESASfaBase> f,
    const ESAOptimParams& optimParams,
    const unsigned int printLevel,
    const HessianCalcMethod hessianMethod,
    const unsigned int hessianNumApproxAccuracy,
    const bool threaded
)
{
    // check a valid solver has been passed thru
    if (
        (s != ModelSolver::DLIB_TR) &&
        // (s != ModelSolver::DLIB_HYBRID_BFGS_NEWTON) &&
        (s != ModelSolver::DLIB_HYBRID_BFGS_TR) &&
        // (s != ModelSolver::DLIB_HYBRID_LBFGS_NEWTON) &&
        (s != ModelSolver::DLIB_HYBRID_LBFGS_TR) &&
        (s != ModelSolver::DLIB_HYBRID_PSO_TR) &&
        (s != ModelSolver::DLIB_HYBRID_PSO_BFGS_TR)
    ) {
        throw std::invalid_argument("invalid solver passed to optimDlib");
    }
    dlib::matrix<double, 0, 1> p = wrap_arma_colvec_to_dlib(par);
    // general stop strategy
    // auto stopStrategy = dlib::objective_delta_stop_strategy(
    //     // delta
    //     optimParams.rel_objfn_change_err_tol,
    //     // max it
    //     optimParams.maxit
    // );
    double out = std::numeric_limits<double>::quiet_NaN();
    double nids = f->getDataObj()->getNids();
    // instansiate the wrapper
    DlibWrapper fn(f, hessianMethod, printLevel, true, hessianNumApproxAccuracy, threaded, optimParams.maxit);
    // try to run the appropriate optimization algorithm
    bool failed = false;
    try {
        if (s == ModelSolver::DLIB_TR) {
            out = dlib_trusted_region(fn, p, optimParams);
        } else if (
            (s == ModelSolver::DLIB_HYBRID_BFGS_NEWTON) ||
            (s == ModelSolver::DLIB_HYBRID_BFGS_TR) ||
            (s == ModelSolver::DLIB_HYBRID_LBFGS_NEWTON) ||
            (s == ModelSolver::DLIB_HYBRID_LBFGS_TR)
        ){
            ModelSolver s1, s2;
            if ((s == ModelSolver::DLIB_HYBRID_BFGS_NEWTON) || (s == ModelSolver::DLIB_HYBRID_BFGS_TR)) {
                s1 = ModelSolver::DLIB_BFGS;
            } else if ((s == ModelSolver::DLIB_HYBRID_LBFGS_NEWTON) || (s == ModelSolver::DLIB_HYBRID_LBFGS_TR)) {
                s1 = ModelSolver::DLIB_LBFGS;
            }
            if ((s == ModelSolver::DLIB_HYBRID_BFGS_NEWTON) || (s == ModelSolver::DLIB_HYBRID_LBFGS_NEWTON)) {
                s2 = ModelSolver::DLIB_NEWTON;
            } else if ((s == ModelSolver::DLIB_HYBRID_BFGS_TR) || (s == ModelSolver::DLIB_HYBRID_LBFGS_TR)) {
                s2 = ModelSolver::DLIB_TR;
            }
            out = dlib_hybrid_maximization(fn, p, optimParams, s1, s2);
        } else if (
            (s == ModelSolver::DLIB_HYBRID_PSO_TR) ||
            (s == ModelSolver::DLIB_HYBRID_PSO_BFGS_TR)
        ) {
            // Stage 1: focused PSO around starting values
            arma::dcolvec startPar = copy_dlib_colvec_to_arma(p);
            arma::vec psoResult = dlib_pso_stage(fn, startPar, optimParams, false, printLevel);
            // Evaluate gnorm at PSO result to decide whether broad stage is needed
            arma::dcolvec psoResultCol(psoResult);
            arma::dmat gTest;
            f->gradHess(psoResultCol, false, &gTest, nullptr, nullptr);
            double gnormTest = std::sqrt(arma::accu(arma::pow(gTest.t(), 2.0)));
            if (gnormTest > optimParams.grad_err_tol_check) {
                if (printLevel > 0) {
                    ESALogger::logger()->info("PSO focused gnorm={:.6f} > tol, running broad stage", gnormTest);
                }
                // Stage 2: broad PSO if focused stage didn't find a good basin
                arma::vec psoResultBroad = dlib_pso_stage(fn, startPar, optimParams, true, printLevel);
                // Take whichever gave a higher LL
                double llFocused = f->operator()(psoResultCol);
                arma::dcolvec psoResultBroadCol(psoResultBroad);
                double llBroad = f->operator()(psoResultBroadCol);
                if (llBroad > llFocused) {
                    psoResult = psoResultBroad;
                }
            }
            // Copy PSO best into dlib parameter vector for TR refinement
            for (arma::uword i = 0; i < psoResult.n_elem; i++) p(i) = psoResult(i);
            fn.setIterToValue(optimParams.pso_max_iter + optimParams.pso_broad_max_iter);
            // Stage 3: optional BFGS warmup then Trust Region
            if (s == ModelSolver::DLIB_HYBRID_PSO_BFGS_TR) {
                out = dlib_hybrid_maximization(fn, p, optimParams, ModelSolver::DLIB_BFGS, ModelSolver::DLIB_TR);
            } else {
                out = dlib_trusted_region(fn, p, optimParams);
            }
        }
    } catch(std::exception& e)
    {
        failed = true;
        #ifdef PYPACKAGE
        // check if there were any buffered messages, if so, flush to python
        esautils::log::flushRingBufferToTarget(spdlog::get("python_bridge"));
        // set the default logger back to the python bridge, so the following are printed
        spdlog::set_default_logger(spdlog::get("python_bridge"));
        #endif // PYPACKAGE
        // failed
        ESALogger::logger()->error("Failed in dlib optimization {}", e.what());
        // copy over p back to armadillo matrix
        arma::dcolvec outPar = copy_dlib_colvec_to_arma(p);
        ESAOptimResultFailed outF(outPar);
        return std::make_unique<ESAOptimResultFailed>(outF);
    }
    #ifdef PYPACKAGE
    // just in case - force back to the python logger
    if (!failed){
        // check if there were any buffered messages, if so, flush to python
        esautils::log::flushRingBufferToTarget(spdlog::get("python_bridge"));
        // set the default logger back to the python bridge, so the following are printed
        spdlog::set_default_logger(spdlog::get("python_bridge"));
    }
    #endif // PYPACKAGE
    // copy over p back to armadillo matrix (which was updateed in place)
    arma::dcolvec thetaOut = copy_dlib_colvec_to_arma(p);
    // check whether hit did not hit maximum convergence
    if (!failed) {
        // recalculate gradient & hessian
        arma::dmat g, h, jac;
        f->gradHess(thetaOut, false, &g, &h, &jac);
        double ll = f->operator()(thetaOut);
        arma::vec gTotal = g.t();
        // double check the gradient norm - just because it exited doesnt mean it converged
        double gnorm = std::sqrt(arma::accu(arma::pow(gTotal, 2.0)));
        bool gnormBelowTol = gnorm < optimParams.grad_err_tol_check;
        if (gnormBelowTol) {
            int N = f->getN();
            int nobs = f->getDataObj()->getNobs();
            ESAOptimResultSuccess success(f->getModelType(), thetaOut, ll, jac, gTotal, h, N, nobs, gnorm);
            return std::make_unique<ESAOptimResultSuccess>(success);
        }
        if (printLevel > 0) {
            ESALogger::logger()->error("Optimization failed: gradnorm is {} and grad is {}", gnorm, gTotal);
        }
        ESAOptimResultFailed outF(thetaOut);
        return std::make_unique<ESAOptimResultFailed>(outF);
    }
    ESAOptimResultFailed outF(thetaOut);
    return std::make_unique<ESAOptimResultFailed>(outF);
}

#endif // WITHDLIB

// ============================================================
// EM with Gauss-Hermite Quadrature
// ============================================================

std::unique_ptr<ESAOptimResult> esaoptimization::optimEM(
    const arma::dcolvec& par,
    std::shared_ptr<ESASfaBase> f,
    const ESAOptimParams& optimParams,
    const unsigned int printLevel,
    const HessianCalcMethod hessCalcMethod
)
{
    auto lcmModel = std::dynamic_pointer_cast<ESASfaLcTre>(f);
    if (!lcmModel) {
        throw std::invalid_argument(
            "optimEM (EM_GHQ solver) requires an ESASfaLcTre model. "
            "Check that the model family is LC_TRE."
        );
    }

    ESASfaLcTreEM emSolver(lcmModel, optimParams, printLevel);
    EMResult emResult = emSolver.run(par);

    if (!emResult.converged && printLevel > 0) {
        ESALogger::logger()->warn(
            "EM_GHQ: did not converge after {} iterations", emResult.nIter
        );
    }

    // Compute final gradient, Hessian, and Jacobian at the converged parameters.
    // When hessCalcMethod is ANALYTICAL, the Louis (1982) observed-data Hessian is
    // used (H_obs = E[H_complete|Y] - Var_posterior[score]).  For BHHH variants the
    // outer-product approximation -J^T J is stored instead.
    arma::dmat g, jac;
    arma::dmat h;
    bool wantLouis = (hessCalcMethod == HessianCalcMethod::ANALYTICAL);
    f->gradHess(emResult.params, false, &g, wantLouis ? &h : nullptr, &jac);

    // For BHHH: build h from the Jacobian outer product
    if (!wantLouis) {
        h = -(jac.t() * jac);
    }

    arma::vec gTotal = g.t();
    double gnorm = arma::norm(gTotal, 2);
    int N    = static_cast<int>(f->getN());
    int nobs = f->getDataObj()->getNobs();

    if (emResult.converged) {
        ESAOptimResultSuccess success(
            f->getModelType(), emResult.params, emResult.logLike,
            jac, gTotal, h, N, nobs, gnorm
        );
        return std::make_unique<ESAOptimResultSuccess>(success);
    } else {
        ESAOptimResultFailed outF(emResult.params);
        return std::make_unique<ESAOptimResultFailed>(outF);
    }
}