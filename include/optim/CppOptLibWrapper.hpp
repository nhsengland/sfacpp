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

#ifndef CPP_OPT_LIB_WRAPPER_HPP
#define CPP_OPT_LIB_WRAPPER_HPP

#include <memory>
#include "sfa/ESASfaBase.hpp"
#include "utils/enums.hpp"
#include "optim/optimparams.hpp"

#ifdef WITHCPPNUMSOLVERS

#include "cppoptlib/function.h"
#include "cppoptlib/solver/solver.h"
#include "cppoptlib/solver/progress.h"

namespace esaoptimwrappers {

    template <class F>
    using FunctionExprXd2 = cppoptlib::function::FunctionCRTP<
        F, double, cppoptlib::function::DifferentiabilityMode::Second
    >;
    template <class F>
    using FunctionExprXd1 = cppoptlib::function::FunctionCRTP<
        F, double, cppoptlib::function::DifferentiabilityMode::First
    >;
    template <class F>
    using FunctionExprXd0 = cppoptlib::function::FunctionCRTP<
        F, double, cppoptlib::function::DifferentiabilityMode::None
    >;

    /**
     * 
     * @note full definition in header to resolve linker error
     */
    template <class FunctionType, class StateType>
    cppoptlib::solver::Progress<FunctionType, StateType> CustomStoppingSolverProgress(const ESAOptimParams& opt)
    {
        cppoptlib::solver::Progress<FunctionType, StateType> progress;
        using ScalarType = typename cppoptlib::solver::Progress<FunctionType, StateType>::ScalarType;
        progress.num_iterations = opt.maxit;
        progress.x_delta = ScalarType{opt.rel_solution_change_err_tol};
        progress.f_delta = ScalarType{opt.rel_objfn_change_err_tol};
        progress.gradient_norm = ScalarType{opt.grad_err_tol};
        progress.x_delta_violations = ScalarType{opt.rel_solution_change_err_tol_violations};
        progress.f_delta_violations = ScalarType{opt.rel_objfn_change_err_tol_violations};
        progress.condition_hessian = ScalarType{opt.condition_hessian};
        progress.constraint_threshold = ScalarType{opt.constraint_threshold};
        progress.status = cppoptlib::solver::Status::NotStarted;
        return progress;
    }

    /**
     * 
     */
    class CppOptLibWrapper : public esaoptimwrappers::FunctionExprXd2<CppOptLibWrapper>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        // constructor
        CppOptLibWrapper(
            std::shared_ptr<ESASfaBase> ptr,
            const HessianCalcMethod method = HessianCalcMethod::NUM_APPROX,
            const unsigned int hessianNumApproxAccuracy = 3
        ) : _ptr(ptr),
            _nobs(ptr->getDataObj()->getNobs()),
            _method(method),
            _hessianNumApproxAccuracy(hessianNumApproxAccuracy)
        {

        }

        ScalarType operator()(const VectorType& x, VectorType* gradient = nullptr, MatrixType* hessian = nullptr) const;

    private:
        std::shared_ptr<ESASfaBase> _ptr;
        unsigned int _nobs;
        HessianCalcMethod _method;
        unsigned int _hessianNumApproxAccuracy;
    };

}

#endif

#endif // CPP_OPT_LIB_WRAPPER_HPP