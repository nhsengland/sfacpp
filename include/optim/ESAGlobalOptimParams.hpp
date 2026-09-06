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
 * @file ESAGlobalOptimParams.hpp
 * @brief Header file for global optimization parameters
 * @date 2025-04-14
 * @author Edmund Haacke
 */

#ifndef ESA_GLOBAL_OPTIM_PARAMS_HPP
#define ESA_GLOBAL_OPTIM_PARAMS_HPP

#include <mutex>
#include <string>
#include "utils/enums.hpp"
#include "optim/optimparams.hpp"

class ESAGlobalOptimParams {

private:
    static ESAGlobalOptimParams* pinstance_;
    static std::mutex mutex_;

protected:
    ESAGlobalOptimParams() : mainOptimParams(ESAOptimParams()),
        mainModelSolver(ModelSolver::DLIB_TR),
        hessianMethod(HessianCalcMethod::ANALYTICAL),
        hessianNumApproxAcc(0),
        optimThreaded(true)
        {};
    ~ESAGlobalOptimParams(){};

public:
    // remove ability to clone
    ESAGlobalOptimParams(ESAGlobalOptimParams& other) = delete;
    // no ability to assign to singleton
    void operator=(const ESAGlobalOptimParams&) = delete;
    // static method to control access to singleton
    static ESAGlobalOptimParams* GetInstance();
    // public attributes that can be modified
    ESAOptimParams mainOptimParams;
    // model solvers
    ModelSolver mainModelSolver;
    // hessian method
    HessianCalcMethod hessianMethod;
    int hessianNumApproxAcc;
    // whether threaded
    bool optimThreaded;
};

#endif // ESA_GLOBAL_OPTIM_PARAMS_HPP