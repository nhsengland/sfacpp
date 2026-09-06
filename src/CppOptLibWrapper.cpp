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

#include "optim/CppOptLibWrapper.hpp"
#include "math/esamath.hpp"
#include "utils/esautils.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"

// template <class FunctionType, class StateType>
// cppoptlib::solver::Progress<FunctionType, StateType> esaoptimwrappers::CustomStoppingSolverProgress(const ESAOptimParamsCppOpt& opt)
// {
    
// }

#ifdef WITHCPPNUMSOLVERS

esaoptimwrappers::CppOptLibWrapper::ScalarType esaoptimwrappers::CppOptLibWrapper::operator()(
    const VectorType& x,
    VectorType* gradient,
    MatrixType* hessian
) const
{
    arma::Col<double> armaX = esautils::eigenToArmaVec<double>(x);
    // log-likelihood
    double ll = -_ptr->operator()(armaX);
    arma::dmat g, h;
    _ptr->gradHess(armaX, 1e-8, true, _method, _hessianNumApproxAccuracy, true, &g, &h);
    if (gradient){
        arma::dmat gr = g.t();
        *gradient = -esautils::armaToEigenVec<double>(gr);
    }
    if (hessian){
        *hessian = -esautils::castEigen(h);
    }
    return ll;
}

#endif // WITHCPPNUMSOLVERS