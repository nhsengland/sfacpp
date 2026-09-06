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

#ifndef ESA_STEP_SIZE_CALLBACK_HPP
#define ESA_STEP_SIZE_CALLBACK_HPP

#include <cmath>
#include "sfa/ESASfaBase.hpp"
#include "data/ESADataPanel.hpp"
#include "utils/esautils.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "math/esamath.hpp"
// ensmallen import conditional
#ifdef WITHENSMALLEN
#include <ensmallen.hpp>
#endif

class ESAStepNullCb
{
public:
    ESAStepNullCb() {}
};

class ESAStepSizeBaseCb
{

public:
    ESAStepSizeBaseCb(
        std::shared_ptr<ESASfaBase> sfaPtr,
        bool analyticalGrad = true,
        unsigned int printLevel = 1,
        const bool threaded = true
    ) : _sfaPtr(sfaPtr), _stepSize(0.0), _printLevel(printLevel), _threaded(threaded)
    {

    }

protected:
    std::shared_ptr<ESASfaBase> _sfaPtr;
    double _stepSize;
    unsigned int _printLevel = 1;
    unsigned int currItr = 1;
    bool _threaded = true;

    void logIteration(double llScore)
    {
        if ((_printLevel >= 2) || (_printLevel >= 1 && (currItr % 10) == 0)){
            ESALogger::logger()->info("Iteration {:>4}: log-likelihood = {:>13.6f}", currItr, llScore);
        }
    }

    void logIterationCoords(const arma::mat& x)
    {
        double ll = this->_sfaPtr->operator()(x);
        logIteration(ll);
    }
};


class ESAStepSizeDifferentiableCb : public ESAStepSizeBaseCb
{
public:
    ESAStepSizeDifferentiableCb(
        std::shared_ptr<ESASfaBase> sfaPtr, 
        bool analyticalGrad = true,
        const unsigned int printLevel = 1,
        const bool threaded = true
    ) : ESAStepSizeBaseCb(sfaPtr, analyticalGrad, printLevel, threaded)
    {

    }

    template <typename OptimizerType, typename FunctionType, typename MatType>
    void BeginOptimization(OptimizerType& optimizer, FunctionType& function, MatType& coords)
    {
        // save initial step size
        _stepSize = optimizer.StepSize();
        // initialize inv hessian
        _invHess = arma::eye(coords.n_rows, coords.n_rows);
    }

    // callback function called after optimizer has taken any step that modifies the coordinates
    template <typename OptimizerType, typename FunctionType, typename MatType>
    bool StepTaken(OptimizerType& optimizer, FunctionType& function, MatType& coords)
    {
        // logIterationCoords(coords);
        // line search 
        // const double wolfe_cons_1 = 1e-3;
        // const double wolfe_cons_2 = 0.9;
        // gradient at current x parameter
        // double nobs = this->_sfaPtr->getDataObj()->getNobs();
        // arma::Col<double> grad = (esamath::colSum(this->_sfaPtr->gradient(coords, 1e-8, true)).t() / nobs);
        // arma::Mat<double> hess0 = (this->_sfaPtr->hessian(coords, HessianCalcMethod::NUM_APPROX, 0) / nobs);
        // arma::dmat hess0 = grad * grad.t();
        // arma::dmat hess1;
        // arma::dmat d0;
        // ESALogger::logger()->trace("hessian is {}", hess0);
        // if (hess0.is_finite()) {
        //     bool isNegDef = esamath::isNegativeDefinite<double>(hess0);
        //     if (isNegDef) {
        //         // negate hessian as it will be +ve definite
        //         hess0 = -hess0;
        //     } else {
        //         // make hessian negative definite
        //         hess0 = esamath::makeNegativeDefinite<double>(hess0);
        //     }
        //     // try to invert the negative of the hessian
        //     bool didMakeInv = false;
        //     bool cantInvHess = true;
        //     try {
        //         double rcond;
        //         bool invSuccess = arma::inv(hess1, rcond, -hess0);
        //         ESALogger::logger()->trace("inv success {}; rcond {}, hess1 {}", invSuccess, rcond, hess1);
        //         if (!esamath::isValidInvertedMatrix<double>(-hess0, hess1) || !invSuccess) {
        //             throw std::runtime_error("Invalid inverted matrix");
        //         } else {
        //             cantInvHess = false;
        //         }
        //     } catch (const std::exception& e) {
        //         cantInvHess = true;
        //         // try and make invertable
        //         hess0 = esamath::makeInvertable<double>(hess0);
        //         didMakeInv = true;
        //     }
        //     if (didMakeInv) {
        //         // try inverting again
        //         try {
        //             double rcond;
        //             bool invSuccess = arma::inv(hess1, rcond, -hess0);
        //             if (!esamath::isValidInvertedMatrix<double>(-hess0, hess1) || !invSuccess) {
        //                 throw std::runtime_error("Invalid inverted matrix");
        //             } else {
        //                 cantInvHess = true;
        //             }
        //         } catch (const std::exception &e) {
        //             cantInvHess = true;
        //         }
        //     } else {
        //         cantInvHess = true;
        //     }
        //     // whether or not the hessian could be inverted
        //     if (cantInvHess) {
        //         ESALogger::logger()->trace("cant Inv Hess");
        //         d0 = grad;
        //     } else {
        //         // can invert
        //         d0 = hess1 * grad;
        //     }
        // }
        // ESALogger::logger()->trace("d0: {}\nhess0: {}\nhess1: {}", d0, hess0, hess1);
        // direction vector
        // arma::Col<double> d = -grad;
        // arma::Col<double> x_p = coords, grad_p = grad;
        // lambda function for
        // std::function<double(const arma::Col<double>&, arma::Col<double> *, void *)> fn = [this](const arma::Col<double>& params, arma::Col<double>* gradOut, void* optData){
        //     double ll = this->_sfaPtr->operator()(params);
        //     if (gradOut){
        //         arma::dmat gr = this->_sfaPtr->gradient(params, 1e-8, true);
        //         double nobs = this->_sfaPtr->getDataObj()->getNobs();
        //         *gradOut = -(esamath::colSum(gr).t() / nobs);
        //     }
        //     return ll;
        // };
        // linesearch
        // linesearch::line_search_mt(1.0, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, fn, nullptr);
        // search direction is - inverse hessian * current gradient
        // arma::Col<double> d = -_invHess * grad;
        // if hessian isn't +ve definite, re-initialize hessian
        // const double phi = arma::dot(grad, d);
        // if ((phi > 0) || std::isnan(phi)) {
        //     _invHess = arma::eye(coords.n_rows, coords.n_rows);
        //     d = -grad;
        // }
        // std::function<double(const arma::Col<double>& x, arma::Col<double>* g)> lgfn = [this](const arma::Col<double>& x, arma::Col<double>* g){
        //     double ll = this->_sfaPtr->operator()(x);
        //     if (g){
        //         arma::dmat gr = this->_sfaPtr->gradient(x, 1e-8, true);
        //         double nobs = this->_sfaPtr->getDataObj()->getNobs();
        //         *g = (esamath::colSum(gr).t() / nobs);
        //     }
        //     return ll;
        // };
        // double s = linesearch::MoreThuente::Search(coords, d, lgfn, 1.0);
        // optimizer.StepSize() = s;
        // next parameter vector 
        // const arma::Col<double> sd = s * d;
        // arma::Col<double> next = coords + sd;
        // calculate the next gradient
        // arma::Col<double> nextGrad;
        // lgfn(next, &nextGrad);
        // update inverse hessian estimate
        // const arma::Col<double> y = nextGrad - coords;
        // const double rho = 1.0 / arma::dot(y, sd);
        // _invHess = (
        //     _invHess - 
        //     rho * (sd * (y.t() * _invHess) + (_invHess * y) * sd.t()) +
        //     rho * (rho * arma::dot(y, _invHess * y) + 1.0) * (sd * sd.t())
        // );
        // update the learning rate
        // double s = 1;
        // ESALogger::logger()->trace("step taken; step size {}; proposed s {}", optimizer.StepSize(), s);
        // currItr++;
        // don't terminate optimization
        return false;
    }

private:
    arma::Mat<double> _invHess;
};

class ESAStepSizeSeperableDifferentiableCb : public ESAStepSizeBaseCb
{
public:
    ESAStepSizeSeperableDifferentiableCb(
        std::shared_ptr<ESASfaBase> sfaPtr, 
        const bool analyticalGrad = true,
        const unsigned int printLevel = 1,
        const bool threaded = true
    ) : ESAStepSizeBaseCb(sfaPtr, analyticalGrad, printLevel, threaded)
    {

    }

    template <typename OptimizerType, typename FunctionType, typename MatType>
    void BeginOptimization(OptimizerType& optimizer, FunctionType& function, MatType& coords)
    {
        // save initial step size
        _stepSize = optimizer.StepSize();
    }

    // callback function called at end of a pass over data for seperable functions
    template <typename OptimizerType, typename FunctionType, typename MatType>
    bool EndEpoch(
        OptimizerType& optimizer,
        FunctionType& function,
        const MatType& coords,
        const size_t epoch,
        const double objective
    )
    {
        logIterationCoords(coords);
        // update the learning rate
        // ESALogger::logger()->trace("epoch {} step size {}", epoch, optimizer.StepSize());
        currItr++;
        // don't terminate optimization
        return false;
    }

};

#endif // ESA_STEP_SIZE_CALLBACK_HPP