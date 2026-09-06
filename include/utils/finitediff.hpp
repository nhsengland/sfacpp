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
 * Based on the logic from cppoptlib by XYZ
 */

#ifndef FINITE_DIFF_HPP
#define FINITE_DIFF_HPP

#include <vector>

// ---- armadillo ----
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif
// ---- end armadillo ----
#ifdef WITHDLIB
#include <dlib/matrix.h>
#endif // WITHDLIB

namespace finitediff {

    /// @brief Calculate gradient by finite differences
    /// @param x0
    /// @param fn
    /// @param accuracy
    /// @returns matrix of gradient
    #ifdef WITHDLIB
    dlib::matrix<double> calculateFiniteGradient(
        const dlib::matrix<double, 0, 1>& x0,
        const std::function<double(const dlib::matrix<double, 0, 1>)>& fn,
        unsigned int accuracy = 0
    );
    #endif // WITHDLIB
    arma::dmat calculateFiniteGradient(
        const arma::dcolvec& x0,
        const std::function<double(const arma::dcolvec&)>& fn,
        unsigned int accuracy = 0
    );

    /// @brief Calculate hessian matrix by finite differences
    /// @param x0
    /// @param fn
    /// @param accuracy
    /// @return matrix of hessian
    #ifdef WITHDLIB
    dlib::matrix<double> calculateFiniteHessian(
        const dlib::matrix<double, 0, 1>& x0,
        const std::function<double(const dlib::matrix<double, 0, 1>)>& fn,
        unsigned int accuracy = 0
    );
    #endif // WITHDLIB
    arma::dmat calculateFiniteHessian(
        const arma::dcolvec& x0,
        const std::function<double(const arma::dcolvec&)>& fn,
        unsigned int accuracy = 0
    );

    /// @brief Calculate hessians per simulation, using finite differences, given gradient
    /// @param x0
    /// @param grad
    /// @return vector of hessian matricies
    #ifdef WITHDLIB
    std::vector<dlib::matrix<double>> calculateFiniteHessianSimsUsingGrad(
        const dlib::matrix<double, 0, 1>& x0,
        const std::function<dlib::matrix<double>(const dlib::matrix<double, 0, 1>)>& grad
    );
    #endif // WITHDLIB
    std::vector<arma::dmat> calculateFiniteHessianSimsUsingGrad(
        const arma::dcolvec& x0,
        const std::function<arma::dmat(const arma::dcolvec&)>& grad
    );

    /// @brief Calculate hessians per simulation, using finite differences
    /// @param x0
    /// @param fn
    /// @param accuracy
    /// @return vector of hessian matricies
    #ifdef WITHDLIB
    std::vector<dlib::matrix<double>> calculateFiniteHessianSims(
        const dlib::matrix<double, 0, 1>& x0,
        const std::function<dlib::matrix<double>(const dlib::matrix<double, 0, 1>)>& fn,
        unsigned int accuracy = 0
    );
    #endif // WITHDLIB
    std::vector<arma::dmat> calculateFiniteHessianSims(
        const arma::dcolvec& x0,
        const std::function<arma::dmat(const arma::dcolvec&)>& fn,
        unsigned int accuracy = 0
    );
}

#endif // FINITE_DIFF_HPP