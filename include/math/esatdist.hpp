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
 * @file esatdist.hpp
 * @brief Header file for the ESATDist class
 * @author Edmund Haacke
 * 
 */

#ifndef ESA_T_DIST_HPP
#define ESA_T_DIST_HPP

// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif //RPACKAGE
// --- end armadillo ---

#ifdef WITHDLIB
#include <dlib/matrix.h>
#endif // WITHDLIB

namespace esatdist {
    
    /// @brief density function for central t-distribution
    /// @param x The value at which to evaluate the density
    /// @param df The degrees of freedom
    /// @return The density at x
    double dt(double x, double df);
    arma::dmat dt(const arma::dmat& x, double df);
    #ifdef WITHDLIB
    dlib::matrix<double> dt(const dlib::matrix<double>& x, double df);
    #endif // WITHDLIB

    /// @brief distribution function for central t-distribution
    /// @param x The value at which to evaluate the distribution
    /// @param df The degrees of freedom
    /// @return The distribution at x
    double pt(double x, double df, double ncp = 0.0, bool lowerTail = true);
    arma::dmat pt(const arma::dmat& x, double df, double ncp = 0.0, bool lowerTail = true);
    #ifdef WITHDLIB
    dlib::matrix<double> pt(const dlib::matrix<double>& x, double df, double ncp = 0.0, bool lowerTail = true);
    #endif // WITHDLIB

    /// @brief quantile function for central t-distribution
    /// @param p The probability at which to evaluate the quantile
    /// @param df The degrees of freedom
    /// @return The quantile at p
    double qt(double p, double df, double ncp = 0.0, bool lowerTail = true);
    arma::dmat qt(const arma::dmat& p, double df, double ncp = 0.0, bool lowerTail = true);
    #ifdef WITHDLIB
    dlib::matrix<double> qt(const dlib::matrix<double>& p, double df, double ncp = 0.0, bool lowerTail = true);
    #endif // WITHDLIB
}

#endif // ESA_T_DIST_HPP