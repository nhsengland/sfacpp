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
 * @file esatdist.cpp
 * @brief Implementation file for t-distribution functions
 * @author Edmund Haacke
 * @date 2025-01-05
 */

#include "math/esatdist.hpp"
#include <random>
#include <cmath>
#include <limits>
// [[Rcpp::depends(BH)]]
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/non_central_t.hpp>


/// Density function for central t-distribution
double esatdist::dt(double x, double df){
    double gamma1 = std::tgamma((df + 1) / 2.0);
    double gamma2 = std::tgamma(df / 2.0);
    double sqrt_pi_df = std::sqrt(df * M_PI);
    return (gamma1 / (gamma2 * sqrt_pi_df)) * std::pow(1 + (x * x) / df, -(df + 1) / 2.0);
}

arma::dmat esatdist::dt(const arma::dmat& x, double df)
{
    arma::dmat result(x.n_rows, x.n_cols);
    for (unsigned int r = 0; r < x.n_rows; r++){
        for (unsigned int c = 0; c < x.n_cols; c++){
            result(r, c) = dt(x(r, c), df);
        }
    }
    return result;
}

#ifdef WITHDLIB
/// Density function for central t-distribution
dlib::matrix<double> esatdist::dt(const dlib::matrix<double>& x, double df){
    dlib::matrix<double> result(x.nr(), x.nc());
    for (long r = 0; r < x.nr(); ++r){
        for (long c = 0; c < x.nc(); ++c){
            result(r, c) = dt(x(r, c), df);
        }
    }
    return result;
}
#endif // WITHDLIB

/// Distribution function for central t-distribution
double esatdist::pt(double x, double df, double ncp, bool lowerTail){
    try{
        if (lowerTail){
            return boost::math::cdf(boost::math::non_central_t(df, ncp), x);
        } else {
            return boost::math::cdf(boost::math::complement(boost::math::non_central_t(df, ncp), x));
        }
    } catch(const std::exception& e){
        std::cerr << e.what() << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }
}

arma::dmat esatdist::pt(const arma::dmat& x, double df, double ncp, bool lowerTail)
{
    arma::dmat result(x.n_rows, x.n_cols);
    for (unsigned int r = 0; r < x.n_rows; r++) {
        for (unsigned int c = 0; c < x.n_cols; c++) {
            result(r, c) = pt(x(r, c), df, ncp, lowerTail);
        }
    }
    return result;
}

#ifdef WITHDLIB
/// Distribution function for central t-distribution
dlib::matrix<double> esatdist::pt(const dlib::matrix<double>& x, double df, double ncp, bool lowerTail){
    dlib::matrix<double> result(x.nr(), x.nc());
    for (long r = 0; r < x.nr(); ++r){
        for (long c = 0; c < x.nc(); ++c){
            result(r, c) = pt(x(r, c), df, ncp, lowerTail);
        }
    }
    return result;
}
#endif // WITHDLIB

/// Quantile function for central t-distribution
double esatdist::qt(double p, double df, double ncp, bool lowerTail){
    try{
        if (lowerTail){
            return boost::math::quantile(boost::math::non_central_t(df, ncp), p);
        } else {
            return boost::math::quantile(boost::math::complement(boost::math::non_central_t(df, ncp), p));
        }
        // boost::math::non_central_t dist(df, ncp);
        // return boost::math::quantile(dist, p);
    } catch (const std::exception& e){
        std::cerr << e.what() << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }
}

arma::dmat esatdist::qt(const arma::dmat&p, double df, double ncp, bool lowerTail)
{
    arma::dmat result(p.n_rows, p.n_cols);
    for (unsigned int r = 0; r < p.n_rows; r++) {
        for (unsigned int c = 0; c < p.n_cols; c++) {
            result(r, c) = qt(p(r, c), df, ncp, lowerTail);
        }
    }
    return result;
}

#ifdef WITHDLIB
/// Quantile function for central t-distribution
dlib::matrix<double> esatdist::qt(const dlib::matrix<double>& p, double df, double ncp, bool lowerTail){
    dlib::matrix<double> result(p.nr(), p.nc());
    for (long r = 0; r < p.nr(); ++r){
        for (long c = 0; c < p.nc(); ++c){
            result(r, c) = qt(p(r, c), df, ncp, lowerTail);
        }
    }
    return result;
}
#endif // WITHDLIB