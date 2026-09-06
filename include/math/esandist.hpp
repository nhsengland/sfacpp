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
 * @file esandist.hpp
 * @brief Header file for the normal distribution utilities
 * @author Edmund Haacke
 * @date 2025-01-13
 */

#ifndef ESA_N_DIST_HPP
#define ESA_N_DIST_HPP

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
// [[Rcpp::depends(BH)]]
#include <boost/math/distributions/normal.hpp>

namespace esandist {

    /// @brief Generate a random draw from a normal distribution
    /// @param mean The mean of the distribution
    /// @param sd The standard deviation of the distribution
    /// @return A matrix with random draw from the normal distribution
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> ndraw(const dlib::matrix<T>& mean, const dlib::matrix<T>& sd, const int seed);
    #endif // WITHDLIB
    template <typename T>
    arma::Mat<T> ndraw(const arma::Mat<T>& mean, const arma::Mat<T>& sd, const int seed);


    /// @brief Compute the density of the normal distribution ala 'dnorm' in R
    /// @param x The matrix of values
    /// @param mean The mean of the distribution
    /// @param sd The standard deviation of the distribution
    /// @return The density of the normal distribution
    /// @example dlib::matrix<double> density = esandist::dnorm(x, 0.0, 1.0);
    // template <typename T>
    #ifdef WITHDLIB
    dlib::matrix<double> dnorm_cpp(const dlib::matrix<double>& x, double mean = 0.0, double sd = 1.0);
    #endif // WITHDLIB

    /// @brief compute the cumulative distribution function of the normal distribution ala 'pnorm' in R
    /// @param x The matrix of values
    /// @param mean The mean of the distribution
    /// @param sd The standard deviation of the distribution
    /// @return The cumulative distribution function of the normal distribution
    /// @example dlib::matrix<double> cdf = esandist::pnorm(x, 0.0, 1.0);
    // template <typename T>
    #ifdef WITHDLIB
    dlib::matrix<double> pnorm_cpp(const dlib::matrix<double>& x, double mean = 0.0, double sd = 1.0);
    #endif // WITHDLIB

    /// @brief Calculate the percentage point function (ppf) - inverse of cdf - percentiles
    /// @param mat Matrix of probabilities
    /// @param mean Mean of the distribution
    /// @param stddev Standard deviation of the distribution
    /// @return Matrix of percentiles
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> ppf(const dlib::matrix<T>& mat, double mean = 0.0, double stddev = 1.0);
    #endif // WITHDLIB
    /// armadillo implementation
    template <typename T>
    arma::dmat ppf(const arma::Base<double, T>& matIn, double mean = 0.0, double stddev = 1.0);
    double ppf(double x, double mean = 0.0, double stddev = 1.0);

    template <typename T1>
    arma::Mat<typename T1::elem_type> normpdf_boost(
        const arma::Base<typename T1::elem_type, T1>& X,
        double mu = 0.0,
        double sigma = 1.0
    )
    {
        typedef typename T1::elem_type eT;
        boost::math::normal_distribution<eT> dist(mu, sigma);
        // force evaluation of X, create deep copy since returning new matrix
        arma::Mat<eT> out = X.get_ref();
        out.transform([&dist](eT val) {
            return boost::math::pdf(dist, val);
        });
        return out;
    }

    template <typename T1>
    arma::Mat<typename T1::elem_type> normcdf_boost(
        const arma::Base<typename T1::elem_type, T1>& X,
        double mu = 0.0,
        double sigma = 1.0
    )
    {
        typedef typename T1::elem_type eT;
        boost::math::normal_distribution<eT> dist(mu, sigma);
        // force evaluation of X, create deep copy since returning new matrix
        arma::Mat<eT> out = X.get_ref();
        out.transform([&dist](eT val) {
            return boost::math::cdf(dist, val);
        });
        return out;
    }

}

#endif // ESA_N_DIST_HPP