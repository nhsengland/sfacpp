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
 * @file esandist.cpp
 * @brief Implementation file for the normal distribution utilities
 * @author Edmund Haacke
 * @date 2025-01-13
 */

#include "math/esandist.hpp"
#include <stdexcept>
#include <random>
#include <math.h>

#ifdef WITHDLIB
#include <dlib/statistics.h>
#endif

/// Generate a random draw from a normal distribution
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esandist::ndraw(const dlib::matrix<T>& mean, const dlib::matrix<T>& sd, const int seed){
    // check whether mean and sd are of the same length
    if (mean.nr() != sd.nr()){
        throw std::invalid_argument("The mean and standard deviation vectors must be of the same length.");
    }
    // expect that sd, mean are only 1 column wide
    if (mean.nc() != 1 || sd.nc() != 1){
        throw std::invalid_argument("The mean and standard deviation vectors must be column vectors.");
    }
    // setup the random number generator
    std::random_device rd;
    std::mt19937 rgen(rd());
    rgen.seed(seed);
    // output matrix
    dlib::matrix<T> output(mean.nr(), 1);
    // iterate thru the number of rows
    for (size_t i = 0; i < mean.nr(); ++i){
        // create a new random number generator with the mean and SD for that row
        std::normal_distribution<T> dist(mean(i), sd(i));
        // draw sample from the distribution
        output(i) = dist(rgen);
    }
    // return output column vector
    return output;
}
// explicit template instantiation
template dlib::matrix<double> esandist::ndraw(const dlib::matrix<double>&, const dlib::matrix<double>&, const int);
template dlib::matrix<float> esandist::ndraw(const dlib::matrix<float>&, const dlib::matrix<float>&, const int);
#endif // WITHDLIB

/// Generate a random draw from a normal distribution - armadillo implementation
template <typename T>
arma::Mat<T> esandist::ndraw(const arma::Mat<T>& mean, const arma::Mat<T>& sd, const int seed)
{
    // checks
    if (mean.n_rows != sd.n_rows) throw std::invalid_argument("Mean and sd vectors must be same length");
    // check that sd, mean are only 1 column wide
    if (mean.n_cols != 1 || sd.n_cols != 1) throw std::invalid_argument("Mean and sd vectors must be column vectors");
    // setup random number
    std::random_device rd;
    std::mt19937 rgen(rd());
    rgen.seed(seed);
    // output matrix 
    arma::Mat<T> out(mean.n_rows, 1);
    for (size_t i = 0; i < mean.n_rows; i++) {
        // create new random number generator with mean & sd for that row
        std::normal_distribution<T> dist(mean.at(i), sd.at(i));
        // draw sampe from the distribution
        out(i) = dist(rgen);
    }
    return out;
}
// explicit template instantisation
template arma::Mat<double> esandist::ndraw(const arma::Mat<double>&, const arma::Mat<double>&, const int);
template arma::Mat<float> esandist::ndraw(const arma::Mat<float>&, const arma::Mat<float>&, const int);

/// Compute the density of the normal distribution ala 'dnorm' in R
// template <typename T>
#ifdef WITHDLIB
dlib::matrix<double> esandist::dnorm_cpp(const dlib::matrix<double>& x, double mean, double sd){
    dlib::matrix<double> result(x.nr(), x.nc());
    double coeff = 1.0 / (sd * std::sqrt(2.0 * M_PI));
    for (long r = 0; r < x.nr(); ++r){
        for (long c = 0; c < x.nc(); ++c){
            double z = (x(r, c) - mean) / sd;
            result(r, c) = coeff * std::exp(-0.5 * z * z);
        }
    }
    return result;
}
#endif // WITHDLIB
// template dlib::matrix<double> esandist::dnorm(const dlib::matrix<double>&, double, double);
// template dlib::matrix<float> esandist::dnorm(const dlib::matrix<double>&, double, double);


/// Compute the cumulative distribution function of the normal distribution ala 'pnorm' in R
// template <typename T>
#ifdef WITHDLIB
dlib::matrix<double> esandist::pnorm_cpp(const dlib::matrix<double>& x, double mean, double sd){
    dlib::matrix<double> result(x.nr(), x.nc());
    for (long r = 0; r < x.nr(); ++r){
        for (long c = 0; c < x.nc(); ++c){
            double z = (x(r, c) - mean) / sd;
            // erfc computes the complementary error function
            result(r, c) = 0.5 * std::erfc(-z / std::sqrt(2.0));
        }
    }
    return result;
}
#endif // WITHDLIB
// template dlib::matrix<double> esandist::pnorm(const dlib::matrix<double>& x, double, double);
// template dlib::matrix<float> esandist::pnorm(const dlib::matrix<float>& x, double, double);

/// Calculate the percentage point function (ppf) - inverse of cdf - percentiles
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esandist::ppf(const dlib::matrix<T>& mat, double mean, double stddev){
    boost::math::normal_distribution<T> normalDist(mean, stddev);
    dlib::matrix<T> result(mat.nr(), mat.nc());
    for (long r = 0; r < mat.nr(); ++r){
        for (long c = 0; c < mat.nc(); ++c){
            T value = mat(r, c);
            // check within valid bounds to avoid overflow error?
            // if (value <= 0.0) {
            //     value = std::numeric_limits<double>::min();
            // } else if (value >= 1.0) {
            //     value = 1.0 - std::numeric_limits<double>::epsilon();
            // }
            if (!std::isnan(value)){
                result(r, c) = boost::math::quantile(normalDist, value);
            } else {
                result(r, c) = std::numeric_limits<T>::quiet_NaN();
            }
            // result(r, c) = dlib::inv_cdf(normal_distribution, mat(r, c));
        }
    }
    return result;
}
template dlib::matrix<double> esandist::ppf(const dlib::matrix<double>&, double, double);
template dlib::matrix<float> esandist::ppf(const dlib::matrix<float>&, double, double);
#endif // WITHDLIB

/// Calculate percentage point function (ppf) - inverse of CDF - percentiles
template <typename T>
arma::dmat esandist::ppf(const arma::Base<double, T>& matIn, double mean, double stddev)
{
    const auto& mat = matIn.get_ref();
    boost::math::normal_distribution<double> normalDist(mean, stddev);
    arma::dmat result(mat);
    result.transform([&normalDist](double val){
        if (!std::isnan(val)) {
            return boost::math::quantile(normalDist, val);
        }
        return std::numeric_limits<double>::quiet_NaN();
    });
    return result;
}
// explicit template instansiation
template arma::dmat esandist::ppf<arma::dmat>(const arma::Base<double, arma::dmat>&, double, double);
template arma::dmat esandist::ppf<arma::subview<double>>(const arma::Base<double, arma::subview<double>>&, double, double);

double esandist::ppf(double x, double mean, double stddev)
{
    boost::math::normal_distribution<double> normalDist(mean, stddev);
    if (!std::isnan(x)) {
        return boost::math::quantile(normalDist, x);
    }
    return std::numeric_limits<double>::quiet_NaN();
}