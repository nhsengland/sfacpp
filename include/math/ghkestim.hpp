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
 * @file ghkestim.hpp
 * @details
 * Header only to avoid one-definition rule (ODR) violation wr.t. arma::Mat
 */
#ifndef ESA_GHK_ESTIM_HPP
#define ESA_GHK_ESTIM_HPP

// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif //RPACKAGE

#include <memory>
#include <stdexcept>
#include "utils/ThreadContext.hpp"
#include "thread_cache/esamathTC.hpp"
// #include "math/manual_linalg.hpp"
#include "math/esandist.hpp"
#include "utils/esautils.hpp"


namespace esamath {

    /**
     * @brief reconfigured ghk_estim from Baduenko GTRE implementation
     */
    template <typename T1, typename T2, typename T3, typename T4>
    inline double ghk_estim(
        const arma::Base<double, T1>& xIn,
        const arma::Base<double, T2>& fromIn,
        const arma::Base<double, T3>& toIn,
        const arma::Base<double, T4>& drawsIn,
        const bool is_cholesky_vcov = true,
        const int nsim = 10000
    )
    {
        // thread-local persistent storage
        ThreadContext* ctx = getContext();
        if (!ctx->wsGhkEstim) {
            ctx->wsGhkEstim = std::make_unique<thread_cache_math::WSGhkEstim>();
        }
        thread_cache_math::WSGhkEstim& ws = *ctx->wsGhkEstim;
        // unwrap armadillo matricies
        const auto& x = xIn.get_ref();
        const auto& from = fromIn.get_ref();
        const auto& to = toIn.get_ref();
        const auto& draws = drawsIn.get_ref();
        // checks - to & from args need to be column or row vectors
        if ((from.n_cols != 1 && from.n_rows != 1) || (to.n_cols != 1 && to.n_rows != 1)) {
            throw std::invalid_argument("The 'from' and 'to' matricies must be a vector");
        }
        int k = x.n_cols;
        // ensure workspace matricies are correct size
        // dont have a definitive view on their final sizes over whole programme
        ws.ensureSize(k, nsim);
        // arma::dmat vcov;
        // arma::subview<double> vcov = ws.vcov.submat(arma::span(0, k - 1), arma::span(0, k - 1));
        arma::dmat vcov;
        if (is_cholesky_vcov) {
            vcov = arma::dmat(x);
        } else {
            // calculate cholesky decomposition - get lower triangular matrix
            vcov = arma::chol(x, "lower").t();
            // manual_chol(x, vcov);
            // manual_linalg::manual_chol(x, vcov);
            // throw std::runtime_error("to implement - insert manual_chol");
        }
        // check the draws - need to have enough columns for the number of columns in the decomp
        if (draws.n_cols < (k - 1) && k > 1) {
            throw std::invalid_argument("Insufficient columns in 'draws'");
        }
        // results vector - fill with ones
        arma::subview<double> prY = ws.prY.submat(arma::span(0, nsim - 1), arma::span(0, 0));
        prY.ones();
        // Use two working vectors for all temporary ops
        arma::subview<double> v1 = ws.w1.submat(0, 0, nsim - 1, 0);
        arma::subview<double> v2 = ws.w2.submat(0, 0, nsim - 1, 0);
        // setup Z history view
        // Ssafety check: if k=1, we don't use z, but we need a valid view reference
        unsigned int z_cols_needed = (k > 1) ? k - 1 : 1;
        arma::subview<double> z = ws.z.submat(0, 0, nsim - 1, z_cols_needed - 1);
        if (k > 1) z.zeros();
        // iterate thru the dimensions
        for (size_t j = 0; j < k; j++) {
            double sigma = vcov(j, j);
            // here, v1 is equivalent to a1; v2 is equivalent to b1
            if (j > 0) {
                v1 = z.cols(0, j - 1) * vcov.submat(j, 0, j, j - 1).t();
                // calculate bounds 
                v2 = (to(j) - v1) / sigma;
                v1 = (from(j) - v1) / sigma;
            } else {
                v1.fill(from(0) / sigma);
                v2.fill(to(0) / sigma);
            }
            // cdf of a1, b1
            v1 = arma::normcdf(v1, 0.0, 1.0);
            v2 = arma::normcdf(v2, 0.0, 1.0);
            // calculate probability
            prY %= (v2 - v1);
            if (j < (k - 1)) {
                v1 += draws.col(j) % (v2 - v1);
                z.col(j) = esandist::ppf(v1);
            }
        }
        // filter any infinite or NaN values
        arma::dmat prYFiltered = esautils::filterRowsInvalidNumbers(prY);
        // return the mean
        return arma::mean(arma::mean(prYFiltered));
    }
}

#endif // ESA_GHK_ESTIM_HPP