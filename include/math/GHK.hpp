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

#ifndef GHK_LIB_HPP
#define GHK_LIB_HPP

#include <armadillo>
#include <cmath>
#include <limits>
#include <algorithm>
#include "esandist.hpp"

namespace ghk {

    // constants
    static const double kThresh = 5.0; 
    static const double m_sqrt1_2 = 0.70710678118654752440;
    static const double inf_neg = -std::numeric_limits<double>::infinity();

    /**
     * @brief robust Log-Sum-Exp
     * calculates log(sum(exp(x))) / N
     */
    inline double log_mean_exp(const arma::vec& log_vals) {
        double max_val = log_vals.max();
        if (max_val == inf_neg) return inf_neg;
        
        double sum = 0.0;
        for (const double val : log_vals) {
            if (val > inf_neg) {
                sum += std::exp(val - max_val);
            }
        }
        return max_val + std::log(sum / double(log_vals.n_elem));
    }

    /**
     * @brief main library function [single threaded]
     */
    template <typename T>
    inline double estim(const arma::mat& L, const arma::vec& a, const arma::vec& b, const arma::Base<double, T>& drawsIn) {
        const auto& draws = drawsIn.get_ref();
        int dim = L.n_rows;
        int nsim = draws.n_rows;
        
        arma::mat z(nsim, dim, arma::fill::zeros);
        arma::vec log_w(nsim, arma::fill::zeros);
        arma::vec mu(nsim);
        arma::vec ca(nsim), cb(nsim);

        for (int i = 0; i < dim; ++i) {
            double L_ii = L(i, i);
            double inv_L_ii = (std::abs(L_ii) > 1e-12) ? 1.0 / L_ii : 0.0;
            // calculate mean
            if (i > 0) {
                // z(nsim x i) * L_row(i x 1) -> mu(nsim x 1)
                mu = z.cols(0, i - 1) * L.submat(i, 0, i, i - 1).t();
            } else {
                mu.zeros();
            }
            // transform bounds
            if (inv_L_ii != 0.0) {
                ca = (a(i) - mu) * inv_L_ii;
                cb = (b(i) - mu) * inv_L_ii;
            } else {
                 ca.fill(inf_neg); 
                 cb.fill(inf_neg);
            }
            // simulation loop
            for (int s = 0; s < nsim; ++s) {
                // skip dead paths immediately
                if (log_w(s) == inf_neg) continue;
                double low = ca(s);
                double high = cb(s);
                //  validity check
                if (low >= high) {
                    log_w(s) = inf_neg;
                    continue;
                }
                double prob_diff = 0.0;
                // calculate 'erfc' vals
                // case 1 - right tail (low > 5.0)
                if (low > kThresh) {
                    double e_low = std::erfc(low * m_sqrt1_2);
                    double e_high = std::erfc(high * m_sqrt1_2);
                    prob_diff = 0.5 * (e_low - e_high);
                    if (prob_diff <= 0.0) {
                        log_w(s) = inf_neg;
                        continue;
                    }
                    log_w(s) += std::log(prob_diff);
                    // simulate Z
                    if (i < dim) {
                        double u = draws(s, i);
                        double target_erfc = e_low * (1.0 - u) + e_high * u;                        
                        double p_target = 1.0 - 0.5 * target_erfc;
                        if (p_target >= 1.0) p_target = 1.0 - 1e-16;
                        z(s, i) = esandist::ppf(p_target);
                    }
                }
                // case 2: left tail (high < -5.0)
                else if (high < -kThresh) {
                    double e_low = std::erfc(-low * m_sqrt1_2);
                    double e_high = std::erfc(-high * m_sqrt1_2);
                    prob_diff = 0.5 * (e_high - e_low);
                    if (prob_diff <= 0.0) {
                        log_w(s) = inf_neg;
                        continue;
                    }
                    log_w(s) += std::log(prob_diff);
                    if (i < dim) {
                        double u = draws(s, i);
                        double target_erfc = e_low + u * (e_high - e_low);
                        
                        double p_target = 0.5 * target_erfc;
                        if (p_target <= 0.0) p_target = 1e-100;
                        z(s, i) = esandist::ppf(p_target);
                    }
                }
                // case 3 central / standard
                else {
                    double p_l = arma::normcdf(low);
                    double p_h = arma::normcdf(high);
                    prob_diff = p_h - p_l;
                    if (prob_diff <= 1e-300) {
                        log_w(s) = inf_neg;
                        continue;
                    }
                    log_w(s) += std::log(prob_diff);
                    if (i < dim) {
                        double u = draws(s, i);
                        double p_target = p_l + u * prob_diff;
                        if (p_target >= 1.0) p_target = 1.0 - 1e-16;
                        if (p_target <= 0.0) p_target = 1e-16;
                        z(s, i) = esandist::ppf(p_target);
                    }
                }
            } // end loop
        }
        return log_mean_exp(log_w);
    }
}
#endif