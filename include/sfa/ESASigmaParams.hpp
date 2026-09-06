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


#ifndef ESA_SIGMA_PARAMS_HPP
#define ESA_SIGMA_PARAMS_HPP

#include <cmath>
#include <optional>
#include <limits>
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"

#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#endif


struct ESASigmaParams {

    // double sigma2uit;
    // double sigma2vit;
    // double sigma;
    // double lambda;
    // std::optional<double> sigmavi0;
    // std::optional<double> sigmaui0;
    double mean_s2uit = std::numeric_limits<double>::quiet_NaN();
    double mean_s2vit = std::numeric_limits<double>::quiet_NaN();
    double mean_s2vi0 = std::numeric_limits<double>::quiet_NaN();
    double mean_s2ui0 = std::numeric_limits<double>::quiet_NaN();
    double s_uit = std::numeric_limits<double>::quiet_NaN();
    double s_vit = std::numeric_limits<double>::quiet_NaN();
    double s_vi0 = std::numeric_limits<double>::quiet_NaN();
    double s_ui0 = std::numeric_limits<double>::quiet_NaN();
    // sigma_uit / sigma_vit
    double lambda = std::numeric_limits<double>::quiet_NaN();
    // sigma_u0 / sigma_v0
    double lambda_0 = std::numeric_limits<double>::quiet_NaN();
    // sigma_u0 / sigmau
    double BigLambda = std::numeric_limits<double>::quiet_NaN();


    // default 
    ESASigmaParams() {}

    /**
     * Constructor
     * @param s2uit log-variance for uit (e.g., exp(zuit * b_zuit))
     * @param s2vit log-variance for vit (e.g., exp(zvit * b_zvit))
     * @param s2vi0 log-variance for vi0 (e.g., exp(zvi0 * b_zvi0))
     * @param s2ui0 log-variance for ui0 (e.g., exp(zui0 * b_zui0))
     */
    ESASigmaParams(
        const arma::dmat& s2uit,
        const arma::dmat& s2vit,
        const std::optional<arma::dmat>& s2vi0 = std::nullopt,
        const std::optional<arma::dmat>& s2ui0 = std::nullopt
    ) {
        if (s2uit.n_cols != 1) throw std::invalid_argument("'s2uit' expected to have 1 column");
        if (s2vit.n_cols != 1) throw std::invalid_argument("'s2vit' expected to have 1 column");
        // calculate mean of s2uit, s2vit
        mean_s2uit = arma::accu(s2uit) / s2uit.n_elem;
        mean_s2vit = arma::accu(s2vit) / s2vit.n_elem;
        s_uit = arma::accu(arma::sqrt(s2uit)) / s2uit.n_elem;
        s_vit = arma::accu(arma::sqrt(s2vit)) / s2vit.n_elem;
        // calculate lambda for each observation
        lambda = arma::accu( (arma::sqrt(s2uit) / arma::sqrt(s2vit)) ) / s2uit.n_elem;
        if (s2vi0) {
            if (s2vi0.value().n_cols != 1) throw std::invalid_argument("'s2vi0' expected to have 1 column");
            mean_s2vi0 = arma::accu(s2vi0.value()) / s2vi0.value().n_elem;
            s_vi0 = arma::accu( arma::sqrt(s2vi0.value()) ) / s2vi0.value().n_elem;
        }
        if (s2ui0) {
            if (s2ui0.value().n_cols != 1) throw std::invalid_argument("'s2ui0' expected to have 1 column");
            mean_s2ui0 = arma::accu(s2ui0.value()) / s2ui0.value().n_elem;
            s_ui0 = arma::accu( arma::sqrt(s2ui0.value()) ) / s2ui0.value().n_elem;
        }
        if (s2vi0.has_value() && s2ui0.has_value()) {
            // calculate lambda0, BigLambda
            lambda_0 = arma::accu( (arma::sqrt(s2ui0.value()) / arma::sqrt(s2vi0.value())) ) / s2vi0.value().n_elem;
            BigLambda = arma::accu( (arma::sqrt(s2ui0.value()) / arma::sqrt(s2uit)) ) / s2ui0.value().n_elem;
        }
    }

};

#endif // ESA_SIGMA_PARAMS_HPP