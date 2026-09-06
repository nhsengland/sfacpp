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
 * @file esamathTC.hpp
 * @details 
 * @author edmund haacke
 * @date 2025-12-15 
*/

#ifndef ESA_MATH_TC_HPP
#define ESA_MATH_TC_HPP

// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif


namespace thread_cache_math {

    struct WSGhkEstim {
        // k x k
        arma::dmat vcov;
        // nsim x (k - 1)
        arma::dmat z;
        // nsim x 1
        // arma::dmat prY, a1, b1, sumCjkNk, a1Internal, b1Internal, dPpfInner;
        arma::dmat prY;
        // nsim x 2 - consolidated workspace
        // arma::dmat work;
        arma::dmat w1, w2;

        void ensureSize(unsigned int k, unsigned int nsim) {
            // k x k
            if (vcov.n_rows < k || vcov.n_cols < k) {
                vcov.set_size(k, k);
            }
            if (z.n_rows < nsim || z.n_cols < (k - 1)) {
                z.set_size(nsim, k - 1);
            }
            if (prY.n_rows < nsim || prY.n_cols < 1) {
                prY.set_size(nsim, 1);
            //     a1.set_size(nsim, 1);
            //     b1.set_size(nsim, 1);
            //     sumCjkNk.set_size(nsim, 1);
            //     a1Internal.set_size(nsim, 1);
            //     b1Internal.set_size(nsim, 1);
            //     dPpfInner.set_size(nsim, 1);
            }
            if (w1.n_rows < nsim || w1.n_cols < 1) w1.set_size(nsim, 1);
            if (w2.n_rows < nsim || w2.n_cols < 1) w2.set_size(nsim, 1);
            // if (work.n_rows < nsim || work.n_cols < 2) {
            //     work.set_size(nsim, 2);
            // }
        }
    };

}

#endif // ESA_MATH_TC_HPP