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
 * @file ESASfaGtreBadTC.hpp
 * @details Structs for thread cache
 */

#ifndef ESA_SFA_GTRE_BAD_TC_HPP
#define ESA_SFA_GTRE_BAD_TC_HPP

#include <vector>
// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif

namespace thread_cache_gtre {

    struct WSDensityHalfNormal {
        // maxT x 1
        arma::dmat xb, sigma2uit, sigmauit, sigma2vit, sigmavit, sigma2vi0, sigmavi0, sigma2ui0, sigmaui0;
        arma::dmat sigma2, sigma, lambda;
        // maxT x nsim
        arma::dmat draw_vi0_scaled, draw_ui0_scaled, epsr, c1In, c2In, cdfVal;

        void ensureSize(unsigned int maxT, unsigned int nsim) {
            // maxT x nsim
            if (draw_vi0_scaled.n_rows != maxT || draw_vi0_scaled.n_cols != nsim) {
                draw_vi0_scaled.set_size(maxT, nsim);
                draw_ui0_scaled.set_size(maxT, nsim);
                epsr.set_size(maxT, nsim);
                c1In.set_size(maxT, nsim);
                c2In.set_size(maxT, nsim);
                cdfVal.set_size(maxT, nsim);
            }
            // maxT x 1
            if (xb.n_rows != maxT || xb.n_cols != 1) {
                xb.set_size(maxT, 1);
                sigma2uit.set_size(maxT, 1);
                sigmauit.set_size(maxT, 1);
                sigma2vit.set_size(maxT, 1);
                sigmavit.set_size(maxT, 1);
                sigma2vi0.set_size(maxT, 1);
                sigmavi0.set_size(maxT, 1);
                sigma2ui0.set_size(maxT, 1);
                sigmaui0.set_size(maxT, 1);
                sigma2.set_size(maxT, 1);
                sigma.set_size(maxT, 1);
                lambda.set_size(maxT, 1);
            }

        }
    };

    struct WSGradHessInner {

        // vector of vector of matricies store hessians over r, i, and t
        std::vector<std::vector<arma::dmat>> hess_itr;
        // vector of matrix for jacobians - nsim elements, with elements of (maxT x nParam)
        std::vector<arma::dmat> jacSims;

        void ensureSize(unsigned int maxT, unsigned int nsim, unsigned int nParams) {
            // iterate thru vector of vector of hessian matricies, and preallocate the internal
            // matricies which are nParams x nParams in size
            if (hess_itr.size() != nsim) {
                hess_itr.resize(nsim);
                // iterate thru, allocate internal matricies
                for (auto& vec : hess_itr) {
                    if (vec.size() != maxT) {
                        vec.resize(maxT);
                    }
                    for (auto& mat : vec) {
                        mat.set_size(nParams, nParams);
                    }
                }
            }
        }
        
        void zeros() {

        }
    };

}

#endif // ESA_SFA_GTRE_BAD_TC_HPP