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
 * @file ESASfaTreGreeneTC.hpp
 * @details structs for thread cache approach
 */

#ifndef ESA_SFA_TRE_GREENE_TC_HPP
#define ESA_SFA_TRE_GREENE_TC_HPP

#include <vector>
#include <algorithm>
// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif

namespace thread_cache {

    // structure to hold temp matricies for calculation, allocated once per thread
    struct WSOperatorInner {
        // dedicated aligned buffers for x, zuit, etc
        // arma::dmat y, x, zuit, zvit, zvi0;
        arma::dmat ldr, lnDen;
        arma::dmat Sr, Kr;
        
        void ensureSize(
            unsigned int maxRow,
            unsigned int maxCol
            // unsigned int nX,
            // unsigned int nZuit,
            // unsigned int nZvit,
            // unsigned int nZvi0
        ) {
            // if (y.n_rows < maxRow || y.n_cols < 1) y.set_size(maxRow, 1);
            // if (x.n_rows < maxRow || x.n_cols < std::max(nX, 1)) x.set_size(maxRow, std::max(nX, 1));
            // if (zuit.n_rows < maxRow || zuit.n_cols < std::max(nZuit, 1)) zuit.set_size(maxRow, std::max(nZuit, 1));
            // if (zvit.n_rows < maxRow || zvit.n_cols < std::max(nZvit, 1)) zvit.set_size(maxRow, std::max(nZvit, 1));
            // if (zvi0.n_rows < maxRow || zvi0.n_cols < std::max(nZvi0, 1)) zvi0.set_size(maxRow, std::max(nZvi0, 1));
            // maxT x nsim
            if (ldr.n_rows < maxRow || ldr.n_cols < maxCol) {
                ldr.set_size(maxRow, maxCol);
                lnDen.set_size(maxRow, maxCol);
            }
            // 1 x nsim
            if (Sr.n_cols < maxCol) {
                Sr.set_size(1, maxCol);
                Kr.set_size(1, maxCol);
            }
        }
        void zeros() {
            ldr.zeros();
            lnDen.zeros();
            Sr.zeros();
            Kr.zeros();
        }
    };

    // structure to hold temp matricies for density calculation
    // allocated once per thread
    struct WSPanelDensityHalfNormal {
        // dedicated aligned buffers for x, zuit, etc
        arma::dmat y, x, zuit, zvit, zvi0;
        // maxT x 10 atlas - so its contiguous in memory
        // arma::dmat atlas;
        arma::dmat a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10;
        arma::dmat a11, a12, a13, a14, a15;
        // larger matricies (nT x nsim)
        arma::dmat a_i, eps_r, c1In, c2In, cdfVal;
        // reserve memory since know maxT (only if too small)
        void ensureSize(
            int maxRow,
            int maxCol,
            int nX,
            int nZuit,
            int nZvit,
            int nZvi0
        ) {
            if (y.n_rows < maxRow || y.n_cols < 1) y.set_size(maxRow, 1);
            if (x.n_rows < maxRow || x.n_cols < std::max(nX, 1)) x.set_size(maxRow, std::max(nX, 1));
            if (zuit.n_rows < maxRow || zuit.n_cols < std::max(nZuit, 1)) zuit.set_size(maxRow, std::max(nZuit, 1));
            if (zvit.n_rows < maxRow || zvit.n_cols < std::max(nZvit, 1)) zvit.set_size(maxRow, std::max(nZvit, 1));
            if (zvi0.n_rows < maxRow || zvi0.n_cols < std::max(nZvi0, 1)) zvi0.set_size(maxRow, std::max(nZvi0, 1));
            // these are maxT x nsim
            if (eps_r.n_rows < maxRow || eps_r.n_cols < maxCol) {
                a_i.set_size(maxRow, maxCol);
                eps_r.set_size(maxRow, maxCol);
                c1In.set_size(maxRow, maxCol);
                c2In.set_size(maxRow, maxCol);
                cdfVal.set_size(maxRow, maxCol);
            }
            if (a1.n_rows < maxRow || a1.n_cols < 1) {
                a1.set_size(maxRow, 1);
                a2.set_size(maxRow, 1);
                a3.set_size(maxRow, 1);
                a4.set_size(maxRow, 1);
                a5.set_size(maxRow, 1);
                a6.set_size(maxRow, 1);
                a7.set_size(maxRow, 1);
                a8.set_size(maxRow, 1);
                a9.set_size(maxRow, 1);
                a10.set_size(maxRow, 1);
                a11.set_size(maxRow, 1);
                a12.set_size(maxRow, 1);
                a13.set_size(maxRow, 1);
                a14.set_size(maxRow, 1);
                a15.set_size(maxRow, 1);
            }
        }
        void zeros() {
            a_i.zeros(); eps_r.zeros();
        }
    };

    // ---- gradient or/and hessian calculation ----
    // structure to hold temp matricies, allocated once per thread
    struct WSGradHessPanel {
        // dedicated aligned buffers for x, zuit, etc
        // arma::dmat y, x, zuit, zvit, zvi0;
        arma::dmat jac;
        arma::dmat dens;
        arma::dmat Qir;
        // hess is nParam x nParam
        arma::dmat hess;

        void ensureSize(
            unsigned int maxRow,
            unsigned int maxCol,
            unsigned int nParams
            // unsigned int nX,
            // unsigned int nZuit,
            // unsigned int nZvit,
            // unsigned int nZvi0
        ) {
            // if (y.n_rows < maxRow || y.n_cols < 1) y.set_size(maxRow, 1);
            // if (x.n_rows < maxRow || x.n_cols < std::max(nX, 1)) x.set_size(maxRow, std::max(nX, 1));
            // if (zuit.n_rows < maxRow || zuit.n_cols < std::max(nZuit, 1)) zuit.set_size(maxRow, std::max(nZuit, 1));
            // if (zvit.n_rows < maxRow || zvit.n_cols < std::max(nZvit, 1)) zvit.set_size(maxRow, std::max(nZvit, 1));
            // if (zvi0.n_rows < maxRow || zvi0.n_cols < std::max(nZvi0, 1)) zvi0.set_size(maxRow, std::max(nZvi0, 1));
            // for maxT x nsim
            if (dens.n_rows != maxRow || dens.n_cols != maxCol) {
                dens.set_size(maxRow, maxCol);
            }
            // for 1 x nsim
            if (Qir.n_rows != 1 || Qir.n_cols != maxCol) {
                Qir.set_size(1, maxCol);
            }
            // final jacobian is maxT x nParams
            if (jac.n_rows != maxRow || jac.n_cols != nParams) {
                jac.set_size(maxRow, nParams);
                jac.zeros();
            }
            // nParam x nParams
            if (hess.n_rows != nParams || hess.n_cols != nParams) {
                hess.set_size(nParams, nParams);
            }
        }
    };
}

#endif // ESA_SFA_TRE_GREENE_TC_HPP