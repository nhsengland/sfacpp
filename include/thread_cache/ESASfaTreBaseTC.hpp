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
 * @file ESASfaTreBaseTC.hpp
 * @details Structs for thread cache approach
 */

#ifndef ESA_SFA_TRE_BASE_TC_HPP
#define ESA_SFA_TRE_BASE_TC_HPP

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

    struct WSWeightFromDensForPanel {
        arma::dmat lnDen;
        arma::dmat Sr, Kr;
        double Smax = 0;
        void ensureSize(unsigned int maxRow, unsigned int maxCol) {
            if (lnDen.n_rows != maxRow || lnDen.n_cols != maxCol) {
                lnDen.set_size(maxRow, maxCol);
            }
            // 1 x nsim
            if (Sr.n_cols < maxCol) {
                Sr.set_size(1, maxCol);
                Kr.set_size(1, maxCol);
            }
        }
        void zeros() {

        }
    };

    struct WSSimulatedHessianForFirm {

        arma::dmat g_bar_i, g_ir_less_gbar_i;
        arma::dmat h_bar_i, hess_comp_2, htmp;

        void ensureSize(unsigned int nParams) {
            // g_ir_bar is 1 x nParams
            if (g_bar_i.n_rows != 1 || g_bar_i.n_cols != nParams) {
                g_bar_i.set_size(1, nParams);
                g_bar_i.zeros();
            }
            // same for g_ir_less_gbar_i
            if (g_ir_less_gbar_i.n_rows != 1 || g_ir_less_gbar_i.n_cols != nParams) {
                g_ir_less_gbar_i.set_size(1, nParams);
                g_ir_less_gbar_i.zeros();
            }
            // h_bar_i is nParam x nParaml as is hess_comp_2
            if (h_bar_i.n_rows != nParams || h_bar_i.n_cols != nParams) {
                h_bar_i.set_size(nParams, nParams);
                h_bar_i.zeros();
                hess_comp_2.set_size(nParams, nParams);
                hess_comp_2.zeros();
                htmp.set_size(nParams, nParams);
            }
        }

        void zeros() {
            g_bar_i.zeros();
            h_bar_i.zeros();
            hess_comp_2.zeros();
        }
    };

    struct WSInternalAnalyticJacHess {
        // dedicate aligned buffers for y, x, zuit etc
        arma::dmat y, x, zuit, zvit, zui0, zvi0, zmuit;
        // scratch matricies
        arma::dmat a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10;
        arma::dmat a11, a12, a13, a14, a15, a16, a17, a18, a19, a20;
        arma::dmat a21, a22, a23, a24, a25, a26, a27, a28, a29, a30;
        arma::dmat a31, a32, a33, a34, a35, a36, a37, a38, a39, a40;
        arma::dmat a41, a42, a43, a44, a45;
        // k x k - reused evry time period
        arma::dmat h;
        // scratchpad for row transposes in hessian
        arma::dmat v1Scratch, v2Scratch;
        // jacobian temporary workspace
        arma::dmat jac;
        // for the simulate hessian, define some scratch matricies
        // for calculating the hessian - need to accumulate it for component 2
        arma::dmat h_bar_i, hessSumNt, hessAccC2;
        arma::dmat g_bar_i;
        arma::dmat gir;

        void ensureSize(int maxT, int k, int maxZ, int nsim, int nX, int nZuit, int nZvit, int nZui0, int nZvi0, int nZmuit = 0) {
            if (y.n_rows < maxT || y.n_cols < 1) y.set_size(maxT, 1);
            if (x.n_rows < maxT || x.n_cols < std::max(nX, 1)) x.set_size(maxT, std::max(nX, 1));
            if (zuit.n_rows < maxT || zuit.n_cols < std::max(nZuit, 1)) zuit.set_size(maxT, std::max(nZuit, 1));
            if (zvit.n_rows < maxT || zvit.n_cols < std::max(nZvit, 1)) zvit.set_size(maxT, std::max(nZvit, 1));
            if (zui0.n_rows < maxT || zui0.n_cols < std::max(nZui0, 1)) zui0.set_size(maxT, std::max(nZui0, 1));
            if (zvi0.n_rows < maxT || zvi0.n_cols < std::max(nZvi0, 1)) zvi0.set_size(maxT, std::max(nZvi0, 1));
            if (nZmuit > 0 && (zmuit.n_rows < maxT || zmuit.n_cols < nZmuit)) zmuit.set_size(maxT, nZmuit);
            if (gir.n_rows < nsim || gir.n_cols < k) {
                gir.set_size(nsim, k);
            }
            if (g_bar_i.n_rows != 1 || g_bar_i.n_cols < k) {
                g_bar_i.set_size(1, k);
            }
            if (h_bar_i.n_rows < k || h_bar_i.n_cols < k) {
                h_bar_i.set_size(k, k);
                hessSumNt.set_size(k, k);
            }
            if (hessAccC2.n_rows < k || hessAccC2.n_cols < k) {
                hessAccC2.set_size(k, k);
            }
            if (a1.n_rows < maxT || a1.n_cols < 1) {
                a1.set_size(maxT, 1);
                a2.set_size(maxT, 1);
                a3.set_size(maxT, 1);
                a4.set_size(maxT, 1);
                a5.set_size(maxT, 1);
                a6.set_size(maxT, 1);
                a7.set_size(maxT, 1);
                a8.set_size(maxT, 1);
                a9.set_size(maxT, 1);
                a10.set_size(maxT, 1);
                a11.set_size(maxT, 1);
                a12.set_size(maxT, 1);
                a13.set_size(maxT, 1);
                a14.set_size(maxT, 1);
                a15.set_size(maxT, 1);
                a16.set_size(maxT, 1);
                a17.set_size(maxT, 1);
                a18.set_size(maxT, 1);
                a19.set_size(maxT, 1);
                a20.set_size(maxT, 1);
                a21.set_size(maxT, 1);
                a22.set_size(maxT, 1);
                a23.set_size(maxT, 1);
                a24.set_size(maxT, 1);
                a25.set_size(maxT, 1);
                a26.set_size(maxT, 1);
                a27.set_size(maxT, 1);
                a28.set_size(maxT, 1);
                a29.set_size(maxT, 1);
                a30.set_size(maxT, 1);
                a31.set_size(maxT, 1);
                a32.set_size(maxT, 1);
                a33.set_size(maxT, 1);
                a34.set_size(maxT, 1);
                a35.set_size(maxT, 1);
                a36.set_size(maxT, 1);
                a37.set_size(maxT, 1);
                a38.set_size(maxT, 1);
                a39.set_size(maxT, 1);
                a40.set_size(maxT, 1);
                a41.set_size(maxT, 1);
                a42.set_size(maxT, 1);
                a43.set_size(maxT, 1);
                a44.set_size(maxT, 1);
                a45.set_size(maxT, 1);
            }
            // hessian buffer
            if (h.n_rows != k || h.n_cols != k) {
                h.set_size(k, k);
            }
            // jacobian buffer
            if (jac.n_rows < maxT || jac.n_cols < k) {
                jac.set_size(maxT, k);
            }
            if (v1Scratch.n_rows < maxZ) {
                v1Scratch.set_size(maxZ, 1);
            }
            if (v2Scratch.n_rows < maxZ) {
                v2Scratch.set_size(maxZ, 1);
            }
        }
    };

}


#endif // ESA_SFA_TRE_BASE_TC_HPP