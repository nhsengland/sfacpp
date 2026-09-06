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
 * @file ESASfaEffGtreTC.hpp
 * @details structs for thread cache approach
 * @author edmund haacke
 * @date 2025-12-15
 */

#ifndef ESA_SFA_EFF_GTRE_TC_HPP
#define ESA_SFA_EFF_GTRE_TC_HPP 

#include <limits>
// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif

namespace thread_cache_eff_gtre {

    struct WSTrePanelEfficiency {
        arma::dmat zui0Dummy;

        void ensureSize(int nT) {
            if (zui0Dummy.n_rows != nT || zui0Dummy.n_cols != 1) {
                zui0Dummy.set_size(nT, 1);
            }
        }
    };

    struct WSGtrePanelEfficiency {
        // data points - so its aligned
        arma::dmat y, x, zuit, zvit, zui0, zvi0;
        // maxT, 1
        arma::dmat xb, sigma2uit, sigma2vit, sigma2ui0, sigma2vi0, eps;
        // maxT + 1, 1
        arma::dmat giveUp, vSigs, from, Rr, expTu_i, tRow, phiBarNumTo;
        // maxT, maxT + 1
        arma::dmat A;
        // maxT + 1, maxT + 1
        arma::dmat V, Vinv, lamInternal, Lambda, t, LambdaChol;
        // maxT, maxT
        arma::dmat sigP1, sigP2, Sigma, SigmaInv;
        // maxT + 1, maxT
        arma::dmat R;
        // 1x1
        arma::dmat tLamt, tRr, expPart, expTu_it;

        void ensureSize(int nT, int nX, int nZuit, int nZvit, int nZui0, int nZvi0) {
            if (y.n_rows != nT || y.n_cols != 1) y.set_size(nT, 1);
            if (x.n_rows != nT || x.n_cols != 1) x.set_size(nT, 1);
            if (zuit.n_rows != nT || zuit.n_cols != 1) zuit.set_size(nT, 1);
            if (zvit.n_rows != nT || zvit.n_cols != 1) zvit.set_size(nT, 1);
            if (zui0.n_rows != nT || zui0.n_cols != 1) zui0.set_size(nT, 1);
            if (zvi0.n_rows != nT || zvi0.n_cols != 1) zvi0.set_size(nT, 1);
            // 1x1
            if (tLamt.n_rows != 1 || tRr.n_cols != 1) {
                tLamt.set_size(1, 1);
                tRr.set_size(1, 1);
                expPart.set_size(1, 1);
                expTu_it.set_size(1, 1);
            }
            // maxT x 1
            if (xb.n_rows != nT || xb.n_cols != 1) {
                xb.set_size(nT, 1);
                sigma2uit.set_size(nT, 1);
                sigma2vit.set_size(nT, 1);
                sigma2ui0.set_size(nT, 1);
                sigma2vi0.set_size(nT, 1);
                eps.set_size(nT, 1);
            }
            // (maxT + 1) x 1
            if (giveUp.n_rows != (nT + 1) || giveUp.n_cols != 1) {
                giveUp.set_size(nT + 1, 1);
                // fill it with NaNs
                giveUp.fill(std::numeric_limits<double>::quiet_NaN());
                vSigs.set_size(nT + 1, 1);
                from.set_size(nT + 1, 1);
                // from always is - infinity
                from.fill(-std::numeric_limits<double>::infinity());
                Rr.set_size(nT + 1, 1);
                expTu_i.set_size(nT + 1, 1);
                tRow.set_size(nT + 1, 1);
                phiBarNumTo.set_size(nT + 1, 1);
            }
            // maxT x (maxT + 1)
            if (A.n_rows != nT || A.n_cols != (nT + 1)) {
                A.set_size(nT, nT + 1);
            }
            // (maxT + 1) x (maxT + 1)
            if (V.n_rows != (nT + 1) || V.n_cols != (nT + 1)) {
                V.set_size(nT + 1, nT + 1);
                Vinv.set_size(nT + 1, nT + 1);
                lamInternal.set_size(nT + 1, nT + 1);
                Lambda.set_size(nT + 1, nT + 1);
                t.set_size(nT + 1, nT + 1);
                LambdaChol.set_size(nT + 1, nT + 1);
            }
            // maxT x max T
            if (sigP1.n_rows != nT || sigP1.n_cols != nT) {
                sigP1.set_size(nT, nT);
                sigP2.set_size(nT, nT);
                Sigma.set_size(nT, nT);
                SigmaInv.set_size(nT, nT);
            }
            // (maxT + 1) x maxT
            if (R.n_rows != (nT + 1) || R.n_cols != nT) {
                R.set_size(nT + 1, nT);
            }
        }

        // void ensureSize(int maxT, int nX, int nZuit, int nZvit, int nZui0, int nZvi0) {
        //     if (y.n_rows < maxT || y.n_cols < 1) y.set_size(maxT, 1);
        //     if (x.n_rows < maxT || x.n_cols < std::max(nX, 1)) x.set_size(maxT, std::max(nX, 1));
        //     if (zuit.n_rows < maxT || zuit.n_cols < std::max(nZuit, 1)) zuit.set_size(maxT, std::max(nZuit, 1));
        //     if (zvit.n_rows < maxT || zvit.n_cols < std::max(nZvit, 1)) zvit.set_size(maxT, std::max(nZvit, 1));
        //     if (zui0.n_rows < maxT || zui0.n_cols < std::max(nZui0, 1)) zui0.set_size(maxT, std::max(nZui0, 1));
        //     if (zvi0.n_rows < maxT || zvi0.n_cols < std::max(nZvi0, 1)) zvi0.set_size(maxT, std::max(nZvi0, 1));
        //     // 1x1
        //     if (tLamt.n_rows != 1 || tRr.n_cols != 1) {
        //         tLamt.set_size(1, 1);
        //         tRr.set_size(1, 1);
        //         expPart.set_size(1, 1);
        //         expTu_it.set_size(1, 1);
        //     }
        //     // maxT x 1
        //     if (xb.n_rows < maxT || xb.n_cols < 1) {
        //         xb.set_size(maxT, 1);
        //         sigma2uit.set_size(maxT, 1);
        //         sigma2vit.set_size(maxT, 1);
        //         sigma2ui0.set_size(maxT, 1);
        //         sigma2vi0.set_size(maxT, 1);
        //         eps.set_size(maxT, 1);
        //     }
        //     // (maxT + 1) x 1
        //     if (giveUp.n_rows < (maxT + 1) || giveUp.n_cols < 1) {
        //         giveUp.set_size(maxT + 1, 1);
        //         // fill it with NaNs
        //         giveUp.fill(std::numeric_limits<double>::quiet_NaN());
        //         vSigs.set_size(maxT + 1, 1);
        //         from.set_size(maxT + 1, 1);
        //         // from always is - infinity
        //         from.fill(-std::numeric_limits<double>::infinity());
        //         Rr.set_size(maxT + 1, 1);
        //         expTu_i.set_size(maxT + 1, 1);
        //         tRow.set_size(maxT + 1, 1);
        //         phiBarNumTo.set_size(maxT + 1, 1);
        //     }
        //     // maxT x (maxT + 1)
        //     if (A.n_rows < maxT || A.n_cols < (maxT + 1)) {
        //         A.set_size(maxT, maxT + 1);
        //     }
        //     // (maxT + 1) x (maxT + 1)
        //     if (V.n_rows < (maxT + 1) || V.n_cols < (maxT + 1)) {
        //         V.set_size(maxT + 1, maxT + 1);
        //         Vinv.set_size(maxT + 1, maxT + 1);
        //         lamInternal.set_size(maxT + 1, maxT + 1);
        //         Lambda.set_size(maxT + 1, maxT + 1);
        //         t.set_size(maxT + 1, maxT + 1);
        //         LambdaChol.set_size(maxT + 1, maxT + 1);
        //     }
        //     // maxT x max T
        //     if (sigP1.n_rows < maxT || sigP1.n_cols < maxT) {
        //         sigP1.set_size(maxT, maxT);
        //         sigP2.set_size(maxT, maxT);
        //         Sigma.set_size(maxT, maxT);
        //         SigmaInv.set_size(maxT, maxT);
        //     }
        //     // (maxT + 1) x maxT
        //     if (R.n_rows < (maxT + 1) || R.n_cols < maxT) {
        //         R.set_size(maxT + 1, maxT);
        //     }
        // }

        void reset() {
            // reset any calculate matricies 
        }

    };

}

#endif // ESA_SFA_EFF_GTRE_TC_HPP