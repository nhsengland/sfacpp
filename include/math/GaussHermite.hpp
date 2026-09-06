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

#ifndef ESA_GAUSS_HERMITE_HPP
#define ESA_GAUSS_HERMITE_HPP

// ---- armadillo ----
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif

#include <cmath>
#include <stdexcept>

/**
 * @brief Gauss-Hermite quadrature nodes and weights for approximating integrals
 *        of the form  ∫ f(x) exp(-x²) dx ≈ Σ_k w_k f(x_k)
 *
 * For TRE random effect  v_i0 ~ N(0, σ²_vi0), change of variables t = v_i0 / (σ√2):
 *   ∫ L(y | v_i0) φ(v_i0/σ) dv_i0 = (1/√π) Σ_k w_k L(y | σ√2 · x_k)
 *
 * The draws matrix passed to panelDensityHalfNormal/TruncNormal stores √2·x_k so that
 * the existing  a_i = sigmavi0 * drawRow  multiplication is unchanged in form.
 * Normalization changes from log(nsim) to log(√π).
 *
 * Supports 7, 10, 15, and 20-point rules.
 * 20-point is recommended for TRE models (accurate to ~machine precision for smooth integrands).
 * Adaptive GHQ extension: build the draws matrix with per-firm centering/scaling (AGQ).
 */
namespace ghq {

// ---- 20-point Gauss-Hermite nodes (positive half; symmetric) ----
// Source: Abramowitz & Stegun table 25.10; Golub-Welsch algorithm
// Nodes satisfy H_20(x_k) = 0 where H_20 is the 20th Hermite polynomial
// Weights w_k are the corresponding quadrature weights (sum to sqrt(pi))

static constexpr unsigned int N_NODES_20 = 20;

// Abscissae x_k (sorted ascending, negative to positive)
static constexpr double NODES_20[N_NODES_20] = {
    -5.38748089001123286201690041000,
    -4.60368244955074427307767524898,
    -3.94476404011562521050857807309,
    -3.34785456235800754461143111894,
    -2.78880605842813048052503375640,
    -2.25497400208927552308233334473,
    -1.73853771211658620678021421019,
    -1.23407621539532300788581768064,
    -0.73747372854539442830775199590,
    -0.24534070830090125859950639050,
     0.24534070830090125859950639050,
     0.73747372854539442830775199590,
     1.23407621539532300788581768064,
     1.73853771211658620678021421019,
     2.25497400208927552308233334473,
     2.78880605842813048052503375640,
     3.34785456235800754461143111894,
     3.94476404011562521050857807309,
     4.60368244955074427307767524898,
     5.38748089001123286201690041000
};

// Weights w_k  (sum(w_k) = sqrt(pi))
static constexpr double WEIGHTS_20[N_NODES_20] = {
    2.22939364553415129252882084981e-13,
    4.39934099227318055916022270680e-10,
    1.08606937076928169399952456069e-07,
    7.80255647853206369414599202019e-06,
    2.28338636016353967257145917907e-04,
    3.24377334223786183218324713235e-03,
    2.48105208874636108821649525589e-02,
    1.09017206020023320013755033535e-01,
    2.86675505362834129719659073965e-01,
    4.62243669600610089650328639562e-01,
    4.62243669600610089650328639562e-01,
    2.86675505362834129719659073965e-01,
    1.09017206020023320013755033535e-01,
    2.48105208874636108821649525589e-02,
    3.24377334223786183218324713235e-03,
    2.28338636016353967257145917907e-04,
    7.80255647853206369414599202019e-06,
    1.08606937076928169399952456069e-07,
    4.39934099227318055916022270680e-10,
    2.22939364553415129252882084981e-13
};

// ---- 15-point rule ----
static constexpr unsigned int N_NODES_15 = 15;

static constexpr double NODES_15[N_NODES_15] = {
    -4.49999070730939155366438053054,
    -3.66995037340445253472922383312,
    -2.96716692790560324848896036069,
    -2.32573248617386083519883948291,
    -1.71999257518648893241583152515,
    -1.13611558521092066631735351753,
    -0.56506958325557585404667399534,
     0.00000000000000000000000000000,
     0.56506958325557585404667399534,
     1.13611558521092066631735351753,
     1.71999257518648893241583152515,
     2.32573248617386083519883948291,
     2.96716692790560324848896036069,
     3.66995037340445253472922383312,
     4.49999070730939155366438053054
};

static constexpr double WEIGHTS_15[N_NODES_15] = {
    1.52247580425351702016898270294e-09,
    1.05911554771106663577520791055e-06,
    1.00004441232499868127324984315e-04,
    2.77806884291277676699644298614e-03,
    3.07800338725460822286790111830e-02,
    1.58488915795935746883839248680e-01,
    4.12028687498898627025891079286e-01,
    5.64100308726417532852625797340e-01,
    4.12028687498898627025891079286e-01,
    1.58488915795935746883839248680e-01,
    3.07800338725460822286790111830e-02,
    2.77806884291277676699644298614e-03,
    1.00004441232499868127324984315e-04,
    1.05911554771106663577520791055e-06,
    1.52247580425351702016898270294e-09
};

// ---- 10-point rule ----
static constexpr unsigned int N_NODES_10 = 10;

static constexpr double NODES_10[N_NODES_10] = {
    -3.43615911883773760332672549432,
    -2.53273167423278979640896079775,
    -1.75668364929988177345140122011,
    -1.03661082978951365417749040639,
    -0.34290132722370460878834030460,
     0.34290132722370460878834030460,
     1.03661082978951365417749040639,
     1.75668364929988177345140122011,
     2.53273167423278979640896079775,
     3.43615911883773760332672549432
};

static constexpr double WEIGHTS_10[N_NODES_10] = {
    7.64043285523262062915936785023e-06,
    1.34364574678123269220156695323e-03,
    3.38743944035299801386767357218e-02,
    2.40138611082314686403780127792e-01,
    6.10862633735325798783564968343e-01,
    6.10862633735325798783564968343e-01,
    2.40138611082314686403780127792e-01,
    3.38743944035299801386767357218e-02,
    1.34364574678123269220156695323e-03,
    7.64043285523262062915936785023e-06
};

// ---- 7-point rule (fast/coarse) ----
static constexpr unsigned int N_NODES_7 = 7;

static constexpr double NODES_7[N_NODES_7] = {
    -2.65196135683523349244708200652,
    -1.67355162876747144503180139830,
    -0.81628788285896466304070975501,
     0.00000000000000000000000000000,
     0.81628788285896466304070975501,
     1.67355162876747144503180139830,
     2.65196135683523349244708200652
};

static constexpr double WEIGHTS_7[N_NODES_7] = {
    9.71781245099519154149528939526e-04,
    5.45155828191270305227500521811e-02,
    4.25607252610127800520717804658e-01,
    8.10264617556807326764876563813e-01,
    4.25607252610127800520717804658e-01,
    5.45155828191270305227500521811e-02,
    9.71781245099519154149528939526e-04
};

/**
 * @brief Get the number of standard GHQ points for a given nQuadPts selection.
 *        Supported values: 7, 10, 15, 20.  Any other value throws.
 */
inline unsigned int validateNQuadPts(unsigned int nQuadPts) {
    if (nQuadPts != 7 && nQuadPts != 10 && nQuadPts != 15 && nQuadPts != 20) {
        throw std::invalid_argument("GHQ: nQuadPts must be 7, 10, 15, or 20");
    }
    return nQuadPts;
}

/**
 * @brief Get raw GHQ nodes (x_k) for a given nQuadPts.
 *        Returns a pointer to the static array.
 */
inline const double* rawNodes(unsigned int nQuadPts) {
    switch (nQuadPts) {
        case  7: return NODES_7;
        case 10: return NODES_10;
        case 15: return NODES_15;
        case 20: return NODES_20;
        default: throw std::invalid_argument("GHQ: unsupported nQuadPts");
    }
}

/**
 * @brief Get raw GHQ weights (w_k) for a given nQuadPts.
 *        sum(w_k) = sqrt(pi).
 */
inline const double* rawWeights(unsigned int nQuadPts) {
    switch (nQuadPts) {
        case  7: return WEIGHTS_7;
        case 10: return WEIGHTS_10;
        case 15: return WEIGHTS_15;
        case 20: return WEIGHTS_20;
        default: throw std::invalid_argument("GHQ: unsupported nQuadPts");
    }
}

/**
 * @brief Build the (nFirms x nQuadPts) draws matrix used in place of Halton draws.
 *
 * Stores sqrt(2) * x_k so that the existing  a_i = sigmavi0 * drawRow  expression
 * in panelDensityHalfNormal/TruncNormal computes v_i0 = σ_vi0 · √2 · x_k correctly.
 *
 * All rows are identical (standard GHQ).  For adaptive GHQ (AGQ), the caller scales
 * and shifts individual rows by the firm-specific posterior mode/curvature.
 *
 * @param nFirms   Number of firms (number of rows)
 * @param nQuadPts Number of quadrature points: 7, 10, 15, or 20
 * @return (nFirms x nQuadPts) arma::dmat
 */
inline arma::dmat buildDrawsMatrix(unsigned int nFirms, unsigned int nQuadPts = 20) {
    validateNQuadPts(nQuadPts);
    const double* nodes = rawNodes(nQuadPts);
    const double sqrt2 = std::sqrt(2.0);
    arma::drowvec row(nQuadPts);
    for (unsigned int k = 0; k < nQuadPts; ++k) {
        row(k) = sqrt2 * nodes[k];
    }
    // replicate the single row nFirms times
    arma::dmat mat(nFirms, nQuadPts);
    for (unsigned int i = 0; i < nFirms; ++i) {
        mat.row(i) = row;
    }
    return mat;
}

/**
 * @brief Build the log-weight row vector log(w_k) for a given nQuadPts.
 *        Used in the GHQ-normalized log-likelihood:
 *          ll_i = Smax + log(Σ_k exp(S_k - Smax) * w_k) - log(sqrt(pi))
 */
inline arma::drowvec logWeightsRow(unsigned int nQuadPts = 20) {
    validateNQuadPts(nQuadPts);
    const double* w = rawWeights(nQuadPts);
    arma::drowvec lw(nQuadPts);
    for (unsigned int k = 0; k < nQuadPts; ++k) {
        lw(k) = std::log(w[k]);
    }
    return lw;
}

/**
 * @brief Normalization constant in log space: log(sqrt(pi)).
 *        GHQ log-likelihood: ll_i = Smax + log(sum_k w_k * K_k) - logNorm()
 *        vs MSL:             ll_i = Smax + log(sum_r K_r)         - log(nsim)
 */
inline double logNorm() {
    return 0.5 * std::log(M_PI);  // log(sqrt(pi))
}

/**
 * @brief Build the (1 x nQuadPts) draw row for firm i under AGHQ.
 *
 * stores v_star + sigma_star * sqrt(2) * x_k so that the existing
 * a_i = sigmavi0(t) * drawRow  expression in panelDensityHalfNormal
 * evaluates a_i[t,k] = sigmavi0(t) * (v_star + sigma_star * sqrt(2) * x_k).
 *
 * The AGHQ normalisation correction log(sigma_star * sqrt(2)) must be added
 * to the log-likelihood separately via aghqLogNormCorrection().
 *
 * @param v_star posterior mode of standardised v_i0 (mean 0, std 1)
 * @param sigma_star posterior standard deviation 1/sqrt(posterior precision)
 * @param nQuadPts number of quadrature points: 7, 10, 15, or 20
 */
inline arma::drowvec buildAGHQRow(double v_star, double sigma_star,
                                   unsigned int nQuadPts = 20)
{
    const double* nodes = rawNodes(nQuadPts);
    const double sqrt2  = std::sqrt(2.0);
    arma::drowvec row(nQuadPts);
    for (unsigned int k = 0; k < nQuadPts; ++k)
        row(k) = v_star + sigma_star * sqrt2 * nodes[k];
    return row;
}

/**
 * @brief Per-firm AGHQ normalization correction (additive, in log space).
 *
 * Standard GHQ normalization: -log(sqrt(pi))
 * AGHQ normalization:         log(sigma_star) - log(sqrt(pi))
 * the extra correction relative to standard GHQ is log(sigma_star).
 *
 * The Jacobian factor sqrt(2) is already accounted for by the z_k^2 term
 * in node_corr; adding it here again would double-count.
 *
 * Add this value to ll_i after the standard Smax + log(sum_k w_k K_k) - logNorm() formula.
 */
inline double aghqLogNormCorrection(double sigma_star) {
    return std::log(sigma_star);
}

} // namespace ghq

#endif // ESA_GAUSS_HERMITE_HPP
