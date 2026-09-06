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
 * @file ESARandEff.hpp
 * @brief Header file for implementation of Random Effects model
 * @author Edmund Haacke
 * @date 2025-01-17
 * @warning Doesn't not perfectly replicate but should be near enough in limited testing
 */

#ifndef ESA_RAND_EFF_HPP
#define ESA_RAND_EFF_HPP

// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif
// --- end armadillo ---
#include "utils/esautils.hpp"


typedef struct RandEffResult {
    arma::dmat params;
    double sigma_eps;
    double sigma_alpha;
    arma::dmat residual;
    arma::dmat reResidual;
    arma::dmat randomEffect;
    arma::dmat statisticalNoise;
} RandEffResult;

class ESARandEff {
public:
    // Constructor
    // ESARandEff(
    //     const arma::dcolvec& y,
    //     const arma::dmat X,
    //     const arma::Col<int> idVec
    // ) : y(y), X(X), idVec(idVec) {};
    ESARandEff(
        const arma::dmat* y,
        const arma::dmat* X,
        const arma::Col<int>* idVec
    ) : y(esautils::makeZeroCopyStrictViewNoOpt(y)),
        X(esautils::makeZeroCopyStrictViewNoOpt(X)),
        idVec(esautils::makeZeroCopyStrictViewColNoOpt<int>(idVec))
    {

    }

    RandEffResult fit() {
        return estimateRandEff(this->y, this->X, this->idVec);
    }

private:
    const arma::dmat y;
    const arma::dmat X;
    const arma::Col<int> idVec;

    /// @brief Demean matrix
    /// @param mat The matrix to demean
    /// @param idVec The vector of IDs
    arma::dmat demeanMatrix(const arma::dmat& mat, const arma::Col<int>& idVec)
    {
        // get the unique IDs
        arma::Col<int> uniqueIDs = esautils::uniqueValsInColVec<int>(idVec);
        // arma::Col<int> uniqueIDs = arma::unique(idVec);
        // output vector
        std::vector<arma::dmat> out(uniqueIDs.n_rows);
        // calculate the column means
        arma::dmat colMeansGrp = esautils::colMeansByGroup<double>(mat, idVec);
        // int currPos = 0;
        // iterate through each panel of individuals
        for (int i = 0; i < uniqueIDs.n_rows; i++){
            // get the panel ID
            int id_i = uniqueIDs(i);
            // auto selectFunc = [id_i](const dlib::matrix<int>& m) -> bool { return m(0, 0) == id_i; };
            // arma::dmat mat_i = esautils::selectMatrixRowsForCondition<double, int>(mat, idVec, selectFunc);
            arma::uvec inds = (idVec == id_i);
            arma::dmat mat_i = esautils::selectMatrixRowsForCondition<double>(mat, inds);
            // calculate the column means
            arma::dmat colMeans = esautils::matrixMeans<double>(mat_i, false);
            // calculate T for the panel
            int t = mat_i.n_rows;
            // multiply col means by column vector of ones to upscale into correct dimension
            arma::dmat colMeansUpscale = arma::dmat(t, 1, arma::fill::ones) * colMeans;
            // demean the matrix
            arma::dmat demeanedMat = mat_i - colMeansUpscale;
            // store in the output matrix
            out[i] = demeanedMat;
        }
        arma::dmat outStack = esautils::stackMatricies<double>(out, true);
        return outStack;
    }

    /// @brief Estimate the Random Effects model
    /// @param y The dependent variable
    /// @param X The independent variables
    /// @param idVec The vector of IDs
    /// @param timeVec The vector of time periods
    RandEffResult estimateRandEff(
        const arma::dcolvec& y,
        const arma::dmat& X,
        const arma::Col<int>& idVec
    )
    {
        // initialisation
        RandEffResult result;
        const int n = y.n_rows;
        const int k = X.n_cols;
        // unique count of group identifiers
        const arma::Col<int> uniqIds = esautils::uniqueValsInColVec<int>(idVec);
        // const arma::Col<int> uniqIds = arma::unique(idVec);
        const int N = uniqIds.n_rows;
        // data structures for within- and between- transformed data
        arma::dcolvec yTilde(n, arma::fill::zeros); // y_it - y_bar_i
        arma::dmat XTilde(n, k, arma::fill::zeros); // X_it - X_bar_i
        arma::dcolvec yBar(N, arma::fill::zeros); // y_bar_i;
        arma::dmat XBar(N, k, arma::fill::zeros); // X_bar_i;
        // final quasi-demeaned data
        arma::dcolvec yStar(n, arma::fill::zeros);
        arma::dmat xStar(n, k, arma::fill::zeros);
        // for group sizes
        arma::vec Tisizes(N);
        // first pass - calculate within & between data => group means & demeaned data
        for (int i = 0; i < N; i++) {
            // locate identifiers matching this id
            // arma::uvec inds1 = (idVec == currId);
            arma::uvec inds = arma::find(idVec == uniqIds(i));
            // group specific data
            arma::dcolvec y_i = y.rows(inds);
            arma::dmat x_i = X.rows(inds);
            double T_i = static_cast<double>(y_i.n_rows);
            Tisizes(i) = T_i;
            // group means
            double y_bar_i = arma::mean(y_i);
            arma::dmat X_bar_i = esautils::matrixMeans<double>(x_i, false);
            // store for the between regression
            yBar(i) = y_bar_i;
            XBar.row(i) = X_bar_i;
            // store for within regression (demeaned data)
            yTilde.rows(inds) = y_i - y_bar_i;
            XTilde.rows(inds) = x_i - (arma::dmat(x_i.n_rows, 1, arma::fill::ones) * X_bar_i);
        }
        // within regression to estimate sigma2_e
        arma::dmat betaWithin = arma::pinv(XTilde) * yTilde;
        arma::dmat eHatTilde = yTilde - XTilde * betaWithin;

        double RSSWithin = arma::dot(eHatTilde, eHatTilde);
        double dfWithin = n - N - k;
        double sigma2e = RSSWithin / dfWithin;
        
        // arma::dmat betaBetween = arma::solve(XBar, yBar);
        arma::dmat betaBetween = arma::pinv(XBar) * yBar;
        arma::dmat uBarHat = yBar - XBar * betaBetween;
        
        double RSSBetween = arma::dot(uBarHat, uBarHat);
        double dfBetween = N - k;
        double sigma2Between = RSSBetween / dfBetween; // Var(v_i + e_bar_i);

        // de-convolute sigma2_v from sigma2_between
        double sumInvTi = arma::sum(1.0 / Tisizes);
        double sigma2Lambda = (1.0 / N) * sumInvTi * sigma2e;
        
        double sigma2v = sigma2Between - sigma2Lambda;
        // ensure variance is non-negative
        sigma2v = std::max(0.0, sigma2v);
        
        // quasi-demeaning & final GLS
        for (int i = 0; i < N; i++) {
            double Ti = Tisizes(i);
            // theta 
            double denom = Ti * sigma2v + sigma2e;
            double thetai = 1.0 - std::sqrt(sigma2e / denom);
            // original data & means
            // arma::uvec inds = (idVec == uniqIds(i));
            arma::uvec inds = arma::find(idVec == uniqIds(i));
            arma::dcolvec y_i = y.rows(inds);
            arma::dmat x_i = X.rows(inds);
            double yBari = yBar(i);
            arma::dmat xBari = XBar.row(i);
            // quasi-demeaning transformation
            yStar.rows(inds) = y_i - thetai * yBari;
            xStar.rows(inds) = x_i - (arma::dmat(x_i.n_rows, 1, arma::fill::ones) * (thetai * xBari));
        }
        // OLS on transformed data is efficient GLS estimator
        arma::dmat beta = arma::pinv(xStar) * yStar;

        // final calculations
        arma::dmat residTransformed = yStar - xStar * beta;
        arma::dmat residComposite = y - X * beta;

        // calculate estimated individual effects
        arma::dcolvec vhati(N);
        for (int i = 0; i < N; i++) {
            double Ti = Tisizes(i);
            arma::uvec inds = arma::find(idVec == uniqIds(i));
            double ubari = arma::mean(residComposite.elem(inds));
            double factor = (Ti * sigma2v) / (sigma2e + Ti * sigma2v);
            vhati(i) = factor * ubari;
        }

        result.params = beta;
        result.sigma_eps = sigma2v;
        result.sigma_alpha = sigma2e;
        result.residual = residTransformed; // residComposite;
        result.reResidual = residComposite; // residTransformed;
        result.randomEffect = vhati;
        // result.statisticalNoise = statNoise;
        return result;
    }
};

#endif // ESA_RAND_EFF_HPP