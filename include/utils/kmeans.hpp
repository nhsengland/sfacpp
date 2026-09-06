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

#ifndef ESA_UTILS_KMEANS_HPP
#define ESA_UTILS_KMEANS_HPP

#include <random>
#include <vector>
#include <limits>
#include <cmath>

#if defined(RPACKAGE)
#include <RcppArmadillo.h>
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif

namespace esautils {

struct KMeansResult {
    // (N,) cluster index per observation
    arma::urowvec assignments;
    // (K x D) final centroids
    arma::dmat centroids;
    // number of iterations to converge [or maxiter]
    unsigned int iterations;
    // whether converged or not
    bool converged;
};

/**
 * @brief K-Means++ centroid initialisation.
 * @param data (N x D) row-major observations
 * @param K number of clusters
 * @param seed RNG seed
 * @return (K x D) initial centroids
 */
inline arma::dmat kmeansppInit(const arma::dmat& data, unsigned int K, unsigned int seed)
{
    unsigned int N = data.n_rows;
    unsigned int D = data.n_cols;
    std::mt19937 rng(seed);
    // matrix of centroids
    arma::dmat centroids(K, D);
    // first centroid follows a uniform random
    std::uniform_int_distribution<unsigned int> uniDist(0, N - 1);
    centroids.row(0) = data.row(uniDist(rng));
    arma::dvec minDist(N, arma::fill::value(std::numeric_limits<double>::infinity()));
    // iterate thru number of clusters
    for (unsigned int k = 1; k < K; ++k) {
        // upd minimum squared distances to existing centroids
        for (unsigned int i = 0; i < N; ++i) {
            arma::drowvec diff = data.row(i) - centroids.row(k - 1);
            double d2 = arma::dot(diff, diff);
            if (d2 < minDist(i)) minDist(i) = d2;
        }
        double totalDist = arma::accu(minDist);
        if (totalDist <= 0.0) {
            // all points coincide — fill remaining centroids with random rows
            for (unsigned int kk = k; kk < K; ++kk)
                centroids.row(kk) = data.row(uniDist(rng));
            return centroids;
        }
        // sample proportional to squared distance
        std::uniform_real_distribution<double> uReal(0.0, totalDist);
        double threshold = uReal(rng);
        double cumSum = 0.0;
        unsigned int chosen = N - 1;
        for (unsigned int i = 0; i < N; ++i) {
            cumSum += minDist(i);
            if (cumSum >= threshold) { chosen = i; break; }
        }
        centroids.row(k) = data.row(chosen);
    }
    return centroids;
}

/**
 * @brief Lloyd's K-Means with K-Means++ initialisation.
 *
 * Empty clusters (or clusters below minClusterSize) are healed by stealing
 * the farthest point from the largest cluster.
 *
 * @param data(N x D) observations, one per row
 * @param K number of clusters
 * @param maxIter maximum Lloyd iterations
 * @param tol convergence: max centroid displacement < tol
 * @param seed RNG seed for K-Means++ init
 * @param minClusterSize minimum acceptable cluster size; undersized clusters trigger healing
 */
inline KMeansResult kmeans(
    const arma::dmat& data,
    unsigned int K,
    unsigned int maxIter = 100,
    double tol = 1e-6,
    unsigned int seed = 1234,
    unsigned int minClusterSize = 2
)
{
    unsigned int N = data.n_rows;
    unsigned int D = data.n_cols;
    // build the result struct
    KMeansResult result;
    result.converged   = false;
    result.iterations  = 0;
    result.assignments = arma::urowvec(N, arma::fill::zeros);
    // if 0 clusters, return
    if (K == 0 || N == 0) return result;
    K = std::min(K, N);
    // initialize centroids
    arma::dmat centroids = kmeansppInit(data, K, seed);
    // iterate thru number of iterations
    for (unsigned int iter = 0; iter < maxIter; ++iter) {
        result.iterations = iter + 1;
        // first: assignment step
        arma::urowvec newAssign(N);
        for (unsigned int i = 0; i < N; ++i) {
            double bestDist = std::numeric_limits<double>::infinity();
            unsigned int bestK = 0;
            for (unsigned int k = 0; k < K; ++k) {
                arma::drowvec diff = data.row(i) - centroids.row(k);
                double d2 = arma::dot(diff, diff);
                if (d2 < bestDist) { bestDist = d2; bestK = k; }
            }
            newAssign(i) = bestK;
        }
        // next calculate the cluster sizes
        arma::uvec clusterSize(K, arma::fill::zeros);
        for (unsigned int i = 0; i < N; ++i) clusterSize(newAssign(i))++;
        // force a minimum cluster size - if its under, nick some points away
        for (unsigned int k = 0; k < K; ++k) {
            if (clusterSize(k) >= minClusterSize) continue;
            // find largest cluster
            arma::uword srcK = clusterSize.index_max();
            if (srcK == k) continue;
            // find farthest point in srcK from its centroid
            double farthestDist = -1.0;
            unsigned int farthestIdx = 0;
            for (unsigned int i = 0; i < N; ++i) {
                if (newAssign(i) != srcK) continue;
                arma::drowvec diff = data.row(i) - centroids.row(srcK);
                double d2 = arma::dot(diff, diff);
                if (d2 > farthestDist) { farthestDist = d2; farthestIdx = i; }
            }
            newAssign(farthestIdx) = k;
            clusterSize(srcK)--;
            clusterSize(k)++;
        }
        // update step
        arma::dmat newCentroids(K, D, arma::fill::zeros);
        arma::uvec newSizes(K, arma::fill::zeros);
        for (unsigned int i = 0; i < N; ++i) {
            unsigned int k = newAssign(i);
            newCentroids.row(k) += data.row(i);
            newSizes(k)++;
        }
        for (unsigned int k = 0; k < K; ++k) {
            if (newSizes(k) > 0)
                newCentroids.row(k) /= static_cast<double>(newSizes(k));
            else
                newCentroids.row(k) = centroids.row(k);  // unchanged if empty
        }
        // check for convergence
        double maxDisp = 0.0;
        for (unsigned int k = 0; k < K; ++k) {
            arma::drowvec disp = newCentroids.row(k) - centroids.row(k);
            double d = std::sqrt(arma::dot(disp, disp));
            if (d > maxDisp) maxDisp = d;
        }
        centroids = std::move(newCentroids);
        result.assignments = std::move(newAssign);
        // if converged, flag and stop iterating
        if (maxDisp < tol) {
            result.converged = true;
            break;
        }
    }
    result.centroids = centroids;
    return result;
}

} // namespace esautils

#endif // ESA_UTILS_KMEANS_HPP
