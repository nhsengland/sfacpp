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

#ifndef ESAUTILS_HPP
#define ESAUTILS_HPP

#include <vector>
#include <functional>

#ifdef WITHDLIB
#include <dlib/matrix.h>
#endif // WITHDLIB

// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#else
#include <armadillo>
#endif
// --- end armadillo ---

#ifdef WITHEIGEN
#ifdef RPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include "Eigen/Core"
#endif // RPACKAGE
#endif //WITHEIGEN

namespace esautils {
    /**
     * @brief strict zerocopy view
    */
    std::optional<arma::dmat> makeZeroCopyStrictView(const std::optional<arma::dmat>& in);
    std::optional<arma::dmat> makeZeroCopyStrictView(const arma::dmat* m);
    /**
     * @brief strict zerocopy view for column
     * @tparam type (int, double)
     */
    template <typename T>
    std::optional<arma::Col<T>> makeZeroCopyStrictView(const std::optional<arma::Col<T>>& in);
    template <typename T>
    std::optional<arma::Col<T>> makeZeroCopyStrictView(const arma::Col<T>* c);
    /**
     * @brief 
     */
    arma::dmat makeZeroCopyStrictViewNoOpt(const arma::dmat* in);
    template <typename T>
    arma::Col<T> makeZeroCopyStrictViewColNoOpt(const arma::Col<T>* c);

    /// @brief Process the sigma2 term
    /// @param par The parameter vector
    /// @param vals The values to process
    /// @param forceExp Whether to force the exponentiation of the sigma2 term
    /// @return The processed sigma2 term
    #ifdef WITHDLIB
    dlib::matrix<double> processSig2Term(const dlib::matrix<double, 0, 1>& par, const dlib::matrix<double>& vals, const bool forceExp = true);
    #endif
    arma::dmat processSig2Term(const arma::dcolvec& par, const arma::dmat& vals, const bool forceExp = true);
    #ifdef WITHEIGEN
    Eigen::MatrixXd processSig2Term(const Eigen::VectorXd& par, const Eigen::MatrixXd& vals, const bool forceExp = true);
    #endif
    
    /// @brief Check if any element in the matrix satisfies a condition
    /// @tparam T 
    /// @param mat 
    /// @param condition 
    /// @return boolean
    #ifdef WITHDLIB
    template <typename T>
    bool any(const dlib::matrix<T>& mat, bool (*condition)(T));
    #endif
    #ifdef WITHEIGEN
    bool any(const Eigen::MatrixXd& mat, bool (*condition)(double));
    #endif

    /// @brief Check if any element in the matrix is infinite
    /// @tparam T 
    /// @param mat 
    /// @return
    #ifdef WITHDLIB
    template <typename T>
    bool any_is_infinite(const dlib::matrix<T>& mat);
    #endif
    #ifdef WITHEIGEN
    bool any_is_infinite(const Eigen::MatrixXd& mat);
    #endif

    /// @brief Convert a matrix from Rcpp Armadillo to Dlib (e.g., copy task)
    /// @tparam T
    /// @param armaMatrix
    /// @return
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> convertToDlibMatrix(const arma::mat& armaMatrix);
    #endif
    
    /// @brief Convert a matrix from Dlib to Rcpp Armadillo (e.g., copy task)
    /// @tparam T
    /// @param dlibMatrix
    /// @return arma::mat
    #ifdef WITHDLIB
    template <typename T>
    arma::mat convertToArmaMatrix(const dlib::matrix<T>& dlibMatrix);
    #endif // WITHDLIB

    /// @brief Get all of the unique elements in a column matrix
    /// @tparam T
    /// @param mat
    /// @return A column matrix with all the unique elements
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T, 0, 1> uniqueValsInColVec(const dlib::matrix<T, 0, 1>& mat);
    #endif
    #ifdef WITHEIGEN
    Eigen::VectorXd uniqueValsInColVec(const Eigen::MatrixXd& mat);
    #endif
    template <typename T>
    arma::Col<T> uniqueValsInColVec(const arma::Col<T>& mat);

    /// @brief Select all rows of a matrix based on a condition
    /// @tparam T1 Type of the underlying matrix
    /// @tparam T2 Type of the associated condition matrix
    /// @param mat Matrix to select rows from
    /// @param condition Condition matrix
    /// @return A matrix with the selected rows from 'mat'
    #ifdef WITHDLIB
    template <typename T1, typename T2>
    dlib::matrix<T1> selectMatrixRowsForCondition(const dlib::matrix<T1>& mat, const dlib::matrix<T2>& condMat, std::function<bool(const dlib::matrix<T2>&)> condition);
    #endif
    #ifdef WITHEIGEN
    Eigen::MatrixXd selectMatrixRowsForCondition(const Eigen::MatrixXd& mat, const Eigen::MatrixXd& condMat, std::function<bool(const Eigen::MatrixXd&)> condition);
    #endif
    /// @tparam T Type of the underlying matrix
    /// @param mat Matrix to select rows from
    /// @param ind Column vector of indicies to select
    /// @return A matrix with the selected rows from 'mat'
    template <typename T>
    arma::Mat<T> selectMatrixRowsForCondition(const arma::Mat<T>& mat, const arma::uvec& ind);

    /// @brief Repeat a column vector as columns
    /// @tparam T Type of the matrix
    /// @param vec The column vector desiring repetition
    /// @param nCols The number of repetitions
    /// @return The matrix with the column vector repeated as columns
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> repeatColVecAsCols(const dlib::matrix<T, 0, 1>& vec, const size_t nCols);
    #endif
    template <typename T>
    arma::Mat<T> repeatColVecAsCols(const arma::Mat<T>& vec, const size_t nCols);
    #ifdef WITHEIGEN
    Eigen::MatrixXd repeatColVecAsCols(const Eigen::VectorXd& vec, const size_t nCols);
    #endif

    /// @brief Shuffle a matrix
    /// @tparam T Type of the matrix
    /// @param mat The matrix to shuffle
    /// @return The shuffled matrix
    #ifdef WITHDLIB
    template <typename T>
    void shuffleMatrix(dlib::matrix<T>& mat, const int seed = 1234);
    #endif
    #ifdef WITHEIGEN
    void shuffleMatrix(Eigen::MatrixXd& mat, const int seed = 1234);
    #endif
    template <typename T>
    void shuffleMatrix(arma::Mat<T>& mat, const int seed = 1234);

    /// @brief Generate a random sample of integers between a range
    /// @param n The highest bound
    /// @param size the number of samples
    /// @param replace whether to sample with replacement
    /// @param seed the seed to use for the random number generator
    std::vector<int> sampleIntegers(const int n, const int size, const bool replace = false, const int seed = 1234);

    /// @brief Rescale a dlib matrix
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> rescaleMatrix(const dlib::matrix<T>& mat, const T lwrBnd, const T uprBnd);
    #endif
    template <typename T>
    arma::Mat<T> rescaleMatrix(const arma::Mat<T>& mat, const T lwrBnd, const T uprBnd);
    #ifdef WITHEIGEN
    Eigen::MatrixXd rescaleMatrix(const Eigen::MatrixXd& mat, const double lwrBnd, const double uprBnd);
    #endif

    /// @brief Filter out invalid numbers (inf and nan)
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> filterRowsInvalidNumbers(const dlib::matrix<T>& mat);
    #endif
    // template <typename T>
    // arma::Mat<T> filterRowsInvalidNumbers(const arma::Mat<T>& mat);
    template <typename T>
    arma::dmat filterRowsInvalidNumbers(const arma::Base<double, T>& matIn);
    #ifdef WITHEIGEN
    Eigen::MatrixXd filterRowsInvalidNumbers(const Eigen::MatrixXd& mat);
    #endif

    /// @brief stack vector of matricies
    /// @tparam T Type of the underlying matrix
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> stackMatricies(const std::vector<dlib::matrix<T>>& matVec, const bool byRow = true);
    #endif
    template <typename T>
    arma::Mat<T> stackMatricies(const std::vector<arma::Mat<T>>& matVec, const bool byRow = true);

    #ifdef WITHEIGEN
    Eigen::MatrixXd stackMatricies(const std::vector<Eigen::MatrixXd>& matVec, const bool byRow = true);
    #endif

    /// @brief Sum up matricies within a vector of matricies
    /// @tparam T Type
    template <typename T>
    arma::Mat<T> sumMatricies(const std::vector<arma::Mat<T>>& matVec);

    /// @brief Means (either rowise or column wise) of a matrix
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> matrixMeans(const dlib::matrix<T>& mat, const bool byRow = true);
    #endif
    template <typename T>
    arma::Mat<T> matrixMeans(const arma::Mat<T>& mat, const bool byRow = true);
    #ifdef WITHEIGEN
    Eigen::MatrixXd matrixMeans(const Eigen::MatrixXd& mat, const bool byRow = true);
    #endif

    /// @brief Calculate column means by grouping variable
    /// @tparam T Type of the matrix
    /// @param mat The matrix to calculate the column means
    /// @param groupVec The grouping variable
    /// @return The column means by group (unique(groupVec), ncol(mat)) in dimensions
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> colMeansByGroup(const dlib::matrix<T>& mat, const dlib::matrix<int>& groupVec);
    #endif
    template <typename T>
    arma::Mat<T> colMeansByGroup(const arma::Mat<T>& mat, const arma::Col<int>& groupVec);
    #ifdef WITHEIGEN
    Eigen::MatrixXd colMeansByGroup(const Eigen::MatrixXd& mat, const Eigen::VectorXi& groupVec);
    #endif

    /// @brief Explode each row by t 
    /// @tparam T Type of the matrix
    /// @param mat The matrix to explode
    /// @param t The number of times to explode each row
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> matrixExplodeRowsOverT(const dlib::matrix<T>& mat, const dlib::matrix<int>& t);
    #endif
    template <typename T>
    arma::Mat<T> matrixExplodeRowsOverT(const arma::Mat<T>& mat, const arma::Col<int>& t);
    #ifdef WITHEIGEN
    Eigen::MatrixXd matrixExplodeRowsOverT(const Eigen::MatrixXd& mat, const Eigen::VectorXi& t);
    #endif // WITHEIGEN

    /// @brief Apply some function to either each row or column of a matrix
    /// @tparam T Type of the matrix
    /// @param mat The matrix to apply the function to
    /// @param margin an int denoting direction - 1 for column, 2 for row
    /// @param stats The matrix to apply
    /// @param func A string denoting the function to apply (*, /, +, -)
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> sweepMatrixElementwise(const dlib::matrix<T>& mat, const int margin, const dlib::matrix<T>& stats, const std::string& func);
    #endif // WITHDLIB
    // arma implementation
    template <typename T1, typename T2>
    arma::dmat sweepMatrixElementwise(
        const arma::Base<double, T1>& matIn,
        const int margin,
        const arma::Base<double, T2>& statsIn,
        const std::string& func
    );
    // arma::Mat<T> sweepMatrixElementwise(const arma::Mat<T>& mat, const int margin, arma::Mat<T>& stats, const std::string& func);
    
    // eigen implementation
    #ifdef WITHEIGEN
    Eigen::MatrixXd sweepMatrixElementwise(const Eigen::MatrixXd& mat, const int margin, const Eigen::MatrixXd& stats, const std::string& func);
    #endif // WITHEIGEN

    /// @brief Replace zeros (up to machine tolerance with a value)
    /// @tparam T Type of the matrix
    /// @param mat The matrix to replace zeros
    /// @param value The value to replace zeros with
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> replaceValuesPrecision(const dlib::matrix<T>& mat, const T value, const T replacement, const double eps = 1e-8);
    #endif // WITHDLIB
    template <typename T>
    arma::Mat<T> replaceValuesPrecision(const arma::Mat<T>& mat, const T value, const T replacement, const double eps = 1e-8);
    #ifdef WITHEIGEN
    Eigen::MatrixXd replaceValuesPrecision(const Eigen::MatrixXd& mat, const double value, const double replacement, const double eps = 1e-8);
    #endif // WITHEIGEN

    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T, 0, 1> replaceValuesPrecision(const dlib::matrix<T, 0, 1>& mat, const T value, const T replacement, const double eps = 1e-8);
    #endif // WITHDLIB
    template <typename T>
    arma::Col<T> replaceValuesPrecision(const arma::Col<T>& mat, const T value, const T replacement, const double eps = 1e-8);
    #ifdef WITHEIGEN
    Eigen::VectorXd replaceValuesPrecision(const Eigen::VectorXd& mat, const double value, const double replacement, const double eps = 1e-8);
    #endif // WITHEIGEN

    #ifdef WITHEIGEN
    /// @brief Convert armadillo matrix to Eigen matrix
    Eigen::MatrixXd castEigen(arma::mat m);
    
    /// @brief Convert Eigen matrix to armadillo matrix
    arma::mat castArma(Eigen::MatrixXd m);


    /**
     * @brief Creates an Eigen::Map that wraps the memory of an Armadillo column vector.
     *
     * This function performs no copy and is a safe, efficient way to use Armadillo data
     * in Eigen-based functions. The returned map is a read-only view into the Armadillo 
     * vector's memory.
     *
     * The lifetime of the returned map should not exceed the lifetime of the source 
     * Armadillo vector.
     *
     * @tparam T The scalar type of the vector elements (e.g., double, float).
     * @param arma_vec The constant Armadillo vector to be mapped.
     * @return A const Eigen::Map that directly uses the memory of arma_vec.
     */
    template <typename T>
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> armaToEigenVec(const arma::Col<T>& m);

    /**
     * @brief Creates a mutable Eigen::Map that wraps the memory of an Armadillo column vector.
     *
     * This function performs no copy. The returned map is a writable view, and modifications
     * made to it will directly affect the original Armadillo vector.
     *
     * The lifetime of the returned map should not exceed the lifetime of the source 
     * Armadillo vector.
     *
     * @tparam T The scalar type of the vector elements (e.g., double, float).
     * @param arma_vec The Armadillo vector to be mapped.
     * @return A mutable Eigen::Map that directly uses the memory of arma_vec.
     */
    template <typename T>
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> armaToEigenVec(arma::Col<T>& m);

    /**
     * @brief Creates an Armadillo column vector that wraps the memory of a const Eigen vector.
     *
     * @warning This function performs no copy. The returned Armadillo vector is a VIEW into the
     * Eigen vector's memory. The user MUST ensure that the original Eigen vector 'eigen_vec'
     * remains alive and in scope for the entire lifetime of the returned Armadillo vector.
     * Destroying the Eigen vector will result in a dangling pointer in the Armadillo vector,
     * leading to undefined behavior. This returns a read-only view.
     *
     * @tparam T The scalar type of the vector elements (e.g., double, float).
     * @param eigen_vec The Eigen vector to be wrapped.
     * @return An arma::Col<T> that directly uses the memory of eigen_vec.
     */
    template <typename T>
    const arma::Col<T> eigenToArmaVec(const Eigen::Matrix<T, Eigen::Dynamic, 1>& m);

    /**
     * @brief Create mutable armadillo column vector, wrapping the memory of an Eigen vector
     * 
     * @warning This is the most dangerous of the conversions. The returned Armadillo vector is a 
     * writable VIEW into the Eigen vector's memory. The user MUST ensure that the original 
     * Eigen vector 'eigen_vec' remains alive and in scope for the entire lifetime of the 
     * returned Armadillo vector. Modifying the returned vector will modify the original Eigen vector.
     *
     * @tparam T The scalar type of the vector elements (e.g., double, float).
     * @param eigen_vec The Eigen vector to be wrapped.
     * @return An arma::Col<T> that directly uses the memory of eigen_vec.
     */
    template <typename T>
    arma::Col<T> eigenToArmaVec(Eigen::Matrix<T, Eigen::Dynamic, 1>& m);
    #endif //WITHEIGEN
}

#endif // ESAUTILS_HPP