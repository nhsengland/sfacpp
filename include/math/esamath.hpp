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

#ifndef ESAMATH_HPP
#define ESAMATH_HPP

// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif //RPACKAGE
// --- end armadillo ---

#include <cmath>

#ifdef WITHDLIB
#include <dlib/matrix.h>
#endif // WITHDLIB

namespace esamath {
    
    /// @brief Function to perform element-wise division of two matrices
    /// @tparam T Type of the matrix
    /// @param mat1 First matrix
    /// @param mat2 Second matrix
    /// @return Matrix of element-wise division
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> pointwise_divide(const dlib::matrix<T>& mat1, const dlib::matrix<T>& mat2);
    #endif // WITHDLIB
    template <typename T>
    arma::Mat<T> pointwise_divide(const arma::Mat<T>& mat1, const arma::Mat<T>& mat2);

    /// @brief Compute the row mean of a matrix
    /// @tparam T Type of the matrix
    /// @param mat Matrix
    /// @return Row mean of the matrix as a column vector - column vector of doubles
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<double, 0, 1> rowMean(const dlib::matrix<T>& mat);
    #endif // WITHDLIB
    template <typename T>
    arma::dcolvec rowMean(const arma::Mat<T>& mat);
    
    /// @brief Multiply elements of each column across the rows
    /// @tparam T Type of the matrix
    /// @param mat Matrix
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<double> colProd(const dlib::matrix<T>& mat);
    #endif // WITHDLIB
    template <typename T>
    arma::dmat colProd(const arma::Mat<T>& mat);

    /// @brief  Sum elements of each column across the rows
    /// @tparam T Type of the matrix
    /// @param mat Matrix
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<double> colSum(const dlib::matrix<T>& mat);
    #endif // WITHDLIB
    template <typename T>
    arma::dmat colSum(const arma::Base<double, T>& matIn);

    /// @brief Sum elements of each row across the columns
    /// @tparam T Type of the matrix
    /// @param mat Matrix
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<double> rowSum(const dlib::matrix<T>& mat);
    #endif // WITHDLIB
    template <typename T>
    arma::dmat rowSum(const arma::Mat<T>& mat);

    /// @brief Check if a matrix is negative definite - lifted from 'npsf' package by Badunenko et al.
    /// @tparam T Type of the matrix
    /// @param mat Matrix
    /// @param tol Tolerance
    /// @return True if the matrix is negative definite
    #ifdef WITHDLIB
    template <typename T>
    bool isNegativeDefinite(const dlib::matrix<T>& mat, double tol = 1e-16);
    #endif // WITHDLIB
    template <typename T>
    bool isNegativeDefinite(const arma::Mat<T>& mat, double tol = 1e-16);

    /// @brief Make a matrix negative definite - lifted from 'npsf' package by Badunenko et al.
    /// @tparam T 
    /// @param mat 
    /// @return
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> makeNegativeDefinite(const dlib::matrix<T>& mat);
    #endif // WITHDLIB
    template <typename T>
    arma::Mat<T> makeNegativeDefinite(const arma::Mat<T>& mat);

    /// @brief Make a matrix invertable - lifted from 'npsf' package by Badunenko et al.
    /// @tparam T 
    /// @param mat 
    /// @return
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> makeInvertable(const dlib::matrix<T>& mat);
    #endif // WITHDLIB
    template <typename T>
    arma::Mat<T> makeInvertable(const arma::Mat<T>& mat);

    /// @brief Invert matrix using pivoted LU decomposition approach
    /// @tparam T 
    /// @param mat Square matrix to invert
    /// @return Inverted matrix
    #ifdef WITHDLIB
    template <typename T>
    dlib::matrix<T> invert_matrix_pivoted_lu(const dlib::matrix<T>& mat);
    #endif // WITHDLIB
    template <typename T>
    arma::Mat<T> invert_matrix_pivoted_lu(const arma::Mat<T>& mat);

    /// @brief Check whether the inverse of a matrix is valid - original multiplied by inverted should yield identity
    /// @tparam T 
    /// @param orig The original matrix
    /// @param inv The inverted matrix
    /// @param tol Tolerance
    /// @return boolean whether or not is valid inverted matrix.
    #ifdef WITHDLIB
    template <typename T>
    bool isValidInvertedMatrix(const dlib::matrix<T>& orig, const dlib::matrix<T>& inv, const double tol = 1e-6);
    #endif // WITHDLIB
    template <typename T>
    bool isValidInvertedMatrix(const arma::Mat<T>& orig, const arma::Mat<T>& inv, const double tol = 1e-6);

    /// @brief Calculate the interquartile range of a matrix
}
#endif // ESAMATH_HPP