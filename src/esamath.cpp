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

// [[Rcpp::depends(BH)]]
#include <memory>
#include <boost/math/distributions/normal.hpp>
#include <dlib/statistics.h>
#include <stdexcept>
#include "math/esamath.hpp"
#include "math/esandist.hpp"
#include "utils/esautils.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "thread_cache/esamathTC.hpp"
#include "utils/ThreadContext.hpp"
// #include "math/manual_linalg.hpp"

/// compute pointwise division
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esamath::pointwise_divide(const dlib::matrix<T>& mat1, const dlib::matrix<T>& mat2) {
    // Ensure the matrices have the same size
    if (mat1.nr() != mat2.nr() || mat1.nc() != mat2.nc()) {
        throw std::invalid_argument("Matrices must have the same dimensions for element-wise division.");
    }
    dlib::matrix<T> result(mat1.nr(), mat1.nc());
    for (long r = 0; r < mat1.nr(); ++r) {
        for (long c = 0; c < mat1.nc(); ++c) {
            result(r, c) = mat1(r, c) / mat2(r, c);
        }
    }
    return result;
}
// template instantiation
template dlib::matrix<double> esamath::pointwise_divide(const dlib::matrix<double>& mat1, const dlib::matrix<double>& mat2);
#endif // WITHDLIB

/// pointwise division for armadillo matrix
template <typename T>
arma::Mat<T> esamath::pointwise_divide(const arma::Mat<T>& mat1, const arma::Mat<T>& mat2)
{
    return mat1 / mat2;
}
template arma::Mat<double> esamath::pointwise_divide(const arma::Mat<double>&, const arma::Mat<double>&);

/// Compute the row mean of a matrix
#ifdef WITHDLIB
template <typename T>
dlib::matrix<double, 0, 1> esamath::rowMean(const dlib::matrix<T>& mat){
    dlib::matrix<double, 0, 1> result(mat.nr());
    for (long r = 0; r < mat.nr(); ++r){
        double sum = 0.0;
        for (long c = 0; c < mat.nc(); ++c){
            sum += mat(r, c);
        }
        result(r) = sum / mat.nc();
    }
    return result;
}
// explicit template instantiation
template dlib::matrix<double, 0, 1> esamath::rowMean(const dlib::matrix<double>& mat);
template dlib::matrix<double, 0, 1> esamath::rowMean(const dlib::matrix<float>& mat);
template dlib::matrix<double, 0, 1> esamath::rowMean(const dlib::matrix<int>& mat);
#endif // WITHDLIB

/// Compute the row mean of a matrix - armadillo implementation
template <typename T>
arma::dcolvec esamath::rowMean(const arma::Mat<T>& mat)
{
    arma::dcolvec result(mat.n_rows);
    for (size_t r = 0; r < mat.n_rows; r++) {
        result(r) = arma::accu(mat.row(r));
    }
    return result;
}
template arma::dcolvec esamath::rowMean<double>(const arma::Mat<double>&);
template arma::dcolvec esamath::rowMean<float>(const arma::Mat<float>&);
template arma::dcolvec esamath::rowMean<int>(const arma::Mat<int>&);

/// Multiply elements of each column down the rows
#ifdef WITHDLIB
template <typename T>
dlib::matrix<double> esamath::colProd(const dlib::matrix<T>& mat){
    if (mat.nr() == 0){
        throw std::invalid_argument("Matrix must have at least one column.");
    }
    if (mat.nr() == 1) return mat;
    dlib::matrix<double, 1, 0> result = dlib::rowm(mat, 0);
    for (size_t r = 1; r < mat.nr(); ++r){
        // result = dlib::pointwise_multiply(result, dlib::rowm(mat, r));
        for (size_t c = 0; c < mat.nc(); ++c){
            result(c) *= mat(r, c);
        }
    }
    return(result);
}
// explicit template instantiation
template dlib::matrix<double> esamath::colProd(const dlib::matrix<double>& mat);
#endif // WITHDLIb

/// Multiply elements of each column down the rows - armadillo implementation
template <typename T>
arma::dmat esamath::colProd(const arma::Mat<T>& mat)
{
    if (mat.n_rows == 0) throw std::invalid_argument("Matrix must have at least one row");
    if (mat.n_rows == 1) return mat;
    arma::dmat res = mat.row(0);
    for (size_t r = 1; r < mat.n_rows; r++) {
        res = res % mat.row(r);
    }
    return res;
}
template arma::dmat esamath::colProd(const arma::Mat<double>&);

/// Sum elements of each column down the rows
#ifdef WITHDLIB
template <typename T>
dlib::matrix<double> esamath::colSum(const dlib::matrix<T>& mat){
    if (mat.nr() == 0){
        throw std::invalid_argument("Matrix must have at least one column.");
    }
    if (mat.nr() == 1) return mat;
    dlib::matrix<double, 1, 0> result = dlib::rowm(mat, 0);
    for (size_t r = 1; r < mat.nr(); ++r){
        for (size_t c = 0; c < mat.nc(); ++c){
            result(c) += mat(r, c);
        }
    }
    return(result);
}
// explicit template instantiation
template dlib::matrix<double> esamath::colSum(const dlib::matrix<double>&);
#endif // WITHDLIB

// Sum elements of each column down the rows - armadillo implementation
template <typename T>
arma::dmat esamath::colSum(const arma::Base<double, T>& matIn)
{
    // unwrap armadillo reference
    const auto& mat = matIn.get_ref();
    if (mat.n_rows == 0) throw std::invalid_argument("Matrix must have at least one row");
    if (mat.n_rows == 1) return mat;
    arma::dmat res = mat.row(0);
    for (size_t r = 1; r < mat.n_rows; r++) {
        res = res + mat.row(r);
    }
    return res;
}
template arma::dmat esamath::colSum<arma::dmat>(const arma::Base<double, arma::dmat>&);
template arma::dmat esamath::colSum<arma::subview<double>>(const arma::Base<double, arma::subview<double>>&);

/// Sum elements of each row across the columns
#ifdef WITHDLIB
template <typename T>
dlib::matrix<double> esamath::rowSum(const dlib::matrix<T>& mat){
    if (mat.nc() == 0){
        throw std::invalid_argument("Matrix must have at least one column.");
    }
    if (mat.nc() == 1) return mat;
    dlib::matrix<double, 0, 1> result = dlib::colm(mat, 0);
    for (size_t c = 1; c < mat.nc(); ++c){
        for (size_t r = 0; r < mat.nr(); ++r){
            result(r) += mat(r, c);
        }
    }
    return(result);
}
template dlib::matrix<double> esamath::rowSum(const dlib::matrix<double>&);
#endif // WITHDLIB

/// Sum elements of each row across the columns
template <typename T>
arma::dmat esamath::rowSum(const arma::Mat<T>& mat)
{
    if (mat.n_cols == 0) throw std::invalid_argument("Matrix must have at least one column");
    if (mat.n_cols == 1) return mat;
    arma::dmat res = mat.col(0);
    for (size_t c = 1; c < mat.n_cols; c++) {
        res = res + mat.col(c);
    }
    return res;
}
template arma::dmat esamath::rowSum(const arma::Mat<double>&);

#ifdef WITHDLIB
template <typename T>
bool esamath::isNegativeDefinite(const dlib::matrix<T>& mat, double tol)
{
    // check that the matrix is fully finite before doing eigenvalue decomposition
    if (!dlib::is_finite(mat)){
        throw std::invalid_argument("Matrix must be fully finite.");
    }
    // eigen values
    dlib::eigenvalue_decomposition<dlib::matrix<T>> eig(mat);
    dlib::matrix<T> eigenValues = eig.get_real_eigenvalues();
    bool anyMet = false;
    for (size_t i = 0; i < mat.nr(); i++){
        if (std::abs(eigenValues(i)) < tol){
            anyMet = true;
            break;
        }
    }
    return !anyMet;
}
// explicit template instantisation
template bool esamath::isNegativeDefinite(const dlib::matrix<double>&, double);
#endif // WITHDLIB

// armadillo implmentation of negative definition check
template <typename T>
bool esamath::isNegativeDefinite(const arma::Mat<T>& mat, double tol)
{
    // throw std::runtime_error("check implementation");
    // check if matrix is fully finite before doing eigenvalue decomposition
    // if (!mat.is_finite()) throw std::invalid_argument("matrix must be finite");
    if (!mat.is_finite()) return false;
    // eigenvalues
    arma::Col<double> eigval;
    bool success = arma::eig_sym(eigval, mat);
    if (!success) return false;
    bool anyMet = false;
    for (size_t i = 0; i < mat.n_rows; i++) {
        if (std::abs(eigval(i)) < tol) {
            anyMet = true;
            break;
        }
    }
    return !anyMet;
}
// explicit template instantisation
template bool esamath::isNegativeDefinite(const arma::Mat<double>&, double);

/// Make a matrix negative definite
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esamath::makeNegativeDefinite(const dlib::matrix<T>& mat)
{
    // check that the matrix is fully finite before doing eigenvalue decomposition
    if (!dlib::is_finite(mat)){
        throw std::invalid_argument("Matrix must be fully finite.");
    }
    // std::cout << "start eigenvalue decomposition" << std::endl;
    // std::cout << mat << std::endl;
    dlib::eigenvalue_decomposition<dlib::matrix<T>> eig(mat);
    // std::cout << "getting eigenvalues" << std::endl;
    dlib::matrix<T> eigenValues = eig.get_real_eigenvalues();
    // std::cout << "getting eigenvectors" << std::endl;
    dlib::matrix<T> eigenVectors = eig.get_pseudo_v();

    dlib::matrix<T> eigenVecTrans = dlib::trans(eigenVectors);
    dlib::matrix<T> comp1 = eigenVecTrans;
    for (size_t i = 0; i < mat.nr(); i++){
        for (size_t j = 0; j < mat.nc(); j++){
            comp1(i, j) = comp1(i, j) * std::abs(eigenValues(i));
        }
    }
    return -1 * dlib::trans(comp1) * eigenVecTrans;
}
template dlib::matrix<double> esamath::makeNegativeDefinite(const dlib::matrix<double>&);
#endif

/// Make matrix negative definite - armadillo implementation
template <typename T>
arma::Mat<T> esamath::makeNegativeDefinite(const arma::Mat<T>& mat)
{
    throw std::runtime_error("check implementation");
    // check matrix fully finite before doing eigenvalue decomposition
    if (!mat.is_finite()) throw std::invalid_argument("Matrix must be fully finite");
    arma::vec eigval;
    arma::mat eigvec;
    bool success = arma::eig_sym(eigval, eigvec, mat);
    if (!success) throw std::invalid_argument("could not eigendecomposiition");
    // arma::mat eigvecT = eigvec.t();
    // arma::mat comp1 = eigvecT;
    arma::mat comp1(eigvec.n_rows, eigvec.n_cols);
    for (size_t i = 0; i < mat.n_rows; i++) {
        for (size_t j = 0; j < mat.n_cols; j++) {
            // comp1(i, j) = comp1(i, j) * std::abs(eigval(i));
            comp1(i, j) = eigvec(i, j) * std::abs(eigval(i));
        }
    }
    // return -1 * comp1.t() * eigvecT;
    return -1 * comp1.t() * eigvec;
}
template arma::Mat<double> esamath::makeNegativeDefinite(const arma::Mat<double>&);

/// Make invertable
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esamath::makeInvertable(const dlib::matrix<T>& mat)
{
    dlib::matrix<T> hess = mat;
    unsigned int k = mat.nc();
    bool ok = false;
    double adj = std::sqrt(std::numeric_limits<double>::epsilon());
    unsigned int i = 0;
    while (!ok && (i < 1000)){
        i++;
        hess = hess + adj * dlib::identity_matrix<T>(k);
        dlib::matrix<T> hessInv;
        bool didInv = true;
        try {
            hessInv = dlib::inv(-hess);
        } catch (const std::exception& e){
            didInv = false;
        }
        ok = didInv;
        adj = adj * 2.0;
    }
    return hess;
}
template dlib::matrix<double> esamath::makeInvertable(const dlib::matrix<double>&);
#endif // WITHDLIB

/// Make matrix invertable - armadillo implementation
template <typename T>
arma::Mat<T> esamath::makeInvertable(const arma::Mat<T>& mat)
{
    arma::Mat<T> hess = mat;
    unsigned int k = mat.n_cols;
    bool ok = false;
    double adj = std::sqrt(std::numeric_limits<double>::epsilon());
    unsigned int i = 0;
    while (!ok && (i < 1000)) {
        i++;
        hess = hess + adj * arma::dmat(k, k, arma::fill::eye);
        arma::Mat<T> hessInv;
        bool didInv = true;
        try {
            hessInv = arma::inv(-hess);
        } catch (const std::exception& e) {
            didInv = false;
        }
        ok = didInv;
        adj = adj * 2.0;
    }
    return hess;
}
template arma::Mat<double> esamath::makeInvertable(const arma::Mat<double>&);

/// Invert matrix using pivoted LU approach
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esamath::invert_matrix_pivoted_lu(const dlib::matrix<T>& mat) {
    // Check if the matrix is square
    if (mat.nr() != mat.nc()) {
        throw std::invalid_argument("Matrix must be square to invert.");
    }

    // Perform LU decomposition with pivoting
    dlib::lu_decomposition<dlib::matrix<T>> lu(mat);

    // Check if the matrix is singular
    if (lu.is_singular()) {
        throw std::runtime_error("Matrix is singular and cannot be inverted.");
    }

    // Create an identity matrix of the same size
    dlib::matrix<T> identity = dlib::identity_matrix<T>(mat.nr());

    // Compute the inverse using the LU decomposition
    dlib::matrix<T> inv_mat = lu.solve(identity);

    return inv_mat;
}
template dlib::matrix<double> esamath::invert_matrix_pivoted_lu(const dlib::matrix<double>&);
#endif // WITHDLIB

/// Invert matrix using pivoted LU approach - armadillo implementation
template <typename T>
arma::Mat<T> esamath::invert_matrix_pivoted_lu(const arma::Mat<T>& mat)
{
    // ESALogger::logger()->warn("inv matrix pivoted LU not tested in armadillo implementation...");
    // check if square matrix
    if (mat.n_rows != mat.n_cols) throw std::invalid_argument("Matrix must be square to invert");
    // LU decomposition with pivoting
    arma::Mat<T> L, U;
    // check if matrix singular?
    arma::Mat<T> identity = arma::Mat<T>(mat.n_rows, mat.n_rows, arma::fill::eye);
    arma::Mat<T> invMat = arma::solve(L, identity);
    return invMat;
}
template arma::Mat<double> esamath::invert_matrix_pivoted_lu(const arma::Mat<double>&);

/// Check whether inverse of a matrix is valid - original multiplied by inverse should yield identity
#ifdef WITHDLIB
template <typename T>
bool esamath::isValidInvertedMatrix(const dlib::matrix<T>& orig, const dlib::matrix<T>& inv, const double tol)
{
    // Check if the matrices are square
    if (orig.nr() != orig.nc() || inv.nr() != inv.nc()) {
        throw std::invalid_argument("Both matrices must be square.");
    }

    // Check if the matrices are the same size
    if (orig.nr() != inv.nr()) {
        throw std::invalid_argument("Both matrices must be the same size.");
    }

    // Check if the original matrix multiplied by the inverse yields the identity matrix
    dlib::matrix<T> prod = orig * inv;
    dlib::matrix<T> identity = dlib::identity_matrix<T>(orig.nr());
    dlib::matrix<T> diff = prod - identity;
    double max_diff = dlib::max(dlib::abs(diff));
    return max_diff < tol;
}
template bool esamath::isValidInvertedMatrix(const dlib::matrix<double>&, const dlib::matrix<double>&, const double);
#endif // WITHDLIB

/// Check whether inverse of a matrix is valid - original multipled by inverse should yield identity - Armadillo implementation
template <typename T>
bool esamath::isValidInvertedMatrix(const arma::Mat<T>& orig, const arma::Mat<T>&inv, const double tol)
{
    // check both are square matricies
    if (orig.n_rows != orig.n_cols || inv.n_rows != inv.n_cols) throw std::invalid_argument("Both matricies must be square.");
    // check same size for both matricies
    if (orig.n_rows != inv.n_rows) throw std::invalid_argument("Both matricies must be the same size.");
    // original matrix multiplied by the inverse should yield the identity matrix
    arma::Mat<T> prod = orig * inv;
    arma::Mat<T> identity(orig.n_rows, orig.n_rows, arma::fill::eye);
    arma::Mat<T> diff = prod - identity;
    double max_diff = arma::abs(diff).max();
    return max_diff < tol;
}
template bool esamath::isValidInvertedMatrix(const arma::Mat<double>&, const arma::Mat<double>&, const double);