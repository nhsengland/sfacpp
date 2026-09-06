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

#ifndef DLIB_2_ARMA_HPP
#define DLIB_2_ARMA_HPP

#if defined(WITHDLIB)

#include <armadillo>
#include <dlib/matrix.h>
#include <iostream>

// --- Armadillo to Dlib ---

/**
 * @brief Creates a non-const Dlib column vector wrapper around an Armadillo colvec.
 *
 * This function is zero-copy.
 *
 * @warning LIFETIME: The returned Dlib wrapper is only valid as long as the
 * original 'arma_vec' exists and is not reallocated.
 * @warning MEMORY LAYOUT: This is safe for column vectors (Nx1) as their
 * in-memory layout is identical in both libraries.
 *
 * @param arma_vec The Armadillo column vector to wrap.
 * @return A non-const dlib::matrix_op that acts as a wrapper.
 */
inline auto wrap_arma_colvec_to_dlib(arma::colvec& arma_vec)
{
    // dlib::mat creates a lightweight wrapper (matrix_op)
    return dlib::mat(arma_vec.memptr(), arma_vec.n_elem, 1);
}

/**
 * @brief Creates a const Dlib column vector wrapper around a const Armadillo colvec.
 *
 * This function is zero-copy.
 *
 * @warning LIFETIME: The returned Dlib wrapper is only valid as long as the
 * original 'arma_vec' exists.
 * @warning MEMORY LAYOUT: Safe for column vectors (Nx1).
 *
 * @param arma_vec The const Armadillo column vector to wrap.
 * @return A const dlib::matrix_op that acts as a wrapper.
 */
inline auto wrap_arma_colvec_to_dlib(const arma::colvec& arma_vec)
{
    // dlib::mat creates a lightweight wrapper (matrix_op)
    return dlib::mat(arma_vec.memptr(), arma_vec.n_elem, 1);
}

inline dlib::matrix<double, 0, 1> copy_arma_colvec_to_dlib(const arma::dcolvec& arma_vec)
{
    dlib::matrix<double, 0, 1> dlibVec;
    dlibVec.set_size(arma_vec.n_elem);
    for (size_t i = 0; i < arma_vec.n_elem; ++i) {
        dlibVec(i) = arma_vec(i);
    }
    return dlibVec;
}

inline dlib::matrix<double, 0, 1> copy_and_negate_colvec_to_dlib(const arma::dcolvec& arma_vec)
{
    dlib::matrix<double, 0, 1> dlibVec;
    dlibVec.set_size(arma_vec.n_elem);
    for (size_t i = 0; i < arma_vec.n_elem; i++) {
        dlibVec(i) = -arma_vec(i);
    }
    return dlibVec;
}

/**
 * @brief Creates a non-const Dlib matrix wrapper around an Armadillo matrix.
 *
 * @warning MAJOR-MISMATCH: Dlib (row-major) will wrap Armadillo's
 * (column-major) memory. Accessing dlib_mat(r, c) will NOT equal
 * arma_mat(r, c) for a general matrix.
 * @warning This is ONLY safe if the matrix is a column vector (N_cols == 1).
 * @warning LIFETIME: The returned Dlib wrapper is only valid as long as the
 * original 'arma_mat' exists and is not reallocated.
 *
 * @param arma_mat The Armadillo matrix to wrap.
 * @return A non-const dlib::matrix_op that acts as a wrapper.
 */
inline auto wrap_arma_mat_to_dlib(arma::mat& arma_mat)
{
    return dlib::mat(arma_mat.memptr(), arma_mat.n_rows, arma_mat.n_cols);
}

/**
 * @brief Creates a const Dlib matrix wrapper around a const Armadillo matrix.
 *
 * @warning See warnings for the non-const version. MAJOR-MISMATCH.
 *
 * @param arma_mat The const Armadillo matrix to wrap.
 * @return A const dlib::matrix_op that acts as a wrapper.
 */
inline auto wrap_arma_mat_to_dlib(const arma::mat& arma_mat)
{
    return dlib::mat(arma_mat.memptr(), arma_mat.n_rows, arma_mat.n_cols);
}


// --- Dlib to Armadillo ---

/**
 * @brief Creates a non-const Armadillo colvec wrapper around a Dlib column vector.
 *
 * This function is zero-copy.
 *
 * @warning LIFETIME: The returned Armadillo vector is only valid as long as the
 * original 'dlib_vec' exists and is not reallocated.
 * @warning MEMORY LAYOUT: Safe for column vectors (Nx1).
 *
 * @param dlib_vec The Dlib column vector (dlib::matrix<T, 0, 1>) to wrap.
 * @return A non-const arma::colvec that shares memory with the dlib vector.
 */
template <typename T>
inline arma::Col<T> wrap_dlib_colvec_to_arma(dlib::matrix<T, 0, 1>& dlib_vec, const bool copy = false)
{
    // arma::Col(ptr, n_elem, copy_aux_mem, strict)
    // copy_aux_mem = false:  Do NOT copy the data. Use the provided memory.
    // strict = false:        Allow wrapping external memory.
    return arma::Col<T>(dlib_vec.begin(), dlib_vec.size(), copy, false);
}
template arma::Col<double> wrap_dlib_colvec_to_arma(dlib::matrix<double, 0, 1>&, const bool);

/**
 * @brief Creates a const Armadillo colvec wrapper around a const Dlib column vector.
 *
 * This function is zero-copy.
 *
 * @warning LIFETIME: The returned Armadillo vector is only valid as long as the
 * original 'dlib_vec' exists.
 * @warning MEMORY LAYOUT: Safe for column vectors (Nx1).
 *
 * @param dlib_vec The const Dlib column vector (dlib::matrix<T, 0, 1>) to wrap.
 * @return A const arma::Col<T> that shares memory with the dlib vector.
 */
template <typename T>
inline const arma::Col<T> wrap_dlib_colvec_to_arma(const dlib::matrix<T, 0, 1>& dlib_vec, const bool copy = false)
{
    // We must const_cast the pointer, as Armadillo's const-wrapper
    // constructor still takes a non-const pointer but promises not to write.
    return arma::Col<T>(const_cast<T*>(dlib_vec.begin()), dlib_vec.size(), copy, false);
}
template const arma::Col<double> wrap_dlib_colvec_to_arma<double>(const dlib::matrix<double, 0, 1>&, const bool);


/**
 * @brief Creates a non-const Armadillo matrix wrapper around a Dlib matrix.
 *
 * @warning MAJOR-MISMATCH: Armadillo (column-major) will wrap Dlib's
 * (row-major) memory. Accessing arma_mat(r, c) will NOT equal
 * dlib_mat(r, c) for a general matrix.
 * @warning This is ONLY safe if the matrix is a column vector (N_cols == 1).
 * @warning LIFETIME: The returned Armadillo matrix is only valid as long as the
 * original 'dlib_mat' exists and is not reallocated.
 *
 * @param dlib_mat The Dlib matrix to wrap.
 * @return A non-const arma::Mat that shares memory with the dlib matrix.
 */
template <typename T, long NR, long NC>
inline arma::Mat<T> wrap_dlib_mat_to_arma(dlib::matrix<T, NR, NC>& dlib_mat)
{
    // copy_aux_mem = false:  Do NOT copy the data. Use the provided memory.
    // strict = false:        Allow wrapping external memory.
    return arma::Mat<T>(dlib_mat.begin(), dlib_mat.nr(), dlib_mat.nc(), false, false);
}

/**
 * @brief Creates a const Armadillo matrix wrapper around a const Dlib matrix.
 *
 * @warning See warnings for the non-const version. MAJOR-MISMATCH.
 *
 * @param dlib_mat The const Dlib matrix to wrap.
 * @return A const arma::Mat that shares memory with the dlib matrix.
 */
template <typename T, long NR, long NC>
inline const arma::Mat<T> wrap_dlib_mat_to_arma(const dlib::matrix<T, NR, NC>& dlib_mat)
{
    // We must const_cast the pointer, as Armadillo's const-wrapper
    // constructor still takes a non-const pointer but promises not to write.
    return arma::Mat<T>(const_cast<T*>(dlib_mat.begin()), dlib_mat.nr(), dlib_mat.nc(), false, false);
}

// --- Dlib to Armadillo (Copying) ---

/**
 * @brief Creates a new Armadillo colvec by COPYING a Dlib column vector.
 *
 * This function performs a deep copy.
 * The memory layout is identical, so this is a straightforward copy.
 *
 * @param dlib_vec The Dlib column vector to copy.
 * @return A new arma::Col<T> containing a copy of the data.
 */
template <typename T>
inline arma::Col<T> copy_dlib_colvec_to_arma(const dlib::matrix<T, 0, 1>& dlib_vec)
{
    // For a column vector, memory layout is contiguous and identical.
    // We can use the Armadillo constructor that copies from a pointer.
    // arma::Col<T>(const T* aux_mem, uword n_elem)
    return arma::Col<T>(dlib_vec.begin(), dlib_vec.size());
}

/**
 * @brief Creates a new Armadillo matrix by COPYING a Dlib matrix.
 *
 * This function performs a deep copy and correctly handles the
 * row-major (Dlib) to column-major (Armadillo) conversion.
 *
 * @param dlib_mat The Dlib matrix to copy.
 * @return A new arma::Mat<T> containing a copy of the data in
 * column-major layout.
 */
template <typename T, long NR, long NC>
inline arma::Mat<T> copy_dlib_mat_to_arma(const dlib::matrix<T, NR, NC>& dlib_mat)
{
    // Dlib is row-major, Arma is col-major.
    // A direct copy constructor will misinterpret the memory layout.
    //
    // We can, however, create a *wrapper* with swapped dimensions,
    // which interprets the row-major memory as a column-major matrix
    // that is the *transpose* of the original dlib matrix.
    //
    // We then transpose this wrapper (.t()), which forces a deep copy
    // and re-arranges the data into the correct column-major layout
    // for the new matrix.

    // 1. Create a const wrapper, requires const_cast for Arma's API.
    //    Note the swapped dimensions: (cols, rows)
    const arma::Mat<T> temp_wrap(
        const_cast<T*>(dlib_mat.begin()), // Pointer to data
        dlib_mat.nc(),                     // Use dlib's COLS as Arma's ROWS
        dlib_mat.nr(),                     // Use dlib's ROWS as Arma's COLS
        false,                             // copy_aux_mem = false (wrap)
        false                              // strict = false (allow wrap)
    );

    // 2. Transpose the wrapper. This forces a copy and
    //    re-arranges the data into correct column-major order.
    //    (e.g., wrap is (A^T), so (A^T)^T = A)
    return temp_wrap.t();
}

#endif // dlib defined

#endif // DLIB_2_ARMA_HPP