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
 * @file numpy_conv.hpp
 * @author edmund haacke
 * @date 2025-12-23
 * @details
 */

#ifndef NUMPY_CONV_HPP
#define NUMPY_CONV_HPP

// only for python package
#ifdef PYPACKAGE

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace numpy_arma_conv {
    // Helper to convert numpy array to armadillo matrix/vector
    template <typename T>
    inline arma::Mat<T> pyToArma(const py::array_t<T>& input, bool copy = false) {
        // safety checks first
        // check dimension - only 1d/2d
        if (input.ndim() > 2) throw std::runtime_error("Input must be 1D or 2D");
        // Check layout for zero-copy - need column major (not row major which is Numpy's default)
        if (!copy && !(input.flags() & py::array::f_style)) {
            throw std::runtime_error("Zero-Copy requires F_CONTIGUOUS numpy array. Use np.asfortranarray()");
        }
        // Request buffer info (check dimensions, strides, etc if needed)
        py::buffer_info buf = input.request();
        // if empty, return an empty matrix
        if (buf.size == 0) return arma::Mat<T>();
        // Create Armadillo matrix using the advanced constructor
        // arma::Mat(aux_mem*, n_rows, n_cols, copy_aux_mem, strict)
        // copy_aux_mem = true -> Allocates new C++ memory and copies data
        // copy_aux_mem = false -> Points to Python memory
        return arma::Mat<T>(
            static_cast<T*>(buf.ptr),
            buf.shape[0], // rows
            buf.shape.size() > 1 ? buf.shape[1] : 1, // cols (handle vectors)
            copy,
            false // strict mode
        );
    }

    // Specialization for Column vectors if you strictly need arma::Col type
    template <typename T>
    inline arma::Col<T> pyToCol(const py::array_t<T>& input, bool copy = true) {
        // request buffer info
        py::buffer_info buf = input.request();
        // if empty, return an empty column vector
        if (buf.size == 0) return arma::Col<T>();
        // return an armadillo column vector (using the advanced constructor)
        return arma::Col<T>(
            static_cast<T*>(buf.ptr),
            buf.size,
            copy,
            false
        );
    }

    // Convert Matrix to 2D Numpy Array
    template <typename T>
    inline py::array_t<T> armaToPy(const arma::Mat<T>& src) {
        // Cast sizeof(T) to signed size_t (ssize_t) to prevent narrowing warnings
        ssize_t element_size = static_cast<ssize_t>(sizeof(T));
        ssize_t rows = static_cast<ssize_t>(src.n_rows);
        ssize_t cols = static_cast<ssize_t>(src.n_cols);
        // 1. Create shape and strides
        std::vector<ssize_t> shape = { rows, cols };
        // Strides must be signed. 
        // Col-Major (Fortran): Step 1 element to go down a row, step n_rows elements to go to next col
        std::vector<ssize_t> strides = { element_size, element_size * rows };
        // Create the Numpy array and Copy data
        return py::array_t<T>(
            shape, 
            strides, 
            src.memptr() 
        );
    }

    // Convert Column Vector to 1D Numpy Array
    template <typename T>
    inline py::array_t<T> colToPy(const arma::Col<T>& src) {
        return py::array_t<T>(
            { static_cast<ssize_t>(src.n_elem) }, // 1D Shape
            { sizeof(T) },                        // Stride
            src.memptr()                          // Copy data
        );
    }

}

#endif // PYPACKAGE

#endif // NUMPY_CONV_HPP