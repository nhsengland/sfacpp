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
 * @file
 */

#ifndef RINTERFACE_UTILS_HPP
#define RINTERFACE_UTILS_HPP

#include <string>
#include <memory>
#include <optional>
#include "utils/enums.hpp"
#include "optim/optimparams.hpp"

#ifdef RPACKAGE
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif // RPACKAGE

namespace rinterface {

    #ifdef RPACKAGE

    // Helper lambda for Nullable Mat -> Arma View
    auto get_mat_view = [](Rcpp::Nullable<Rcpp::NumericMatrix>& param) -> std::optional<arma::mat> {
        if (param.isNotNull()) {
            Rcpp::NumericMatrix m(param);
            return std::make_optional<arma::mat>(const_cast<double*>(m.begin()), m.nrow(), m.ncol(), false, true);
        }
        return std::nullopt;
    };
    // Helper lambda for Nullable Vec -> Arma View
    auto get_vec_view = [](Rcpp::Nullable<Rcpp::NumericVector>& param) -> std::optional<arma::vec> {
        if (param.isNotNull()) {
            Rcpp::NumericVector v(param);
            return std::make_optional<arma::vec>(const_cast<double*>(v.begin()), v.size(), false, true);
        }
        return std::nullopt;
    };
    // Helper lambda for Nullable IntegerVec -> Arma iVec View
    auto get_ivec_view = [](Rcpp::Nullable<Rcpp::IntegerVector>& param) -> std::optional<arma::ivec> {
        if (param.isNotNull()) {
            Rcpp::IntegerVector v(param);
            // Armadillo 'ivec' is Col<int>. R 'IntegerVector' stores 32-bit ints.
            // Direct memory mapping is safe.
            return std::make_optional<arma::ivec>(const_cast<int*>(v.begin()), v.size(), false, true);
        }
        return std::nullopt;
    };
    
    /// @brief Extract an element from a Rcpp List by key, and convert to a C++ type. Wrap in std optional
    /// @tparam T 
    /// @param l An Rcpp::List
    /// @param k A key in the list
    /// @return 
    template <typename T>
    std::optional<T> elementFromRcppList(Rcpp::List& l, const std::string& k);

    /**
     * @brief Parse optimization parameters
     * @param optimParams Instance of ESAOptimParams to load params into
     * @param optimOpts an optional Rcpp list from R
     * @param unsigned int seed
     */
    void setupOptimParams(
        ESAOptimParams& optimParams,
        Rcpp::Nullable<Rcpp::List> optimOpts,
        const int seed
    );

    /// @brief Process and convert input data matricies to dlib matricies
    /// @param y 
    /// @param x 
    /// @param zmuit_ 
    /// @param zuit_ 
    /// @param zvit_ 
    /// @param zui0_ 
    /// @param zvi0_ 
    /// @param startVals_ 
    /// @param idVec_ 
    /// @param timeVec_ 
    /// @param yDlib 
    /// @param xDlib 
    /// @param zmuitDlib 
    /// @param zuitDlib 
    /// @param zvitDlib 
    /// @param zui0Dlib 
    /// @param zvi0Dlib 
    /// @param idVecDlib 
    /// @param timeVecDlib 

    void processDataMatricies(
        const arma::dcolvec& y,
        const arma::dmat& x,
        arma::dmat& zmuit,
        arma::dmat& zuit,
        arma::dmat& zvit,
        arma::dmat& zui0,
        arma::dmat& zvi0,
        arma::Col<int>& idVec,
        arma::Col<int>& timeVec,
        const ESASfaModelType& mT,
        const arma::colvec& y_,
        const arma::mat& x_,
        Rcpp::Nullable<arma::mat> zmuit_ = R_NilValue,
        Rcpp::Nullable<arma::mat> zuit_ = R_NilValue,
        Rcpp::Nullable<arma::mat> zvit_ = R_NilValue,
        Rcpp::Nullable<arma::mat> zui0_ = R_NilValue,
        Rcpp::Nullable<arma::mat> zvi0_ = R_NilValue,
        Rcpp::Nullable<arma::mat> startVals_ = R_NilValue,
        Rcpp::Nullable<arma::colvec> idVec_ = R_NilValue,
        Rcpp::Nullable<arma::colvec> timeVec_ = R_NilValue
    );

    #endif // RPACKAGE
}

#endif // RINTERFACE_UTILS_HPP