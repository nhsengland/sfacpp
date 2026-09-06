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

#include <set>
#include <cmath>
#include <random>
#include <algorithm>
#include <dlib/rand.h>
#include "utils/esautils.hpp"
#include "math/esamath.hpp"



// strict no-ccopy views
std::optional<arma::dmat> esautils::makeZeroCopyStrictView(const std::optional<arma::dmat>& in)
{
    if (!in.has_value()) return std::nullopt;
    const arma::dmat& m = in.value();
    // safety check against nullptr
    if (m.memptr() == nullptr) return std::nullopt;
    // new view using pointer from the input view
    return std::make_optional<arma::dmat>(
        const_cast<double*>(m.memptr()),
        m.n_rows,
        m.n_cols,
        false,
        true
    );
}

std::optional<arma::dmat> esautils::makeZeroCopyStrictView(const arma::dmat* m)
{
    if (m == nullptr) return std::nullopt;
    return std::make_optional<arma::dmat>(
        const_cast<double*>(m->memptr()),
        m->n_rows,
        m->n_cols,
        false,
        true
    );
}

template <typename T>
std::optional<arma::Col<T>> esautils::makeZeroCopyStrictView(const std::optional<arma::Col<T>>& in)
{
    if (!in.has_value()) return std::nullopt;
    const arma::Col<T>& c = in.value();
    if (c.memptr() == nullptr) return std::nullopt;
    // new view using pointer from input view
    return std::make_optional<arma::Col<T>>(
        const_cast<T*>(c.memptr()),
        c.n_rows,
        false,
        true
    );
}

template <typename T>
std::optional<arma::Col<T>> esautils::makeZeroCopyStrictView(const arma::Col<T>* c)
{
    if (c == nullptr) return std::nullopt;
    if (c->memptr() == nullptr) return std::nullopt;
    return std::make_optional<arma::Col<T>>(
        const_cast<T*>(c->memptr()),
        c->n_rows,
        false,
        true
    );
}
// explicit template instantization
template std::optional<arma::Col<int>> esautils::makeZeroCopyStrictView(const std::optional<arma::Col<int>>&);
template std::optional<arma::Col<unsigned int>> esautils::makeZeroCopyStrictView(const std::optional<arma::Col<unsigned int>>&);
template std::optional<arma::Col<double>> esautils::makeZeroCopyStrictView(const std::optional<arma::Col<double>>&);
template std::optional<arma::Col<int>> esautils::makeZeroCopyStrictView(const arma::Col<int>*);
template std::optional<arma::Col<unsigned int>> esautils::makeZeroCopyStrictView(const arma::Col<unsigned int>*);
template std::optional<arma::Col<double>> esautils::makeZeroCopyStrictView(const arma::Col<double>*);

arma::dmat esautils::makeZeroCopyStrictViewNoOpt(const arma::dmat* m)
{
    if (m == nullptr) throw std::runtime_error("got a nullptr when making a view");
    return arma::dmat(
        const_cast<double*>(m->memptr()),
        m->n_rows,
        m->n_cols,
        false,
        true
    );
}

template <typename T>
arma::Col<T> esautils::makeZeroCopyStrictViewColNoOpt(const arma::Col<T>* c)
{
    if (c == nullptr) throw std::runtime_error("got a nullptr for colint view creation");
    return arma::Col<T>(
        const_cast<T*>(c->memptr()),
        c->n_rows,
        false,
        true
    );
}
template arma::Col<int> esautils::makeZeroCopyStrictViewColNoOpt<int>(const arma::Col<int>*);
template arma::Col<double> esautils::makeZeroCopyStrictViewColNoOpt<double>(const arma::Col<double>*);

#ifdef WITHDLIB
/// Process sigma2 term - dlib implementation
dlib::matrix<double> esautils::processSig2Term(const dlib::matrix<double, 0, 1>& par, const dlib::matrix<double>& vals, const bool forceExp) {
    dlib::matrix<double> retVal = vals * par;
    if (par.nr() > 1 || forceExp){
        // at least one more parameter than a constant - take exponent
        return dlib::exp(retVal);
    }
    // otherwise, dont return - since its just the sigma2 term
    return retVal;
}
#endif // WITHDLIB
/// Process sigma2 term - armadillo implementation
arma::dmat esautils::processSig2Term(const arma::dcolvec& par, const arma::dmat& vals, const bool forceExp) {
    arma::dmat retVal = vals * par;
    if (par.n_rows > 1 || forceExp) {
        return arma::exp(retVal);
    }
    return retVal;
}
#ifdef WITHEIGEN
/// Process sigma2 term - eigen implementation
Eigen::MatrixXd esautils::processSig2Term(const Eigen::VectorXd& par, const Eigen::MatrixXd& vals, const bool forceExp){
    Eigen::MatrixXd retVal = vals * par;
    if (par.rows() > 1 || forceExp){
        // take exponential of retVal
        return retVal.array().exp();
    }
    return retVal;
}
#endif //WITHEIGEN

/// Check if any element in the matrix satisfies a condition
#ifdef WITHDLIB
template <typename T>
bool esautils::any(const dlib::matrix<T>& mat, bool (*condition)(T)) {
    for (size_t i = 0; i < mat.nr(); ++i) {
        for (size_t j = 0; j < mat.nc(); ++j) {
            if (condition(mat(i, j))) return true;
        }
    }
    return false;
}
template bool esautils::any<double>(const dlib::matrix<double>& mat, bool (*condition)(double));
template bool esautils::any<int>(const dlib::matrix<int>& mat, bool (*condition)(int));
template bool esautils::any<float>(const dlib::matrix<float>& mat, bool (*condition)(float));
#endif // WITHDLIB

#ifdef WITHEIGEN
bool esautils::any(const Eigen::MatrixXd& mat, bool (*condition)(double)){
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            if (condition(mat(i, j))) return true;
        }
    }
    return false;
}
#endif // WITHEIGEN

/// Check if any element in the matrix is infinite
#ifdef WITHDLIB
template <typename T>
bool esautils::any_is_infinite(const dlib::matrix<T>& mat) {
    return esautils::any<T>(mat, [](T x) { return std::isinf(x); });
}
// explicit template instantiation
template bool esautils::any_is_infinite<double>(const dlib::matrix<double>& mat);
#endif // WITHDLIB

#ifdef WITHEIGEN
bool esautils::any_is_infinite(const Eigen::MatrixXd& mat){
    return esautils::any(mat, [](double x) { return std::isinf(x); });
}
#endif // /WITHEIGEN

/// Convert a matrix from Rcpp Armadillo to dlib (e.g., copy task)
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esautils::convertToDlibMatrix(const arma::mat& armaMatrix){
    dlib::matrix<T> dlibMatrix(armaMatrix.n_rows, armaMatrix.n_cols);
    for (size_t i = 0; i < armaMatrix.n_rows; ++i) {
        for (size_t j = 0; j < armaMatrix.n_cols; ++j) {
            dlibMatrix(i, j) = armaMatrix(i, j);
        }
    }
    return dlibMatrix;
}
// explicit template instantiation
template dlib::matrix<double> esautils::convertToDlibMatrix<double>(const arma::mat& armaMatrix);
template dlib::matrix<int> esautils::convertToDlibMatrix<int>(const arma::mat& armaMatrix);
template dlib::matrix<float> esautils::convertToDlibMatrix<float>(const arma::mat& armaMatrix);
#endif // WITHDLIB

/// Convert a matrix from dlib to Rcpp Armadillo (e.g., copy task)
#ifdef WITHDLIB
template <typename T>
arma::mat esautils::convertToArmaMatrix(const dlib::matrix<T>& dlibMatrix){
    arma::mat armaMatrix(dlibMatrix.nr(), dlibMatrix.nc());
    for (size_t i = 0; i < dlibMatrix.nr(); ++i) {
        for (size_t j = 0; j < dlibMatrix.nc(); ++j) {
            armaMatrix(i, j) = dlibMatrix(i, j);
        }
    }
    return armaMatrix;
}
// explicit template instantiation
template arma::mat esautils::convertToArmaMatrix<double>(const dlib::matrix<double>& dlibMatrix);
template arma::mat esautils::convertToArmaMatrix<int>(const dlib::matrix<int>& dlibMatrix);
#endif // WITHDLIB

/// Get all of the unique elements in a column matrix
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T, 0, 1> esautils::uniqueValsInColVec(const dlib::matrix<T, 0, 1>& mat){
    std::set<T> unique_elements(mat.begin(), mat.end());
    dlib::matrix<T, 0, 1> result(unique_elements.size());
    std::copy(unique_elements.begin(), unique_elements.end(), result.begin());
    return result;
}
// explicit template instantiation
template dlib::matrix<double, 0, 1> esautils::uniqueValsInColVec<double>(const dlib::matrix<double, 0, 1>& mat);
template dlib::matrix<int, 0, 1> esautils::uniqueValsInColVec<int>(const dlib::matrix<int, 0, 1>& mat);
template dlib::matrix<float, 0, 1> esautils::uniqueValsInColVec<float>(const dlib::matrix<float, 0, 1>& mat);
#endif // WITHDLIB

#ifdef WITHEIGEN
Eigen::VectorXd esautils::uniqueValsInColVec(const Eigen::MatrixXd& mat){
    std::set<double> uniqueElements;
    for (size_t r = 0; r < mat.rows(); ++r){
        for (size_t c = 0; c < mat.cols(); ++c){
            uniqueElements.insert(mat(r, c));
        }
    }
    // convert set to Eigen vector
    Eigen::VectorXd result(uniqueElements.size());
    int idx = 0;
    for (const double& val : uniqueElements){
        result(idx++) = val;
    }
    return result;
}
#endif // WITHEIGEN

template <typename T>
arma::Col<T> esautils::uniqueValsInColVec(const arma::Col<T>& mat)
{
    arma::Col<T> uniq = arma::unique(mat);
    return uniq;
}
template arma::Col<int> esautils::uniqueValsInColVec<int>(const arma::Col<int>&);
template arma::Col<unsigned int> esautils::uniqueValsInColVec<unsigned int>(const arma::Col<unsigned int>&);
template arma::Col<double> esautils::uniqueValsInColVec<double>(const arma::Col<double>&);

/// From a matrix, select all rows based on a condition in another matrix of equal length
#ifdef WITHDLIB
template <typename T1, typename T2>
dlib::matrix<T1> esautils::selectMatrixRowsForCondition(const dlib::matrix<T1>& mat, const dlib::matrix<T2>& condMat, std::function<bool(const dlib::matrix<T2>&)> condition){
    // check both matricies have same number of rows
    if (mat.nr() != condMat.nr()){
        throw std::invalid_argument("Matrix and condition matrix must have the same number of rows");
    }
    std::vector<dlib::matrix<T1>> selected_rows;
    for (size_t i = 0; i < mat.nr(); ++i) {
        dlib::matrix<T1> row = dlib::rowm(mat, i);
        dlib::matrix<T2> condRow = dlib::rowm(condMat, i);
        if (condition(condRow)){
            selected_rows.push_back(row);
        }
    }
    // create a new matrix to hold the selected row
    dlib::matrix<T1> result(selected_rows.size(), mat.nc());
    for (size_t i = 0; i < selected_rows.size(); ++i) {
        dlib::set_rowm(result, i) = selected_rows[i];
    }
    return result;
}
// dlib - explicit template instantiation
template dlib::matrix<double> esautils::selectMatrixRowsForCondition<double, double>(const dlib::matrix<double>& mat, const dlib::matrix<double>& condMat, std::function<bool(const dlib::matrix<double>&)> condition);
template dlib::matrix<double> esautils::selectMatrixRowsForCondition<double, int>(const dlib::matrix<double>& mat, const dlib::matrix<int>& condMat, std::function<bool(const dlib::matrix<int>&)> condition);
#endif // WITHDLIB

// Armadillo implementation
template <typename T>
arma::Mat<T> esautils::selectMatrixRowsForCondition(const arma::Mat<T>& mat, const arma::uvec& ind)
{
    std::vector<unsigned int> nonzeros;
    for (unsigned int i = 0; i < ind.n_rows; i++){
        if (ind(i) != 0) nonzeros.push_back(i);
    }
    arma::uvec nonzeroInds = arma::conv_to<arma::uvec>::from(nonzeros);
    return mat.rows(nonzeroInds);
}
// armadillo - explicit template instantisation
template arma::Mat<double> esautils::selectMatrixRowsForCondition<double>(const arma::Mat<double>&, const arma::uvec&);

// Eigen implementation
#ifdef WITHEIGEN
Eigen::MatrixXd esautils::selectMatrixRowsForCondition(const Eigen::MatrixXd& mat, const Eigen::MatrixXd& condMat, std::function<bool(const Eigen::MatrixXd&)> condition){
    // check both matricies have same number of rows
    if (mat.rows() != condMat.rows()){
        throw std::invalid_argument("Matrix and condition matrix must have the same number of rows");
    }
    std::vector<Eigen::VectorXd> selected_rows;
    for (size_t i = 0; i < mat.rows(); ++i) {
        Eigen::VectorXd row = mat.row(i);
        Eigen::VectorXd condRow = condMat.row(i);
        if (condition(condRow)){
            selected_rows.push_back(row);
        }
    }
    // create a new matrix to hold the selected row
    Eigen::MatrixXd result(selected_rows.size(), mat.cols());
    for (size_t i = 0; i < selected_rows.size(); ++i) {
        result.row(i) = selected_rows[i];
    }
    return result;
}
#endif // WITHEIGEN

/// Repeat column vector as columns - dlib implementation
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esautils::repeatColVecAsCols(const dlib::matrix<T, 0, 1>& vec, const size_t nCols){
    dlib::matrix<T> result(vec.nr(), nCols);
    for (size_t i = 0; i < nCols; ++i){
        dlib::set_colm(result, i) = vec;
    }
    return result;
}
// explicit template instantiation
template dlib::matrix<double> esautils::repeatColVecAsCols<double>(const dlib::matrix<double, 0, 1>& vec, const size_t nCols);
template dlib::matrix<int> esautils::repeatColVecAsCols<int>(const dlib::matrix<int, 0, 1>& vec, const size_t nCols);
#endif // WITHDLIB

/// Repeat column vector as columns - armadillo implementation
template <typename T>
arma::Mat<T> esautils::repeatColVecAsCols(const arma::Mat<T>& vec, const size_t nCols)
{
    if (vec.n_cols > 1) throw std::runtime_error("matrix has more than 1 column for repeating");
    arma::Mat<T> res(vec.n_rows, nCols);
    res.each_col() = vec;
    return res;
}
// arma - explicit template instansitation
template arma::Mat<double> esautils::repeatColVecAsCols<double>(const arma::Mat<double>&, const size_t);
template arma::Mat<int> esautils::repeatColVecAsCols<int>(const arma::Mat<int>&, const size_t);

/// Shuffle a matrix
#ifdef WITHDLIB
template <typename T>
void esautils::shuffleMatrix(dlib::matrix<T>& mat, const int seed){
    dlib::rand rnd;
    for (long i = 0; i < mat.nr(); ++i) {
        for (long j = 0; j < mat.nc(); ++j) {
            long i1 = rnd.get_random_32bit_number() % mat.nr();
            long j1 = rnd.get_random_32bit_number() % mat.nc();
            std::swap(mat(i, j), mat(i1, j1));
        }
    }
}
// explicit template instantiation
template void esautils::shuffleMatrix<double>(dlib::matrix<double>&, const int);
#endif // WITHDLIB

template <typename T>
void esautils::shuffleMatrix(arma::Mat<T>& mat, const int seed)
{
    std::mt19937 engine(seed);
    std::uniform_int_distribution<unsigned int> row_dist(0, mat.n_rows - 1);
    std::uniform_int_distribution<unsigned int> col_dist(0, mat.n_cols - 1);
    for (size_t i = 0; i < mat.n_rows; i++) {
        for (size_t j = 0; j < mat.n_cols; j++) {
            unsigned int i1 = row_dist(engine);
            unsigned int j1 = col_dist(engine);
            std::swap(mat.at(i, j), mat.at(i1, j1));
        }
    }
}
template void esautils::shuffleMatrix<double>(arma::Mat<double>&, const int);

/// Generate a random sample of integers between a range
std::vector<int> esautils::sampleIntegers(const int n, const int size, const bool replace, const int seed){
    // check that both 'n' and 'size' are positive integers
    if ((n <= 0) || (size <= 0)){
        throw std::runtime_error("'n' and 'size' must be positive integers");
    }
    std::vector<int> result(size);
    std::mt19937 gen(seed);
    if (replace){
        // handle situation with replacement
        std::uniform_int_distribution<int> dis(0, n - 1);
        for (int i = 0; i < size; ++i){
            result[i] = dis(gen);
        }
    } else {
        // without replacement
        std::vector<int> pool(n);
        std::iota(pool.begin(), pool.end(), 0);
        std::shuffle(pool.begin(), pool.end(), gen);
        result.assign(pool.begin(), pool.begin() + size);
    }
    return result;
}

/// Rescale a dlib matrix
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esautils::rescaleMatrix(const dlib::matrix<T>& mat, const T lwrBnd, const T uprBnd){
    dlib::matrix<T> result(mat.nr(), mat.nc());
    double matMin = dlib::min(mat);
    double matMax = dlib::max(mat);
    double matRange = matMax - matMin;
    double bndRange = uprBnd - lwrBnd;
    for (size_t r = 0; r < mat.nr(); ++r){
        for (size_t c = 0; c < mat.nc(); ++c){
            double currVal = mat(r, c);
            double c2 = (currVal - matMin) / matRange;
            double v = lwrBnd + c2 * bndRange;
            if (std::isnan(v)){
                v = lwrBnd;
            }
            result(r, c) = v;
        }
    }
    return result;
}
// explicit template instantiation
template dlib::matrix<double> esautils::rescaleMatrix<double>(const dlib::matrix<double>&, const double, const double);
#endif // WITHDLIB

/// Rescale an armadillo matrix
template <typename T>
arma::Mat<T> esautils::rescaleMatrix(const arma::Mat<T>& mat, const T lwrBnd, const T uprBnd) {
    arma::Mat<T> result(mat.n_rows, mat.n_cols);
    double matMin = mat.min();
    double matMax = mat.max();
    double matRange = matMax - matMin;
    double bndRange = uprBnd - lwrBnd;
    for (size_t r = 0; r < mat.n_rows; ++r){
        for (size_t c = 0; c < mat.n_cols; ++c){
            double currVal = mat.at(r, c);
            double c2 = (currVal - matMin) / matRange;
            double v = lwrBnd + c2 * bndRange;
            if (std::isnan(v)){
                v = lwrBnd;
            }
            result(r, c) = v;
        }
    }
    return result;
}
template arma::Mat<double> esautils::rescaleMatrix(const arma::Mat<double>&, const double, const double);

/// Filter out invalid numbers (inf and nan)
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esautils::filterRowsInvalidNumbers(const dlib::matrix<T>& mat){
    std::vector<long> validRows;
    for (size_t r = 0; r < mat.nr(); ++r){
        bool hasInf = false;
        for (size_t c = 0; c < mat.nc(); ++c){
            if (std::isinf(mat(r, c)) || std::isnan(mat(r, c))){
                hasInf = true;
                break;
            }
        }
        if (!hasInf){
            validRows.push_back(r);
        }
    }
    dlib::matrix<T> result(validRows.size(), mat.nc());
    for (size_t i = 0; i < validRows.size(); ++i){
        dlib::set_rowm(result, i) = dlib::rowm(mat, validRows[i]);
    }
    return result;
}
// explicit template instantisation
template dlib::matrix<double> esautils::filterRowsInvalidNumbers<double>(const dlib::matrix<double>&);
#endif // WITHDLIB

/// Filter out invalid numbers (inf, nan) - armadillo implementation
// template <typename T>
// arma::Mat<T> esautils::filterRowsInvalidNumbers(const arma::Mat<T>& mat) {
template <typename T>
arma::dmat esautils::filterRowsInvalidNumbers(const arma::Base<double, T>& matIn){
    const auto& mat = matIn.get_ref();
    std::vector<long> validRows;
    for (size_t r = 0; r < mat.n_rows; r++) {
        bool hasInf = false;
        for (size_t c = 0; c < mat.n_cols; c++) {
            if (std::isinf(mat.at(r, c)) || std::isnan(mat.at(r, c))){
                hasInf = true;
                break;
            }
        }
        if (!hasInf){
            validRows.push_back(r);
        }
    }
    arma::dmat result(validRows.size(), mat.n_cols);
    for (size_t i = 0; i < validRows.size(); i++) {
        result.row(i) = mat.row(validRows[i]);
    }
    return result;
}
template arma::dmat esautils::filterRowsInvalidNumbers<arma::dmat>(const arma::Base<double, arma::dmat>&);
template arma::dmat esautils::filterRowsInvalidNumbers<arma::subview<double>>(const arma::Base<double, arma::subview<double>>&);
// template arma::Mat<double> esautils::filterRowsInvalidNumbers<double>(const arma::Mat<double>&);

/// Stack vector of matricies
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esautils::stackMatricies(const std::vector<dlib::matrix<T>>& matVec, const bool byRow){
    if (matVec.size() == 0){
        throw std::invalid_argument("No matricies to stack");
    }
    // start using the dimensions of the first matrix
    unsigned int totalRows = matVec[0].nr();
    unsigned int totalCols = matVec[0].nc();
    for (size_t i = 1; i < matVec.size(); ++i){
        // check the dimensions of the matricies, that they are stackable
        // if using rowwise stacking - the number of columns must be the same across all
        // if using columnwise stacking, the number of rows across all must be the same
        if (byRow){
            if (matVec[i].nc() != totalCols){
                throw std::invalid_argument("Matricies must have the same number of columns for rowwise stacking");
            }
            totalRows += matVec[i].nr();
        } else {
            if (matVec[i].nr() != totalRows){
                throw std::invalid_argument("Matricies must have the same number of rows for columnwise stacking");
            }
            totalCols += matVec[i].nc();
        }
    }
    // empty results vector
    dlib::matrix<T> result(totalRows, totalCols);
    // start the stacking
    unsigned int currPos = 0;
    for (size_t i = 0; i < matVec.size(); i++){
        unsigned int currR = matVec[i].nr();
        unsigned int currC = matVec[i].nc();
        if (byRow){
            // if by row, stack the matricies vertically
            dlib::set_subm(result, dlib::range(currPos, currPos + currR - 1), dlib::range(0, currC - 1)) = matVec[i];
            currPos += currR;
            // dlib::set_subm(result, dlib::range(i * matVec[i].nr(), (i + 1) * matVec[i].nr() - 1), dlib::range(0, matVec[i].nc() - 1)) = matVec[i];
        } else {
            // if by column, stack the matricies horizontally
            dlib::set_subm(result, dlib::range(0, currR - 1), dlib::range(currPos, currPos + currC - 1)) = matVec[i];
            currPos += currC;
            // dlib::set_subm(result, dlib::range(0, matVec[i].nr() - 1), dlib::range(i * matVec[i].nc(), (i + 1) * matVec[i].nc() - 1)) = matVec[i];
        }
    }
    return result;
}
// dlib - explicit template instantiation
template dlib::matrix<double> esautils::stackMatricies<double>(const std::vector<dlib::matrix<double>>&, const bool);
template dlib::matrix<int> esautils::stackMatricies<int>(const std::vector<dlib::matrix<int>>&, const bool);
#endif // WITHDLIB

// armadillo matrix implementation of stackMatricies function
template <typename T>
arma::Mat<T> esautils::stackMatricies(const std::vector<arma::Mat<T>>& matVec, const bool byRow)
{
    if (matVec.size() == 0) throw std::invalid_argument("No matricies to stack");
    // use dimensions of the first matrix
    unsigned int totalRows = matVec[0].n_rows;
    unsigned int totalCols = matVec[0].n_cols;
    // iterate thru elements - check dimensions that they are stackable
    // for rowwise stacking - number of columns must be the same across all
    // for columnwise stacking - number of rows across must all be the same
    for (size_t i = 1; i < matVec.size(); i++) {
        if (byRow) {
            if (matVec[i].n_cols != totalCols) {
                throw std::invalid_argument("Matricies must have the same number of columns for rowwise stacking");
            }
            totalRows += matVec[i].n_rows;
        } else {
            if (matVec[i].n_rows != totalRows) {
                throw std::invalid_argument("Matricies must have the same number of rows for columnwise stacking");
            }
            totalCols += matVec[i].n_cols;
        }
    }
    // empty results matrix
    arma::Mat<T> result(totalRows, totalCols);
    // start stacking
    unsigned int currPos = 0;
    for (size_t i = 0; i < matVec.size(); i++){
        unsigned int currR = matVec[i].n_rows;
        unsigned int currC = matVec[i].n_cols;
        if (byRow) {
            // if by row, stack the matricies vertically
            result.submat(arma::span(currPos, (currPos + currR - 1)), arma::span(0, (currC - 1))) = matVec[i];
            currPos += currR;
        } else {
            // if by column, stack the matricies horizontally
            result.submat(arma::span(0, (currR - 1)), arma::span(currPos, (currPos + currC - 1))) = matVec[i];
            currPos += currC;
        }
    }
    return result;
}
// armadillo - explicit template instantisation
template arma::Mat<double> esautils::stackMatricies<double>(const std::vector<arma::Mat<double>>&, const bool);
template arma::Mat<int> esautils::stackMatricies<int>(const std::vector<arma::Mat<int>>&, const bool);

/// armadillo implementation of summation of matrices in a vector of matricies
template <typename T>
arma::Mat<T> esautils::sumMatricies(const std::vector<arma::Mat<T>>& matVec)
{
    arma::Mat<T> out;
    for (size_t i = 0; i < matVec.size(); i++) {
        if (i == 0) {
            out = matVec[i];
        } else {
            out += matVec[i];
        }
    }
    return out;
}
// armadillo - explicit template instantisation
template arma::Mat<double> esautils::sumMatricies<double>(const std::vector<arma::Mat<double>>&);

/// Means (either rowise or column wise) of a matrix
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esautils::matrixMeans(const dlib::matrix<T>& mat, const bool byRow){
    if (byRow){
        dlib::matrix<T> result(mat.nr(), 1);
        for (size_t i = 0; i < mat.nr(); ++i){
            result(i, 0) = dlib::mean(dlib::rowm(mat, i));
        }
        return result;
    } else {
        // columnwise
        dlib::matrix<T> result(1, mat.nc());
        for (size_t i = 0; i < mat.nc(); ++i){
            result(0, i) = dlib::mean(dlib::colm(mat, i));
        }
        return result;
    }
}
// explicit template instantiation
template dlib::matrix<double> esautils::matrixMeans<double>(const dlib::matrix<double>&, const bool);
template dlib::matrix<int> esautils::matrixMeans<int>(const dlib::matrix<int>&, const bool);
#endif // WITHDLIB

/// Means (either rowwise, or columnwise) of a matrix
template <typename T>
arma::Mat<T> esautils::matrixMeans(const arma::Mat<T>& mat, const bool byRow)
{
    if (byRow) {
        arma::Mat<T> result(mat.n_rows, 1);
        for (size_t i = 0; i < mat.n_rows; i++) {
            result(i, 0) = arma::mean(mat.row(i));
        }
        return result;
    } else {
        // columnwise
        arma::Mat<T> result(1, mat.n_cols);
        for (size_t i = 0; i < mat.n_cols; i++) {
            result(0, i) = arma::mean(mat.col(i));
        }
        return result;
    }
}
template arma::Mat<double> esautils::matrixMeans<double>(const arma::Mat<double>&, const bool);

/// Column means by grouping variable
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esautils::colMeansByGroup(const dlib::matrix<T>& mat, const dlib::matrix<int>& groupVec)
{
    // unique IDs
    dlib::matrix<int, 0, 1> uniqueIDs = esautils::uniqueValsInColVec<int>(groupVec);
    // create the output matrix
    dlib::matrix<double> out(uniqueIDs.nr(), mat.nc());
    // iterate through each group
    for (int i = 0; i < uniqueIDs.nr(); ++i){
        // panel ID
        int id_i = uniqueIDs(i);
        // select data from mat based on the groupVec
        auto sel = [id_i](const dlib::matrix<int>& m) -> bool { return m(0, 0) == id_i; };
        dlib::matrix<double> mat_i = esautils::selectMatrixRowsForCondition<double, int>(mat, groupVec, sel);
        // calculate the column means
        dlib::matrix<double> colMeans = esautils::matrixMeans<double>(mat_i, false);
        // store in the output matrix
        dlib::set_rowm(out, i) = colMeans;
    }
    return out;
}
// explicit template instantiation
template dlib::matrix<double> esautils::colMeansByGroup<double>(const dlib::matrix<double>&, const dlib::matrix<int>&);
#endif // WITHDLIB

/// Column means by grouping variable
template <typename T>
arma::Mat<T> esautils::colMeansByGroup(const arma::Mat<T>& mat, const arma::Col<int>& groupVec)
{
    arma::Col<int> uniqueIds = arma::unique(groupVec);
    // create output matrix
    arma::Mat<T> out(uniqueIds.n_rows, mat.n_cols);
    // iterate thru each group
    for (size_t i = 0; i < uniqueIds.n_rows; i++) {
        // panel ID
        int id_i = uniqueIds.at(i);
        arma::uvec inds = (uniqueIds == id_i);
        arma::Mat<T> mat_i = esautils::selectMatrixRowsForCondition<double>(mat, inds);
        // calculate column means
        arma::Mat<T> colMeans = esautils::matrixMeans<double>(mat_i, false);
        out.row(i) = colMeans;
    }
    return out;
}
template arma::Mat<double> esautils::colMeansByGroup(const arma::Mat<double>&, const arma::Col<int>&);

/// Explode a matrix's rows over t
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esautils::matrixExplodeRowsOverT(const dlib::matrix<T>& mat, const dlib::matrix<int>& t)
{
    // verify the number of rows in both arguments are equal
    if (mat.nr() != t.nr()){
        throw std::invalid_argument("The number of rows in the matrix and t must be equal");
    }
    // vector of matricies to store outputs
    std::vector<dlib::matrix<T>> outVec(mat.nr());
    // iterate through each row
    for (unsigned int i = 0; i < mat.nr(); i++){
        // get the number of time periods
        int t_i = t(i);
        if (t_i < 0){
            throw std::invalid_argument("t must be a positive integer, it isn't at index " + std::to_string(i));
        }
        // repeat the row t_i times
        dlib::matrix<T> row = dlib::rowm(mat, i);
        dlib::matrix<T> repRow = dlib::ones_matrix<T>(t_i, 1) * row;
        outVec[i] = repRow;
    }
    // stack the output into single matrix
    return esautils::stackMatricies<T>(outVec, true);
}
// explicit template instantiation
template dlib::matrix<double> esautils::matrixExplodeRowsOverT<double>(const dlib::matrix<double>&, const dlib::matrix<int>&);
#endif

/// Explode a matrix's rows over t - armadillo implementation
template <typename T>
arma::Mat<T> esautils::matrixExplodeRowsOverT(const arma::Mat<T>& mat, const arma::Col<int>& t)
{
    // verify number of rows in both arguments are equal
    if (mat.n_rows != t.n_rows) throw std::invalid_argument("Number of rows in matrix and 't' must be equal");
    std::vector<arma::Mat<T>> outVec(mat.n_rows);
    // iterate thru each row
    for (size_t i = 0; i < mat.n_rows; i++) {
        // get number of time periods
        int t_i = t.at(i);
        if (t_i < 0) throw std::invalid_argument("t must be positive integer, it isn't at index " + std::to_string(i));
        // repeat the row t_i times
        arma::Mat<T> row = mat.row(i);
        arma::Mat<T> repRow = arma::Mat<T>(t_i, 1, arma::fill::ones) * row;
        outVec[i] = repRow;
    }
    // stack output into single matrix
    return esautils::stackMatricies<T>(outVec, true);
}
template arma::Mat<double> esautils::matrixExplodeRowsOverT<double>(const arma::Mat<double>&, const arma::Col<int>&);

/// Apply some function to either each row or column of a matrix - dlib implentation
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esautils::sweepMatrixElementwise(
    const dlib::matrix<T>& mat,
    const int margin,
    const dlib::matrix<T>& stats,
    const std::string& func
)
{
    // check arguments...
    if (margin != 1 && margin != 2){
        throw std::invalid_argument("margin must be either 1 or 2");
    }
    if (func != "*" && func != "/" && func != "+" && func != "-"){
        throw std::invalid_argument("func must be one of *, /, +, -");
    }
    // create the output matrix
    dlib::matrix<T> out(mat.nr(), mat.nc());
    if (margin == 1){
        // columnwise
        // first check that number of rows in stats is equal to the input matrix
        if (stats.nr() != mat.nr()){
            throw std::invalid_argument("The number of rows in stats must be equal to the number of rows in the input matrix");
        }
        // stats should have 1 column
        if (stats.nc() != 1){
            throw std::invalid_argument("stats must have 1 column");
        }
        for (size_t c = 0; c < mat.nc(); ++c){
            // handle the different functions
            if (func == "*"){
                // elementwise multiplication - columnwise
                dlib::set_colm(out, c) = dlib::pointwise_multiply(dlib::colm(mat, c), stats);
            } else if (func == "/"){
                // elementwise division - columnwise
                dlib::set_colm(out, c) = esamath::pointwise_divide<T>(dlib::colm(mat, c), stats);
            } else if (func == "+"){
                // elementwise addition - columnwise
                dlib::set_colm(out, c) = dlib::colm(mat, c) + stats;
            } else if (func == "-"){
                // elementwise subtraction - columnwise
                dlib::set_colm(out, c) = dlib::colm(mat, c) - stats;
            }
        }
    } else if (margin == 2){
        // rowwise
        // check number of columns align between input and stats
        if (stats.nc() != mat.nc()){
            throw std::invalid_argument("The number of columns in stats must be equal to the number of columns in the input matrix");
        }
        // stats should have 1 row
        if (stats.nr() != 1){
            throw std::invalid_argument("stats must have 1 row");
        }
        for (size_t r = 0; r < mat.nr(); ++r){
            // handle the different functions
            if (func == "*"){
                // elementwise multiplication - rowwise
                dlib::set_rowm(out, r) = dlib::pointwise_multiply(dlib::rowm(mat, r), stats);
            } else if (func == "/"){
                // elementwise division - rowwise
                dlib::set_rowm(out, r) = esamath::pointwise_divide<T>(dlib::rowm(mat, r), stats);
            } else if (func == "+"){
                // elementwise addition - rowwise
                dlib::set_rowm(out, r) = dlib::rowm(mat, r) + stats;
            } else if (func == "-"){
                // elementwise subtraction - rowwise
                dlib::set_rowm(out, r) = dlib::rowm(mat, r) - stats;
            }
        }
    }
    return out;
}
// explicit template instantiation
template dlib::matrix<double> esautils::sweepMatrixElementwise(const dlib::matrix<double>&, const int, const dlib::matrix<double>&, const std::string&);
#endif // WITHDLIB

/// Apply some function to either each row or column of a matrix - armadillo implentation 
// template <typename T>
// arma::Mat<T> esautils::sweepMatrixElementwise(const arma::Mat<T>& mat, const int margin, arma::Mat<T>& stats, const std::string& func)
template <typename T1, typename T2>
arma::dmat esautils::sweepMatrixElementwise(
    const arma::Base<double, T1>& matIn,
    const int margin,
    const arma::Base<double, T2>& statsIn,
    const std::string& func
)
{
    const auto& mat = matIn.get_ref();
    const auto& stats = statsIn.get_ref();
    // check arguments
    if (margin != 1 && margin != 2) throw std::invalid_argument("margin must be 1, or 2");
    if (func != "*" && func != "/" && func != "+" && func != "-") throw std::invalid_argument("func must be one of *, /, +, -");
    // create output matrix
    arma::dmat out(mat);
    if (margin == 1) {
        // columnwise application
        // check no. rows in stats = input matrix
        if (stats.n_rows != mat.n_rows) throw std::invalid_argument("rows in 'stats' must equal rows in 'mat'");
        // stats should also only have one column
        if (stats.n_cols != 1) throw std::invalid_argument("stats must have 1 column");
        // iterate thru the columns, and apply function
        out.each_col([&func, &stats](arma::vec& b) {
            if (func == "*") {
                // elementwise multiplication
                b = b % stats;
            } else if (func == "/") {
                // elementwise division
                b = b / stats;
            } else if (func == "+") {
                // elementwise addition
                b = b + stats;
            } else if (func == "-") {
                // elementwise subtraction
                b = b - stats;
            }
        });
    } else if (margin == 2) {
        // rowwise application
        // check no. columns align between inputs and stats
        if (stats.n_cols != mat.n_cols) throw std::invalid_argument("cols in 'stats' must equal cols in 'mat'");
        // stats should have one row
        if (stats.n_rows != 1) throw std::invalid_argument("stats must have 1 row");
        out.each_row([&func, &stats](arma::drowvec& b) {
            if (func == "*") {
                // elementwise multiplication
                b = b % stats;
            } else if (func == "/") {
                // elementwise division
                b = b / stats;
            } else if (func == "+") {
                // elementwise addition
                b = b + stats;
            } else if (func == "-") {
                // elementwise subtraction
                b = b - stats;
            }
        });
    }
    return out;
}
// explicit template instantisation
// template arma::Mat<double> esautils::sweepMatrixElementwise(const arma::Mat<double>&, const int, arma::Mat<double>&, const std::string&);
template arma::dmat esautils::sweepMatrixElementwise<arma::dmat, arma::dmat>(
    const arma::Base<double, arma::dmat>&, const int, const arma::Base<double, arma::dmat>&, const std::string&
);
template arma::dmat esautils::sweepMatrixElementwise<arma::dmat, arma::subview<double>>(
    const arma::Base<double, arma::dmat>&, const int, const arma::Base<double, arma::subview<double>>&, const std::string&
);
template arma::dmat esautils::sweepMatrixElementwise<arma::subview<double>, arma::dmat>(
    const arma::Base<double, arma::subview<double>>&, const int, const arma::Base<double, arma::dmat>&, const std::string&
);
template arma::dmat esautils::sweepMatrixElementwise<arma::subview<double>, arma::subview<double>>(
    const arma::Base<double, arma::subview<double>>&, const int, const arma::Base<double, arma::subview<double>>&, const std::string&
);


/// Replace zeros up to machine tolerance - dlib implementation
#ifdef WITHDLIB
template <typename T>
dlib::matrix<T> esautils::replaceValuesPrecision(const dlib::matrix<T>& mat, const T value, const T replacement, const double eps)
{
    dlib::matrix<T> result(mat.nr(), mat.nc());
    for (size_t i = 0; i < mat.nr(); ++i){
        for (size_t j = 0; j < mat.nc(); ++j){
            // if (std::abs(mat(i, j) - value) < std::numeric_limits<T>::epsilon()){
            if (std::abs(mat(i, j) - value) < eps){
                result(i, j) = replacement;
            } else {
                result(i, j) = mat(i, j);
            }
        }
    }
    return result;
}
template dlib::matrix<double> esautils::replaceValuesPrecision(const dlib::matrix<double>&, const double, const double, const double);
template dlib::matrix<int> esautils::replaceValuesPrecision(const dlib::matrix<int>&, const int, const int, const double);
#endif // WITHDLIB

/// Replace zeros up to machine tolerance - armadillo implementation
template <typename T>
arma::Mat<T> esautils::replaceValuesPrecision(const arma::Mat<T>& mat, const T value, const T replacement, const double eps)
{
    arma::Mat<T> out(mat);
    out.transform([&value, &replacement, &eps](T v) {
        return (std::abs((v - value)) < eps) ? replacement : v;
    });
    return out;
}
template arma::Mat<double> esautils::replaceValuesPrecision(const arma::Mat<double>& , const double, const double, const double);

#ifdef WITHDLIB
template <typename T>
dlib::matrix<T, 0, 1> esautils::replaceValuesPrecision(const dlib::matrix<T, 0, 1>& mat, const T value, const T replacement, const double eps)
{
    dlib::matrix<T, 0, 1> result(mat.nr());
    for (size_t i = 0; i < mat.nr(); ++i){
       // if (std::abs(mat(i) - value) < std::numeric_limits<T>::epsilon()){
        if (std::abs(mat(i) - value) < eps){
            result(i) = replacement;
        } else {
            result(i) = mat(i);
        }
    }
    return result;
}
template dlib::matrix<double, 0, 1> esautils::replaceValuesPrecision<double>(const dlib::matrix<double, 0, 1>&, const double, const double, const double);
template dlib::matrix<int, 0, 1> esautils::replaceValuesPrecision(const dlib::matrix<int, 0, 1>&, const int, const int, const double);
#endif // WITHDLIB

template <typename T>
arma::Col<T> esautils::replaceValuesPrecision(const arma::Col<T>& mat, const T value, const T replacement, const double eps)
{
    arma::Mat<T> out(mat);
    out.transform([&value, &replacement, &eps](T v) {
        return (std::abs((v - value)) < eps) ? replacement : v;
    });
    return out;
}
template arma::Col<double> esautils::replaceValuesPrecision<double>(const arma::Col<double>&, const double, const double, const double);

#ifdef WITHEIGEN
/// Convert armadillo matrix to Eigen matrix
Eigen::MatrixXd esautils::castEigen(arma::mat m) {
    Eigen::MatrixXd out = Eigen::Map<Eigen::MatrixXd>(m.memptr(), m.n_rows, m.n_cols);
    return out;
}

/// Convert Eigen matrix to armadillo matrix
arma::mat esautils::castArma(Eigen::MatrixXd m) {
    arma::mat out = arma::mat(m.data(), m.rows(), m.cols(), false, false);
    return out;
}

/// Create eigen map, which wraps memory of armadillo column vector
template <typename T>
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> esautils::armaToEigenVec(const arma::Col<T>& m)
{
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(m.memptr(), m.n_rows);
}
template Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> esautils::armaToEigenVec(const arma::Col<double>&);
template Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>> esautils::armaToEigenVec(const arma::Col<float>&);

/// create mutable eigen map - wraps memory of armadillo column vector
template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> esautils::armaToEigenVec(arma::Col<T>& m)
{
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(m.memptr(), m.n_rows);
}
template Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> esautils::armaToEigenVec(arma::Col<double>&);
template Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> esautils::armaToEigenVec(arma::Col<float>&);

/// Create armadillo column vector that wraps the memory of a const eigen vector
template <typename T>
const arma::Col<T> esautils::eigenToArmaVec(const Eigen::Matrix<T, Eigen::Dynamic, 1>& m)
{
    return arma::Col<T>(const_cast<T*>(m.data()), m.size(), false, true);
}
template const arma::Col<double> esautils::eigenToArmaVec(const Eigen::Matrix<double, Eigen::Dynamic, 1>&);
template const arma::Col<float> esautils::eigenToArmaVec(const Eigen::Matrix<float, Eigen::Dynamic, 1>&);

/// create mutable armadillo column vector that wraps the memory of an eigen vector
template <typename T>
arma::Col<T> esautils::eigenToArmaVec(Eigen::Matrix<T, Eigen::Dynamic, 1>& m)
{
    return arma::Col<T>(m.data(), m.size(), false, true);
}
template arma::Col<double> esautils::eigenToArmaVec(Eigen::Matrix<double, Eigen::Dynamic, 1>&);
template arma::Col<float> esautils::eigenToArmaVec(Eigen::Matrix<float, Eigen::Dynamic, 1>&);

#endif // WITHEIGEN