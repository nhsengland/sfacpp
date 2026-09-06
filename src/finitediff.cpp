/**
 * 
 * Based on the logic from cppoptlib by Patrick Wieschollek
 * https://github.com/PatWie/CppNumericalSolvers
 */

#include "utils/finitediff.hpp"
#include <array>
#include <limits>
#include <string>
#include <exception>
#include "math/esamath.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"

#ifdef WITHDLIB
dlib::matrix<double> finitediff::calculateFiniteGradient(
    const dlib::matrix<double, 0, 1>& x0,
    const std::function<double(const dlib::matrix<double, 0, 1>)>& fn,
    unsigned int accuracy
)
{
    // machine precision
    constexpr double machineEps = std::numeric_limits<double>::epsilon();
    // coefficients for finite difference formulas of increasing accuracy
    // accuracy can be 0, 1, 2, 3
    if (accuracy > 3){
        throw std::invalid_argument("'accuracy' must be 0, 1, 2, or 3.");
    }
    static const std::array<std::vector<double>, 4> coeff = {
        {
            {1.0, -1.0},
            {1.0, -8.0, 8.0, -1.0},
            {-1.0, 9.0, -45.0, 45.0, -9.0, 1.0},
            {3.0, -32.0, 168.0, -672.0, 672.0, -168.0, 32.0, -3.0}
        }
    };
    static const std::array<std::vector<double>, 4> coeff2 = {
        {
            {1.0, -1.0},
            {-2.0, -1.0, 1.0, 2.0},
            {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0},
            {-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0}
        }
    };
    static const std::array<double, 4> dd = {2.0, 12.0, 60.0, 840.0};
    // output gradient
    dlib::matrix<double> grad(1, x0.nr());
    // copy of parameter vector
    dlib::matrix<double, 0, 1> x = x0;
    const int innerSteps = 2 * (accuracy + 1);
    for (unsigned int d = 0; d < x0.nr(); d++){
        // compute coordinate-dependent stepsize
        double h = std::sqrt(machineEps) * std::max(std::abs(x0(d)), 1.0);
        double ddVal = dd[accuracy] * h;
        grad(1, d) = 0.0;
        for (int s = 0; s < innerSteps; s++){
            double tmp = x(d);
            x(d) += coeff2[accuracy][s] * h;
            grad(1, d) += coeff[accuracy][s] * fn(x);
            x(d) = tmp;
        }
        grad(1, d) /= ddVal;
    }
    return grad;
}
#endif // WITHDLIB

/// calculate gradient by finite differences - armadillo implementation
arma::dmat finitediff::calculateFiniteGradient(
    const arma::dcolvec& x0,
    const std::function<double(const arma::dcolvec&)>& fn,
    unsigned int accuracy
)
{
    // machine precision
    constexpr double machineEps = std::numeric_limits<double>::epsilon();
    // coefficients for finite difference formulas of increasing accuracy
    // accuracy can be 0, 1, 2, 3
    if (accuracy > 3){
        throw std::invalid_argument("'accuracy' must be 0, 1, 2, or 3.");
    }
    static const std::array<std::vector<double>, 4> coeff = {
        {
            {1.0, -1.0},
            {1.0, -8.0, 8.0, -1.0},
            {-1.0, 9.0, -45.0, 45.0, -9.0, 1.0},
            {3.0, -32.0, 168.0, -672.0, 672.0, -168.0, 32.0, -3.0}
        }
    };
    static const std::array<std::vector<double>, 4> coeff2 = {
        {
            {1.0, -1.0},
            {-2.0, -1.0, 1.0, 2.0},
            {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0},
            {-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0}
        }
    };
    static const std::array<double, 4> dd = {2.0, 12.0, 60.0, 840.0};
    // output gradient
    arma::dmat grad(1, x0.n_rows);
    // copy of parameter vector
    arma::dcolvec x = x0;
    const int innerSteps = 2 * (accuracy + 1);
    for (unsigned int d = 0; d < x0.n_rows; d++){
        // compute coordinate-dependent stepsize
        double h = std::sqrt(machineEps) * std::max(std::abs(x0.at(d)), 1.0);
        double ddVal = dd[accuracy] * h;
        grad.at(1, d) = 0.0;
        for (int s = 0; s < innerSteps; s++){
            double tmp = x(d);
            x.at(d) += coeff2[accuracy][s] * h;
            grad.at(1, d) += coeff[accuracy][s] * fn(x);
            x.at(d) = tmp;
        }
        grad.at(1, d) /= ddVal;
    }
    return grad;
}

/// calculate hessian per simulation, using gradient function - dlib implementation
#ifdef WITHDLIB
std::vector<dlib::matrix<double>> finitediff::calculateFiniteHessianSimsUsingGrad(
    const dlib::matrix<double, 0, 1>& x0,
    const std::function<dlib::matrix<double>(const dlib::matrix<double, 0, 1>)>& grad
)
{
    // numerical approximation of the hessian using gradient
    constexpr double machineEps = std::numeric_limits<double>::epsilon();
    unsigned int n = x0.nr();
    // local copy of the parameter vector
    dlib::matrix<double, 0, 1> x = x0;
    // 2d matrix of dlib matricies
    std::vector<std::vector<dlib::matrix<double>>> hessElements(n, std::vector<dlib::matrix<double>>(n));
    // central difference approximation
    unsigned int mtxDepth = 0;
    for (unsigned int i = 0; i < n; i++){
        // setp size for ith coordinate
        double hi = std::sqrt(machineEps) * (1.0 + std::abs(x0(i)));
        x = x0;
        x(i) += hi;
        dlib::matrix<double> grad_plus = grad(x);
        // ESALogger::logger()->trace("grad plus x {}", x);
        if (mtxDepth == 0) mtxDepth = grad_plus.nr();
        // grad_plus ∈ ℝ^(nsim x k)
        dlib::matrix<double> grad_plus_i = dlib::colm(grad_plus, i);
        x = x0;
        x(i) -= hi;
        // ESALogger::logger()->trace("grad minus x {}", x);
        dlib::matrix<double> grad_minus = grad(x);
        // grad_minus ∈ ℝ^(nsim x k)
        dlib::matrix<double> grad_minus_i = dlib::colm(grad_minus, i);
        // ESALogger::logger()->trace("grad plus i {} grad minus i {}", grad_plus_i, grad_minus_i);
        dlib::matrix<double> gdiff = (grad_plus_i - grad_minus_i) / (4.0 * hi);
        // ESALogger::logger()->trace("diff {}", gdiff);
        dlib::matrix<double> dfdxx = 2.0 * ((grad_plus_i - grad_minus_i) / (4.0 * hi));
        hessElements[i][i] = std::move(dfdxx);
        for (unsigned int j = i + 1; j < n; j++){
            double hj = std::sqrt(machineEps) * (1.0 + std::abs(x0(j)));
            x = x0;
            x(j) += hj;
            dlib::matrix<double> ghjp = grad(x);
            // ghjp ∈ ℝ^(nsim x k)
            dlib::matrix<double> ghjp_i = dlib::colm(ghjp, i);
            x = x0;
            x(j) -= hj;
            dlib::matrix<double> ghjm = grad(x);
            // ghjm ∈ ℝ^(nsim x k)
            dlib::matrix<double> ghjm_i = dlib::colm(ghjm, i);
            // for g_j - we can use matricies from i; 
            dlib::matrix<double> ghip_j = dlib::colm(grad_plus, j);
            dlib::matrix<double> ghim_j = dlib::colm(grad_minus, j);
            dlib::matrix<double> dfdxixj = ((ghjp_i - ghjm_i) / (4.0 * hj)) + ((ghip_j - ghim_j) / (4.0 * hi));
            hessElements[i][j] = std::move(dfdxixj);
        }
    }
    if (hessElements.size() == 0) throw std::runtime_error("hessElements has zero size, something went wrong");
    if (hessElements[0].size() == 0) throw std::runtime_error("second dimension of hessElements has zero size, something went wrong");
    std::vector<dlib::matrix<double>> hessians(mtxDepth);
    for (unsigned int i = 0; i < mtxDepth; i++){
        dlib::matrix<double> hess(n, n);
        for (unsigned int j = 0; j < n; j++){
            // diagonal elements
            hess(j, j) = hessElements[j][j](i, 0);
            for (unsigned int k = j + 1; k < n; k++){
                // off-diagonal elements
                hess(j, k) = hessElements[j][k](i, 0);
                hess(k, j) = hessElements[j][k](i, 0);
            }
        }
        hessians[i] = hess;
    }
    return hessians;
}
#endif // WITHDLIB

/// calculate hessian per simulation, using finite differences, given gradient function - armadillo implementation
std::vector<arma::dmat> finitediff::calculateFiniteHessianSimsUsingGrad(
    const arma::dcolvec& x0,
    const std::function<arma::dmat(const arma::dcolvec&)>& grad
)
{
    // numerical approximation of the hessian using gradient
    constexpr double machineEps = std::numeric_limits<double>::epsilon();
    unsigned int n = x0.n_rows;
    // local copy of the parameter vector
    arma::dcolvec x = x0;
    // 2d matrix of dlib matricies
    std::vector<std::vector<arma::dmat>> hessElements(n, std::vector<arma::dmat>(n));
    // central difference approximation
    unsigned int mtxDepth = 0;
    for (unsigned int i = 0; i < n; i++){
        // setp size for ith coordinate
        double hi = std::sqrt(machineEps) * (1.0 + std::abs(x0(i)));
        x = x0;
        x.at(i) += hi;
        arma::dmat grad_plus = grad(x);
        // ESALogger::logger()->trace("grad plus x {}", x);
        if (mtxDepth == 0) mtxDepth = grad_plus.n_rows;
        // grad_plus ∈ ℝ^(nsim x k)
        arma::dmat grad_plus_i = grad_plus.col(i);
        x = x0;
        x.at(i) -= hi;
        // ESALogger::logger()->trace("grad minus x {}", x);
        arma::dmat grad_minus = grad(x);
        // grad_minus ∈ ℝ^(nsim x k)
        arma::dmat grad_minus_i = grad_minus.col(i);
        // ESALogger::logger()->trace("grad plus i {} grad minus i {}", grad_plus_i, grad_minus_i);
        arma::dmat gdiff = (grad_plus_i - grad_minus_i) / (4.0 * hi);
        // ESALogger::logger()->trace("diff {}", gdiff);
        arma::dmat dfdxx = 2.0 * ((grad_plus_i - grad_minus_i) / (4.0 * hi));
        hessElements[i][i] = std::move(dfdxx);
        for (unsigned int j = i + 1; j < n; j++){
            double hj = std::sqrt(machineEps) * (1.0 + std::abs(x0(j)));
            x = x0;
            x(j) += hj;
            arma::dmat ghjp = grad(x);
            // ghjp ∈ ℝ^(nsim x k)
            arma::dmat ghjp_i = ghjp.col(i);
            x = x0;
            x(j) -= hj;
            arma::dmat ghjm = grad(x);
            // ghjm ∈ ℝ^(nsim x k)
            arma::dmat ghjm_i = ghjm.col(i);
            // for g_j - we can use matricies from i; 
            arma::dmat ghip_j = grad_plus.col(j);
            arma::dmat ghim_j = grad_minus.col(j);
            arma::dmat dfdxixj = ((ghjp_i - ghjm_i) / (4.0 * hj)) + ((ghip_j - ghim_j) / (4.0 * hi));
            hessElements[i][j] = std::move(dfdxixj);
        }
    }
    if (hessElements.size() == 0) throw std::runtime_error("hessElements has zero size, something went wrong");
    if (hessElements[0].size() == 0) throw std::runtime_error("second dimension of hessElements has zero size, something went wrong");
    std::vector<arma::dmat> hessians(mtxDepth);
    for (unsigned int i = 0; i < mtxDepth; i++){
        arma::dmat hess(n, n);
        for (unsigned int j = 0; j < n; j++){
            // diagonal elements
            hess(j, j) = hessElements[j][j](i, 0);
            for (unsigned int k = j + 1; k < n; k++){
                // off-diagonal elements
                hess(j, k) = hessElements[j][k](i, 0);
                hess(k, j) = hessElements[j][k](i, 0);
            }
        }
        hessians[i] = hess;
    }
    return hessians;
}

/// calculate hessian per simulation, using finite differences - dlib implementation
#ifdef WITHDLIB
std::vector<dlib::matrix<double>> finitediff::calculateFiniteHessianSims(
    const dlib::matrix<double, 0, 1>& x0,
    const std::function<dlib::matrix<double>(const dlib::matrix<double, 0, 1>)>& fn,
    unsigned int accuracy
)
{
    // ESALogger::logger()->trace("Calculate Finite Hessian Sims with accuracy {}", accuracy);
    // machine precision
    constexpr double machineEps = std::numeric_limits<double>::epsilon();
    unsigned int n = x0.nr();
    // 2d matrix of dlib matricies - each element in the std::vector corresponds to an element in the n x n hessian
    // matrix; with each dlib matrix corresponding to a nsim x 1 (?) matrix for simulations
    std::vector<std::vector<dlib::matrix<double>>> hessElements(n, std::vector<dlib::matrix<double>>(n));
    // local copy of the parameter vector
    dlib::matrix<double, 0, 1> x = x0;
    dlib::matrix<double> f0 = fn(x0);
    // check what was returned from the function is compatible
    // if neither row or column is 1, then something went wrong
    if (f0.nc() != 1 && f0.nr() != 1){
        throw std::runtime_error("fn returned a " + std::to_string(f0.nr()) + " x " + std::to_string(f0.nc()) + " matrix, expecting either row or col to be 1");
    }
    bool usingRows = (f0.nr() > f0.nc());
    unsigned int mtxDepth = std::max(f0.nr(), f0.nc());
    if (accuracy == 0){
        // basic central difference approximation
        for (unsigned int i = 0; i < n; i++){
            // adaptive step size for the ith coordinate
            // double hi = std::sqrt(machineEps) * std::max(std::abs(x0(i)), 1.0);
            double hi = std::sqrt(machineEps) * (1.0 + std::abs(x0(i)));
            // diagonal - standard second derivative
            // x = x0;
            // x(i) += hi;
            // dlib::matrix<double> fpl = fn(x);
            // x = x0;
            // x(i) -= hi;
            // dlib::matrix<double> fmi = fn(x);
            // hessElements[i][i] = (fpl - 2.0 * f0 + fmi) / (hi * hi);
            x = x0;
            x(i) += 2.0 * hi;
            dlib::matrix<double> f2p = fn(x);
            x = x0;
            x(i) += hi;
            dlib::matrix<double> fp = fn(x);
            x = x0;
            x(i) -= hi;
            dlib::matrix<double> fm = fn(x);
            x = x0;
            x(i) -= 2.0 * hi;
            dlib::matrix<double> f2m = fn(x);
            hessElements[i][i] = (-f2p + 16.0 * fp - 30.0 * f0 + 16.0 * fm - f2m) / (12.0 * hi * hi);
            for (unsigned int j = i + 1; j < n; j++){
                // off-diagnonal elements
                // double hj = std::sqrt(machineEps) * std::max(std::abs(x0(j)), 1.0);
                double hj = std::sqrt(machineEps) * (1.0 + std::abs(x0(i)));
                x = x0;
                x(i) += hi;
                x(j) += hj;
                dlib::matrix<double> fpp = fn(x);
                x = x0;
                x(i) += hi;
                x(j) -= hj;
                dlib::matrix<double> fpm = fn(x);
                x = x0;
                x(i) -= hi;
                x(j) += hj;
                dlib::matrix<double> fmp = fn(x);
                x = x0;
                x(i) -= hi;
                x(j) -= hj;
                dlib::matrix<double> fmm = fn(x);
                dlib::matrix<double> d2f = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
                // hessian matrix is symmetrical
                hessElements[i][j] = d2f;
                // hessElements[j][i] = d2f;
            }
        }
    } else {
        // higher order finite difference approximation
        for (unsigned int i = 0; i < n; i++){
            double hi = std::sqrt(machineEps) * std::max(std::abs(x0(i)), 1.0);
            x = x0;
            x(i) += hi;
            dlib::matrix<double> fp = fn(x);
            x = x0;
            x(i) -= hi;
            dlib::matrix<double> fm = fn(x);
            hessElements[i][i] = (fp - 2.0 * f0 + fm) / (hi * hi);
            for (unsigned int j = i + 1; j < n; j++){
                double hj = std::sqrt(machineEps) * std::max(std::abs(x0(j)), 1.0);
                double h = (hi + hj) / 2.0;
                double tmpi = x0(i), tmpj = x0(j);
                dlib::matrix<double> term1, term2, term3, term4;
                // term 1
                x = x0;
                x(i) = tmpi + 1.0 * h;
                x(j) = tmpj - 2.0 * h;
                term1 = fn(x);
                x = x0;
                x(i) = tmpi + 2.0 * h;
                x(j) = tmpj - 1.0 * h;
                term1 += fn(x);
                x = x0;
                x(i) = tmpi - 2.0 * h;
                x(j) = tmpj + 1.0 * h;
                term1 += fn(x);
                x = x0;
                x(i) = tmpi - 1.0 * h;
                x(j) = tmpj + 2.0 * h;
                term1 += fn(x);
                // term2: +63 * (f(tmpi-1h, tmpj-2h) + f(tmpi-2h, tmpj-1h) + f(tmpi+1h,
                // tmpj+2h) + f(tmpi+2h, tmpj+1h))
                x = x0;
                x(i) = tmpi - 1.0 * h;
                x(j) = tmpj - 2.0 * h;
                term2 = fn(x);
                x = x0;
                x(i) = tmpi - 2.0 * h;
                x(j) = tmpj - 1.0 * h;
                term2 += fn(x);
                x = x0;
                x(i) = tmpi + 1.0 * h;
                x(j) = tmpj + 2.0 * h;
                term2 += fn(x);
                x = x0;
                x(i) = tmpi + 2.0 * h;
                x(j) = tmpj + 1.0 * h;
                term2 += fn(x);
                // term3: +44 * (f(tmpi+2h, tmpj-2h) + f(tmpi-2h, tmpj+2h) - f(tmpi-2h,
                // tmpj-2h) - f(tmpi+2h, tmpj+2h))
                x = x0;
                x(i) = tmpi + 2.0 * h;
                x(j) = tmpj - 2.0 * h;
                term3 = fn(x);
                x = x0;
                x(i) = tmpi - 2.0 * h;
                x(j) = tmpj + 2.0 * h;
                term3 += fn(x);
                x = x0;
                x(i) = tmpi - 2.0 * h;
                x(j) = tmpj - 2.0 * h;
                term3 -= fn(x);
                x = x0;
                x(i) = tmpi + 2.0 * h;
                x(j) = tmpj + 2.0 * h;
                term3 -= fn(x);
                // term4: +74 * (f(tmpi-1h, tmpj-1h) + f(tmpi+1h, tmpj+1h) - f(tmpi+1h,
                // tmpj-1h) - f(tmpi-1h, tmpj+1h))
                x = x0;
                x(i) = tmpi - 1.0 * h;
                x(j) = tmpj - 1.0 * h;
                term4 = fn(x);
                x = x0;
                x(i) = tmpi + 1.0 * h;
                x(j) = tmpj + 1.0 * h;
                term4 += fn(x);
                x = x0;
                x(i) = tmpi + 1.0 * h;
                x(j) = tmpj - 1.0 * h;
                term4 -= fn(x);
                x = x0;
                x(i) = tmpi - 1.0 * h;
                x(j) = tmpj + 1.0 * h;
                term4 -= fn(x);
                dlib::matrix<double> mixed = (
                    (-63.0 * term1 + 63.0 * term2 + 44.0 * term3 + 74.0 * term4) /
                    (600.0 * h * h)
                );
                dlib::matrix<double> tmp = (-63.0 * term1 + 63.0 * term2 + 44.0 * term3 + 74.0 * term4);
                // ESALogger::logger()->trace("i: {}, j: {}: {} / {}", i, j, tmp, (600.0 * h * h));
                // ESALogger::logger()->trace("i: {}, j: {}: result {}", i, j, mixed);
                // ESALogger::logger()->trace("term 1: {}\nterm 2: {}\nterm3: {}\nterm4: {}", term1, term2, term3, term4);
                // hessian is symmetric
                hessElements[i][j] = mixed;
                // hessElements[j][i] = mixed;
            }
        }
    }
    // finally, convert to a vector of hessian dlib matrices
    // find size of the dlib::matrix in first element
    if (hessElements.size() == 0) throw std::runtime_error("hessElements has zero size, something went wrong");
    if (hessElements[0].size() == 0) throw std::runtime_error("second dimension of hessElements has zero size, something went wrong");
    std::vector<dlib::matrix<double>> hessians(mtxDepth);
    for (unsigned int i = 0; i < mtxDepth; i++){
        // iterating depth dlib matrix in the 2d vector
        dlib::matrix<double> hess(n, n);
        // row based or column based for the final dlib matrix
        unsigned int ridx = (usingRows ? i : 0);
        unsigned int cidx = (usingRows ? 0 : i);
        for (unsigned int j = 0; j < n; j++){
            // diagonal elements
            hess(j, j) = hessElements[j][j](ridx, cidx);
            for (unsigned int k = j + 1; k < n; k++){
                // off-diagonal elements
                hess(j, k) = hessElements[j][k](ridx, cidx);
                hess(k, j) = hessElements[j][k](ridx, cidx);
            }
        }
        hessians[i] = hess;
    }
    return hessians;
}
#endif // WITHDLIB

/// calculate hessian per simulation using finite differences - armadillo implementation
std::vector<arma::dmat> finitediff::calculateFiniteHessianSims(
    const arma::dcolvec& x0,
    const std::function<arma::dmat(const arma::dcolvec&)>& fn,
    unsigned int accuracy
)
{
    // machine precision
    constexpr double machineEps = std::numeric_limits<double>::epsilon();
    unsigned int n = x0.n_rows;
    // 2d matrix of dlib matricies - each element in the std::vector corresponds to an element in the n x n hessian
    // matrix; with each dlib matrix corresponding to a nsim x 1 (?) matrix for simulations
    std::vector<std::vector<arma::dmat>> hessElements(n, std::vector<arma::dmat>(n));
    // local copy of the parameter vector
    arma::dcolvec x = x0;
    arma::dmat f0 = fn(x0);
    // check what was returned from the function is compatible
    // if neither row or column is 1, then something went wrong
    if (f0.n_cols != 1 && f0.n_rows != 1){
        throw std::runtime_error("fn returned a " + std::to_string(f0.n_rows) + " x " + std::to_string(f0.n_cols) + " matrix, expecting either row or col to be 1");
    }
    bool usingRows = (f0.n_rows > f0.n_cols);
    unsigned int mtxDepth = std::max(f0.n_rows, f0.n_cols);
    if (accuracy == 0){
        // basic central difference approximation
        for (unsigned int i = 0; i < n; i++){
            // adaptive step size for the ith coordinate
            // double hi = std::sqrt(machineEps) * std::max(std::abs(x0(i)), 1.0);
            double hi = std::sqrt(machineEps) * (1.0 + std::abs(x0(i)));
            // diagonal - standard second derivative
            // x = x0;
            // x(i) += hi;
            // dlib::matrix<double> fpl = fn(x);
            // x = x0;
            // x(i) -= hi;
            // dlib::matrix<double> fmi = fn(x);
            // hessElements[i][i] = (fpl - 2.0 * f0 + fmi) / (hi * hi);
            x = x0;
            x.at(i) += 2.0 * hi;
            arma::dmat f2p = fn(x);
            x = x0;
            x.at(i) += hi;
            arma::dmat fp = fn(x);
            x = x0;
            x.at(i) -= hi;
            arma::dmat fm = fn(x);
            x = x0;
            x.at(i) -= 2.0 * hi;
            arma::dmat f2m = fn(x);
            hessElements[i][i] = (-f2p + 16.0 * fp - 30.0 * f0 + 16.0 * fm - f2m) / (12.0 * hi * hi);
            for (unsigned int j = i + 1; j < n; j++){
                // off-diagnonal elements
                // double hj = std::sqrt(machineEps) * std::max(std::abs(x0(j)), 1.0);
                double hj = std::sqrt(machineEps) * (1.0 + std::abs(x0.at(i)));
                x = x0;
                x.at(i) += hi;
                x.at(j) += hj;
                arma::dmat fpp = fn(x);
                x = x0;
                x.at(i) += hi;
                x.at(j) -= hj;
                arma::dmat fpm = fn(x);
                x = x0;
                x.at(i) -= hi;
                x.at(j) += hj;
                arma::dmat fmp = fn(x);
                x = x0;
                x.at(i) -= hi;
                x.at(j) -= hj;
                arma::dmat fmm = fn(x);
                arma::dmat d2f = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
                // hessian matrix is symmetrical
                hessElements[i][j] = d2f;
                // hessElements[j][i] = d2f;
            }
        }
    } else {
        // higher order finite difference approximation
        for (unsigned int i = 0; i < n; i++){
            double hi = std::sqrt(machineEps) * std::max(std::abs(x0.at(i)), 1.0);
            x = x0;
            x.at(i) += hi;
            arma::dmat fp = fn(x);
            x = x0;
            x.at(i) -= hi;
            arma::dmat fm = fn(x);
            hessElements[i][i] = (fp - 2.0 * f0 + fm) / (hi * hi);
            for (unsigned int j = i + 1; j < n; j++){
                double hj = std::sqrt(machineEps) * std::max(std::abs(x0.at(j)), 1.0);
                double h = (hi + hj) / 2.0;
                double tmpi = x0.at(i), tmpj = x0.at(j);
                arma::dmat term1, term2, term3, term4;
                // term 1
                x = x0;
                x.at(i) = tmpi + 1.0 * h;
                x.at(j) = tmpj - 2.0 * h;
                term1 = fn(x);
                x = x0;
                x.at(i) = tmpi + 2.0 * h;
                x.at(j) = tmpj - 1.0 * h;
                term1 += fn(x);
                x = x0;
                x.at(i) = tmpi - 2.0 * h;
                x.at(j) = tmpj + 1.0 * h;
                term1 += fn(x);
                x = x0;
                x.at(i) = tmpi - 1.0 * h;
                x.at(j) = tmpj + 2.0 * h;
                term1 += fn(x);
                // term2: +63 * (f(tmpi-1h, tmpj-2h) + f(tmpi-2h, tmpj-1h) + f(tmpi+1h,
                // tmpj+2h) + f(tmpi+2h, tmpj+1h))
                x = x0;
                x.at(i) = tmpi - 1.0 * h;
                x.at(j) = tmpj - 2.0 * h;
                term2 = fn(x);
                x = x0;
                x.at(i) = tmpi - 2.0 * h;
                x.at(j) = tmpj - 1.0 * h;
                term2 += fn(x);
                x = x0;
                x.at(i) = tmpi + 1.0 * h;
                x.at(j) = tmpj + 2.0 * h;
                term2 += fn(x);
                x = x0;
                x.at(i) = tmpi + 2.0 * h;
                x.at(j) = tmpj + 1.0 * h;
                term2 += fn(x);
                // term3: +44 * (f(tmpi+2h, tmpj-2h) + f(tmpi-2h, tmpj+2h) - f(tmpi-2h,
                // tmpj-2h) - f(tmpi+2h, tmpj+2h))
                x = x0;
                x.at(i) = tmpi + 2.0 * h;
                x.at(j) = tmpj - 2.0 * h;
                term3 = fn(x);
                x = x0;
                x.at(i) = tmpi - 2.0 * h;
                x.at(j) = tmpj + 2.0 * h;
                term3 += fn(x);
                x = x0;
                x.at(i) = tmpi - 2.0 * h;
                x.at(j) = tmpj - 2.0 * h;
                term3 -= fn(x);
                x = x0;
                x.at(i) = tmpi + 2.0 * h;
                x.at(j) = tmpj + 2.0 * h;
                term3 -= fn(x);
                // term4: +74 * (f(tmpi-1h, tmpj-1h) + f(tmpi+1h, tmpj+1h) - f(tmpi+1h,
                // tmpj-1h) - f(tmpi-1h, tmpj+1h))
                x = x0;
                x.at(i) = tmpi - 1.0 * h;
                x.at(j) = tmpj - 1.0 * h;
                term4 = fn(x);
                x = x0;
                x.at(i) = tmpi + 1.0 * h;
                x.at(j) = tmpj + 1.0 * h;
                term4 += fn(x);
                x = x0;
                x.at(i) = tmpi + 1.0 * h;
                x.at(j) = tmpj - 1.0 * h;
                term4 -= fn(x);
                x = x0;
                x.at(i) = tmpi - 1.0 * h;
                x.at(j) = tmpj + 1.0 * h;
                term4 -= fn(x);
                arma::dmat mixed = (
                    (-63.0 * term1 + 63.0 * term2 + 44.0 * term3 + 74.0 * term4) /
                    (600.0 * h * h)
                );
                arma::dmat tmp = (-63.0 * term1 + 63.0 * term2 + 44.0 * term3 + 74.0 * term4);
                // ESALogger::logger()->trace("i: {}, j: {}: {} / {}", i, j, tmp, (600.0 * h * h));
                // ESALogger::logger()->trace("i: {}, j: {}: result {}", i, j, mixed);
                // ESALogger::logger()->trace("term 1: {}\nterm 2: {}\nterm3: {}\nterm4: {}", term1, term2, term3, term4);
                // hessian is symmetric
                hessElements[i][j] = mixed;
                // hessElements[j][i] = mixed;
            }
        }
    }
    // finally, convert to a vector of hessian dlib matrices
    // find size of the dlib::matrix in first element
    if (hessElements.size() == 0) throw std::runtime_error("hessElements has zero size, something went wrong");
    if (hessElements[0].size() == 0) throw std::runtime_error("second dimension of hessElements has zero size, something went wrong");
    std::vector<arma::dmat> hessians(mtxDepth);
    for (unsigned int i = 0; i < mtxDepth; i++){
        // iterating depth dlib matrix in the 2d vector
        arma::dmat hess(n, n);
        // row based or column based for the final dlib matrix
        unsigned int ridx = (usingRows ? i : 0);
        unsigned int cidx = (usingRows ? 0 : i);
        for (unsigned int j = 0; j < n; j++){
            // diagonal elements
            hess.at(j, j) = hessElements[j][j](ridx, cidx);
            for (unsigned int k = j + 1; k < n; k++){
                // off-diagonal elements
                hess.at(j, k) = hessElements[j][k].at(ridx, cidx);
                hess.at(k, j) = hessElements[j][k].at(ridx, cidx);
            }
        }
        hessians[i] = hess;
    }
    return hessians;
}

/// calculate hessian using finite differences
#ifdef WITHDLIB
dlib::matrix<double> finitediff::calculateFiniteHessian(
    const dlib::matrix<double, 0, 1>& x0,
    const std::function<double(const dlib::matrix<double, 0, 1>)>& fn,
    unsigned int accuracy
)
{
    // machine precision
    constexpr double machineEps = std::numeric_limits<double>::epsilon();
    unsigned int n = x0.nr();
    // output matrix - hessian
    dlib::matrix<double> hessian(n, n);
    // local copy of parameter vector
    dlib::matrix<double, 0, 1> x = x0;
    double f0 = fn(x0);
    if (accuracy == 0){
        // basic central difference approximation
        for (unsigned int i = 0; i < n; i++){
            // adaptive stepsize for ith coordinate
            double hi = std::sqrt(machineEps) * std::max(std::abs(x0(i)), 1.0);
            // diagonal - standard second derivative
            x = x0;
            x(i) += hi;
            double fpl = fn(x);
            x = x0;
            x(i) -= hi;
            double fmi = fn(x);
            hessian(i, i) = (fpl - 2.0 * f0 + fmi) / (hi * hi);
            for (unsigned int j = i + 1; j < n; j++){
                double hj = std::sqrt(machineEps) * std::max(std::abs(x0(j)), 1.0);
                // off-diagonals - central differences
                x = x0;
                x(i) += hi;
                x(j) += hj;
                double fpp = fn(x);
                x = x0;
                x(i) += hi;
                x(j) -= hj;
                double fpm = fn(x);
                x = x0;
                x(i) -= hi;
                x(j) += hj;
                double fmp = fn(x);
                x = x0;
                x(i) -= hi;
                x(j) -= hj;
                double fmm = fn(x);
                double d2f = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
                // hessian is symmetrical
                hessian(i, j) = d2f;
                hessian(j, i) = d2f;
            }
        }
    } else {
        // higher order finite difference approximation
        for (unsigned int i = 0; i < n; i++){
            double hi = std::sqrt(machineEps) * std::max(std::abs(x0(i)), 1.0);
            x = x0;
            x(i) += hi;
            double fp = fn(x);
            x = x0;
            x(i) -= hi;
            double fm = fn(x);
            hessian(i, i) = (fp - 2.0 * f0 + fm) / (hi * hi);
            for (unsigned int j = i + 1; j < n; j++){
                double hj = std::sqrt(machineEps) * std::max(std::abs(x0(j)), 1.0);
                double h = (hi + hj) / 2.0;
                double tmpi = x0(i), tmpj = x0(j);
                double term1 = 0.0, term2 = 0.0, term3 = 0.0, term4 = 0.0;
                // term1: -63 * (f(tmpi+1h, tmpj-2h) + f(tmpi+2h, tmpj-1h) + f(tmpi-2h,
                // tmpj+1h) + f(tmpi-1h, tmpj+2h))
                x = x0;
                x(i) = tmpi + 1.0 * h;
                x(j) = tmpj - 2.0 * h;
                term1 += fn(x);
                x = x0;
                x(i) = tmpi + 2.0 * h;
                x(j) = tmpj - 1.0 * h;
                term1 += fn(x);
                x = x0;
                x(i) = tmpi - 2.0 * h;
                x(j) = tmpj + 1.0 * h;
                term1 += fn(x);
                x = x0;
                x(i) = tmpi - 1.0 * h;
                x(j) = tmpj + 2.0 * h;
                term1 += fn(x);
                // term2: +63 * (f(tmpi-1h, tmpj-2h) + f(tmpi-2h, tmpj-1h) + f(tmpi+1h,
                // tmpj+2h) + f(tmpi+2h, tmpj+1h))
                x = x0;
                x(i) = tmpi - 1.0 * h;
                x(j) = tmpj - 2.0 * h;
                term2 += fn(x);
                x = x0;
                x(i) = tmpi - 2.0 * h;
                x(j) = tmpj - 1.0 * h;
                term2 += fn(x);
                x = x0;
                x(i) = tmpi + 1.0 * h;
                x(j) = tmpj + 2.0 * h;
                term2 += fn(x);
                x = x0;
                x(i) = tmpi + 2.0 * h;
                x(j) = tmpj + 1.0 * h;
                term2 += fn(x);
                // term3: +44 * (f(tmpi+2h, tmpj-2h) + f(tmpi-2h, tmpj+2h) - f(tmpi-2h,
                // tmpj-2h) - f(tmpi+2h, tmpj+2h))
                x = x0;
                x(i) = tmpi + 2.0 * h;
                x(j) = tmpj - 2.0 * h;
                term3 += fn(x);
                x = x0;
                x(i) = tmpi - 2.0 * h;
                x(j) = tmpj + 2.0 * h;
                term3 += fn(x);
                x = x0;
                x(i) = tmpi - 2.0 * h;
                x(j) = tmpj - 2.0 * h;
                term3 -= fn(x);
                x = x0;
                x(i) = tmpi + 2.0 * h;
                x(j) = tmpj + 2.0 * h;
                term3 -= fn(x);
                // term4: +74 * (f(tmpi-1h, tmpj-1h) + f(tmpi+1h, tmpj+1h) - f(tmpi+1h,
                // tmpj-1h) - f(tmpi-1h, tmpj+1h))
                x = x0;
                x(i) = tmpi - 1.0 * h;
                x(j) = tmpj - 1.0 * h;
                term4 += fn(x);
                x = x0;
                x(i) = tmpi + 1.0 * h;
                x(j) = tmpj + 1.0 * h;
                term4 += fn(x);
                x = x0;
                x(i) = tmpi + 1.0 * h;
                x(j) = tmpj - 1.0 * h;
                term4 -= fn(x);
                x = x0;
                x(i) = tmpi - 1.0 * h;
                x(j) = tmpj + 1.0 * h;
                term4 -= fn(x);
                // combine weighted terms
                double mixed = (
                    (-63.0 * term1 + 63.0 * term2 + 44.0 * term3 + 74.0 * term4) /
                    (600.0 * h * h)
                );
                // hessian is symmetric
                hessian(i, j) = mixed;
                hessian(j, i) = mixed;
            }
        }
    }
    return hessian;
}
#endif // WITHDLIB

/// calculate hessian using finite differences - armadillo implementation
arma::dmat finitediff::calculateFiniteHessian(
    const arma::dcolvec& x0,
    const std::function<double(const arma::dcolvec&)>& fn,
    unsigned int accuracy
)
{
    // machine precision
    constexpr double machineEps = std::numeric_limits<double>::epsilon();
    unsigned int n = x0.n_rows;
    // output matrix - hessian
    arma::dmat hessian(n, n);
    // local copy of parameter vector
    arma::dcolvec x = x0;
    double f0 = fn(x0);
    if (accuracy == 0){
        // basic central difference approximation
        for (unsigned int i = 0; i < n; i++){
            // adaptive stepsize for ith coordinate
            double hi = std::sqrt(machineEps) * std::max(std::abs(x0.at(i)), 1.0);
            // diagonal - standard second derivative
            x = x0;
            x.at(i) += hi;
            double fpl = fn(x);
            x = x0;
            x.at(i) -= hi;
            double fmi = fn(x);
            hessian(i, i) = (fpl - 2.0 * f0 + fmi) / (hi * hi);
            for (unsigned int j = i + 1; j < n; j++){
                double hj = std::sqrt(machineEps) * std::max(std::abs(x0.at(j)), 1.0);
                // off-diagonals - central differences
                x = x0;
                x.at(i) += hi;
                x.at(j) += hj;
                double fpp = fn(x);
                x = x0;
                x.at(i) += hi;
                x.at(j) -= hj;
                double fpm = fn(x);
                x = x0;
                x.at(i) -= hi;
                x.at(j) += hj;
                double fmp = fn(x);
                x = x0;
                x.at(i) -= hi;
                x.at(j) -= hj;
                double fmm = fn(x);
                double d2f = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
                // hessian is symmetrical
                hessian(i, j) = d2f;
                hessian(j, i) = d2f;
            }
        }
    } else {
        // higher order finite difference approximation
        for (unsigned int i = 0; i < n; i++){
            double hi = std::sqrt(machineEps) * std::max(std::abs(x0.at(i)), 1.0);
            x = x0;
            x.at(i) += hi;
            double fp = fn(x);
            x = x0;
            x.at(i) -= hi;
            double fm = fn(x);
            hessian(i, i) = (fp - 2.0 * f0 + fm) / (hi * hi);
            for (unsigned int j = i + 1; j < n; j++){
                double hj = std::sqrt(machineEps) * std::max(std::abs(x0.at(j)), 1.0);
                double h = (hi + hj) / 2.0;
                double tmpi = x0.at(i), tmpj = x0.at(j);
                double term1 = 0.0, term2 = 0.0, term3 = 0.0, term4 = 0.0;
                // term1: -63 * (f(tmpi+1h, tmpj-2h) + f(tmpi+2h, tmpj-1h) + f(tmpi-2h,
                // tmpj+1h) + f(tmpi-1h, tmpj+2h))
                x = x0;
                x.at(i) = tmpi + 1.0 * h;
                x.at(j) = tmpj - 2.0 * h;
                term1 += fn(x);
                x = x0;
                x.at(i) = tmpi + 2.0 * h;
                x.at(j) = tmpj - 1.0 * h;
                term1 += fn(x);
                x = x0;
                x.at(i) = tmpi - 2.0 * h;
                x.at(j) = tmpj + 1.0 * h;
                term1 += fn(x);
                x = x0;
                x.at(i) = tmpi - 1.0 * h;
                x.at(j) = tmpj + 2.0 * h;
                term1 += fn(x);
                // term2: +63 * (f(tmpi-1h, tmpj-2h) + f(tmpi-2h, tmpj-1h) + f(tmpi+1h,
                // tmpj+2h) + f(tmpi+2h, tmpj+1h))
                x = x0;
                x.at(i) = tmpi - 1.0 * h;
                x.at(j) = tmpj - 2.0 * h;
                term2 += fn(x);
                x = x0;
                x.at(i) = tmpi - 2.0 * h;
                x.at(j) = tmpj - 1.0 * h;
                term2 += fn(x);
                x = x0;
                x.at(i) = tmpi + 1.0 * h;
                x.at(j) = tmpj + 2.0 * h;
                term2 += fn(x);
                x = x0;
                x.at(i) = tmpi + 2.0 * h;
                x.at(j) = tmpj + 1.0 * h;
                term2 += fn(x);
                // term3: +44 * (f(tmpi+2h, tmpj-2h) + f(tmpi-2h, tmpj+2h) - f(tmpi-2h,
                // tmpj-2h) - f(tmpi+2h, tmpj+2h))
                x = x0;
                x.at(i) = tmpi + 2.0 * h;
                x.at(j) = tmpj - 2.0 * h;
                term3 += fn(x);
                x = x0;
                x.at(i) = tmpi - 2.0 * h;
                x.at(j) = tmpj + 2.0 * h;
                term3 += fn(x);
                x = x0;
                x.at(i) = tmpi - 2.0 * h;
                x.at(j) = tmpj - 2.0 * h;
                term3 -= fn(x);
                x = x0;
                x.at(i) = tmpi + 2.0 * h;
                x.at(j) = tmpj + 2.0 * h;
                term3 -= fn(x);
                // term4: +74 * (f(tmpi-1h, tmpj-1h) + f(tmpi+1h, tmpj+1h) - f(tmpi+1h,
                // tmpj-1h) - f(tmpi-1h, tmpj+1h))
                x = x0;
                x.at(i) = tmpi - 1.0 * h;
                x.at(j) = tmpj - 1.0 * h;
                term4 += fn(x);
                x = x0;
                x.at(i) = tmpi + 1.0 * h;
                x.at(j) = tmpj + 1.0 * h;
                term4 += fn(x);
                x = x0;
                x.at(i) = tmpi + 1.0 * h;
                x.at(j) = tmpj - 1.0 * h;
                term4 -= fn(x);
                x = x0;
                x.at(i) = tmpi - 1.0 * h;
                x.at(j) = tmpj + 1.0 * h;
                term4 -= fn(x);
                // combine weighted terms
                double mixed = (
                    (-63.0 * term1 + 63.0 * term2 + 44.0 * term3 + 74.0 * term4) /
                    (600.0 * h * h)
                );
                // hessian is symmetric
                hessian(i, j) = mixed;
                hessian(j, i) = mixed;
            }
        }
    }
    return hessian;
}