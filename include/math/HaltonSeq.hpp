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

#ifndef HALTON_SEQ_HPP
#define HALTON_SEQ_HPP

#include <iostream>
// ---- armadillo ----
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif
// ---- end armadillo ----
#include <cmath>
#include <vector>
#include <limits>
#include <functional>
#include <stdexcept>
#include <algorithm> // for std::shuffle
#include <random>    // for std::mt19937
#include <boost/math/distributions/normal.hpp>

// Helper to check if a number is prime
inline bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i = i + 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    return true;
}

class HaltonSeq {
private:
    arma::ivec d; 
    arma::vec r;
    arma::ivec perm;
    
    int base;
    int skip;
    int len;             
    int current_step;   
    bool is_scrambled; // whether or not to use scrambling

    boost::math::normal_distribution<double> normalDist;
    
    /**
     * @brief inverse CDF
     */
    std::function<double(double)> invcdf = [this](double x){
        if (
            (x < std::numeric_limits<double>::epsilon()) ||
            (x >= (1.0 - std::numeric_limits<double>::epsilon()))
        ){
            x = std::numeric_limits<double>::epsilon();
        }
        return boost::math::quantile(normalDist, x);
    };

    int increment_digits() {
        int j = 0; 
        while (d(j) == base - 1) {
            d(j) = 0;
            j++;
            if(j >= d.n_elem) throw std::runtime_error("Halton sequence depth exceeded.");
        }
        d(j) += 1;
        return j;
    }

    /**
     * @brief Calculates the value for standard Halton.
     * Uses the optimized recursive 'r' vector update method.
     */
    double get_standard_halton(int changed_index) {
        // update the remainder vector 'r' based on the changed digit
        if (changed_index >= 1) {
            r(changed_index - 1) = (double)(d(changed_index) + r(changed_index)) / base;
        }
        if (changed_index >= 2) {
            for (int k = changed_index - 2; k >= 0; k--) {
                r(k) = r(k + 1) / base;
            }
        }
        return (d(0) + r(0)) / base;
    }

    /**
     * @brief Calculates the value for scrambled Halton.
     * Applies the permutation sigma to digits: sum( sigma(d_k) * base^(-k-1) )
     */
    double get_scrambled_halton() {
        double val = 0.0;
        double f = 1.0 / base;
        for (size_t k = 0; k < d.n_elem; ++k) {
            // apply permutation to the current digit d(k)
            int scrambled_digit = perm(d(k));
            val += scrambled_digit * f;
            f /= base;
        }
        return val;
    }

public:
    /**
     * Constructor
     * @param scramble_seed: Seed for the digit permutation. 
     * IMPORTANT: In MSLE, change this seed per dimension/base to ensure independence.
     */
    HaltonSeq(
        int base_in,
        int length_in,
        int skip_in = 5000,
        bool scramble_in = false,
        int scramble_seed = 0,
        std::function<double(double)> invcdf_in = [](double x){ return x; }
    ) : base(base_in),
        skip(skip_in),
        len(length_in),
        current_step(0), 
        is_scrambled(scramble_in),
        invcdf(invcdf_in)
    {
        if (!isPrime(base)) throw std::invalid_argument("Base number must be prime");
        if (skip < 0) throw std::domain_error("Skip must be non-negative");
        normalDist = boost::math::normal_distribution<double>(0.0, 1.0);
        long long S = skip + length_in;
        int D = (int)std::ceil(std::log(S) / std::log(base));
        d = arma::zeros<arma::ivec>(D + 1);
        r = arma::zeros<arma::vec>(D + 1);
        // scrambling setup
        if (is_scrambled) {
            // create a vector [0, 1, ..., base-1]
            std::vector<int> p(base);
            std::iota(p.begin(), p.end(), 0);
            // shuffle it deterministically
            std::mt19937 g(scramble_seed); 
            std::shuffle(p.begin(), p.end(), g);
            // store in armadillo vec
            perm = arma::conv_to<arma::ivec>::from(p);
        }
        // halton burn-in (still iterate thru steps, but ignore values)
        for (int i = 0; i < skip; ++i) {
            next_raw(); 
        }
        // reset for length
        current_step = 0; 
    }

    int length() const { return len; }

    /**
     * Internal helper to advance state and get raw [0,1] value
     */
    double next_raw() {
        int changed_idx = increment_digits();
        if (is_scrambled) {
            return get_scrambled_halton();
        } else {
            return get_standard_halton(changed_idx);
        }
    }

    /** 
     * @brief Get the next deterministic value with Inverse CDF applied
    */
    double next() {
        if (current_step >= len) {
            throw std::out_of_range("HaltonSeq iterator exhausted");
        }
        current_step++;
        double draw = next_raw();
        return invcdf(draw);
        // double u = next_raw();
        // u = std::min(1.0 - eps, std::max(eps, u));
        // return invcdf(u);
    }

    /**
     * @brief Fill function
     * @param x
     * @param shuffle_output whether or not to shuffle the output vec
     * @param epsilon eps
     */
    void fill(arma::vec& x, bool shuffle_output = false, const double epsilon = 1e-12) {//const double epsilon = std::numeric_limits<double>::epsilon()) {
        for(size_t i = 0; i < x.n_elem; ++i) {
            if (current_step < len) {
                x(i) = next();
            } else {
                break; 
            }
        }
        // shuffle if was desired
        if (shuffle_output) {
            x = arma::shuffle(x);
        }
        x.clamp(epsilon, 1.0 - epsilon);
    }

    /**
     * Static helper
     */
    static arma::vec generate(int base, int n, int skip = 5000, bool scrambled = false, int seed = 1234, bool shuffle_output = false) {
        // Note: We use 'seed + base' as the default scramble seed to ensuring different bases 
        // get different permutations automatically if the user passes the same seed to all dimensions.
        HaltonSeq hs(base, n, skip, scrambled, seed + base);
        arma::vec result(n);
        hs.fill(result, shuffle_output);
        return result;
    }
};

#endif //HALTON_SEQ_HPP