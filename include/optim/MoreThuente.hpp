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
 * @file MoreThuente.hpp
 * @date 2025-12-29
 * @details More-Thuente style linesearch
 */

#ifndef MORE_THUENTE_HPP
#define MORE_THUENTE_HPP

#include <iostream>

#ifdef WITHDLIB
#include <dlib/matrix.h>
#endif // WITHDLIB

namespace linesearch {

    /**
     * @brief fit cubic polynomial to two points and find minimum
     */
    inline double cubicMinimizer(
        double a_lo, double f_lo, double g_lo,
        double a_hi, double f_hi, double g_hi
    )
    {
        double d1 = g_lo + g_hi - 3.0 * (f_lo - f_hi) / (a_lo - a_hi);
        double det = d1 * d1 - g_lo * g_hi;
        double d2 = std::sqrt(std::max(0.0, det));
        if (a_hi < a_lo) d2 = -d2;
        double num = g_hi + d2 - d1;
        double den = g_hi - g_lo + 2.0 * d2;
        double frac = num / den;
        return a_hi - (a_hi - a_lo) * frac;
    }

#ifdef WITHDLIB

    /**
     * @brief slope of function along search direction
     */
    inline double derDeriv(const dlib::matrix<double, 0, 1>& g, const dlib::matrix<double, 0, 1>& d)
    {
        return dlib::dot(g, d);
    }

    /**
     * @brief zoom function
     */
    template <typename fn, typename fn_der>
    inline double zoom(
        fn& f,
        fn_der& der,
        const dlib::matrix<double, 0, 1>& xCurr,
        const dlib::matrix<double, 0, 1>& direction,
        double fx,
        double gxDir,
        double a_lo,
        double f_lo,
        double g_lo,
        double a_hi,
        double f_hi,
        double g_hi,
        double c1,
        double c2,
        dlib::matrix<double, 0, 1>& gOut
    ) {
        int maxIt = 20;
        for (int i = 0; i < maxIt; i++) {
            // cubic minimizer
            double aj = cubicMinimizer(a_lo, f_lo, g_lo, a_hi, f_hi, g_hi);
            double interval = std::abs(a_hi - a_lo);
            double dist_lo = std::abs(aj - a_lo);
            double dist_hi = std::abs(aj - a_hi);
            // 
            if (std::isnan(aj) || dist_lo < 0.1 * interval || dist_hi < 0.1 * interval) {
                aj = (a_lo + a_hi) / 2.0;
            }
            // evaluate fn at trial step
            dlib::matrix<double, 0, 1> xNext = xCurr + aj * direction;
            double fj = f(xNext);
            // gradient at trial step
            gOut = der(xNext);
            // slop of fn along search direction
            double gj = derDeriv(gOut, direction);
            // armijo (sufficient decrease) condition
            double armijoTarget = fx + c1 * aj * gxDir;
            if (fj > armijoTarget || fj >= f_lo) {
                // step too big - minimum between a_lo and aj
                a_hi = aj;
                f_hi = fj;
                g_hi = gj;
            } else {
                // check curvature
                if (std::abs(gj) <= -c2 * gxDir) {
                    // successful - return
                    return aj;
                }
                // update bracket
                // +ve slope - passed minimum (a_hi = aj)
                // -ve slope - havent reached minimum (a_lo = aj)
                if (gj * (a_hi - a_lo) >= 0) {
                    a_hi = a_lo;
                    f_hi = f_lo;
                    g_hi = g_lo;
                }
                a_lo = aj;
                f_lo = fj;
                g_lo = gj;
            }
        }
        // fallback
        return a_lo;
    }

    template <typename fn, typename fn_der>
    inline double StrongWolfeLineSearch(
        const fn& f,
        const fn_der& der,
        const dlib::matrix<double, 0, 1>& x,
        const dlib::matrix<double, 0, 1>& d,
        double fval,
        const dlib::matrix<double, 0, 1>& g,
        double& alphaOut, // output
        dlib::matrix<double, 0, 1>& gOut,
        double c1 = 1e-4,
        double c2 = 0.9,
        int maxIt = 20
    )
    {
        double alphaPrev = 0.0;
        // newton step first
        double alpha1 = 1.0;
        double fprev = fval;
        double f0 = fval;
        double f1 = f0;
        // slope of fn in search direction
        double g0 = derDeriv(g, d);
        double gprev = g0;
        // safety check - decent direction
        if (g0 > 0) {
            // pointing uphill
            alphaOut = 0.0;
            return fval;
        }
        // iterations
        for (int i = 0; i < maxIt; i++) {
            // param vector at step
            dlib::matrix<double, 0, 1> x1 = x + alpha1 * d;
            // function evaluated at this
            f1 = f(x1);
            // protection block - if function evaluated to Nan/Inf likelihood score
            bool isInvalid = !std::isfinite(f1);
            // loop until find a finite function value
            while (isInvalid) {
                // backtrack
                alpha1 *= 0.5;
                // if step becomes neglible, give up
                if (alpha1 < 1e-16) {
                    alphaOut = 0.0;
                    return fval;
                }
                x1 = x + alpha1 * d;
                f1 = f(x1);
                isInvalid = !std::isfinite(f1);
            }
            // f1 is valid - compute gradient at this point
            gOut = der(x1);
            dlib::matrix<double, 0, 1> gvec1 = der(x1);
            // also protect against inf/nan in the gradient [cliffs]
            while (!dlib::is_finite(gOut)) {
                alpha1 *= 0.5;
                // protection if step size goes tiny
                if (alpha1 < 1e-16) {
                    alphaOut = 0.0;
                    return fval;
                }
                x1 = x + alpha1 * d;
                // score of new param vector
                f1 = f(x1);
                // reevaluate gradient at new param vector
                gOut = der(x1);
            }
            double g1 = derDeriv(gOut, d);
            // check armijo violation/fn increase
            if (f1 > f0 + c1 * alpha1 * g0 || (i > 0 && f1 >= fprev)) {
                // minimum between alphaPrev and alpha1
                // zoom 
                alphaOut = zoom(
                    f, der, x, d, f0, g0,
                    alphaPrev, fprev, gprev,
                    alpha1, f1, g1,
                    c1, c2, gOut
                );
                return f(x + alphaOut * d);
            }
            // check strong wolfe curvature
            if (std::abs(g1) <= -c2 * g0) {
                alphaOut = alpha1;
                return f1;
            }
            // check if slope is +ve - if so, apply zoom
            if (g1 >= 0) {
                alphaOut = zoom(
                    f, der, x, d, f0, g0,
                    alpha1, f1, g1,
                    alphaPrev, f0, g0,
                    c1, c2, gOut
                );
                return f(x + alphaOut * d);
            }
            // step expansion - double the step
            alphaPrev = alpha1;
            fprev = f1;
            gprev = g1;
            alpha1 *= 2.0;
            if (alpha1 >= 50.0) alpha1 = 50.0;
        }
        alphaOut = alpha1;
        return f1;
    }

#endif // WITHDLIB

}

#endif // MORE_THUENTE