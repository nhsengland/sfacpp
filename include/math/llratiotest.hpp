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
 * @file llratiotest.hpp
 * @author edmund haacke
 * @date 2025-12-25
 * @details
 * - function to run a log-likelihood ratio test
 */

#ifndef LL_RATIO_TEST_HPP
#define LL_RATIO_TEST_HPP

#include <cmath>
#include <string>
#include <boost/math/distributions/chi_squared.hpp>
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"

namespace postestimation {

    struct LogLikeRatioTest {
        // deviance (D)
        double stat;
        // significance
        double pval;
        // degrees of freedom difference
        int df;
    };

    /**
     * Perform a log-likelihood ratio test
     * 
     * @param ll0 the log-likelihood of the first model
     * @param ll1 the log-likelihood score of the alternative model
     * @param nparams0 number of parameters in the first model
     * @param nparams1 number of parameters in the second model
     * @return LogLikeRatioTest
     */
    inline LogLikeRatioTest logLikelihoodRatioTest(double ll0, double ll1, int nparam0, int nparam1)
    {
        // first, firgure out what is null - the restricted model, vs alternative (less restricted model);
        double llNull, llAlt;
        int nparamNull, nparamAlt;
        int diff = nparam1 - nparam0;
        if (diff > 0) {
            // if diff > 0; then nparam1 is the alternate, and nparam0 is the restricted
            llNull = ll0;
            llAlt = ll1;
            nparamNull = nparam0;
            nparamAlt = nparam1;
        } else if (diff < 0) {
            // if diff > 0, then nparam0 is the alternative, and nparam1 is the restricted
            llNull = ll1;
            llAlt = ll0;
            nparamNull = nparam1;
            nparamAlt = nparam0;
        } else {
            // if diff is 0, then raise an error
            throw std::runtime_error("no difference in number of parameters: nparam0: " + std::to_string(nparam0) + ", nparam1: " + std::to_string(nparam1));
        }
        // calculate degrees of freedom
        int df = nparamAlt - nparamNull;
        // test statistic : D = -2 ln(L0/L1) = 2*(L1 - L0)
        double statistic = 2.0 * (llAlt - llNull);
        // should be llAlt>=llNull; could be some numerical noise
        if (statistic < 0) statistic = 0.0;
        // pval using chisquared dist
        boost::math::chi_squared dist(df);
        double pval = boost::math::cdf(boost::math::complement(dist, statistic));
        return {statistic, pval, df};
    }

}

#endif // LL_RATIO_TEST_HPP