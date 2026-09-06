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

#ifndef ESA_SFA_LCM_CROSS_HPP
#define ESA_SFA_LCM_CROSS_HPP

#include <memory>
#include <vector>
#include <optional>

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

#include "sfa/ESASfaBase.hpp"
#include "utils/lcmutils.hpp"
#include "sfa/ESASigmaParams.hpp"
#include "data/ESADataLCM.hpp"
#include "utils/enums.hpp"


class ESASfaLcmCross : public ESASfaBase {

public:

    ESASfaLcmCross(
        std::shared_ptr<ESADataPanelLCM> lcmDataPtr,
        const int s
    );

    double operator()(const arma::dcolvec& params, const bool exceptNotFinite = false) const override;
    double operator()(const arma::dcolvec& params, const arma::Col<int>& subsetIdents) const override;

    void gradHess(
        const arma::dcolvec& params,
        const bool exceptNotFinite = false,
        arma::dmat* gradOut = nullptr,
        arma::dmat* hessOut = nullptr,
        arma::dmat* jacOut = nullptr
    ) const override;

    void gradHess(
        const arma::dcolvec& params,
        const arma::Col<int>& subsetIdents,
        const double step = 1e-8,
        const bool analyticalGrad = true,
        const HessianCalcMethod hessMethod = HessianCalcMethod::ANALYTICAL,
        const unsigned int accuracy = 0,
        const bool threaded = false,
        arma::dmat* gradOut = nullptr,
        arma::dmat* hessOut = nullptr,
        arma::dmat* jacOut = nullptr
    ) const override;

    arma::dcolvec startingValues() const override;
    ESASigmaParams getSigmaParams(const arma::dcolvec& par) const override;
    double getN() const override;

    static double densityHalfNormal(double eps, double sigma2u, double sigma2v, int s);

private:

    std::shared_ptr<ESADataPanelLCM> lcmDataPtr;

    /**
     * @brief Gradient of log-density for half-normal, per observation
     * @param y Scalar dependent variable
     * @param x Row vector of covariates (1 x nX)
     * @param zuit Row vector of inefficiency determinants (1 x nZuit)
     * @param zvit Row vector of noise determinants (1 x nZvit)
     * @param par_c Class-specific parameter vector [beta | b_zuit | b_zvit]
     * @param s Production (+1) or cost (-1)
     * @return Row vector gradient (1 x nParams_c)
     */
    arma::rowvec gradLogDensityHalfNormal(
        double y_val,
        const arma::rowvec& x_row,
        const arma::rowvec& zuit_row,
        const arma::rowvec& zvit_row,
        const arma::dcolvec& b_x,
        const arma::dcolvec& b_zuit,
        const arma::dcolvec& b_zvit,
        int s
    ) const;
};

#endif // ESA_SFA_LCM_CROSS_HPP
