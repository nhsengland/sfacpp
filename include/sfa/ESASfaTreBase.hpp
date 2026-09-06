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

#ifndef ESA_SFA_TRE_BASE_HPP
#define ESA_SFA_TRE_BASE_HPP

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

#include <optional>
#include <vector>
#include "sfa/HaltonSettings.hpp"
#include "sfa/ESASfaBase.hpp"
#include "data/ESADataBase.hpp"
#include "utils/enums.hpp"


class ESASfaTreBase : public ESASfaBase {

public:

    ESASfaTreBase(
        const std::shared_ptr<ESADataBase> dataObjPtr,
        const int s,
        const int nsim = 100,
        const int seed = 1234,
        const HaltonSettings hsetting = HaltonSettings()
    );

    ESASfaTreBase(
        const std::shared_ptr<ESADataBase> dataObjPtr,
        const int s,
        const int nsim = 100,
        const int seed = 1234,
        const std::shared_ptr<arma::dmat> haltonDrawPtr = nullptr,
        const HaltonSettings hsetting = HaltonSettings()
    );

    /**
     * @brief Calculate starting values for TRE, GTRE model
     * @return column vector of starting parameters
     */
    arma::dcolvec startingValues() const override;

    const HaltonSettings* getHaltonSettings() const override {
        return &hsettings;
    }

protected:

    const int seed;
    const int nsim;
    const bool obsUseSameHaltonDraw;
    HaltonSettings hsettings;

    /**
     * @brief Return a (1 x nsim) row vector of weights for a panel (based on density)
     * @details Use softmax function
     * @param out should be a fully allocated maxT x nsim matrix
     */
    template <typename T>
    void weightFromDensForPanel(const arma::Base<double, T>& ldIn, arma::dmat& out) const;

    /**
     * @brief For an individual panel, calculate both gradient and hessian analytically for heteroskedastic model
     * @note applicable for the heteroskedastic model
     * @param idx Current index position of firm identifier
     * @param mT Enumeration of the model type
     * @param par Column vector of the parameters
     * @param y Column vector of the dependent variable
     * @param x Matrix of independent variables
     * @param zmuit Optionally, matrix of determinants (if any) of pre-trunacted mean (trunc-normal model only)
     * @param zuit Matrix of determinants affecting variance of inefficiency component
     * @param zvit Matrix of determinants affecting variance of random noise component
     * @param zvi0 Matrix of determinants affecting variance of time-invariant latent firm heterogeneity
     * @param draws Matrix of halton draws
     * @param zui0 Optionally, matrix of determinants affecting variance of time-invariant inefficiency
     * @param jacOut Pointer for outputting vector of matricies for jacobian over nsim
     * @param hessOut Pointer for outputting vector of vector of matrices for hessian over nT and nsim
     */
    template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZui0, typename TZvi0>
    void internalAnalyticJacHess(
        const unsigned int idx,
        const ESASfaModelType mT,
        const arma::dcolvec& par,
        // accept any armadillo matrix/view
        const arma::Base<double, TY>& yIn,
        const arma::Base<double, TX>& xIn,
        // optionals
        const std::optional<TZmuit>& zmuitIn,
        const arma::Base<double, TZuit>& zuitIn,
        const arma::Base<double, TZvit>& zvitIn,
        const arma::Base<double, TZvi0>& zvi0In,
        const arma::dmat& draws,
        const std::optional<TZui0>& zui0In,
        const arma::Base<double, arma::subview<double>>& QirIn,
        arma::dmat* jacOut = nullptr,
        arma::dmat* hessOut = nullptr
    ) const;

    /**
     * @brief Analytical gradient and Hessian for the truncated-normal TRE model.
     * @details Derived in documentation/truncated_normal_tre.md.
     *          Parameter order: β | b_u | b_v | b_v0 | b_μ
     */
    template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0>
    void internalAnalyticJacHessTN(
        const unsigned int idx,
        const arma::dcolvec& par,
        const arma::Base<double, TY>& yIn,
        const arma::Base<double, TX>& xIn,
        const arma::Base<double, TZmuit>& zmuitIn,
        const arma::Base<double, TZuit>& zuitIn,
        const arma::Base<double, TZvit>& zvitIn,
        const arma::Base<double, TZvi0>& zvi0In,
        const arma::dmat& draws,
        const arma::Base<double, arma::subview<double>>& QirIn,
        arma::dmat* jacOut = nullptr,
        arma::dmat* hessOut = nullptr
    ) const;

};

#endif // ESA_SFA_TRE_BASE_HPP