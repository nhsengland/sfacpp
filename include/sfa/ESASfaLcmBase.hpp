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

#ifndef ESA_SFA_LCM_BASE_HPP
#define ESA_SFA_LCM_BASE_HPP

#include <memory>
#include <vector>
#include <optional>
#include <cmath>
#include <algorithm>
#include <numeric>

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

#include "data/ESADataLCM.hpp"
#include "sfa/ESASfaBase.hpp"
#include "sfa/HaltonSettings.hpp"
#include "utils/enums.hpp"
#include "utils/lcmutils.hpp"

class ESASfaLcmBase : public ESASfaBase {

public:

    /**
     * @brief class constructor
     * @param dataObjPtr Pointer to an ESADataPanelLCM object
     * @param s whether cost or production function
     * @param nsim number of simulations (for TRE-based models)
     * @param seed seed
     * @param hsetting HaltonSettings struct
     */
    ESASfaLcmBase(
        const std::shared_ptr<ESADataPanelLCM> dataObjPtr,
        const int s,
        const int nsim = 100,
        const int seed = 1234,
        const HaltonSettings hsetting = HaltonSettings()
    );

    /**
     * @brief constructor
     * @param dataObjPtr Pointer to an ESADataPanelLCM object
     * @param s whether cost or production function
     * @param nsim number of simulations (for TRE-based models)
     * @param seed seed
     * @param haltonDrawPtr Pointer to an armadillo matrix containing the halton draws
     * @param hsetting HaltonSettings struct
     */
    ESASfaLcmBase(
        const std::shared_ptr<ESADataPanelLCM> dataObjPtr,
        const int s,
        const int nsim = 100,
        const int seed = 1234,
        const std::shared_ptr<arma::dmat> haltonDrawPtr = nullptr,
        const HaltonSettings hsetting = HaltonSettings()
    );

    /**
     * @brief return the halton settings used
     * @return Pointer to HaltonSetting
     */
    const HaltonSettings* getHaltonSettings() const override {
        return &hsettings;
    }

protected:

    const int seed;
    const int nsim;
    const bool obsUseSameHaltonDraw;
    HaltonSettings hsettings;
    std::shared_ptr<arma::dmat> haltonDraws;

    /**
     * @brief derive the analytical first-order derivative, jacobian, and hessian for a panel
     * @tparam TY is template type for yIn
     * @tparam TX is template type for xIn
     * @tparam TZuit is template type for zuitIn
     * @tparam TZvit is template type for zvitIn
     * @tparam TZvi0 is template type for zvi0In
     * @param idx the index of the current panel
     * @param mT the model type
     * @param par column vector of parameters
     * @param yIn matrix or view for dependent variable
     * @param xIn matrix or view for factors of production
     * @param zuitIn matrix or view of variables determining time-varying inefficiency
     * @param zvitIn matrix or view of variables determining stochastic noise
     * @param zvi0In matrix or view of variables determining latent firm heterogeneity
     * @param draws matrix of halton draws
     * @param QirIn view for Qir matrix
     * @param jacOut pointer to write jacobian out to
     * @param hessOut pointer to write hessian out to
     */
    template <typename TY, typename TX, typename TZuit, typename TZvit, typename TZvi0>
    void internalAnalyticJacHess(
        const unsigned int idx,
        const ESASfaModelType mT,
        const arma::dcolvec& par,
        // accept any armadillo matrix or view
        const arma::Base<double, TY>& yIn,
        const arma::Base<double, TX>& xIn,
        const arma::Base<double, TZuit>& zuitIn,
        const arma::Base<double, TZvit>& zvitIn,
        const arma::Base<double, TZvi0>& zvi0In,
        const arma::dmat& draws,
        const arma::Base<double, arma::subview<double>>& QirIn,
        arma::dmat* jacOut = nullptr,
        arma::dmat* hessOut = nullptr
    ) const;

    /**
     * @brief derive the analytical first-order derivative, jacobian, and hessian for a panel
     * @tparam TY is template type for yIn
     * @tparam TX is template type for xIn
     * @tparam TZmuit is template type for zmuitIn
     * @tparam TZuit is template type for zuitIn
     * @tparam TZvit is template type for zvitIn
     * @tparam TZvi0 is template type for zvi0In
     * @param idx the index of the current panel
     * @param mT the model type
     * @param par column vector of parameters
     * @param yIn matrix or view for dependent variable
     * @param xIn matrix or view for factors of production
     * @param zmuitIn matrix or view of variables determining mean of truncated inefficiency component
     * @param zuitIn matrix or view of variables determining time-varying inefficiency
     * @param zvitIn matrix or view of variables determining stochastic noise
     * @param zvi0In matrix or view of variables determining latent firm heterogeneity
     * @param draws matrix of halton draws
     * @param QirIn view for Qir matrix
     * @param jacOut pointer to write jacobian out to
     * @param hessOut pointer to write hessian out to
     */
    template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0>
    void internalAnalyticJacHessTN(
        const unsigned int idx,
        const ESASfaModelType mT,
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

}; // class ESASfaLcmBase

#endif // ESA_SFA_LCM_BASE_HPP
