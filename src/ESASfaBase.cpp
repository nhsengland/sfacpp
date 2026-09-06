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
 * @file ESASfaBase.cpp
 * @brief ESASfaBase class implementation file
 * @date 2024-12-25
 * @author Edmund Haacke
 */

#include <limits>
#include <cmath>
#include "sfa/ESASfaBase.hpp"
#include "utils/log/logs.hpp"
#include "data/ESADataCross.hpp"
#include "data/ESADataPanel.hpp"

/// Constructor
ESASfaBase::ESASfaBase(const std::shared_ptr<ESADataBase> dataObjPtr, const double s) : dataObjPtr(dataObjPtr), s(s)
{
    
};

double ESASfaBase::operator()(const arma::dcolvec& params, const bool exceptNotFinite) const
{
    return std::numeric_limits<double>::quiet_NaN();
}

double ESASfaBase::operator()(const arma::dcolvec& params, const arma::Col<int>& subsetIdents) const
{
    return std::numeric_limits<double>::quiet_NaN();
}

/// N
double ESASfaBase::getN() const
{
    // try casting data ptr to ESADataPanel
    if (dynamic_cast<ESADataPanel *>(dataObjPtr.get())) {
        // data ptr is from ESADataPanel
        // dereference ptr to underlying data obj
        ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
        // return total number of firms
        return dataObj.getNids();
    } else if (dynamic_cast<ESADataCross *>(dataObjPtr.get())){
        // data ptr is from ESADataCross
        // dereference ptr to underlying data obj
        ESADataCross& dataObj = (ESADataCross&)*dataObjPtr;
        // return total number of observations
        return dataObj.getNobs();
    }
    ESALogger::logger()->warn("Failed to find N in ESASfaBase");
    return std::numeric_limits<double>::quiet_NaN();
}

// /// Jacobian of the objective function
// arma::dmat ESASfaBase::jacobian(const arma::dcolvec& params, const double step, const bool isAnalytical) const
// {
//     // not implemented here - implemented within subclasses
//     return arma::mat();
// }

// arma::dmat ESASfaBase::jacobian(const arma::dcolvec& params, const arma::Col<int>& subsetIdents, const double step, const bool isAnalytical) const
// {
//     // not implemented here - implemented within subclasses
//     return arma::mat();
// }

// /// Gradient of the objective function
// arma::dmat ESASfaBase::gradient(const arma::dcolvec& params, const double step, const bool isAnalytical) const
// {
//     // not implemented here - implemented within subclasses
//     return arma::mat();
// }

// arma::dmat ESASfaBase::gradient(
//     const arma::dcolvec& params,
//     const arma::Col<int>& subsetIdents,
//     const double step,
//     const bool isAnalytical
// ) const
// {
//     // not implemented here, implemented within subclasses
//     return arma::mat();
// }

// /// Hessian of the objective function
// arma::dmat ESASfaBase::hessian(
//     const arma::dcolvec& params,
//     const HessianCalcMethod method,
//     const unsigned int accuracy,
//     const bool threaded
// ) const
// {
//     // not implement here - implemented within subclasses
//     return arma::mat();
// }

// arma::dmat ESASfaBase::hessian(
//     const arma::dcolvec& params,
//     const arma::Col<int>& subsetIdents,
//     const HessianCalcMethod method,
//     const unsigned int accuracy,
//     const bool threaded
// ) const
// {
//     // not implemented here, implemented within subclasses
//     return arma::mat();
// }

/// Gradient and hessian together
void ESASfaBase::gradHess(
    const arma::dcolvec& params,
    const bool exceptNotFinite,
    arma::dmat* gradOut,
    arma::dmat* hessOut,
    arma::dmat* jacOut
) const
{
    // not implemented here, implemented within subclasses
    if (gradOut) *gradOut = arma::dmat();
    if (hessOut) *hessOut = arma::dmat();
    if (jacOut) *jacOut = arma::dmat();
}

void ESASfaBase::gradHess(
    const arma::dcolvec& params,
    const arma::Col<int>& subsetIdents,
    const double step,
    const bool analyticalGrad,
    const HessianCalcMethod hessMethod,
    const unsigned int accuracy,
    const bool threaded,
    arma::dmat* gradOut,
    arma::dmat* hessOut,
    arma::dmat* jacOut
) const
{
    // not implemented here, implemented in subclasses
    if (gradOut) *gradOut = arma::dmat();
    if (hessOut) *hessOut = arma::dmat();
    if (jacOut) *jacOut = arma::dmat();
}

/// Calculate reduced form analytical gradient and hessian
void ESASfaBase::gradHessReduced(
    const arma::dcolvec& params,
    const bool threaded,
    arma::dmat* gradOut,
    arma::dmat* jacOut,
    arma::dmat* hessOut
) const
{
    // not implemented here, implemented in subclasses
    if (gradOut) *gradOut = arma::dmat();
    if (jacOut) *jacOut = arma::dmat();
    if (hessOut) *hessOut = arma::dmat();
}

void ESASfaBase::gradHessReduced(
    const arma::dcolvec& params,
    const arma::Col<int>& subsetIdents,
    const bool threaded,
    arma::dmat* gradOut,
    arma::dmat* jacOut,
    arma::dmat* hessOut
) const
{
    // not implemented here, implemented in subclasses
    if (gradOut) *gradOut = arma::dmat();
    if (jacOut) *jacOut = arma::dmat();
    if (hessOut) *hessOut = arma::dmat();
}

/// starting values for maximum likelihood estimation
arma::dcolvec ESASfaBase::startingValues() const
{
    // dereference pointer to data object
    ESADataBase& dataObj = (ESADataBase&)*dataObjPtr;
    // default method for generating starting values is to undertake a standard OLS regression
    // for the inputs - e.g., solve y=bX for b; and set remaining coefficients to zero.
    arma::dcolvec b_x;
    try {
        b_x = arma::solve(dataObj.getX(), dataObj.getY());
    } catch (const std::exception& e) {
        ESALogger::logger()->error("error inverting matrix to calculate starting values {}", e.what());
        // set b_x to zero
        b_x = arma::zeros(dataObj.getX().n_cols);
    }
    return b_x;
}

/// sigma values
ESASigmaParams ESASfaBase::getSigmaParams(const arma::dcolvec& par) const
{
    // return empty; implement in subclasses;
    return ESASigmaParams();
}

template <typename TY, typename TZu, typename TZv>
MoMResult ESASfaBase::getMoMComponents(
    const arma::Base<double, TY>& yIn,
    const std::optional<TZu>& zuOpt,
    const std::optional<TZv>& zvOpt,
    const std::optional<double>& m2override
) const
{
    // unwrap armadillo references
    const auto& y = yIn.get_ref();
    // constants
    const double n = y.n_elem;
    const double pi = arma::datum::pi;
    // calculate moments
    double m3 = arma::accu(arma::pow(y, 3)) / n;
    double m2 = arma::accu(arma::pow(y, 2)) / n;
    if (m2override.has_value()) m2 = m2override.value();
    // adjust skewness for m3 if necessary
    if (m3 * this->s >= 0) m3 = -0.0001 * this->s;
    // su2init: [ m3 / (sqrt(2.0 / pi) * (1.0 - 4.0/pi)) ]^(2/3)
    double termDenom = std::sqrt(2.0 / pi) * (1.0 - 4.0 / pi);
    double su2init = std::pow(m3 / termDenom, 2.0);
    su2init = std::pow(su2init, (1.0 / 3.0));
    if (!std::isfinite(su2init) || su2init < 0.0) su2init = 0.1;
    double sv2init = m2 - (1.0 - 2.0/pi) * su2init;
    if (!std::isfinite(sv2init) || sv2init < 0.0) sv2init = 0.1;
    // for hetroskedasticity
    arma::dcolvec b_zu_est, b_zv_est;
    if (zuOpt.has_value()) {
        arma::dcolvec y1 = 0.5 * arma::log(arma::pow((arma::pow(y, 2) - sv2init) / (1.0 - 2.0/pi), 2.0));
        b_zu_est = arma::pinv(zuOpt.value()) * y1;
    }
    if (zvOpt.has_value()) {
        arma::dcolvec y2 = 0.5 * arma::log(arma::pow(arma::pow(y, 2) - (1.0 - 2.0/pi) * su2init, 2));
        b_zv_est = arma::pinv(zvOpt.value()) * y2;
    }
    return {su2init, sv2init, b_zu_est, b_zv_est};
}

// ===========================================================================
//                      EXPLICIT TEMPLATE INSTANTISATION
// ===========================================================================

template MoMResult ESASfaBase::getMoMComponents<arma::dmat, arma::dmat, arma::dmat>(
    const arma::Base<double, arma::dmat>&,
    const std::optional<arma::dmat>&,
    const std::optional<arma::dmat>&,
    const std::optional<double>&
) const;
template MoMResult ESASfaBase::getMoMComponents<arma::subview<double>, arma::subview<double>, arma::subview<double>>(
    const arma::Base<double, arma::subview<double>>&,
    const std::optional<arma::subview<double>>&,
    const std::optional<arma::subview<double>>&,
    const std::optional<double>&
) const;
template MoMResult ESASfaBase::getMoMComponents<arma::dmat, arma::subview<double>, arma::subview<double>>(
    const arma::Base<double, arma::dmat>&,
    const std::optional<arma::subview<double>>&,
    const std::optional<arma::subview<double>>&,
    const std::optional<double>&
) const;