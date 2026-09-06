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
 * @file ESASfaTfeChen.cpp
 * @author Edmund Haacke
 * @date 2025-03-09
 * @brief Implementation file for ESASfaTfeChen class
 */

#include "sfa/ESASfaTfeChen.hpp"
#include "data/ESADataPanel.hpp"

// contructor method
ESASfaTfeChen::ESASfaTfeChen(std::shared_ptr<ESADataBase> dataObjPtr, const double s) : ESASfaBase(dataObjPtr, s)
{
    // check if dataObjPtr can be legitimately cast to instance of ESADataPanel
    // as this uses the newer generation of the data class (and is incompatible)
    // with previous version.
    if (!dynamic_cast<ESADataPanel*>(dataObjPtr.get())){
        throw std::invalid_argument("data object is not instance of ESADataPanel");
    }
}

// operator() - objective function - override
double ESASfaTfeChen::operator()(const arma::dcolvec& params, const bool exceptNotFinite) const
{
    return 0.0;
}

// // gradient function - override
// arma::dmat ESASfaTfeChen::gradient(
//     const arma::dcolvec& params,
//     const double step,
//     const bool isAnalytical
// ) const
// {

// }

// // hessian matrix - override
// arma::dmat ESASfaTfeChen::hessian(
//     const arma::dcolvec& params,
//     const HessianCalcMethod method,
//     const unsigned int accuracy,
//     const bool threaded
// ) const
// {
    
// }

// starting values for maximum likelihood estimation
arma::dcolvec ESASfaTfeChen::startingValues() const
{
    return arma::dcolvec();
}

// ---- half-normal distribution ----
// log-likelihood function for half-normal distribution
