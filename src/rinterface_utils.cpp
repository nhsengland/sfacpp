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
 * 
 */

#include "rinterface/rinterface_utils.hpp"
#include "optim/ESAGlobalOptimParams.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"

#ifdef RPACKAGE

// extract an element from a Rcpp List by key, and convert to a C++ type. Wrap in std optional.
template <typename T>
std::optional<T> rinterface::elementFromRcppList(Rcpp::List& l, const std::string& k){
    if (l.containsElementNamed(k.c_str())){
        try {
            return std::make_optional<T>(Rcpp::as<T>(l[k]));
        } catch (std::exception& e){
            throw std::invalid_argument("Error parsing '" + k + "' " + std::string(e.what()));
        }
    }
    return std::nullopt;
}
// explicit template instantization
template std::optional<std::string> rinterface::elementFromRcppList<std::string>(Rcpp::List&, const std::string&);
template std::optional<unsigned int> rinterface::elementFromRcppList<unsigned int>(Rcpp::List&, const std::string&);
template std::optional<int> rinterface::elementFromRcppList<int>(Rcpp::List&, const std::string&);
template std::optional<double> rinterface::elementFromRcppList<double>(Rcpp::List&, const std::string&);

// setup the optimization parameters
void rinterface::setupOptimParams(
    ESAOptimParams& optimParams,
    Rcpp::Nullable<Rcpp::List> optimOpts,
    const int seed
)
{
    optimParams.seed = seed;
    // check if optimization options present
    if (optimOpts.isNotNull()) {
        // user provided optimization options
        Rcpp::List optimOptsList = Rcpp::as<Rcpp::List>(optimOpts);
        // check for each of the elements 
        std::vector<std::string> optimNames = Rcpp::as<std::vector<std::string>>(optimOptsList.names());
        std::vector<std::string> optimNamesLegit = {
            "maxit",
            "grad_err_tol",
            "grad_err_tol_check",
            "rel_solution_change_err_tol",
            "rel_solution_change_err_tol_violations",
            "rel_objfn_change_err_tol",
            "rel_objfn_change_err_tol_violations",
            "condition_hessian",
            "constraint_threshold",
            "tr_radius",
            "wolfe_cons_1",
            "wolfe_cons_2",
            "step_size"
        };
        std::vector<std::string> optimInvalid;
        std::copy_if(optimNames.begin(), optimNames.end(), std::back_inserter(optimInvalid), [&optimNamesLegit](const std::string& s){
            return std::find(optimNamesLegit.begin(), optimNamesLegit.end(), s) == optimNamesLegit.end();
        });
        // iterate through any elements in optimInvalid, and notify the user that they are invalid / won't be processed
        if (optimInvalid.size() > 0){
            std::string invalidElems = std::accumulate(optimInvalid.begin(), optimInvalid.end(), std::string(), [](const std::string& a, const std::string& b){
                return a + (a.length() > 0 ? "," : "") + b; 
            });
            ESALogger::logger()->warn("The following elements in 'optimOpts' are invalid and will be ignored: {}", invalidElems);
        }
        // max number of iterations
        std::optional<unsigned int> optMaxIt = elementFromRcppList<unsigned int>(optimOptsList, "maxit");
        if (optMaxIt) optimParams.maxit = optMaxIt.value();
        // gradient error tolerance
        std::optional<double> optGradErrTol = elementFromRcppList<double>(optimOptsList, "grad_err_tol");
        if (optGradErrTol) optimParams.grad_err_tol = optGradErrTol.value();
        // check after alleged convergence
        std::optional<double> optGradErrCheckTol = elementFromRcppList<double>(optimOptsList, "grad_err_tol_check");
        if (optGradErrCheckTol) optimParams.grad_err_tol_check = optGradErrCheckTol.value();
        // relative solution change tolerance - error tolerance controlling how small the proportional
        // change in the solution vector before convergence
        std::optional<double> optRelSolChngErrTol = elementFromRcppList<double>(optimOptsList, "rel_solution_change_err_tol");
        if (optRelSolChngErrTol) optimParams.rel_solution_change_err_tol = optRelSolChngErrTol.value();
        // cppoptlib specific - how many violations to tolerate
        std::optional<double> optRelSolChngErrTolNViol = elementFromRcppList<double>(optimOptsList, "rel_solution_change_err_tol_violations");
        if (optRelSolChngErrTolNViol) optimParams.rel_solution_change_err_tol_violations = optRelSolChngErrTolNViol.value();
        // relative objective function change tolerance
        std::optional<double> relObjFnChngErrTol = elementFromRcppList<double>(optimOptsList, "rel_objfn_change_err_tol");
        if (relObjFnChngErrTol) optimParams.rel_objfn_change_err_tol = relObjFnChngErrTol.value();
        // cppoptlib specific - how many violations of the above to tolerate
        std::optional<double> relObjFnChngErrTolNViol = elementFromRcppList<double>(optimOptsList, "rel_objfn_change_err_tol_violations");
        if (relObjFnChngErrTolNViol) optimParams.rel_objfn_change_err_tol_violations = relObjFnChngErrTolNViol.value();
        //
        // other settings for ccpoptlib
        std::optional<double> optCondHess = elementFromRcppList<double>(optimOptsList, "condition_hessian");
        if (optCondHess) optimParams.condition_hessian = optCondHess.value();
        std::optional<double> optConsThres = elementFromRcppList<double>(optimOptsList, "constraint_threshold");
        if (optConsThres) optimParams.constraint_threshold = optConsThres.value();
        //
        // trust region for dlib
        std::optional<double> optTrRadius = elementFromRcppList<double>(optimOptsList, "tr_radius");
        if (optTrRadius) optimParams.tr_radius = optTrRadius.value();
        //
        // line search tuning parameters
        std::optional<double> optWolfe1 = elementFromRcppList<double>(optimOptsList, "wolfe_cons_1");
        if (optWolfe1) optimParams.wolfe_cons_1 = optWolfe1.value();
        std::optional<double> optWolfe2 = elementFromRcppList<double>(optimOptsList, "wolfe_cons_2");
        if (optWolfe2) optimParams.wolfe_cons_2 = optWolfe2.value();
        //
        // step size / learning rate [mainly used by ensmalled for first order methods]
        std::optional<double> optStepSize = elementFromRcppList<double>(optimOptsList, "step_size");
        if (optStepSize) optimParams.step_size = optStepSize.value();
    }
}

void rinterface::processDataMatricies(
    const arma::dcolvec& y,
    const arma::dmat& x,
    arma::dmat& zmuit,
    arma::dmat& zuit,
    arma::dmat& zvit,
    arma::dmat& zui0,
    arma::dmat& zvi0,
    arma::Col<int>& idVec,
    arma::Col<int>& timeVec,
    const ESASfaModelType& mT,
    const arma::colvec& y_,
    const arma::mat& x_,
    Rcpp::Nullable<arma::dmat> zmuit_,
    Rcpp::Nullable<arma::dmat> zuit_,
    Rcpp::Nullable<arma::dmat> zvit_,
    Rcpp::Nullable<arma::dmat> zui0_,
    Rcpp::Nullable<arma::dmat> zvi0_,
    Rcpp::Nullable<arma::dmat> startVals_,
    Rcpp::Nullable<arma::colvec> idVec_,
    Rcpp::Nullable<arma::colvec> timeVec_
)
{
    // model family, distribution
    ESASfaModelDistribution mD = ESAEnums::getDistribution(mT);
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    // tre, gtre, tfe should all have id, time vectors
    if ((mF == ESASfaModelFamily::TFE) || (mF == ESASfaModelFamily::TRE) || (mF == ESASfaModelFamily::GTRE)) {
        if (idVec_.isNotNull()) {
            idVec = Rcpp::as<arma::Col<int>>(idVec_);
        } else {
            throw std::invalid_argument("'idVec' must be provided for TFE, TRE, GTRE models");
        }
        if (timeVec_.isNotNull()) {
            timeVec = Rcpp::as<arma::Col<int>>(timeVec_);
        } else {
            throw std::invalid_argument("'timeVec' must be provided for TFE, TRE, GTRE mdoels");
        }
    }
    // zuit, zvit should be present for all classes
    if (mF == ESASfaModelFamily::TFE || mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE || mF == ESASfaModelFamily::CROSS) {
        if (zuit_.isNotNull()) {
            zuit = Rcpp::as<arma::dmat>(zuit_);
        } else {
            throw std::invalid_argument("'zuit' must be provided for TFE, TRE, GTRE, and cross-sectional models.");
        }
        if (zvit_.isNotNull()) {
            zvit = Rcpp::as<arma::dmat>(zvit_);
        } else {
            throw std::invalid_argument("'zvit' must be provided for TFE, TRE, GTRE, and cross-sectional models.");
        }
    }
    // TRE, GTRE require zvi0
    if (mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE) {
        if (zvi0_.isNotNull()) {
            zvi0 = Rcpp::as<arma::dmat>(zvi0_);
        } else {
            throw std::invalid_argument("'zvi0' must be provided for TRE, GTRE models.");
        }
    }
    // GTRE should have something for zui0
    if (mF == ESASfaModelFamily::GTRE) {
        if (zui0_.isNotNull()) {
            zui0 = Rcpp::as<arma::dmat>(zui0_);
        } else {
            throw std::invalid_argument("'zui0' must be provided for GTRE model.");
        }
    }
    // truncated normal distribution - require zmuit
    if (mD == ESASfaModelDistribution::TNORM) {
        if (zmuit_.isNotNull()) {
            zmuit = Rcpp::as<arma::dmat>(zmuit_);
        } else {
            throw std::invalid_argument("'zmuit' must be provided for truncated normal models");
        }
    }
}

#endif // RPACKAGE