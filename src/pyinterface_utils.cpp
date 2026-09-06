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
 * @file pyinterface_utils.cpp
 * @author edmund haacke
 * @date 2025-12-08
 * @details implemention file for pyinterface_utils.hpp
 */

#include "pyinterface/pyinterface_utils.hpp"

#ifdef PYPACKAGE

template <typename T>
std::optional<T> pyinterface::elementFromPyDict(py::dict& l, const std::string& k)
{
    if (l.contains(k) && !l[k.c_str()].is_none()) {
        try {
            return std::make_optional<T>(py::cast<T>(l[k.c_str()]));
        } catch (std::exception& e){
            throw std::invalid_argument("Error parsing '" + k + "' " + std::string(e.what()));
        }
    }
    return std::nullopt;
}
// explicit template instantization
template std::optional<std::string> pyinterface::elementFromPyDict<std::string>(py::dict&, const std::string&);
template std::optional<unsigned int> pyinterface::elementFromPyDict<unsigned int>(py::dict&, const std::string&);
template std::optional<int> pyinterface::elementFromPyDict<int>(py::dict&, const std::string&);
template std::optional<double> pyinterface::elementFromPyDict<double>(py::dict&, const std::string&);

void pyinterface::setupOptimParams(
    ESAOptimParams& optimParams,
    std::optional<py::dict> optimOpts,
    const int seed
)
{
    optimParams.seed = seed;
    // check if optimization options are present
    if (optimOpts) {
        // user provided optimization options
        // max number of iterations
        std::optional<unsigned int> optMaxIt = elementFromPyDict<unsigned int>(optimOpts.value(), "maxit");
        if (optMaxIt) optimParams.maxit = optMaxIt.value();
        // gradient error tolerance
        std::optional<double> optGradErrTol = elementFromPyDict<double>(optimOpts.value(), "grad_err_tol");
        if (optGradErrTol) optimParams.grad_err_tol = optGradErrTol.value();
        // relative solution change tolerance - error tolerance controlling how small the proportional
        // change in the solution vector before convergence
        std::optional<double> optRelSolChngErrTol = elementFromPyDict<double>(optimOpts.value(), "rel_solution_change_err_tol");
        if (optRelSolChngErrTol) optimParams.rel_solution_change_err_tol = optRelSolChngErrTol.value();
        // cppoptlib specific - how many violations to tolerate
        std::optional<double> optRelSolChngErrTolNViol = elementFromPyDict<double>(optimOpts.value(), "rel_solution_change_err_tol_violations");
        if (optRelSolChngErrTolNViol) optimParams.rel_solution_change_err_tol_violations = optRelSolChngErrTolNViol.value();
        // relative objective function change tolerance
        std::optional<double> relObjFnChngErrTol = elementFromPyDict<double>(optimOpts.value(), "rel_objfn_change_err_tol");
        if (relObjFnChngErrTol) optimParams.rel_objfn_change_err_tol = relObjFnChngErrTol.value();
        // cppoptlib specific - how many violations of the above to tolerate
        std::optional<double> relObjFnChngErrTolNViol = elementFromPyDict<double>(optimOpts.value(), "rel_objfn_change_err_tol_violations");
        if (relObjFnChngErrTolNViol) optimParams.rel_objfn_change_err_tol_violations = relObjFnChngErrTolNViol.value();
        //
        // other settings for ccpoptlib
        std::optional<double> optCondHess = elementFromPyDict<double>(optimOpts.value(), "condition_hessian");
        if (optCondHess) optimParams.condition_hessian = optCondHess.value();
        std::optional<double> optConsThres = elementFromPyDict<double>(optimOpts.value(), "constraint_threshold");
        if (optConsThres) optimParams.constraint_threshold = optConsThres.value();
        //
        // trust region for dlib
        std::optional<double> optTrRadius = elementFromPyDict<double>(optimOpts.value(), "tr_radius");
        if (optTrRadius) optimParams.tr_radius = optTrRadius.value();
        //
        // line search tuning parameters
        std::optional<double> optWolfe1 = elementFromPyDict<double>(optimOpts.value(), "wolfe_cons_1");
        if (optWolfe1) optimParams.wolfe_cons_1 = optWolfe1.value();
        std::optional<double> optWolfe2 = elementFromPyDict<double>(optimOpts.value(), "wolfe_cons_2");
        if (optWolfe2) optimParams.wolfe_cons_2 = optWolfe2.value();
        //
        // step size / learning rate [mainly used by ensmalled for first order methods]
        std::optional<double> optStepSize = elementFromPyDict<double>(optimOpts.value(), "step_size");
        if (optStepSize) optimParams.step_size = optStepSize.value();
        // ---- EM parameters ----
        std::optional<unsigned int> emMaxIter = elementFromPyDict<unsigned int>(optimOpts.value(), "em_max_iter");
        if (emMaxIter) optimParams.em_max_iter = emMaxIter.value();
        std::optional<double> emTol = elementFromPyDict<double>(optimOpts.value(), "em_tol");
        if (emTol) optimParams.em_tol = emTol.value();
        std::optional<unsigned int> emNQuad = elementFromPyDict<unsigned int>(optimOpts.value(), "em_nquad_pts");
        if (emNQuad) optimParams.em_nquad_pts = emNQuad.value();
        std::optional<unsigned int> emSegMaxIter = elementFromPyDict<unsigned int>(optimOpts.value(), "em_seg_max_iter");
        if (emSegMaxIter) optimParams.em_seg_max_iter = emSegMaxIter.value();
        std::optional<unsigned int> emClassMaxIter = elementFromPyDict<unsigned int>(optimOpts.value(), "em_class_max_iter");
        if (emClassMaxIter) optimParams.em_class_max_iter = emClassMaxIter.value();
    }
}


void pyinterface::processDataMatricies(
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
    std::optional<arma::mat> zmuit_,
    std::optional<arma::mat> zuit_,
    std::optional<arma::mat> zvit_,
    std::optional<arma::mat> zui0_,
    std::optional<arma::mat> zvi0_,
    std::optional<arma::mat> startVals_,
    std::optional<arma::Col<int>> idVec_,
    std::optional<arma::Col<int>> timeVec_
)
{
    // model family, distribution
    ESASfaModelDistribution mD = ESAEnums::getDistribution(mT);
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    // tre, gtre, tfe should all have id, time vectors
    if ((mF == ESASfaModelFamily::TFE) || (mF == ESASfaModelFamily::TRE) || (mF == ESASfaModelFamily::GTRE)) {
        if (idVec_) {
            // idVec = arma::conv_to<arma::Col<int>>::from(idVec_.value());
            idVec = idVec_.value();
        } else {
            throw std::invalid_argument("'idVec' must be provided for TFE, TRE, GTRE models");
        }
        if (timeVec_) {
            // timeVec = arma::conv_to<arma::Col<int>>::from(timeVec_.value());
            timeVec = timeVec_.value();
        } else {
            throw std::invalid_argument("'timeVec' must be provided for TFE, TRE, GTRE mdoels");
        }
    }
    // zuit, zvit should be present for all classes
    if (mF == ESASfaModelFamily::TFE || mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE || mF == ESASfaModelFamily::CROSS) {
        if (zuit_) {
            zuit = zuit_.value();
        } else {
            throw std::invalid_argument("'zuit' must be provided for TFE, TRE, GTRE, and cross-sectional models.");
        }
        if (zvit_) {
            zvit = zvit_.value();
        } else {
            throw std::invalid_argument("'zvit' must be provided for TFE, TRE, GTRE, and cross-sectional models.");
        }
    }
    // TRE, GTRE require zvi0
    if (mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE) {
        if (zvi0_) {
            zvi0 = zvi0_.value();
        } else {
            throw std::invalid_argument("'zvi0' must be provided for TRE, GTRE models.");
        }
    }
    // GTRE should have something for zui0
    if (mF == ESASfaModelFamily::GTRE) {
        if (zui0_) {
            zui0 = zui0_.value();
        } else {
            throw std::invalid_argument("'zui0' must be provided for GTRE model.");
        }
    }
    // truncated normal distribution - require zmuit
    if (mD == ESASfaModelDistribution::TNORM) {
        if (zmuit_) {
            zmuit = zmuit_.value();
        } else {
            throw std::invalid_argument("'zmuit' must be provided for truncated normal models");
        }
    }
}

#endif // PYPACKAGE