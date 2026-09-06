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

#include <stdexcept>
#include "utils/enums.hpp"


ESASfaModelType ESAEnums::getModelTypeForNameAndDist(const std::string& name, const std::string& dist)
{
    // first check the names
    if (name == "tfe"){
        // check the distribution
        if (dist == "hnorm"){
            return ESASfaModelType::TFE_HNORM_ZUIT;
        } else if (dist == "tnorm"){
            return ESASfaModelType::TFE_TNORM_ZUIT;
        } else if (dist == "exp"){
            return ESASfaModelType::TFE_EXP_ZUIT;
        } else if (dist == "gamma"){
            return ESASfaModelType::TFE_GAMMA_ZUIT;
        }
        throw std::invalid_argument("'" + dist + "' is not a recognized distribution for a True Fixed Effects model. Please use 'hnorm', 'tnorm', 'exp', or 'gamma'.");
    } else if (name == "tre"){
        if (dist == "hnorm"){
            return ESASfaModelType::TRE_HNORM_ZUIT;
        } else if (dist == "tnorm"){
            return ESASfaModelType::TRE_TNORM_ZUIT;
        } else if (dist == "exp"){
            return ESASfaModelType::TRE_EXP_ZUIT;
        } else if (dist == "gamma"){
            return ESASfaModelType::TRE_GAMMA_ZUIT;
        }
        throw std::invalid_argument("'" + dist + "' is not a recognized distribution for a True Random Effects model. Please use 'hnorm', 'tnorm', 'exp', or 'gamma'.");
    } else if (name == "gtre"){
        if (dist == "hnorm"){
            return ESASfaModelType::GTRE_HNORM_ZUIT_ZUI0;
        }
        throw std::invalid_argument("'" + dist + "' is not a recognized distribution for a Generalized True Random Effects model. Please use 'hnorm'.");
    } else if (name == "cross"){
        if (dist == "hnorm"){
            return ESASfaModelType::CROSS_HNORM_ZUIT;
        } else if (dist == "tnorm"){
            return ESASfaModelType::CROSS_TNORM_ZUIT;
        }
    } else if (name == "lccross" || name == "lc_cross") {
        if (dist == "hnorm") return ESASfaModelType::CROSS_LC_HNORM;
        else if (dist == "tnorm") return ESASfaModelType::CROSS_LC_TNORM;
    } else if (name == "lctre" || name == "lc_tre") {
        if (dist == "hnorm") return ESASfaModelType::LC_TRE_HNORM;
        else if (dist == "tnorm") return ESASfaModelType::LC_TRE_TNORM;
    } else if (name == "lmarkov") {
        if (dist == "hnorm") return ESASfaModelType::LC_MARKOV_TRE_HNORM;
        else if (dist == "tnorm") return ESASfaModelType::LC_MARKOV_TRE_TNORM;
    }
    // raise an exception 
    throw std::invalid_argument("'" + name + "' is not a recognized model type. Please use 'tfe', 'tre', or 'gtre'.");
}


ESASfaModelFamily ESAEnums::getModelFamily(const ESASfaModelType& mT){
    if (mT == ESASfaModelType::TFE_HNORM_ZUIT || mT == ESASfaModelType::TFE_TNORM_ZUIT || mT == ESASfaModelType::TFE_EXP_ZUIT || mT == ESASfaModelType::TFE_GAMMA_ZUIT){
        return ESASfaModelFamily::TFE;
    } else if (mT == ESASfaModelType::TRE_HNORM_ZUIT || mT == ESASfaModelType::TRE_TNORM_ZUIT || mT == ESASfaModelType::TRE_EXP_ZUIT || mT == ESASfaModelType::TRE_GAMMA_ZUIT){
        return ESASfaModelFamily::TRE;
    } else if (mT == ESASfaModelType::GTRE_HNORM_ZUIT_ZUI0){
        return ESASfaModelFamily::GTRE;
    } else if (mT == ESASfaModelType::CROSS_HNORM_ZUIT || mT == ESASfaModelType::CROSS_TNORM_ZUIT){
        return ESASfaModelFamily::CROSS;
    } else if (mT == ESASfaModelType::CROSS_LC_HNORM || mT == ESASfaModelType::CROSS_LC_TNORM) {
        return ESASfaModelFamily::LC_X;
    } else if (mT == ESASfaModelType::LC_TRE_HNORM || mT == ESASfaModelType::LC_TRE_TNORM) {
        return ESASfaModelFamily::LC_TRE;
    } else if (mT == ESASfaModelType::LC_MARKOV_TRE_HNORM || mT == ESASfaModelType::LC_MARKOV_TRE_TNORM) {
        return ESASfaModelFamily::LM_TRE;
    }
    return ESASfaModelFamily::FAMILY_UNKNOWN;
}

ESASfaModelDistribution ESAEnums::getDistribution(const ESASfaModelType& mT){
    if (
        (mT == ESASfaModelType::TFE_HNORM_ZUIT) || 
        (mT == ESASfaModelType::TRE_HNORM_ZUIT) || 
        (mT == ESASfaModelType::GTRE_HNORM_ZUIT_ZUI0) ||
        (mT == ESASfaModelType::CROSS_HNORM_ZUIT) ||
        (mT == ESASfaModelType::CROSS_LC_HNORM) ||
        (mT == ESASfaModelType::LC_TRE_HNORM) ||
        (mT == ESASfaModelType::LC_MARKOV_TRE_HNORM)
    ){
        return ESASfaModelDistribution::HNORM;
    } else if (
        (mT == ESASfaModelType::TFE_TNORM_ZUIT) ||
        (mT == ESASfaModelType::TRE_TNORM_ZUIT) ||
        (mT == ESASfaModelType::CROSS_TNORM_ZUIT) ||
        (mT == ESASfaModelType::CROSS_LC_TNORM) ||
        (mT == ESASfaModelType::LC_TRE_TNORM) ||
        (mT == ESASfaModelType::LC_MARKOV_TRE_TNORM)
    ){
        return ESASfaModelDistribution::TNORM;
    } else if (mT == ESASfaModelType::TFE_EXP_ZUIT || mT == ESASfaModelType::TRE_EXP_ZUIT){
        return ESASfaModelDistribution::EXP;
    } else if (mT == ESASfaModelType::TFE_GAMMA_ZUIT || mT == ESASfaModelType::TRE_GAMMA_ZUIT){
        return ESASfaModelDistribution::GAMMA;
    }
    return ESASfaModelDistribution::DIST_UNKNOWN;
}

ESASfaMarginalEffectType ESAEnums::getMarginalEffectType(const std::string& name){
    if (name == "wang"){
        return ESASfaMarginalEffectType::MEFF_WANG;
    } else if (name == "kumbhakar"){
        return ESASfaMarginalEffectType::MEFF_KUMBHAKAR;
    }
    return ESASfaMarginalEffectType::MEFF_NONE;
}

bool ESAEnums::isHalfNormalModel(const ESASfaModelType& mT){
    return ESAEnums::getDistribution(mT) == ESASfaModelDistribution::HNORM;
}

bool ESAEnums::isTruncNormalModel(const ESASfaModelType& mT){
    return ESAEnums::getDistribution(mT) == ESASfaModelDistribution::TNORM;
}

bool ESAEnums::isExponentialModel(const ESASfaModelType& mT){
    return ESAEnums::getDistribution(mT) == ESASfaModelDistribution::EXP;
}

bool ESAEnums::isGammaModel(const ESASfaModelType& mT){
    return ESAEnums::getDistribution(mT) == ESASfaModelDistribution::GAMMA;
}

std::string ESAEnums::strForModelType(const ESASfaModelType& mT){
    switch (mT){
        case ESASfaModelType::TFE_HNORM_ZUIT:
            return "True Fixed Effects with Half-Normal inefficiency";
        case ESASfaModelType::TFE_TNORM_ZUIT:
            return "True Fixed Effects with Truncated-Normal inefficiency";
        case ESASfaModelType::TFE_EXP_ZUIT:
            return "True Fixed Effects with Exponential distributed inefficiency";
        case ESASfaModelType::TFE_GAMMA_ZUIT:
            return "True Fixed Effects with Gamma distributed inefficiency";
        case ESASfaModelType::TRE_HNORM_ZUIT:
            return "True Random Effects with Half-Normal inefficiency";
        case ESASfaModelType::TRE_TNORM_ZUIT:
            return "True Random Effects with Truncated-Normal inefficiency";
        case ESASfaModelType::TRE_EXP_ZUIT:
            return "True Random Effects with Exponential distributed inefficiency";
        case ESASfaModelType::TRE_GAMMA_ZUIT:
            return "True Random Effects with Gamma distributed inefficiency";
        case ESASfaModelType::GTRE_HNORM_ZUIT_ZUI0:
            return "Generalized True Random Effects with Half-Normal persistent and transient inefficiency";
        case ESASfaModelType::CROSS_HNORM_ZUIT:
            return "Cross-sectional with Half-Normal inefficiency";
        case ESASfaModelType::CROSS_TNORM_ZUIT:
            return "Cross-sectional with Truncated-Normal inefficiency";
        case ESASfaModelType::CROSS_LC_HNORM:
            return "Latent class cross-sectional with Half-Normal inefficiency";
        case ESASfaModelType::CROSS_LC_TNORM:
            return "Latent class cross-sectional with Truncated-Normal inefficiency";
        case ESASfaModelType::LC_TRE_HNORM:
            return "Latent class True Random Effects with Half-Normal inefficiency";
        case ESASfaModelType::LC_TRE_TNORM:
            return "Latent class True Random Effects with Truncated-Normal inefficiency";
        case ESASfaModelType::LC_MARKOV_TRE_HNORM:
            return "Latent Markov True Random Effects with Half-Normal inefficiency";
        case ESASfaModelType::LC_MARKOV_TRE_TNORM:
            return "Latent Markov True Random Effects with Truncated-Normal inefficiency";
        default:
            return "MODEL_UNKNOWN";
    }
    return "MODEL_UNKNOWN";
}

bool ESAEnums::isModelTypeAndLocationCompatible(const ESASfaModelType& mT, const ESASfaModelTermLocation& loc){
    // common to all models
    bool common = (
        loc == ESASfaModelTermLocation::X ||
        loc == ESASfaModelTermLocation::ZUIT ||
        loc == ESASfaModelTermLocation::ZVIT ||
        loc == ESASfaModelTermLocation::ZVI0
    );
    switch (mT){
        case ESASfaModelType::TFE_HNORM_ZUIT:
            return common;
        case ESASfaModelType::TFE_TNORM_ZUIT:
            return (common || loc == ESASfaModelTermLocation::MUUIT);
        case ESASfaModelType::TRE_HNORM_ZUIT:
            return common;
        case ESASfaModelType::TRE_TNORM_ZUIT:
            return (common || loc == ESASfaModelTermLocation::MUUIT);
        case ESASfaModelType::GTRE_HNORM_ZUIT_ZUI0:
            return (common || loc == ESASfaModelTermLocation::ZUI0 || loc == ESASfaModelTermLocation::MUUIT);
        default:
            return common;
    }
}

HessianCalcMethod ESAEnums::getHessianCalcMethod(const std::string &m)
{
    if (m == "bhhh_num_approx"){
        return HessianCalcMethod::BHHH_APPROX_NUM_APPROX_GRAD;
    } else if (m == "bhhh_analytical"){
        return HessianCalcMethod::BHHH_APPROX_ANALYTICAL_GRAD;
    } else if (m == "num_approx"){
        return HessianCalcMethod::NUM_APPROX;
    } else if (m == "analytical"){
        return HessianCalcMethod::ANALYTICAL;
    } else if (m == "num_approx_hess_grad"){
        return HessianCalcMethod::NUM_APPROX_WITH_NUM_APPROX_GRAD;
    }
    return HessianCalcMethod::NUM_APPROX;
}

std::string ESAEnums::strForHessianCalcMethod(const HessianCalcMethod m)
{
    switch (m){
        case HessianCalcMethod::ANALYTICAL:
            return "Hessian is analytical";
        case HessianCalcMethod::NUM_APPROX:
            return "Hessian is numerically approximated";
        case HessianCalcMethod::BHHH_APPROX_ANALYTICAL_GRAD:
            return "Hessian is BHHH (analytical gradient)";
        case HessianCalcMethod::BHHH_APPROX_NUM_APPROX_GRAD:
            return "Hessian is BHHH (num approx gradient)";
        case HessianCalcMethod::NUM_APPROX_WITH_NUM_APPROX_GRAD:
            return "Hessian is numerically approximated (num approx gradient)";
        default:
            return "Unknown Hessian";
    }
}

bool ESAEnums::isAnalyticalGrad(const HessianCalcMethod& m)
{
    switch (m) {
        case HessianCalcMethod::ANALYTICAL:
            return true;
        case HessianCalcMethod::NUM_APPROX:
            return true;
        case HessianCalcMethod::BHHH_APPROX_ANALYTICAL_GRAD:
            return true;
        case HessianCalcMethod::BHHH_APPROX_NUM_APPROX_GRAD:
            return false;
        case HessianCalcMethod::NUM_APPROX_WITH_NUM_APPROX_GRAD:
            return false;
        default:
            return false;
    }
}

std::string ESAEnums::strForModelSolver(const ModelSolver& s)
{
    switch (s){
        case ModelSolver::CPPOPTLIB_NEWTON:
            return "Newton (cppoptlib)";
        case ModelSolver::CPPOPTLIB_BFGS:
            return "BFGS (cppoptlib)";
        case ModelSolver::CPPOPTLIB_CONJUGATED_GD:
            return "Conjugated gradient descent (cppoptlib)";
        case ModelSolver::CPPOPTLIB_GD:
            return "Gradient descent (cppoptlib)";
        case ModelSolver::CPPOPTLIB_LBFGS:
            return "Limited memory BFGS (cppoptlib)";
        case ModelSolver::CPPOPTLIB_LBFGSB:
            return "Limited memory BFGS with box constraints (cppoptlib)";
        case ModelSolver::CPPOPTLIB_NELDER_MEAD:
            return "Nelder Mead (cppoptlib)";
        case ModelSolver::OPTIMLIB_BFGS:
            return "BFGS (optimlib)";
        case ModelSolver::OPTIMLIB_LBFGS:
            return "Limited memory BFGS (optimlib)";
        case ModelSolver::OPTIMLIB_CG:
            return "Conjugated gradient (optimlib)";
        case ModelSolver::OPTIMLIB_GD:
            return "Gradient descent (optimlib)";
        case ModelSolver::OPTIMLIB_NEWTON:
            return "Newton (optimlib)";
        case ModelSolver::ENS_ADABELIEF:
            return "AdaBelief (ensmallen)";
        case ModelSolver::DLIB_TR:
            return "Trusted region (dlib)";
        case ModelSolver::DLIB_NEWTON:
            return "Newton (dlib)";
        case ModelSolver::DLIB_BFGS:
            return "BFGS (dlib)";
        case ModelSolver::DLIB_LBFGS:
            return "Limited memory BFGS (dlib)";
        case ModelSolver::DLIB_CG:
            return "Conjugate gradient (dlib)";
        case ModelSolver::DLIB_HYBRID_BFGS_TR:
            return "BFGS → Trusted region (dlib)";
        case ModelSolver::DLIB_HYBRID_BFGS_NEWTON:
            return "BFGS → Newton (dlib)";
        case ModelSolver::DLIB_HYBRID_LBFGS_TR:
            return "Limited memory BFGS → Trusted region (dlib)";
        case ModelSolver::DLIB_HYBRID_LBFGS_NEWTON:
            return "Limited memory BFGS → Newton (dlib)";
        case ModelSolver::DLIB_HYBRID_PSO_TR:
            return "Particle swarm (custom) → Trusted region (dlib)";
        case ModelSolver::DLIB_HYBRID_PSO_NEWTON:
            return "Particle swarm (custom) → Newton (dlib)";
        case ModelSolver::DLIB_HYBRID_PSO_BFGS_TR:
            return "Particle swarm (custom) → BFGS → Trusted region (dlib)";
        case ModelSolver::DLIB_HYBRID_PSO_BFGS_NEWTON:
            return "Particle swarm (custom) → BFGS → Newton (dlib)";
        case ModelSolver::EM_GHQ:
            return "EM with Gauss-Hermite quadrature (LC-TRE)";
        default:
            return "Unknown model solver";
    }
}

ModelSolverLib ESAEnums::libForModelSolver(const ModelSolver& s)
{
    if (s == ModelSolver::EM_GHQ) return ModelSolverLib::EM_LIB;
    int i = static_cast<int>(s);
    if (i >= 1 && i <= 7) return ModelSolverLib::CPPOPTLIB;
    else if (i >= 8 && i <= 12) return ModelSolverLib::OPTIMLIB;
    else if (i >= 13 && i <= 17) return ModelSolverLib::ENS;
    else if (i >= 18) return ModelSolverLib::DLIB;
    return ModelSolverLib::UNKNOWN;
}

// the default model solver, depends on what packages are included
ModelSolver ESAEnums::getDefaultModelSolver()
{
    #if defined(WITHDLIB)
    return ModelSolver::DLIB_TR;
    #elif defined(WITHOPTIMLIB)
    return ModelSolver::OPTIMLIB_BFGS;
    #elif defined(WITHCPPNUMSOLVERS)
    return ModelSolver::CPPOPTLIB_NEWTON;
    #else
    return ModelSolver::ENS_GD;
    #endif
}

ModelSolver ESAEnums::getModelSolverForMethodAndLib(const std::string& meth, const std::string& lib)
{
    // EM_GHQ is library-independent — matched before library dispatch
    if (meth == "em_ghq") return ModelSolver::EM_GHQ;
    // handle library first
    #ifdef WITHCPPNUMSOLVERS
    if (lib == "cppoptlib" || lib == "cppnumericalsolvers"){
        // CPPOPTLIB
        if (meth == "newton") return ModelSolver::CPPOPTLIB_NEWTON;
        else if (meth == "bfgs") return ModelSolver::CPPOPTLIB_BFGS;
        else if (meth == "cgd") return ModelSolver::CPPOPTLIB_CONJUGATED_GD;
        else if (meth == "gd") return ModelSolver::CPPOPTLIB_GD;
        else if (meth == "lbfgs") return ModelSolver::CPPOPTLIB_LBFGS;
        else if (meth == "lbfgsb") return ModelSolver::CPPOPTLIB_LBFGSB;
        else if (meth == "nm") return ModelSolver::CPPOPTLIB_NELDER_MEAD;
        else throw std::invalid_argument("'" + meth + "' is not a valid method for '" + lib + "'");
    } 
    #endif // WITHCPPNUMSOLVERS
    #ifdef WITHOPTIMLIB
    if (lib == "optimlib" || lib == "optim") {
        // OPTIMLIB
        if (meth == "bfgs") return ModelSolver::OPTIMLIB_BFGS;
        else if (meth == "lbfgs") return ModelSolver::OPTIMLIB_LBFGS;
        else if (meth == "cg") return ModelSolver::OPTIMLIB_CG;
        else if (meth == "gd") return ModelSolver::OPTIMLIB_GD;
        else if (meth == "newton") return ModelSolver::OPTIMLIB_NEWTON;
        else throw std::invalid_argument("'" + meth + "' is not a valid method for '" + lib + "'");
    }
    #endif // WITHOPTIMLIB
    #ifdef WITHENSMALLEN
    if (lib == "ens" || lib == "ensmallen") {
        // ensmallen
        if (meth == "adabelief") return ModelSolver::ENS_ADABELIEF;
        else if (meth == "adam") return ModelSolver::ENS_ADAM;
        else if (meth == "sgdr") return ModelSolver::ENS_SGDR;
        else if (meth == "gd") return ModelSolver::ENS_GD;
        else if (meth == "sa") return ModelSolver::ENS_SA;
        else throw std::invalid_argument("'" + meth + "' is not a valid method for '" + lib + "'");
    }
    #endif // WITHENSMALLEN
    #ifdef WITHDLIB
    if (lib == "dlib") {
        // dlib
        if (meth == "tr") return ModelSolver::DLIB_TR;
        else if (meth == "bfgs") return ModelSolver::DLIB_BFGS;
        else if (meth == "lbfgs") return ModelSolver::DLIB_LBFGS;
        else if (meth == "cg") return ModelSolver::DLIB_CG;
        else if (meth == "newton") return ModelSolver::DLIB_NEWTON;
        else if (meth == "hybrid_bfgs_tr") return ModelSolver::DLIB_HYBRID_BFGS_TR;
        else if (meth == "hybrid_lbfgs_tr") return ModelSolver::DLIB_HYBRID_LBFGS_TR;
        else if (meth == "hybrid_bfgs_newton") return ModelSolver::DLIB_HYBRID_BFGS_NEWTON;
        else if (meth == "hybrid_lbfgs_newton") return ModelSolver::DLIB_HYBRID_LBFGS_NEWTON;
        else if (meth == "bfgs_tr") return ModelSolver::DLIB_HYBRID_BFGS_TR;
        else if (meth == "lbfgs_tr") return ModelSolver::DLIB_HYBRID_LBFGS_TR;
        else if (meth == "hybrid_pso_tr" || meth == "pso_tr") return ModelSolver::DLIB_HYBRID_PSO_TR;
        else if (meth == "hybrid_pso_newton" || meth == "pso_newton") return ModelSolver::DLIB_HYBRID_PSO_NEWTON;
        else if (meth == "hybrid_pso_bfgs_tr" || meth == "pso_bfgs_tr") return ModelSolver::DLIB_HYBRID_PSO_BFGS_TR;
        else if (meth == "hybrid_pso_bfgs_newton" || meth == "pso_bfgs_newton") return ModelSolver::DLIB_HYBRID_PSO_BFGS_NEWTON;
        else throw std::invalid_argument("'" + meth + "' is not a valid method for '" + lib + "'");
    }
    #endif // WITHDLIB
    throw std::invalid_argument("'" + lib + "' is not a recognised argument. Must be one of 'optimlib', 'cppoptlib', or 'ensmallen'");
}

int ESAEnums::modelFamilyToInt(const ESASfaModelFamily mF) {
    return static_cast<int>(mF);
}

ESASfaModelFamily ESAEnums::intToModelFamily(const int value) {
    // check bounds
    if (value < 0 || value > ESASfaModelFamily::FAMILY_UNKNOWN) {
        return ESASfaModelFamily::FAMILY_UNKNOWN;
    }
    return static_cast<ESASfaModelFamily>(value);
}

ESAPsoTopology ESAEnums::psoTopologyForStr(const std::string& s)
{
    if (s == "global") return ESAPsoTopology::GLOBAL;
    else if (s == "ring") return ESAPsoTopology::RING;
    else if (s == "vn" || s == "vonneumann" || s == "von_neumann") return ESAPsoTopology::VONNEUMANN;
    else if (s == "graph") return ESAPsoTopology::GRAPH;
    else if (s == "graph_history") return ESAPsoTopology::GRAPH_HISTORY;
    else if (s == "dynamic") return ESAPsoTopology::DYNAMIC;
    throw std::invalid_argument("'" + s + "' is not a recognized topology for particle swarm optimization");
}