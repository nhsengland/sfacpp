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

#ifndef ENUMS_HPP
#define ENUMS_HPP

#include <string>

enum ModelSolver : int {
    CPPOPTLIB_NEWTON = 1,
    CPPOPTLIB_BFGS = 2,
    CPPOPTLIB_CONJUGATED_GD = 3,
    CPPOPTLIB_GD = 4,
    CPPOPTLIB_LBFGS = 5,
    CPPOPTLIB_LBFGSB = 6,
    CPPOPTLIB_NELDER_MEAD = 7,
    OPTIMLIB_BFGS = 8,
    OPTIMLIB_LBFGS = 9,
    OPTIMLIB_CG = 10,
    OPTIMLIB_GD = 11,
    OPTIMLIB_NEWTON = 12,
    ENS_ADABELIEF = 13,
    ENS_ADAM = 14,
    ENS_SGDR = 15,
    ENS_GD = 16,
    ENS_SA = 17,
    DLIB_TR = 18,
    DLIB_NEWTON = 19,
    DLIB_BFGS = 20,
    DLIB_LBFGS = 21,
    DLIB_CG = 22,
    DLIB_HYBRID_BFGS_TR = 23,
    DLIB_HYBRID_BFGS_NEWTON = 24,
    DLIB_HYBRID_LBFGS_TR = 25,
    DLIB_HYBRID_LBFGS_NEWTON = 26,
    DLIB_HYBRID_PSO_TR = 27,
    DLIB_HYBRID_PSO_NEWTON = 28,
    DLIB_HYBRID_PSO_BFGS_TR = 29,
    DLIB_HYBRID_PSO_BFGS_NEWTON = 30,
    EM_GHQ = 31  // EM with Gauss-Hermite quadrature (LC_TRE models)
};

enum ModelSolverLib {
    CPPOPTLIB = 1,
    OPTIMLIB = 2,
    ENS = 3,
    DLIB = 4,
    EM_LIB = 5,
    UNKNOWN = 6
};

enum ModelSolverStatus {
    SUCCESS = 1,
    FAILED = 2
};

enum HessianCalcMethod {
    // numerical approximation of hessian
    NUM_APPROX = 1,
    // analytical hessian
    ANALYTICAL = 2, 
    // BHHH approximation, using analytical gradient
    BHHH_APPROX_ANALYTICAL_GRAD = 3, 
    // BHHH approximation, using numerical approximation of gradient
    BHHH_APPROX_NUM_APPROX_GRAD = 4,
    // numerical approximation of hessian, with numerical approx of gradient
    NUM_APPROX_WITH_NUM_APPROX_GRAD = 5
};

enum ESASfaModelType {
    // Cross-sectional models
    CROSS_HNORM_ZUIT, // Half-normal
    CROSS_TNORM_ZUIT, // Truncated-normal
    // True Fixed Effects - Greene 2005
    TFE_HNORM_ZUIT, // Half-normal
    TFE_TNORM_ZUIT, // Truncated-normal
    TFE_EXP_ZUIT, // Exponential
    TFE_GAMMA_ZUIT, // Gamma
    // True Random Effects - Greene 2005
    TRE_HNORM_ZUIT, // Half-normal
    TRE_TNORM_ZUIT, // Truncated-normal
    TRE_EXP_ZUIT, // Exponential
    TRE_GAMMA_ZUIT, // Gamma
    // Generalized True Random Effects - Various authors
    GTRE_HNORM_ZUIT_ZUI0, // Half-normal
    // Latent class models
    CROSS_LC_HNORM, // cross-sectional, latent class, half-normal
    CROSS_LC_TNORM, // cross-sectional, latent class, trunc-normal
    LC_TRE_HNORM, // latent-class true random effects, half-normal
    LC_TRE_TNORM, // latent-class true random effects, trunc-normal
    LC_MARKOV_TRE_HNORM, // latent markov true random effects, half-normal
    LC_MARKOV_TRE_TNORM, // latent markov true random effects, trunc-normal
    MODEL_UNKNOWN
};

enum ESASfaModelDistribution {
    HNORM,
    TNORM,
    EXP,
    GAMMA,
    DIST_UNKNOWN
};

enum ESASfaModelFamily {
    CROSS,
    TFE,
    TRE,
    GTRE,
    LC_X,
    LC_TRE,
    LM_TRE,
    FAMILY_UNKNOWN
};

enum ESASfaModelTermLocation {
    Y,
    X,
    ZUIT,
    ZVIT,
    ZUI0,
    ZVI0,
    MUUIT,
    TERM_UNKNOWN_LOC
};

enum ESASfaMarginalEffectType {
    MEFF_WANG,
    MEFF_KUMBHAKAR,
    MEFF_NONE
};

/**
 * @brief Topologies for particle swarm optimization
 */
enum ESAPsoTopology {
    GLOBAL,
    RING,
    VONNEUMANN,
    GRAPH,
    GRAPH_HISTORY,
    DYNAMIC
};

namespace ESAEnums {

    ESASfaModelType getModelTypeForNameAndDist(const std::string& name, const std::string& dist);
    ESASfaModelFamily getModelFamily(const ESASfaModelType& mT);
    ESASfaModelDistribution getDistribution(const ESASfaModelType& mT);
    ESASfaMarginalEffectType getMarginalEffectType(const std::string& name);
    bool isHalfNormalModel(const ESASfaModelType& mT);
    bool isTruncNormalModel(const ESASfaModelType& mT);
    bool isExponentialModel(const ESASfaModelType& mT);
    bool isGammaModel(const ESASfaModelType& mT);
    std::string strForModelType(const ESASfaModelType& mT);
    bool isModelTypeAndLocationCompatible(const ESASfaModelType& mT, const ESASfaModelTermLocation& loc);

    HessianCalcMethod getHessianCalcMethod(const std::string& m);
    std::string strForHessianCalcMethod(const HessianCalcMethod m);
    bool isAnalyticalGrad(const HessianCalcMethod& m);

    std::string strForModelSolver(const ModelSolver& s);
    ModelSolverLib libForModelSolver(const ModelSolver& s);

    ModelSolver getDefaultModelSolver();

    ModelSolver getModelSolverForMethodAndLib(const std::string& meth, const std::string& lib);

    int modelFamilyToInt(const ESASfaModelFamily mF);
    ESASfaModelFamily intToModelFamily(const int value);

    ESAPsoTopology psoTopologyForStr(const std::string& s);
}

#endif // ENUMS_HPP