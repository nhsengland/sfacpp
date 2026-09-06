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

// File: ESASfaMeff.hpp
// Author: Edmund Haacke
// Date: 2024-12-28
// Description:
//    Parent class for calculating marginal effects of the inefficiency on the determinants.

#ifndef ESA_SFA_MEFF_HPP
#define ESA_SFA_MEFF_HPP

#include <vector>
#include <string>
#include <memory>
// --- armadillo ---
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif //RPACKAGE
// --- end armadillo ---

#include "utils/enums.hpp"
#include "sfa/ESASfaBase.hpp"
#include "data/ESADataBase.hpp"
#include "data/ESADataPanel.hpp"
#include "model/ESASfaModelTerms.hpp"
#include "optim/ESAOptimResult.hpp"

typedef struct ESASfaMeffReturn {
    arma::dmat marginalEffects;
    std::vector<std::string> columnNames;
} ESASfaMeffReturn;

typedef struct ESASfaMeffCIReturn {
    arma::dmat lowerCI, upperCI;
    std::vector<std::string> columnNames;
} ESASfaMeffCIReturn;

class ESASfaMeff {

public:
    /// Constructor
    ESASfaMeff(const std::shared_ptr<ESADataBase> dataObjPtr, const int s, const int nsim, const int seed);

    // Destructor
    virtual ~ESASfaMeff() = default;

    /// @brief Calculate the marginal effects of the inefficiency on the determinant
    virtual ESASfaMeffReturn marginalEffects(const arma::dcolvec& par, const ESASfaModelTerms& modelTerms) const;

    /// @brief Calculate the confidence intervals for the marginal effects using bootstrap method
    /// @param par The estimated parameter vector
    /// @param modelTerms The model terms
    /// @param optimParams The optimization parameters for ML estimation (BHHH)
    /// @param repetitions The number of bootstrap repetitions
    /// @return The marginal effects and their confidence intervals
    std::unique_ptr<ESASfaMeffCIReturn> bootstrappedCIs(
        const std::shared_ptr<ESASfaBase> mdlPtr,
        const arma::dcolvec& par,
        const ESASfaModelTerms& modelTerms,
        double confidenceLevel,
        const int bsReps,
        const bool useCustomStartVals = true
    );

protected:

    const std::shared_ptr<ESADataBase> dataObjPtr;
    const int _s;
    const int _nsim;
    const int _seed;

    /**
     * @brief Generate a pseudo sample for bootstrapping
     * @param par Parameter vector
     * @param seed Seed for reproducability
     * @param prodCost Whether production (1) or cost (-1) function
     * @return Instance of ESADataPanel
     */
    ESADataPanel generatePseudoSample(
        const arma::dcolvec& par,
        const int seed,
        const int prodCost
    ) const;

    /**
     * @brief private method to fit a model for the bootstrap sample, return success OR nullptr
     * @param pseudoSample A shared pointer to ESASfaData object containing boostrapped sample
     * @param optStartVals Optionally, a column vector of starting values to use to fit a single bootstrap model
     * @param modelFamily The model family to fit
     * @param prodCost Whether production or cost function
     * @param numSims Number of simulations (for MSL for TRE and GTRE)
     * @param seed Seed for reproducability
     * @return Pointer to optim success object, or nullptr
     */
    std::unique_ptr<ESAOptimResultSuccess> fitBsModel(
        const std::shared_ptr<ESASfaBase>& mdlPtr,
        const std::shared_ptr<ESADataPanel> pseudoSample,
        const std::optional<arma::dcolvec>&optStartVals,
        const ESASfaModelFamily modelFamily,
        const int prodCost,
        const int numSims,
        const int seed
    );

    /**
     * @brief Run boostrap simulations
     * @param par parameter (column) vector
     * @param bootstrapReps number of bootstrap replications
     * @param seed seed for reproducability
     * @return vector of marginal effect returns
     */
    std::vector<ESASfaMeffReturn> runBootstrapSims(
        const arma::dcolvec& par,
        const unsigned int bootstrapReps,
        const int seed
    ) const;
};

#endif // ESA_SFA_MEFF_HPP