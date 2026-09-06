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

#ifndef ESA_SFA_EFF_GTRE_HPP
#define ESA_SFA_EFF_GTRE_HPP

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
#include <memory>
#include <optional>
#include "data/ESADataBase.hpp"
// #include "utils/enums.hpp"
#include "effscores.hpp"


class ESASfaEffGtre {
public:
    /// Constructor
    ESASfaEffGtre(const std::shared_ptr<ESADataBase>& dataObjPtr, const double s);
    // ESASfaEffGtre() {}

    /// calculate the overall efficiency scores
    /**
     * @brief Calculate overall efficiency scores
     * @param par the parameter vector at which to estimate
     * @param nsim the number of simulations for GHK
     * @param haltonStart Offset on my100008Primes
     * @param seed
     * @param threaded bool whether or not to use threading
     */
    std::unique_ptr<ESASfaEffScores> efficiencyScores(
        const arma::dcolvec& par,
        const int nsim = 2000,
        const int haltonStart = 0,
        const int seed = 7,
        const bool threaded = false
    );

private:
    const std::shared_ptr<ESADataBase>& dataObjPtr;
    const double s;
    const ESASfaModelType mT;

    /// @brief Calculate the efficiency score for the GTRE model for a given panel
    ///         Methodlogy based on Colombi et al, 2014
    /// @param par 
    /// @param y 
    /// @param x 
    /// @param zuit 
    /// @param zvit 
    /// @param zui0 
    /// @param zvi0 
    /// @return 
    template <typename TY, typename TX, typename TZuit, typename TZvit, typename TZui0, typename TZvi0>
    void gtrePanelEfficiency(
        const unsigned int idx,
        const arma::dcolvec& par,
        const arma::Base<double, TY>& yIn,
        const arma::Base<double, TX>& xIn,
        const arma::Base<double, TZuit>& zuitIn,
        const arma::Base<double, TZvit>& zvitIn,
        const arma::Base<double, TZui0>& zui0In,
        const arma::Base<double, TZvi0>& zvi0In,
        const arma::dmat& haltonDraw,
        arma::dmat& out,
        // extracted coefficients
        const arma::dcolvec& b_x,
        const arma::dcolvec& b_zuit,
        const arma::dcolvec& b_zvit,
        const arma::dcolvec& b_zvi0,
        const arma::dcolvec& b_zui0
    ) const;

    /**
     * @brief Calculate the efficiency score for the TRE mode, based on the
     * panel methodology based on Colombi et al, 2014
     * @param idx Current panel index [firm]
     * @param par Column vector of parameters
     * @param y Column vector of the dependent variable
     * @param x Matrix of independent variables
     * @param zuit Matrix of determinants of inefficiency
     * @param zvit Matrix of determinants of random noise
     * @param zvi0 Matrix of determinants of latent firm heterogeneity
     * @param nsim number of simulations
     * @param haltonStart 
     * @param seed Seed for halton draw
     */
    template <typename TX, typename TY, typename TZuit, typename TZvit, typename TZvi0>
    void trePanelEfficiency(
        const unsigned int idx,
        const arma::dcolvec& par,
        const arma::Base<double, TY>& yIn,
        const arma::Base<double, TX>& xIn,
        const arma::Base<double, TZuit>& zuitIn,
        const arma::Base<double, TZvit>& zvitIn,
        const arma::Base<double, TZvi0>& zvi0In,
        const arma::dmat& haltonDraw,
        arma::dmat& out,
        // extracted coefficients
        const arma::dcolvec& b_x,
        const arma::dcolvec& b_zuit,
        const arma::dcolvec& b_zvit,
        const arma::dcolvec& b_zvi0,
        const arma::dcolvec& b_zui0
    ) const;
};

#endif // ESA_SFA_EFF_GTRE_HPP