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
 * @file ESASfaLcmBaseTC.hpp
 * @details Structs for thread cache approach
 */

 #ifndef ESA_SFA_LCM_BASE_TC_HPP
 #define ESA_SFA_LCM_BASE_TC_HPP

namespace thread_cache_lcm {

    /**
     * @brief Thread cache for calculating LCM derivatives
     */
    struct WSLcmInternalAnalyticJacHess {
        // dedicated aligned buffers for y, x, zuit etc
        arma::dmat y, x, zuit, zvit, zvi0;

        /**
         * @brief resize all matricies
         * @param maxT the maximimum number of observations for a firm
         * @param k total number of variables
         * @param maxZ 
         * @param nsim number of simulations
         * @param nX number of variables in the frontier
         * @param nZuit number of variables for determinants of time-varying inefficiency
         * @param nZvit number of variables for determinants of stochastic noise
         * @param nZvi0 number of variables for determinants of latent firm effects
         */
        void ensureSize(int maxT, int k, int maxZ, int nsim, int nX, int nZuit, int nZvit, int nZvi0)
        {
            
        }
    };

    /**
     * @brief Thread cache for calculating LCM derivatives
     */
    struct WSLcmInternalAnalyticJacHessTN {
        // dedicated aligned buffers for y, x, zuit etc
        arma::dmat y, x, zmuit, zuit, zvit, zvi0;
        
        /**
         * @brief resize all matricies
         * @param maxT the maximimum number of observations for a firm
         * @param k total number of variables
         * @param maxZ 
         * @param nsim number of simulations
         * @param nX number of variables in the frontier
         * @param nZmuit number of variables determining mean of truncated-normal inefficiency component
         * @param nZuit number of variables for determinants of time-varying inefficiency
         * @param nZvit number of variables for determinants of stochastic noise
         * @param nZvi0 number of variables for determinants of latent firm effects
         */
        void ensureSize(int maxT, int k, int maxZ, int nsim, int nX, int nZmuit, int nZuit, int nZvit, int nZvi0)
        {

        }
    };

} // namespace thread_cache

 #endif // ESA_SFA_LCM_BASE_TC_HPP