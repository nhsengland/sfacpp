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

#include <memory>
#include "sfa/ESASfaLcmBase.hpp"
#include "math/esandist.hpp"
#include "math/esamath.hpp"
#include "math/primes.hpp"
#include "math/HaltonSeq.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "utils/ThreadContext.hpp"
#include "thread_cache/ESASfaTreBaseTC.hpp"

// constructor 1
ESASfaLcmBase::ESASfaLcmBase(
    const std::shared_ptr<ESADataPanelLCM> dataObjPtr,
    const int s,
    const int nsim,
    const int seed,
    const HaltonSettings hsetting
) : ESASfaBase(dataObjPtr, s),
    seed(seed),
    nsim(nsim),
    obsUseSameHaltonDraw(hsetting.obsUseSameHaltonDraw),
    hsettings(hsetting)
{
    // check type is specific for LCM
    if (!dynamic_cast<ESADataPanelLCM *>(dataObjPtr.get())) {
        throw std::invalid_argument("data object is not of type ESADataPanelLCM");
    }
    // dereference pointer to underlying data object
    ESADataPanelLCM& dataObj = (ESADataPanelLCM&)*dataObjPtr;
    ESASfaModelFamily mF = ESAEnums::getModelFamily(dataObj.getModelType());
    // number of firms - only relevant for the panel models
    if (mF == ESASfaModelFamily::LC_TRE || mF == ESASfaModelFamily::LM_TRE) {
        // setup halton draws
        double nids = dataObj.getNids();
        if (hsettings.burnin < 0) throw std::invalid_argument("halton burnin cannot be less than zero");
        if (hsettings.start < 0) throw std::invalid_argument("halton start cannot be less than zero");
        int haltonMainPrime = hsettings.base;
        if (!hsettings.useBase && hsettings.start >= 100007) {
            throw std::invalid_argument("start position for halton main out of bounds, try reducing");
        }
        if (!hsettings.useBase) haltonMainPrime = my100008Primes[hsettings.start];
        // main halton draw
        arma::dmat h = HaltonSeq::generate(haltonMainPrime, (nsim * nids), hsettings.burnin, hsettings.scrambled, seed, hsettings.shuffle);
        h = arma::reshape(h, nsim, nids).t();
        // calculate the mean of the halton draw
        double mn = arma::mean(arma::mean(h));
        ESALogger::logger()->info("The mean of the halton draw {:.5f}", mn);
        // apply ppf function to whole halton draw to map
        h = esandist::ppf(h, 0.0, 1.0);
        this->haltonDraws = std::make_shared<arma::dmat>(h);
    }
}

// secondary constructor, where pass in a shared ptr to an already derived halton matrix
ESASfaLcmBase::ESASfaLcmBase(
    const std::shared_ptr<ESADataPanelLCM> dataObjPtr,
    const int s,
    const int nsim,
    const int seed,
    const std::shared_ptr<arma::dmat> haltonDrawPtr,
    const HaltonSettings hsetting
) : ESASfaBase(dataObjPtr, s),
    seed(seed),
    nsim(nsim),
    obsUseSameHaltonDraw(hsetting.obsUseSameHaltonDraw),
    hsettings(hsetting)
{
    // check type is specific for LCM
    if (!dynamic_cast<ESADataPanelLCM *>(dataObjPtr.get())) {
        throw std::invalid_argument("data object is not of type ESADataPanelLCM");
    }
    // dereference pointer to underlying data object
    ESADataPanelLCM& dataObj = (ESADataPanelLCM&)*dataObjPtr;
    ESASfaModelFamily mF = ESAEnums::getModelFamily(dataObj.getModelType());
    // number of firms - only relevant for the panel models
    if (mF == ESASfaModelFamily::LC_TRE || mF == ESASfaModelFamily::LM_TRE) {
        // check halton draws
        double nids = dataObj.getNids();
        if (haltonDrawPtr == nullptr) throw std::invalid_argument("'haltonDrawPtr' cannot be 'nullptr'");
        // check dimensions of the halton draw matrix
        if (haltonDrawPtr->n_rows < nids || haltonDrawPtr->n_cols < nsim) {
            throw std::invalid_argument("invalid matrix size for halton draws");
        }
        this->haltonDraws = haltonDrawPtr;
    }
}

// analytical first-order derivative for half-normal
template <typename TY, typename TX, typename TZuit, typename TZvit, typename TZvi0>
void ESASfaLcmBase::internalAnalyticJacHess(
    const unsigned int idx,
    const ESASfaModelType mT,
    const arma::dcolvec& par,
    const arma::Base<double, TY>& yIn,
    const arma::Base<double, TX>& xIn,
    const arma::Base<double, TZuit>& zuitIn,
    const arma::Base<double, TZvit>& zvitIn,
    const arma::Base<double, TZvi0>& zvi0In,
    const arma::dmat& draws,
    const arma::Base<double, arma::subview<double>>& QirIn,
    arma::dmat* jacOut,
    arma::dmat* hessOut
) const
{
    // persistent variables per thread
    ThreadContext* ctx = getContext();
    // check if empty or not
    if (!ctx->lcmGradPan) {
        ctx->lcmGradPan = std::make_unique<thread_cache_lcm::WSLcmInternalAnalyticJacHess>();
    }
    // dereference ptr for the thread cache struct for the LCM grad
    thread_cache_lcm::WSLcmInternalAnalyticJacHess& ws = *ctx->lcmGradPan;
    // dereference ptr to data object
    ESADataPanelLCM& dataObj = (ESADataPanelLCM&)*dataObjPtr;
    // TODO: implement half-normal analytical gradient for LCM base
}

// analytical first-order derivative for truncated-normal
template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0>
void ESASfaLcmBase::internalAnalyticJacHessTN(
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
    arma::dmat* jacOut,
    arma::dmat* hessOut
) const
{
    // persistent variables on specific thread
    ThreadContext* ctx = getContext();
    // TODO: implement truncated-normal analytical gradient for LCM base
}
