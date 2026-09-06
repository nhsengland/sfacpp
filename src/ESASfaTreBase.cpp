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
 * @file ESASfaTreBase.cpp
 */
#include <sstream>
#include "sfa/ESASfaTreBase.hpp"
#include "data/ESADataPanel.hpp"
#include "math/esandist.hpp"
#include "math/esamath.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "math/HaltonSeq.hpp"
#include "math/primes.hpp"
#include "math/esandist.hpp"
#include "utils/ThreadContext.hpp"
#include "thread_cache/ESASfaTreBaseTC.hpp"
#include "regression/ESAFixedEff.hpp"
#include "regression/ESARandEff.hpp"

ESASfaTreBase::ESASfaTreBase(
    const std::shared_ptr<ESADataBase> dataObjPtr,
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
    if (!dynamic_cast<ESADataPanel *>(dataObjPtr.get())) {
        throw std::invalid_argument("data object is not of type ESADataPanel");
    }
    // dereference pointer to underlying data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    ESASfaModelFamily mF = ESAEnums::getModelFamily(dataObj.getModelType());
    // number of firms
    double nids = dataObj.getNids();
    // check haltonBurnin
    if (hsetting.burnin < 0) throw std::invalid_argument("halton burnin cannot be less than zero");
    if (hsetting.start < 0) throw std::invalid_argument("halton start cannot be less than zero");
    // 
    int haltonMainPrime = hsetting.base;
    if (!hsetting.useBase && hsetting.start >= 100007) {
        throw std::invalid_argument("start position for halton main out of bounds, try reducing");
    }
    if (!hsetting.useBase) haltonMainPrime = my100008Primes[hsetting.start];
    // main halton draw irrespectively
    arma::dmat hseq = HaltonSeq::generate(haltonMainPrime, (nsim * nids), hsetting.burnin, hsetting.scrambled, seed, hsetting.shuffle);
    // hseq = hseq.reshape(nids, nsim);
    hseq = arma::reshape(hseq, nsim, nids).t();
    // mtrx to store output 
    arma::dmat h;
    if (mF == ESASfaModelFamily::GTRE){
        // process halton draw for ui0
        int haltonUi0MainPrime = hsetting.ui0Base;
        if (!hsetting.ui0UseBase && hsetting.ui0Start >= 100007) {
            throw std::invalid_argument("start position for halton ui0 out of bounds, try reducing");
        }
        if (!hsetting.ui0UseBase) haltonUi0MainPrime = my100008Primes[hsetting.ui0Start];
        // // second halton draw
        arma::dmat hseq2 = HaltonSeq::generate(haltonUi0MainPrime, (nsim * nids), hsetting.burnin, hsetting.scrambled, seed, hsetting.shuffle);
        // hseq2 = hseq2.reshape(nids, nsim);
        hseq2 = arma::reshape(hseq2, nsim, nids).t();
        h = arma::dmat(nids * 2, nsim);
        h.submat(arma::span(0, nids - 1), arma::span(0, nsim - 1)) =  hseq;
        h.submat(arma::span(nids, nids * 2 - 1), arma::span(0, nsim - 1)) = hseq2;
        double correl = arma::as_scalar(arma::cor(arma::vectorise(hseq), arma::vectorise(hseq2)));
        double m1 = arma::mean(arma::mean(hseq));
        double m2 = arma::mean(arma::mean(hseq2));
        ESALogger::logger()->info("Mean of draw 1: {:.5f}, draw 2: {:.5f}. Correlation between draws: {:.5f}", m1, m2, correl);
    } else {
        h = std::move(hseq);
        double mn = arma::mean(arma::mean(h));
        ESALogger::logger()->info("The mean of the halton draw {:.5f}", mn);
    }
    // apply ppf function to whole halton draw to map
    h = esandist::ppf(h, 0.0, 1.0);
    this->haltonDraws = std::make_shared<arma::dmat>(h);
}

// secondary constructor where pass in a shared ptr to an already derived halton matrix
ESASfaTreBase::ESASfaTreBase(
    const std::shared_ptr<ESADataBase> dataObjPtr,
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
    if (!dynamic_cast<ESADataPanel *>(dataObjPtr.get())) {
        throw std::invalid_argument("data object is not of type ESADataPanel");
    }
    // dereference pointer to underlying data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    ESASfaModelFamily mF = ESAEnums::getModelFamily(dataObj.getModelType());
    // number of firms
    double nids = dataObj.getNids();
    if (haltonDrawPtr == nullptr) {
        throw std::invalid_argument("'haltonDrawPtr' cannot be 'nullptr'");
    }
    // check size of the halton draws to ensure appropriately size
    if (mF == ESASfaModelFamily::GTRE) {
        // expecting nids * 2 x nsim
        if (haltonDrawPtr->n_rows < (nids * 2) || haltonDrawPtr->n_cols < nsim) {
            throw std::invalid_argument("Invalid matrix size for halton GTRE");
        }
    } else {
        if (haltonDrawPtr->n_rows < nids || haltonDrawPtr->n_cols < nsim) {
            throw std::invalid_argument("Invalid matrix size for halton");
        }
    }
    this->haltonDraws = haltonDrawPtr;
}

// Starting values for the GTRE/TRE model
arma::dcolvec ESASfaTreBase::startingValues() const
{
    // dereference ptr to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    ESASfaModelFamily mF = ESAEnums::getModelFamily(dataObj.getModelType());
    ESASfaModelDistribution mD = ESAEnums::getDistribution(dataObj.getModelType());
    const arma::Col<int> idents = esautils::makeZeroCopyStrictViewColNoOpt<int>(dataObj.getIdVecPtr()); // dataObj.getIdVec();
    const arma::Col<int> uniqIdents = arma::unique(idents);
    const int N = uniqIdents.n_rows;
    const int nobs = dataObj.getNobs();
    // firstly, estimate a fixed effects (within estimator)
    // regression X on y (both should be present)
    // ESAFixedEff feMdl(dataObj.getYPtr(), dataObj.getXPtr(), dataObj.getIdVecPtr());
    // FixedEffResult feRes = feMdl.fit();
    ESARandEff reMdl(dataObj.getYPtr(), dataObj.getXPtr(), dataObj.getIdVecPtr());
    RandEffResult reRes = reMdl.fit();

    // ESALogger::logger()->info("RE result: beta: {}", feRes.params);
    // decompose residuals into transient vs persistent
    arma::dcolvec residTransient(nobs);
    arma::dcolvec residPersistExpand(nobs);
    arma::dcolvec residPersistUniq(N);
    // for the persistent model, need to collapse to group means
    std::optional<arma::dmat> zui0Collapse = dataObj.getZui0().has_value() ? std::make_optional<arma::dmat>(N, dataObj.getNZui0()) : std::nullopt;
    std::optional<arma::dmat> zvi0Collapse = dataObj.getZvi0().has_value() ? std::make_optional<arma::dmat>(N, dataObj.getNZvi0()) : std::nullopt;
    for (int i = 0; i < N; i++) {
        int currId = uniqIdents(i);
        arma::uvec inds = arma::find(idents == currId);
        // persistent component is the mean of the total residual per group
        // double grpMean = arma::mean(feRes.residual.elem(inds));
        double grpMean = arma::mean(reRes.reResidual.elem(inds));
        residPersistUniq(i) = grpMean;
        residPersistExpand.elem(inds).fill(grpMean);
        // transient component is the deviation from group means
        // residTransient.elem(inds) = feRes.residual.elem(inds) - grpMean;
        residTransient.elem(inds) = reRes.reResidual.elem(inds) - grpMean;
        // collapse to group means
        if (dataObj.getZui0().has_value()) {
            zui0Collapse.value().row(i) = arma::mean(dataObj.getZui0().value().rows(inds), 0);
        }
        if (dataObj.getZvi0().has_value()) {
            zvi0Collapse.value().row(i) = arma::mean(dataObj.getZvi0().value().rows(inds), 0);
        }
    }
    // apply MoM estimators for transient inefficiency [which both TRE & GTRE should estimate]
    MoMResult transRes = getMoMComponents(residTransient, dataObj.getZuit(), dataObj.getZvit());
    if(transRes.sigma2u < 1e-4) transRes.sigma2u = 0.05;
    if(transRes.sigma2v < 1e-4) transRes.sigma2v = 0.05;
    // ESALogger::logger()->info(
    //     "Transient MoM Result:\n sigma2u {} sigma2v {}\n b_zu {}\n b_zv {}, \n ln sigma2u {} ln sigma2v {}",
    //     transRes.sigma2u, transRes.sigma2v, transRes.b_zu, transRes.b_zv, std::log(transRes.sigma2u), std::log(transRes.sigma2v)
    // );
    // for TRE
    double sigmavi0 = arma::var(residPersistUniq);
    // ESALogger::logger()->info("sigmavi0 {}", sigmavi0);
    // 
    double avgT = (double)nobs / (double)N;
    double varBetween = arma::var(residPersistUniq);
    double vif = (transRes.sigma2u + transRes.sigma2v) / avgT;
    double m2Pers = arma::accu(arma::pow(residPersistUniq, 2)) / N;
    double m2Override = std::max(m2Pers - vif, varBetween * 0.2);
    // apply MoM estimators for persistent inefficiency
    MoMResult persistRes = getMoMComponents(residPersistUniq, zui0Collapse, zvi0Collapse, std::make_optional(m2Override));
    // MoMResult persistRes = getMoMComponents(residPersistUniq, zui0Collapse, zvi0Collapse);
    // MoMResult persistRes = getMoMComponents(residPersistExpand, dataObj.getZui0(), dataObj.getZvi0());
    // ESALogger::logger()->info(
    //     "Persistent MoM Result:\n sigma2ui0 {} sigma2vi0 {} \n b_zui0 {} \n b_zvi0 {} \n ln sigma2u {} ln sigma2v {}",
    //     persistRes.sigma2u, persistRes.sigma2v, persistRes.b_zu, persistRes.b_zv, std::log(persistRes.sigma2u), std::log(persistRes.sigma2v)
    // );
    // Think it might be better to underestimate sigmas than over
    const double scale = 0.8;
    // calculate intercept shift for the FE model
    double meanUTrans = std::sqrt(2.0 / arma::datum::pi) * std::sqrt(transRes.sigma2u);
    double meanUPers = 0.0;
    if (mF == ESASfaModelFamily::GTRE) {
        meanUPers = std::sqrt(2.0 / arma::datum::pi) * std::sqrt(persistRes.sigma2u);
    }
    // update the intercept shift
    // feRes.params(0) += (this->s * (meanUTrans + meanUPers) * scale);
    reRes.params(0) += (this->s * (meanUTrans + meanUPers) * scale);
    // ---- build the starting values ----
    int nX = dataObj.getNX(), nZmuit = (mD == ESASfaModelDistribution::TNORM) ? dataObj.getNZmuit() : 0;
    // common to both TRE, GTRE
    int nZuit = dataObj.getNZuit(), nZvit = dataObj.getNZvit(), nZvi0 = dataObj.getNZvi0();
    // only for GTRE
    int nZui0 = (mF == ESASfaModelFamily::GTRE) ? dataObj.getNZui0() : 0;
    int cntr = 0;
    // create the column vector
    arma::dcolvec start(nX + nZmuit + nZuit + nZvit + nZvi0 + nZui0);
    // add frontier components from the FE estimator
    // start.rows(0, nX - 1) = feRes.params;
    start.rows(0, nX - 1) = reRes.params;
    cntr += nX;
    // zmuit (if present) - just fill 0s
    if (nZmuit > 0) {
        start.rows(cntr, cntr + nZmuit - 1).zeros();
        cntr += nZmuit;
    }
    // zuit - use the log of the moment sigma2u (for transient) IF only 1 element, otherwise, use the coefficient vector
    if (nZuit == 1) {
        start(cntr) = std::log(transRes.sigma2u) * scale;
    } else {
        start.rows(cntr, cntr + nZuit - 1) = transRes.b_zu;
        // overwrite the intercept
        start(cntr) = std::log(transRes.sigma2u) * scale;
    }
    cntr += nZuit;
    // zvit - use the log of the moment sigma2v for transient IF only 1 element, otherwise the coefficient vector
    if (mF == ESASfaModelFamily::TRE && nZvit == 1){
        start(cntr) = std::log(sigmavi0);
    } else if (nZvit == 1) {
        start(cntr) = std::log(transRes.sigma2v) * scale;
    } else {
        start.rows(cntr, cntr + nZvit - 1) = transRes.b_zv;
        start(cntr) = std::log(transRes.sigma2v) * scale;
    }
    cntr += nZvit;
    // zvi0 - same principal as above, just for the transient moment
    if (nZvi0 == 1) {
        // start(cntr) = std::log(persistRes.sigma2v);
        start.rows(cntr, cntr + nZvi0 - 1) = persistRes.b_zv * scale;
    } else {
        start.rows(cntr, cntr + nZvi0 - 1) = persistRes.b_zv * scale;
        // start(cntr) = std::log(persistRes.sigma2v);
    }
    cntr += nZvi0;
    // zui0 - for the GTRE model only
    if (nZui0 == 1) {
        // start(cntr) = std::log(persistRes.sigma2u);
        start.rows(cntr, cntr + nZui0 - 1) = persistRes.b_zu * scale;
    } else if (nZui0 > 1) {
        start.rows(cntr, cntr + nZui0 - 1) = persistRes.b_zu * scale;
        // start(cntr) = std::log(persistRes.sigma2u);
    }
    // ESALogger::logger()->info("Starting values are {}", start);
    return start;
    
}

template <typename T>
void ESASfaTreBase::weightFromDensForPanel(const arma::Base<double, T>& ldIn, arma::dmat& out) const
{
    // dereference ptr to data obj
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    // unwrap armadillo references
    const auto& ld = ldIn.get_ref();

    ThreadContext* ctx = getContext();
    if (!ctx->treBaseWeightDens) {
        ctx->treBaseWeightDens = std::make_unique<thread_cache::WSWeightFromDensForPanel>();
    }
    thread_cache::WSWeightFromDensForPanel& ws = *ctx->treBaseWeightDens;
    // thread_cache::WSWeightFromDensForPanel* debug_ws = &ws;
    int maxT = dataObj.getMaxT();
    int nT = ld.n_rows;
    int ns = this->nsim;
    ws.ensureSize(maxT, ns);
    // subview to only relevant part of lnDen (where nT < maxT)
    arma::subview<double> lnDenView = ws.lnDen.submat(arma::span(0, nT - 1), arma::span(0, ns - 1));
    // calculate ln of the density
    lnDenView = arma::log(ld);
    // claculate S_r - sum over t (1 x nsim)
    // write into subview to stop armadillo shrinking the allocated memory
    arma::subview<double> srView = ws.Sr.submat(arma::span(0, 0), arma::span(0, ns - 1));
    srView = esamath::colSum(lnDenView);
    // find max value, Smax
    ws.Smax = srView.max();
    // write into subview to stop armadillo shrinking allocated memory
    arma::subview<double> krView = ws.Kr.submat(arma::span(0, 0), arma::span(0, ns - 1));
    krView = arma::exp(srView - ws.Smax);
    // out should be 1 x nsim, but check it
    if (out.n_cols != ns || out.n_rows != 1) {
        out.set_size(1, ns);
    }
    out = krView / arma::accu(krView);
}

// THIS IS FOR THE HETEROSKEDASTIC MODEL
template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZui0, typename TZvi0>
void ESASfaTreBase::internalAnalyticJacHess(
    const unsigned int idx,
    const ESASfaModelType mT,
    const arma::dcolvec& par,
    // accept any armadillo matrix/view
    const arma::Base<double, TY>& yIn,
    const arma::Base<double, TX>& xIn,
    // optionals
    const std::optional<TZmuit>& zmuitIn,
    const arma::Base<double, TZuit>& zuitIn,
    const arma::Base<double, TZvit>& zvitIn,
    const arma::Base<double, TZvi0>& zvi0In,
    const arma::dmat& draws,
    const std::optional<TZui0>& zui0In,
    const arma::Base<double, arma::subview<double>>& QirIn,
    arma::dmat* jacOut,
    arma::dmat* hessOut
) const
{
    // persistent variables on specific thread
    ThreadContext* ctx = getContext();
    // check if empty or not
    if (!ctx->treBaseJacHess){
        ctx->treBaseJacHess = std::make_unique<thread_cache::WSInternalAnalyticJacHess>();
    }
    thread_cache::WSInternalAnalyticJacHess& ws = *ctx->treBaseJacHess;
    // > dereference pointer to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    bool isGtre = (mF == ESASfaModelFamily::GTRE);
    // 
    int drawIdxPos = this-> obsUseSameHaltonDraw ? 0 : idx;
    int ui0DrawsBase = 0;
    int nIds = dataObj.getNids();
    if (isGtre) ui0DrawsBase = this->obsUseSameHaltonDraw ? 1 : nIds + idx;
    // checks that wont be out of bounds for halton draw - differs between TRE and GTRE
    if (isGtre){
        if (draws.n_rows <= ui0DrawsBase){
            throw std::invalid_argument("draws must have at least " + std::to_string(ui0DrawsBase) + " rows");
        }
    } else {
        if (draws.n_rows <= drawIdxPos) throw std::invalid_argument("'draws' must be of length 'ident' at minimum");
    }
    if (draws.n_cols < this->nsim) throw std::invalid_argument("'draws' must be of width 'nsim' at minimum");
    // unwrap armadillo base references to usable objects
    const auto& Qir = QirIn.get_ref();
    // > for Qir, gir check appropriate sizes
    // Qir should be 1 x nsim in size
    int ns = this->nsim;
    if (Qir.n_rows != 1 || Qir.n_cols != ns) throw std::invalid_argument("'Qir is incorrect dimensions, expect (1 x nsim)");
    int nT = yIn.get_ref().n_rows;
    if (nT == 0) return;
    // extract coefficients
    arma::dcolvec b_x = dataObj.paramX(par);
    if (!dataObj.paramZuit(par)) throw std::invalid_argument("missing 'zuit'");
    arma::dcolvec b_zuit = dataObj.paramZuit(par).value();
    if (!dataObj.paramZvit(par)) throw std::invalid_argument("missing 'zvit'");
    arma::dcolvec b_zvit = dataObj.paramZvit(par).value();
    if (!dataObj.paramZvi0(par)) throw std::invalid_argument("missing 'zvi0'");
    arma::dcolvec b_zvi0 = dataObj.paramZvi0(par).value();
    // calculate m so far
    int nX = b_x.n_rows, nZuit = b_zuit.n_rows, nZvit = b_zvit.n_rows, nZvi0 = b_zvi0.n_rows, nZui0 = 0;
    arma::dcolvec b_zui0;
    // arma::dmat zui0Acc;
    // if (isGtre && zui0) {
    if (isGtre && zui0In){
        if (!dataObj.paramZui0(par)) throw std::invalid_argument("missing 'zui0'");
        b_zui0 = dataObj.paramZui0(par).value();
        nZui0 = b_zui0.n_rows;
        // zui0Acc = zui0.value();
    }
    int k = nX + nZuit + nZvit + nZvi0 + nZui0;
    int maxT = dataObj.getMaxT();
    int maxZ = std::max({nX, nZuit, nZvit, nZvi0, nZui0});
    // resize matricies in workspace if needed
    int requiredRows = (nT > maxT) ? nT : maxT;
    ws.ensureSize(requiredRows, k, maxZ, ns, nX, nZuit, nZvit, nZui0, nZvi0);
    // subviews to aligned buffers
    arma::subview<double> y = ws.y.rows(arma::span(0, nT - 1));
    // protect incase one of the variables has count 0, so -1 would lead to underflow
    arma::subview<double> x = ws.x.submat(arma::span(0, nT - 1), arma::span(0, std::max(nX - 1, 0)));
    arma::subview<double> zuit = ws.zuit.submat(arma::span(0, nT - 1), arma::span(0, std::max(nZuit - 1, 0)));
    arma::subview<double> zvit = ws.zvit.submat(arma::span(0, nT - 1), arma::span(0, std::max(nZvit - 1, 0)));
    arma::subview<double> zui0 = ws.zui0.submat(arma::span(0, nT - 1), arma::span(0, std::max(nZui0 - 1, 0)));
    arma::subview<double> zvi0 = ws.zvi0.submat(arma::span(0, nT - 1), arma::span(0, std::max(nZvi0 - 1, 0)));
    // copy into buffer (which is aligned)
    y = yIn.get_ref();
    x = xIn.get_ref();
    zuit = zuitIn.get_ref();
    zvit = zvitIn.get_ref();
    zvi0 = zvi0In.get_ref();
    if (isGtre && zui0In) {
        zui0 = zui0In.value();
    } else {
        zui0.zeros();
    }
    // ---- output matricies ----
    // instead of preallocating lots of memory to calculate the simulated hessian,
    // calculate it directly here - first check the size of the jac, hess
    if (jacOut) {
        // expect minimum nT x k
        if (jacOut->n_rows < nT || jacOut->n_cols < k) {
            jacOut->set_size(nT, k);
        }
        // since we will write directly into the destination, through accumulation, zero out
        (*jacOut).zeros();
    }
    // hessian should be k x k
    if (hessOut) {
        if (hessOut->n_rows < k || hessOut->n_cols < k) {
            hessOut->set_size(k, k);
        }
    }
    
    // --- aliasing (mapping atlas cols) ---
    // invariants
    arma::subview<double> xb = ws.a1.submat(0, 0, nT-1, 0);
    arma::subview<double> sigma2uit = ws.a2.submat(0, 0, nT-1, 0);
    arma::subview<double> sigmauit = ws.a3.submat(0, 0, nT-1, 0);
    arma::subview<double> sigma2vit = ws.a38.submat(0, 0, nT-1, 0);
    arma::subview<double> sigmavit = ws.a4.submat(0, 0, nT-1, 0);
    arma::subview<double> sigma2vi0 = ws.a5.submat(0, 0, nT-1, 0);
    arma::subview<double> sigmavi0 = ws.a6.submat(0, 0, nT-1, 0);
    arma::subview<double> sigma2ui0 = ws.a7.submat(0, 0, nT-1, 0);
    arma::subview<double> sigmaui0 = ws.a8.submat(0, 0, nT-1, 0);
    arma::subview<double> sigma2 = ws.a9.submat(0, 0, nT-1, 0);
    arma::subview<double> sigma = ws.a10.submat(0, 0, nT-1, 0);
    arma::subview<double> lambda = ws.a11.submat(0, 0, nT-1, 0);
    arma::subview<double> lamDivSig = ws.a12.submat(0, 0, nT-1, 0);
    // simulation specifics
    arma::subview<double> eps = ws.a13.submat(0, 0, nT-1, 0);
    arma::subview<double> epsDivSigma2 = ws.a14.submat(0, 0, nT-1, 0);
    arma::subview<double> epsDivSigma = ws.a15.submat(0, 0, nT-1, 0);
    arma::subview<double> Aitr = ws.a16.submat(0, 0, nT-1, 0);
    arma::subview<double> Mitr = ws.a17.submat(0, 0, nT-1, 0);
    // first derivative scalars
    arma::subview<double> dBeta1 = ws.a18.submat(0, 0, nT-1, 0);
    arma::subview<double> f_sigma = ws.a19.submat(0, 0, nT-1, 0);
    arma::subview<double> f_lambda = ws.a20.submat(0, 0, nT-1, 0);
    // Reuse columns for term aggregation
    arma::subview<double> term1 = ws.a21.submat(0, 0, nT-1, 0);
    arma::subview<double> f_kappa_grad = ws.a39.submat(0, 0, nT-1, 0); 
    // actual calculations
    // simplying sigma2u and sigma2v into lambda and sigma2
    xb = x * b_x;
    // sigma2uit - variance of the time-varying inefficiency
    sigma2uit = esautils::processSig2Term(b_zuit, zuit);
    sigmauit = arma::sqrt(sigma2uit);
    // sigma2vit - variance of the stochastic noise component
    sigma2vit = esautils::processSig2Term(b_zvit, zvit);
    sigmavit = arma::sqrt(sigma2vit);
    // sigma2vi0 - variance of the firm effect
    sigma2vi0 = esautils::processSig2Term(b_zvi0, zvi0);
    sigmavi0 = arma::sqrt(sigma2vi0);
    // sigma2ui0 - variance of the persistent inefficiency
    if (isGtre) {
        sigma2ui0 = esautils::processSig2Term(b_zui0, zui0);
        sigmaui0 = arma::sqrt(sigma2ui0);
    }
    sigma2 = sigma2uit + sigma2vit;
    sigma = arma::sqrt(sigma2);
    // lambda - sigu / sigv
    lambda = sigmauit / sigmavit;
    lamDivSig = lambda / sigma;
    // access jacobian buffer view
    arma::subview<double> jac = ws.jac.submat(arma::span(0, nT - 1), arma::span(0, k - 1));
    // view to store accumulated hessian (h_bar_i)
    arma::subview<double> h_bar_i = ws.h_bar_i.submat(arma::span(0, k - 1), arma::span(0, k - 1));
    // zero it out (so its not contaminated by previous runs)
    h_bar_i.zeros();
    // view for gir - nsim x k matrix, containing the gradient (e.g., sum of jacobian per simulation)
    arma::subview<double> gir = ws.gir.submat(arma::span(0, ns - 1), arma::span(0, k - 1));
    // view to store g_bar_i (1 x k) in size
    arma::subview<double> g_bar_i = ws.g_bar_i.submat(arma::span(0, 0), arma::span(0, k - 1));
    // zero this out, since accumlate to it
    g_bar_i.zeros();
    // --- replications for number of simulations ----
    for (int ns = 0; ns < nsim; ns++) {
        // ---- PRECALCULATE USEFUL TERMS ----
        // output matrix ∈ ℝ^(nT x m)
        // draw processing - scalar extraction
        double vi0Norm = draws(drawIdxPos, ns);
        double ui0Norm = 0.0;
        if (isGtre) {
            ui0Norm = std::abs(draws(ui0DrawsBase, ns));
        }
        eps = y - xb - (sigmavi0 * vi0Norm);
        if (isGtre) {
            eps += (s * sigmaui0 * ui0Norm);
        }
        // eps / sigma2; eps / sigma
        epsDivSigma2 = eps / sigma2;
        epsDivSigma = eps / sigma;
        // calculate Aitr
        Aitr = -s * epsDivSigma % lambda;
        // calculate Mitr with stability check
        for(int i=0; i<nT; ++i) {
            double a = Aitr(i);
            if(a < -37.0) {
                double a2 = a*a;
                Mitr(i) = -a - (1.0/a) + (2.0/(a*a2)); 
            } else {
                Mitr(i) = arma::normpdf(a) / arma::normcdf(a);
            }
        }
        // ---- FIRST ORDER PARTIAL DERIVATIVES ----
        // partial derivatives
        // z = zvit; p = zuit; g = zvi0
        dBeta1 = epsDivSigma2 + (s * Mitr % lamDivSig);
        // wrt x
        // f_beta = esautils::sweepMatrixElementwise(x, 1, dBeta1, "*");
        // >>>> ∂lnP/∂sigma <<<<
        // arma::subview<double> f_sigma = ws.f_sigma.submat(arma::span(0, nT - 1), arma::span(0, 0));
        f_sigma = (1.0 / sigma) % (arma::pow(epsDivSigma, 2.0) - (Mitr % Aitr) - 1.0);
        // >>>> ∂lnP/∂lambda <<<<
        // arma::subview<double> f_lambda = ws.f_lambda.submat(arma::span(0, nT - 1), arma::span(0, 0));
        f_lambda = -s * Mitr % epsDivSigma;
        // fill jacobian directly
        jac.cols(0, nX - 1) = esautils::sweepMatrixElementwise(x, 1, dBeta1, "*");
        int curPos = nX;
        // ---- zuit component ----
        if (nZuit > 0) {
            term1 = (f_lambda % (0.5 * lambda)) + (f_sigma % (0.5 * sigma2uit / sigma));
            jac.cols(curPos, curPos + nZuit - 1) = esautils::sweepMatrixElementwise(zuit, 1, term1, "*");
            curPos += nZuit;
        }
        // ---- zvit component ----
        if (nZvit > 0) {
            term1 = (f_lambda % (-0.5 * lambda)) + (f_sigma % (0.5 * sigma2vit / sigma));
            jac.cols(curPos, curPos + nZvit - 1) = esautils::sweepMatrixElementwise(zvit, 1, term1, "*");
            curPos += nZvit;
        }
        // ---- zvi0 component ----
        if (nZvi0 > 0) {
            // f_g = dBeta1 * vi0 * 0.5 * sigmav0 * zvi0
            // term1 = scalar multiplier vector
            term1 = dBeta1 * vi0Norm % (0.5 * sigmavi0);
            jac.cols(curPos, curPos + nZvi0 - 1) = esautils::sweepMatrixElementwise(zvi0, 1, term1, "*");
            curPos += nZvi0;
        }
        // ---- zui0 component {gtre only} ----
        if (isGtre && nZui0 > 0) {
            // f_kappa = s * dBeta1 * |ui0| * 0.5 * sigmau0 * zui0
            // term1 = s * dBeta1 * std::abs(ui0Norm) % (0.5 * sigmaui0); // wrong
            // term1 = -s * dBeta1 * std::abs(ui0Norm) % (0.5 * sigmaui0);
            f_kappa_grad = - s * dBeta1 * std::abs(ui0Norm) % (0.5 * sigmaui0);
            jac.cols(curPos, curPos + nZui0 - 1) = esautils::sweepMatrixElementwise(zui0, 1, f_kappa_grad, "*");
        }
        // write directly into the destination buffer
        if (jacOut) {
            // multiply by weight, Qir, and accumulate
            arma::subview<double> jacOutView = (jacOut->submat(arma::span(0, nT - 1), arma::span(0, k - 1)));
            jacOutView += (Qir(0, ns) * jac);
        }
        if (!hessOut) continue;
        // only bother calculating gir when deriving the hessian, since it is needed for the hessian
        // weighted average gradient (e.g, sum of jacobian), per simulation
        gir.row(ns) = esamath::colSum(jac);
        // weighted average gradient, overall
        g_bar_i += (Qir(0, ns) * gir.row(ns));
        // ---- SECOND ORDER PARTIAL DERIVATIVES ----
        // ---- second derivative scalars in atlas ----
        arma::subview<double> Ditr = ws.a22.submat(0, 0, nT-1, 0);
        arma::subview<double> f_beta_beta1 = ws.a23.submat(0, 0, nT-1, 0);
        arma::subview<double> dBetadSigma1 = ws.a24.submat(0, 0, nT-1, 0);
        arma::subview<double> dBetadLambda1 = ws.a25.submat(0, 0, nT-1, 0);
        arma::subview<double> f_sigma_sigma = ws.a26.submat(0, 0, nT-1, 0);
        arma::subview<double> f_lambda_lambda = ws.a27.submat(0, 0, nT-1, 0);
        arma::subview<double> f_sigma_lambda = ws.a28.submat(0, 0, nT-1, 0);
        arma::subview<double> dBetadSigmaV0 = ws.a29.submat(0, 0, nT-1, 0);
        arma::subview<double> dSigmadSigmaV0 = ws.a30.submat(0, 0, nT-1, 0);
        arma::subview<double> dLambdadSigmaV0 = ws.a31.submat(0, 0, nT-1, 0);
        arma::subview<double> f_sigmaw_sigmaw = ws.a32.submat(0, 0, nT-1, 0);
        arma::subview<double> dBetadSigmaH = ws.a33.submat(0, 0, nT-1, 0);
        arma::subview<double> dSigmadSigmaH = ws.a34.submat(0, 0, nT-1, 0);
        arma::subview<double> dLambdadSigmaH = ws.a35.submat(0, 0, nT-1, 0);
        arma::subview<double> f_sigmah_sigmah= ws.a36.submat(0, 0, nT-1, 0);
        // others - reuse term1/term2
        // calculate Ditr which is the derivative of Mitr
        // IMR(x) = PDF(x)/CDF(x); IMR'(x) = -x.IMR(x) - IMR(x)^2
        Ditr = -(Aitr % Mitr) - arma::pow(Mitr, 2.0);
        // common hessian terms 1 - lambda^2 * D
        term1 = 1.0 - (arma::pow(lambda, 2.0) % Ditr); 
        arma::subview<double> term2_view = ws.a37.submat(0, 0, nT-1, 0);
        term2_view = Mitr + (Aitr % Ditr);
        f_beta_beta1 = -(1.0 / sigma2) % term1;
        dBetadSigma1 = (1.0 / sigma2) % (-2.0 * epsDivSigma - (s * lambda % term2_view));
        dBetadLambda1 = (s * 1.0 / sigma) % term2_view;
        f_sigma_sigma = -(1.0 / sigma2) % (3.0 * arma::pow(epsDivSigma, 2.0) - (Aitr % (Aitr % Ditr + 2.0 * Mitr)) - 1.0);
        f_lambda_lambda = -arma::pow(epsDivSigma, 2.0) % Mitr % (Aitr + Mitr);
        f_sigma_lambda = (s * epsDivSigma2) % term2_view;
        dBetadSigmaV0 = (vi0Norm / sigma2) % (arma::pow(lambda, 2.0) % Ditr - 1.0);
        dSigmadSigmaV0 = (vi0Norm / sigma2) % (-2.0 * epsDivSigma - lambda % term2_view);
        dLambdadSigmaV0 = (vi0Norm / sigma) % term2_view;
        f_sigmaw_sigmaw = -arma::pow(vi0Norm / sigma, 2.0) % term1;
        if (isGtre) {
            // modified the signs on these
            dBetadSigmaH = -s * (std::abs(ui0Norm) / sigma2) % (arma::pow(lambda, 2.0) % Ditr - 1.0);
            f_sigmah_sigmah = - arma::pow(std::abs(ui0Norm) / sigma, 2.0) % term1;
            dSigmadSigmaH = -s * (std::abs(ui0Norm) / sigma2) % (-2.0 * epsDivSigma - lambda % term2_view);
            dLambdadSigmaH = -s * (std::abs(ui0Norm) / sigma) % term2_view;
        }
        // >> for overall simulated hessian
        // reset hessian over nt to zero (since it's an accumulate one)
        arma::subview<double> hessSumNt = ws.hessSumNt.submat(arma::span(0, k - 1), arma::span(0, k - 1));
        hessSumNt.zeros();
        // >> iterate over nT iterations
        for (int tt = 0; tt < nT; tt++) {
            // reference single k x k hessian buffer
            arma::dmat& H = ws.h;
            H.zeros();
            // reusable scalars
            double lam = lambda(tt), sig = sigma(tt); // sig2 = sigma2(tt);
            double sig2u = sigma2uit(tt), sig2v = sigma2vit(tt);
            double sigv0 = sigmavi0(tt), sigu0 = sigmaui0(tt);
            // X Transpose (into Scratch 1)
            arma::subview<double> xT = ws.v1Scratch.submat(0, 0, nX-1, 0);
            xT = x.row(tt).t();
            // Beta-Beta
            H.submat(0, 0, nX-1, nX-1) = f_beta_beta1(tt) * (xT * xT.t());
            // Beta-Zuit
            if (nZuit > 0) {
                arma::subview<double> zT = ws.v2Scratch.submat(0, 0, nZuit-1, 0);
                zT = zuit.row(tt).t();
                double scalar = (dBetadLambda1(tt) * 0.5 * lam) + (dBetadSigma1(tt) * 0.5 * sig2u / sig);
                // Direct assignment using outer product
                H.submat(0, nX, nX-1, nX+nZuit-1) = scalar * (xT * zT.t());
                H.submat(nX, 0, nX+nZuit-1, nX-1) = scalar * (zT * xT.t());
            }
            // Beta-Zvit
            if (nZvit > 0) {
                arma::subview<double> zT = ws.v2Scratch.submat(0, 0, nZvit-1, 0);
                zT = zvit.row(tt).t();
                double scalar = (dBetadLambda1(tt) * -0.5 * lam) + (dBetadSigma1(tt) * 0.5 * sig2v / sig);
                int sc = nX + nZuit;
                H.submat(0, sc, nX-1, sc+nZvit-1) = scalar * (xT * zT.t());
                H.submat(sc, 0, sc+nZvit-1, nX-1) = scalar * (zT * xT.t());
            }
            // Beta-Zvi0
            if (nZvi0 > 0) {
                arma::subview<double> zT = ws.v2Scratch.submat(0, 0, nZvi0-1, 0);
                zT = zvi0.row(tt).t();
                double scalar = dBetadSigmaV0(tt) * 0.5 * sigv0;
                int sc = nX + nZuit + nZvit;
                H.submat(0, sc, nX-1, sc+nZvi0-1) = scalar * (xT * zT.t());
                H.submat(sc, 0, sc+nZvi0-1, nX-1) = scalar * (zT * xT.t());
            }
            // Beta-Zui0
            if (isGtre && nZui0 > 0) {
                arma::subview<double> zT = ws.v2Scratch.submat(0, 0, nZui0-1, 0);
                zT = zui0.row(tt).t();
                double scalar = dBetadSigmaH(tt) * 0.5 * sigu0;
                int sc = nX + nZuit + nZvit + nZvi0;
                H.submat(0, sc, nX-1, sc+nZui0-1) = scalar * (xT * zT.t());
                H.submat(sc, 0, sc+nZui0-1, nX-1) = scalar * (zT * xT.t());
            }
            if (nZuit > 0) {
                // Must ensure Z is in Scratch 1 for symmetric outer product
                arma::subview<double> zT = ws.v1Scratch.submat(0, 0, nZuit-1, 0);
                zT = zuit.row(tt).t();
                double dL = 0.5 * lam;
                double dS = 0.5 * sig2u / sig;
                double scalar = f_lambda_lambda(tt)*dL*dL + f_sigma_sigma(tt)*dS*dS 
                              + 2.0*f_sigma_lambda(tt)*dL*dS 
                              + f_lambda(tt)*0.25*lam + f_sigma(tt)*((sig2u*(sig2u+2.0*sig2v))/(4.0*std::pow(sig, 3.0)));
                int sc = nX;
                H.submat(sc, sc, sc+nZuit-1, sc+nZuit-1) = scalar * (zT * zT.t());
            }
            if (nZvit > 0) {
                arma::subview<double> zT = ws.v1Scratch.submat(0, 0, nZvit-1, 0);
                zT = zvit.row(tt).t();
                double dL = -0.5 * lam;
                double dS = 0.5 * sig2v / sig;
                double scalar = f_lambda_lambda(tt)*dL*dL + f_sigma_sigma(tt)*dS*dS 
                              + 2.0*f_sigma_lambda(tt)*dL*dS 
                              + f_lambda(tt)*0.25*lam + f_sigma(tt)*((sig2v*(sig2v+2.0*sig2u))/(4.0*std::pow(sig, 3.0)));
                int sc = nX + nZuit;
                H.submat(sc, sc, sc+nZvit-1, sc+nZvit-1) = scalar * (zT * zT.t());
            }
            // Zuit-Zvit
            if (nZuit > 0 && nZvit > 0) {
                arma::subview<double> zTu = ws.v1Scratch.submat(0, 0, nZuit-1, 0);
                zTu = zuit.row(tt).t();
                arma::subview<double> zTv = ws.v2Scratch.submat(0, 0, nZvit-1, 0);
                zTv = zvit.row(tt).t();
                double dLu = 0.5 * lam, dSu = 0.5 * sig2u / sig;
                double dLv = -0.5 * lam, dSv = 0.5 * sig2v / sig;
                double scalar = f_lambda_lambda(tt)*dLu*dLv + f_sigma_sigma(tt)*dSu*dSv
                              + f_sigma_lambda(tt)*(dLu*dSv + dSu*dLv)
                              + f_lambda(tt)*(-0.25*lam) + f_sigma(tt)*(-(sig2u*sig2v)/(4.0*std::pow(sig, 3.0)));
                int scU = nX;
                int scV = nX + nZuit;
                H.submat(scU, scV, scU+nZuit-1, scV+nZvit-1) = scalar * (zTu * zTv.t());
                H.submat(scV, scU, scV+nZvit-1, scU+nZuit-1) = scalar * (zTv * zTu.t());
            }
            // Zvi0-Zvi0
            if (nZvi0 > 0) {
                arma::subview<double> zT = ws.v1Scratch.submat(0, 0, nZvi0-1, 0);
                zT = zvi0.row(tt).t();
                double scalar = 0.25 * sigv0 * (sigv0 * f_sigmaw_sigmaw(tt) + (dBeta1(tt) * vi0Norm));
                int sc = nX + nZuit + nZvit;
                H.submat(sc, sc, sc+nZvi0-1, sc+nZvi0-1) = scalar * (zT * zT.t());
            }
            // Zvi0-Zuit
            if (nZvi0 > 0 && nZuit > 0) {
                arma::subview<double> zTg = ws.v1Scratch.submat(0, 0, nZvi0-1, 0);
                zTg = zvi0.row(tt).t();
                arma::subview<double> zTu = ws.v2Scratch.submat(0, 0, nZuit-1, 0);
                zTu = zuit.row(tt).t();
                double dSw = 0.5 * sigv0;
                double scalar = dSw * (dLambdadSigmaV0(tt) * 0.5 * lam + dSigmadSigmaV0(tt) * 0.5 * sig2u / sig);
                int scU = nX;
                int scG = nX + nZuit + nZvit;
                H.submat(scU, scG, scU+nZuit-1, scG+nZvi0-1) = scalar * (zTu * zTg.t());
                H.submat(scG, scU, scG+nZvi0-1, scU+nZuit-1) = scalar * (zTg * zTu.t());
            }
            // ZVI0-ZVIT
            if (nZvi0 > 0 && nZvit > 0) {
                arma::subview<double> zTg = ws.v1Scratch.submat(0, 0, nZvi0-1, 0);
                zTg = zvi0.row(tt).t();
                arma::subview<double> zTv = ws.v2Scratch.submat(0, 0, nZvit-1, 0);
                zTv = zvit.row(tt).t();
                double dSw = 0.5 * sigv0;
                double scalar = dSw * (dLambdadSigmaV0(tt) * -0.5 * lam + dSigmadSigmaV0(tt) * 0.5 * sig2v / sig);
                int scV = nX + nZuit;
                int scG = nX + nZuit + nZvit;
                H.submat(scV, scG, scV+nZvit-1, scG+nZvi0-1) = scalar * (zTv * zTg.t());
                H.submat(scG, scV, scG+nZvi0-1, scV+nZvit-1) = scalar * (zTg * zTv.t());
            }
            // 12. GTRE BLOCKS (ZUI0)
            if (isGtre && nZui0 > 0) {
                int scK = nX + nZuit + nZvit + nZvi0;
                arma::subview<double> zTk = ws.v1Scratch.submat(0, 0, nZui0-1, 0);
                zTk = zui0.row(tt).t();
                // K-K
                double f_sh = s * dBeta1(tt) * std::abs(ui0Norm);
                // double scalarK = 0.25 * sigu0 * (sigu0 * f_sigmah_sigmah(tt) + f_sh);
                double scalarK = 0.25 * sigu0 * (sigu0 * f_sigmah_sigmah(tt) - f_sh); // FIXED 2025-12-20
                H.submat(scK, scK, scK+nZui0-1, scK+nZui0-1) = scalarK * (zTk * zTk.t());
                // K-U
                if (nZuit > 0) {
                    arma::subview<double> zTu = ws.v2Scratch.submat(0, 0, nZuit-1, 0);
                    zTu = zuit.row(tt).t();
                    double scalar = 0.5 * sigu0 * (dLambdadSigmaH(tt) * 0.5 * lam + dSigmadSigmaH(tt) * 0.5 * sig2u / sig);
                    H.submat(nX, scK, nX+nZuit-1, scK+nZui0-1) = scalar * (zTu * zTk.t());
                    H.submat(scK, nX, scK+nZui0-1, nX+nZuit-1) = scalar * (zTk * zTu.t());
                }
                // K-V
                if (nZvit > 0) {
                    arma::subview<double> zTv = ws.v2Scratch.submat(0, 0, nZvit-1, 0);
                    zTv = zvit.row(tt).t();
                    double scalar = 0.5 * sigu0 * (dLambdadSigmaH(tt) * -0.5 * lam + dSigmadSigmaH(tt) * 0.5 * sig2v / sig);
                    int scV = nX + nZuit;
                    H.submat(scV, scK, scV+nZvit-1, scK+nZui0-1) = scalar * (zTv * zTk.t());
                    H.submat(scK, scV, scK+nZui0-1, scV+nZvit-1) = scalar * (zTk * zTv.t());
                }
                // K-G
                if (nZvi0 > 0) {
                    arma::subview<double> zTg = ws.v2Scratch.submat(0, 0, nZvi0-1, 0);
                    zTg = zvi0.row(tt).t();
                    // double scalar = s * (-(std::abs(ui0Norm)/sig) * (vi0Norm/sig) * term1(tt)) * 0.5 * sigu0 * 0.5 * sigv0;
                    double scalar = s * ((std::abs(ui0Norm)/sig) * (vi0Norm/sig) * term1(tt)) * 0.5 * sigu0 * 0.5 * sigv0; // FIXED 2025-12-20
                    int scG = nX + nZuit + nZvit;
                    H.submat(scG, scK, scG+nZvi0-1, scK+nZui0-1) = scalar * (zTg * zTk.t());
                    H.submat(scK, scG, scK+nZui0-1, scG+nZvi0-1) = scalar * (zTk * zTg.t());
                }
            }
            // accumulate the hessian over time period
            hessSumNt += H;
        }
        // ---- finish iterating over nT - still iterating over nsim ----
        h_bar_i += (Qir(0, ns) * hessSumNt);
    }
    // finalise the hessian matrix
    if (hessOut) {
        // we have gir, h_bar_i, and g_bar_i
        // iterate over number of simulations again
        // view to the second component of the hessian matrix
        arma::subview<double> hess_comp_2 = ws.hessAccC2.submat(arma::span(0, k - 1), arma::span(0, k - 1));
        // since accumlating it, need to reset 0 to prevent contamination from previous iterations
        hess_comp_2.zeros();
        for (int ns = 0; ns < nsim; ns++) {
            // weighted average of the outer product of the gradient less g_bar_i
            hess_comp_2 += Qir(0, ns) * (
                (gir.row(ns) - g_bar_i).t() * (gir.row(ns) - g_bar_i)
            );
        }
        // write to destination buffer
        *hessOut = h_bar_i + hess_comp_2;
    }
}

// Truncated Normal
template <typename TY, typename TX, typename TZmuit, typename TZuit, typename TZvit, typename TZvi0>
void ESASfaTreBase::internalAnalyticJacHessTN(
    const unsigned int idx,
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
    ThreadContext* ctx = getContext();
    if (!ctx->treBaseJacHess)
        ctx->treBaseJacHess = std::make_unique<thread_cache::WSInternalAnalyticJacHess>();
    thread_cache::WSInternalAnalyticJacHess& ws = *ctx->treBaseJacHess;
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;

    int drawIdxPos = this->obsUseSameHaltonDraw ? 0 : idx;
    if (draws.n_rows <= drawIdxPos)
        throw std::invalid_argument("internalAnalyticJacHessTN: draws too small");
    if (draws.n_cols < this->nsim)
        throw std::invalid_argument("internalAnalyticJacHessTN: draws: not enough columns");

    const auto& Qir = QirIn.get_ref();
    int ns = this->nsim;
    int nT = (int)yIn.get_ref().n_rows;
    if (nT == 0) return;

    // extract parameters
    arma::dcolvec b_x     = dataObj.paramX(par);
    if (!dataObj.paramZuit(par))  throw std::invalid_argument("internalAnalyticJacHessTN: missing 'zuit'");
    arma::dcolvec b_zuit  = dataObj.paramZuit(par).value();
    if (!dataObj.paramZvit(par))  throw std::invalid_argument("internalAnalyticJacHessTN: missing 'zvit'");
    arma::dcolvec b_zvit  = dataObj.paramZvit(par).value();
    if (!dataObj.paramZvi0(par))  throw std::invalid_argument("internalAnalyticJacHessTN: missing 'zvi0'");
    arma::dcolvec b_zvi0  = dataObj.paramZvi0(par).value();
    if (!dataObj.paramZmuit(par)) throw std::invalid_argument("internalAnalyticJacHessTN: missing 'zmuit'");
    arma::dcolvec b_zmuit = dataObj.paramZmuit(par).value();

    int nX = (int)b_x.n_rows, nZuit = (int)b_zuit.n_rows, nZvit = (int)b_zvit.n_rows;
    int nZvi0 = (int)b_zvi0.n_rows, nZmuit = (int)b_zmuit.n_rows;
    int k    = nX + nZuit + nZvit + nZvi0 + nZmuit;
    int maxT = dataObj.getMaxT();
    int maxZ = std::max({nX, nZuit, nZvit, nZvi0, nZmuit});
    int requiredRows = (nT > maxT) ? nT : maxT;

    ws.ensureSize(requiredRows, k, maxZ, ns, nX, nZuit, nZvit, 0, nZvi0, nZmuit);

    // copy inputs into aligned workspace buffers
    arma::subview<double> y     = ws.y.rows(arma::span(0, nT-1));
    arma::subview<double> x     = ws.x.submat(arma::span(0,nT-1), arma::span(0,std::max(nX-1,0)));
    arma::subview<double> zuit  = ws.zuit.submat(arma::span(0,nT-1), arma::span(0,std::max(nZuit-1,0)));
    arma::subview<double> zvit  = ws.zvit.submat(arma::span(0,nT-1), arma::span(0,std::max(nZvit-1,0)));
    arma::subview<double> zvi0  = ws.zvi0.submat(arma::span(0,nT-1), arma::span(0,std::max(nZvi0-1,0)));
    arma::subview<double> zmuit = ws.zmuit.submat(arma::span(0,nT-1), arma::span(0,std::max(nZmuit-1,0)));
    y     = yIn.get_ref();
    x     = xIn.get_ref();
    zuit  = zuitIn.get_ref();
    zvit  = zvitIn.get_ref();
    zvi0  = zvi0In.get_ref();
    zmuit = zmuitIn.get_ref();

    // output buffer setup
    if (jacOut) {
        if (jacOut->n_rows < (unsigned)nT || jacOut->n_cols < (unsigned)k) jacOut->set_size(nT, k);
        jacOut->zeros();
    }
    if (hessOut) {
        if (hessOut->n_rows < (unsigned)k || hessOut->n_cols < (unsigned)k) hessOut->set_size(k, k);
    }

    // =========================================================
    // INVARIANT QUANTITIES (atlas slot assignments for TN)
    //   a1:xb  a2:sig2u  a3:sigu  a38:sig2v  a4:sigv
    //   a5:sig2v0  a6:sigv0  a9:sig2  a10:sig  a11:lam
    //   a12:lam/sig  a40:mu  a41:C  a42:Mc  a43:Dc  a44:term2C
    // =========================================================
    arma::subview<double> xb = ws.a1.submat(0,0,nT-1,0);
    arma::subview<double> sigma2uit = ws.a2.submat(0,0,nT-1,0);
    arma::subview<double> sigmauit = ws.a3.submat(0,0,nT-1,0);
    arma::subview<double> sigma2vit = ws.a38.submat(0,0,nT-1,0);
    arma::subview<double> sigmavit = ws.a4.submat(0,0,nT-1,0);
    arma::subview<double> sigma2vi0 = ws.a5.submat(0,0,nT-1,0);
    arma::subview<double> sigmavi0 = ws.a6.submat(0,0,nT-1,0);
    arma::subview<double> sigma2 = ws.a9.submat(0,0,nT-1,0);
    arma::subview<double> sigma = ws.a10.submat(0,0,nT-1,0);
    arma::subview<double> lambda = ws.a11.submat(0,0,nT-1,0);
    arma::subview<double> lamDivSig = ws.a12.submat(0,0,nT-1,0);
    // TN-specific invariants
    arma::subview<double> mu = ws.a40.submat(0,0,nT-1,0);
    arma::subview<double> C_tn = ws.a41.submat(0,0,nT-1,0);  // C = mu/sigma_u
    arma::subview<double> Mc = ws.a42.submat(0,0,nT-1,0);  // phi(C)/Phi(C)
    arma::subview<double> Dc = ws.a43.submat(0,0,nT-1,0);  // Mc'
    arma::subview<double> term2C = ws.a44.submat(0,0,nT-1,0);  // Mc + C*Dc

    xb = x * b_x;
    sigma2uit = esautils::processSig2Term(b_zuit, zuit);
    sigmauit = arma::sqrt(sigma2uit);
    sigma2vit = esautils::processSig2Term(b_zvit, zvit);
    sigmavit = arma::sqrt(sigma2vit);
    sigma2vi0 = esautils::processSig2Term(b_zvi0, zvi0);
    sigmavi0 = arma::sqrt(sigma2vi0);
    sigma2 = sigma2uit + sigma2vit;
    sigma = arma::sqrt(sigma2);
    lambda = sigmauit / sigmavit;
    lamDivSig = lambda / sigma;
    mu = zmuit * b_zmuit;
    C_tn = mu / sigmauit;           // C = mu / sigma_u

    // Mc = phi(C)/Phi(C) with tail approximation for small Phi(C)
    for (int i = 0; i < nT; ++i) {
        double c = C_tn(i);
        if (c < -37.0) {
            double c2 = c*c;
            Mc(i) = -c - (1.0/c) + (2.0/(c*c2));
        } else {
            Mc(i) = arma::normpdf(c) / arma::normcdf(c);
        }
    }
    Dc = -(C_tn % Mc) - arma::pow(Mc, 2.0);   // D_C = M_C'
    term2C = Mc + C_tn % Dc;                        // M_C + C*D_C

    // =========================================================
    // PER-DRAW SLOT ASSIGNMENTS
    //   a13:eps  a14:A=(eps+mu)/sig  a16:B  a17:Mb  a18:dBeta1
    //   a19:f_sig  a20:f_lam  a21:f_mu  a22:Db  a23:term1
    //   a24:term2_sig  a25:term2_lam
    // =========================================================
    arma::subview<double> eps = ws.a13.submat(0,0,nT-1,0);
    arma::subview<double> A_tn = ws.a14.submat(0,0,nT-1,0);
    arma::subview<double> B_tn = ws.a16.submat(0,0,nT-1,0);
    arma::subview<double> Mb = ws.a17.submat(0,0,nT-1,0);
    arma::subview<double> dBeta1 = ws.a18.submat(0,0,nT-1,0);
    arma::subview<double> f_sigma = ws.a19.submat(0,0,nT-1,0);
    arma::subview<double> f_lambda = ws.a20.submat(0,0,nT-1,0);
    arma::subview<double> f_mu = ws.a21.submat(0,0,nT-1,0);
    arma::subview<double> Db = ws.a22.submat(0,0,nT-1,0);
    arma::subview<double> term1 = ws.a23.submat(0,0,nT-1,0);
    arma::subview<double> term2_sig = ws.a24.submat(0,0,nT-1,0);
    arma::subview<double> term2_lam = ws.a25.submat(0,0,nT-1,0);

    // =========================================================
    // SECOND-ORDER SCALAR SLOT ASSIGNMENTS (also per-draw)
    //   a26:f_bb1  a27:dBdSig  a28:dBdLam  a29:dBdMu
    //   a30:dBdSigV0  a31:dSigdSigV0  a32:dLamdSigV0  a33:f_swsw
    //   a34:f_sigsig  a35:f_lamlam  a36:f_siglam
    //   a37:dMudSig  a45:dMudLam  a39:f_mumu
    // =========================================================
    arma::subview<double> f_bb1 = ws.a26.submat(0,0,nT-1,0);
    arma::subview<double> dBdSig = ws.a27.submat(0,0,nT-1,0);
    arma::subview<double> dBdLam = ws.a28.submat(0,0,nT-1,0);
    arma::subview<double> dBdMu = ws.a29.submat(0,0,nT-1,0);
    arma::subview<double> dBdSigV0 = ws.a30.submat(0,0,nT-1,0);
    arma::subview<double> dSigdSigV0 = ws.a31.submat(0,0,nT-1,0);
    arma::subview<double> dLamdSigV0 = ws.a32.submat(0,0,nT-1,0);
    arma::subview<double> f_swsw = ws.a33.submat(0,0,nT-1,0);
    arma::subview<double> f_sigsig = ws.a34.submat(0,0,nT-1,0);
    arma::subview<double> f_lamlam = ws.a35.submat(0,0,nT-1,0);
    arma::subview<double> f_siglam = ws.a36.submat(0,0,nT-1,0);
    arma::subview<double> dMudSig = ws.a37.submat(0,0,nT-1,0);
    arma::subview<double> dMudLam = ws.a45.submat(0,0,nT-1,0);
    arma::subview<double> f_mumu = ws.a39.submat(0,0,nT-1,0);

    // =========================================================
    // ACCUMULATORS
    // =========================================================
    arma::subview<double> jac = ws.jac.submat(arma::span(0,nT-1), arma::span(0,k-1));
    arma::subview<double> h_bar_i = ws.h_bar_i.submat(arma::span(0,k-1), arma::span(0,k-1));
    h_bar_i.zeros();
    arma::subview<double> gir = ws.gir.submat(arma::span(0,ns-1), arma::span(0,k-1));
    arma::subview<double> g_bar_i = ws.g_bar_i.submat(arma::span(0,0), arma::span(0,k-1));
    g_bar_i.zeros();

    // =========================================================
    // SIMULATION LOOP
    // =========================================================
    // reusable scratch for jacobian column construction
    arma::subview<double> term_scratch = ws.a15.submat(0,0,nT-1,0);
    for (int r = 0; r < ns; ++r) {
        double vi0Norm = draws(drawIdxPos, r);
        // --- composite residual ---
        eps = y - xb - (sigmavi0 * vi0Norm);
        // --- A = (eps + s*mu) / sigma  [includes s for cost/production frontier] ---
        A_tn = (eps + (this->s * mu)) / sigma;
        // --- B = mu/(lambda*sigma) - s*eps*lambda/sigma ---
        //       = (mu - s*eps*lambda^2) / (lambda*sigma)
        B_tn = mu / (lambda % sigma) - (this->s * eps % lambda / sigma);
        // --- Mb = phi(B)/Phi(B) with tail approximation ---
        for (int i = 0; i < nT; ++i) {
            double b = B_tn(i);
            if (b < -37.0) {
                double b2 = b*b;
                Mb(i) = -b - (1.0/b) + (2.0/(b*b2));
            } else {
                Mb(i) = arma::normpdf(b) / arma::normcdf(b);
            }
        }
        // --- first-order scalars ---
        // dBeta1 = A/sigma + s*lambda*Mb/sigma = (eps+s*mu)/sigma2 + s*lambda*Mb/sigma
        dBeta1  = A_tn / sigma + (this->s * lambda % Mb / sigma);
        // f_sigma = (1/sigma) * (A^2 - Mb*B + Mc*C - 1)
        f_sigma = (1.0 / sigma) % (arma::pow(A_tn, 2.0) - Mb % B_tn + Mc % C_tn - 1.0);
        // f_lambda = -Mb*(mu + s*eps*lambda^2)/(lambda^2*sigma) + Mc*C/(lambda*(1+lambda^2))
        f_lambda = -Mb % (mu + (this->s * eps % arma::pow(lambda, 2.0))) / (arma::pow(lambda, 2.0) % sigma)
                   + Mc % C_tn / (lambda % (1.0 + arma::pow(lambda, 2.0)));
        // f_mu = -s*A/sigma + Mb/(lambda*sigma) - Mc/sigma_u
        f_mu = -(this->s * A_tn) / sigma + Mb / (lambda % sigma) - Mc / sigmauit;
        // --- fill jacobian row-by-row ---
        // canonical parameter order: [beta | zmuit | zuit | zvit | zvi0]
        // beta block: x * dBeta1
        jac.cols(0, nX-1) = esautils::sweepMatrixElementwise(x, 1, dBeta1, "*");
        int curPos = nX;
        // zmuit block: f_mu * zmuit
        if (nZmuit > 0) {
            jac.cols(curPos, curPos+nZmuit-1) = esautils::sweepMatrixElementwise(zmuit, 1, f_mu, "*");
            curPos += nZmuit;
        }
        // zuit block: [f_lambda*(lam/2) + f_sigma*(sig2u/(2*sig))] * zuit
        if (nZuit > 0) {
            term_scratch = (f_lambda % (0.5 * lambda)) + (f_sigma % (0.5 * sigma2uit / sigma));
            jac.cols(curPos, curPos+nZuit-1) = esautils::sweepMatrixElementwise(zuit, 1, term_scratch, "*");
            curPos += nZuit;
        }
        // zvit block: [f_lambda*(-lam/2) + f_sigma*(sig2v/(2*sig))] * zvit
        if (nZvit > 0) {
            term_scratch = (f_lambda % (-0.5 * lambda)) + (f_sigma % (0.5 * sigma2vit / sigma));
            jac.cols(curPos, curPos+nZvit-1) = esautils::sweepMatrixElementwise(zvit, 1, term_scratch, "*");
            curPos += nZvit;
        }
        // zvi0 block: dBeta1 * vi0Norm * (sigmav0/2) * zvi0
        if (nZvi0 > 0) {
            term_scratch = dBeta1 * vi0Norm % (0.5 * sigmavi0);
            jac.cols(curPos, curPos+nZvi0-1) = esautils::sweepMatrixElementwise(zvi0, 1, term_scratch, "*");
            curPos += nZvi0;
        }
        // accumulate weighted jacobian
        if (jacOut) {
            arma::subview<double> jacOutView = jacOut->submat(arma::span(0,nT-1), arma::span(0,k-1));
            jacOutView += (Qir(0, r) * jac);
        }
        if (!hessOut) continue;
        // weighted gradient for this simulation
        gir.row(r)  = esamath::colSum(jac);
        g_bar_i += (Qir(0, r) * gir.row(r));
        // =====================================================
        // SECOND-ORDER SCALARS (all nT x 1, per draw)
        // =====================================================
        Db = -(B_tn % Mb) - arma::pow(Mb, 2.0);          // D_B = M_B'
        term1 = 1.0 - (arma::pow(lambda, 2.0) % Db);
        term2_sig = Mb + B_tn % Db;                              // M_B + B*D_B
        // term2_lam = M_B - D_B*(mu + s*eps*lam^2)/(lam*sigma)
        term2_lam = Mb - Db % (mu + (this->s * eps % arma::pow(lambda, 2.0))) / (lambda % sigma);
        // f_beta_beta1 = -term1 / sigma^2
        f_bb1 = -(term1 / sigma2);
        // dBetadSigma = -(1/sig2)*(2A + s*lam*term2_sig)
        dBdSig = -(1.0 / sigma2) % (2.0 * A_tn + (this->s * lambda % term2_sig));
        // dBetadLambda = (s/sig)*term2_lam
        dBdLam = (this->s / sigma) % term2_lam;
        // dBetadMu = s*(1 + D_B) / sigma^2  [= ∂(dBeta1)/∂mu]
        dBdMu = this->s * (1.0 + Db) / sigma2;
        // dBetadSigmaV0 = (vi0/sig2)*(lam^2*D_B - 1)
        dBdSigV0 = (vi0Norm / sigma2) % (arma::pow(lambda, 2.0) % Db - 1.0);
        // dSigmadSigmaV0 = (vi0/sig2)*(-2A - lam*term2_sig)
        dSigdSigV0 = (vi0Norm / sigma2) % (-2.0 * A_tn - lambda % term2_sig);
        // dLambdadSigmaV0 = (vi0/sig)*term2_lam
        dLamdSigV0 = (vi0Norm / sigma) % term2_lam;
        // f_sigmaw_sigmaw = -(vi0/sig)^2 * term1
        f_swsw = -arma::pow(vi0Norm / sigma, 2.0) % term1;
        // f_sigma_sigma = -(1/sig2)*(3A^2 - B*(B*Db+2Mb) + C*(C*Dc+2Mc) - 1)
        // Note: C-group has +C(CDc+2Mc) because ln Phi(C) enters with a minus sign
        f_sigsig = -(1.0 / sigma2) % (
                        3.0 * arma::pow(A_tn, 2.0)
                        - B_tn % (B_tn % Db + 2.0 * Mb)
                        + C_tn % (C_tn % Dc + 2.0 * Mc)
                        - 1.0
                   );

        // f_lambda_lambda: D_B*(dB/dlam)^2 + M_B*(d2B/dlam2) - D_C*(dC/dlam)^2 - M_C*(d2C/dlam2)
        // where dB/dlam = -(mu+s*eps*lam^2)/(lam^2*sig), dC/dlam = -C/(lam*(1+lam^2))
        //       d2B/dlam2 = 2*mu/(lam^3*sig)
        //       d2C/dlam2 = mu*(2+3*lam^2)/(sig*lam^3*(1+lam^2)^{3/2})
        // reuse term_scratch for dB/dlam, then dC/dlam
        term_scratch = -(mu + (this->s * eps % arma::pow(lambda,2.0))) / (arma::pow(lambda,2.0) % sigma);
        f_lamlam     = Db % arma::pow(term_scratch, 2.0);
        f_lamlam    += Mb % (2.0 * mu) / (arma::pow(lambda, 3.0) % sigma);
        term_scratch = -C_tn / (lambda % (1.0 + arma::pow(lambda,2.0)));
        f_lamlam    -= Dc % arma::pow(term_scratch, 2.0);
        f_lamlam    -= Mc % (mu % (2.0 + 3.0 * arma::pow(lambda, 2.0)))
                        / (sigma % arma::pow(lambda, 3.0) % arma::pow(1.0 + arma::pow(lambda, 2.0), 1.5));
        // f_sigma_lambda = (mu+s*eps*lam^2)*term2_sig/(lam^2*sig^2)
        //                  - C*term2C/(sig*lam*(1+lam^2))
        f_siglam = (mu + (this->s * eps % arma::pow(lambda,2.0))) % term2_sig
                        / (arma::pow(lambda,2.0) % sigma2)
                   - C_tn % term2C / (sigma % lambda % (1.0 + arma::pow(lambda,2.0)));
        // dMudSigma = 2sA/sig2 - term2_sig/(lam*sig2) + term2C/(sigu*sig)
        dMudSig  = 2.0 * this->s * A_tn / sigma2
                   - term2_sig / (lambda % sigma2)
                   + term2C / (sigmauit % sigma);
        // dMudLambda = -D_B*(mu+s*eps*lam^2)/(lam^3*sig^2)
        //             - M_B/(lam^2*sig)
        //             + (D_C*C + M_C)/(sigu*lam*(1+lam^2))
        dMudLam  = -Db % (mu + (this->s * eps % arma::pow(lambda,2.0)))
                        / (arma::pow(lambda,3.0) % sigma2)
                   - Mb / (arma::pow(lambda,2.0) % sigma)
                   + (Dc % C_tn + Mc) / (sigmauit % lambda % (1.0 + arma::pow(lambda,2.0)));
        // f_mu_mu = -1/sig2 + D_B/(lam^2*sig^2) - D_C/sig2_u
        f_mumu   = -1.0 / sigma2
                   + Db / (arma::pow(lambda,2.0) % sigma2)
                   - Dc / sigma2uit;
        // =====================================================
        // HESSIAN BLOCK ASSEMBLY (inner loop over time periods)
        // =====================================================
        arma::subview<double> hessSumNt = ws.hessSumNt.submat(arma::span(0,k-1), arma::span(0,k-1));
        hessSumNt.zeros();
        // Hessian offsets matching canonical order: [beta | zmuit | zuit | zvit | zvi0]
        const int offM = nX; // zmuit
        const int offU = nX + nZmuit;  // zuit
        const int offV = nX + nZmuit + nZuit; // zvit
        const int offG = nX + nZmuit + nZuit + nZvit; // zvi0

        for (int tt = 0; tt < nT; ++tt) {
            arma::dmat& H = ws.h;
            H.zeros();

            double lam = lambda(tt), sig = sigma(tt), sig2 = sigma2(tt);
            double sig2u = sigma2uit(tt), sig2v = sigma2vit(tt);
            double sigv0 = sigmavi0(tt), sigu = sigmauit(tt);
            double dSu = 0.5 * sig2u / sig, dLu = 0.5 * lam;
            double dSv = 0.5 * sig2v / sig, dLv = -0.5 * lam;
            // --- scratch vectors for outer products ---
            arma::subview<double> xT    = ws.v1Scratch.submat(0,0,nX-1,0);
            xT = x.row(tt).t();
            // 1. beta-beta
            H.submat(0,0,nX-1,nX-1) = f_bb1(tt) * (xT * xT.t());
            // 2. beta-zuit
            if (nZuit > 0) {
                arma::subview<double> zT = ws.v2Scratch.submat(0,0,nZuit-1,0);
                zT = zuit.row(tt).t();
                double sc = dBdLam(tt)*dLu + dBdSig(tt)*dSu;
                H.submat(0,offU,nX-1,offU+nZuit-1) = sc * (xT*zT.t());
                H.submat(offU,0,offU+nZuit-1,nX-1) = sc * (zT*xT.t());
            }
            // 3. beta-zvit
            if (nZvit > 0) {
                arma::subview<double> zT = ws.v2Scratch.submat(0,0,nZvit-1,0);
                zT = zvit.row(tt).t();
                double sc = dBdLam(tt)*dLv + dBdSig(tt)*dSv;
                H.submat(0,offV,nX-1,offV+nZvit-1) = sc * (xT*zT.t());
                H.submat(offV,0,offV+nZvit-1,nX-1) = sc * (zT*xT.t());
            }
            // 4. beta-zvi0
            if (nZvi0 > 0) {
                arma::subview<double> zT = ws.v2Scratch.submat(0,0,nZvi0-1,0);
                zT = zvi0.row(tt).t();
                double sc = dBdSigV0(tt) * 0.5 * sigv0;
                H.submat(0,offG,nX-1,offG+nZvi0-1) = sc * (xT*zT.t());
                H.submat(offG,0,offG+nZvi0-1,nX-1) = sc * (zT*xT.t());
            }
            // 5. beta-zmuit (NEW)
            if (nZmuit > 0) {
                arma::subview<double> zT = ws.v2Scratch.submat(0,0,nZmuit-1,0);
                zT = zmuit.row(tt).t();
                double sc = dBdMu(tt);
                H.submat(0,offM,nX-1,offM+nZmuit-1) = sc * (xT*zT.t());
                H.submat(offM,0,offM+nZmuit-1,nX-1) = sc * (zT*xT.t());
            }
            // 6. zuit-zuit
            if (nZuit > 0) {
                arma::subview<double> zT = ws.v1Scratch.submat(0,0,nZuit-1,0);
                zT = zuit.row(tt).t();
                double sc = f_lamlam(tt)*dLu*dLu + f_sigsig(tt)*dSu*dSu
                          + 2.0*f_siglam(tt)*dLu*dSu
                          + f_lambda(tt)*0.25*lam
                          + f_sigma(tt)*(sig2u*(sig2u + 2.0*sig2v))/(4.0*std::pow(sig,3.0));
                H.submat(offU,offU,offU+nZuit-1,offU+nZuit-1) = sc * (zT*zT.t());
            }
            // 7. zvit-zvit
            if (nZvit > 0) {
                arma::subview<double> zT = ws.v1Scratch.submat(0,0,nZvit-1,0);
                zT = zvit.row(tt).t();
                double sc = f_lamlam(tt)*dLv*dLv + f_sigsig(tt)*dSv*dSv
                          + 2.0*f_siglam(tt)*dLv*dSv
                          + f_lambda(tt)*0.25*lam
                          + f_sigma(tt)*(sig2v*(sig2v + 2.0*sig2u))/(4.0*std::pow(sig,3.0));
                H.submat(offV,offV,offV+nZvit-1,offV+nZvit-1) = sc * (zT*zT.t());
            }
            // 8. zuit-zvit
            if (nZuit > 0 && nZvit > 0) {
                arma::subview<double> zTu = ws.v1Scratch.submat(0,0,nZuit-1,0);
                zTu = zuit.row(tt).t();
                arma::subview<double> zTv = ws.v2Scratch.submat(0,0,nZvit-1,0);
                zTv = zvit.row(tt).t();
                double sc = f_lamlam(tt)*dLu*dLv + f_sigsig(tt)*dSu*dSv
                          + f_siglam(tt)*(dLu*dSv + dSu*dLv)
                          + f_lambda(tt)*(-0.25*lam)
                          + f_sigma(tt)*(-(sig2u*sig2v))/(4.0*std::pow(sig,3.0));
                H.submat(offU,offV,offU+nZuit-1,offV+nZvit-1) = sc * (zTu*zTv.t());
                H.submat(offV,offU,offV+nZvit-1,offU+nZuit-1) = sc * (zTv*zTu.t());
            }
            // 9. zvi0-zvi0
            if (nZvi0 > 0) {
                arma::subview<double> zT = ws.v1Scratch.submat(0,0,nZvi0-1,0);
                zT = zvi0.row(tt).t();
                double sc = 0.25 * sigv0 * (sigv0 * f_swsw(tt) + dBeta1(tt) * vi0Norm);
                H.submat(offG,offG,offG+nZvi0-1,offG+nZvi0-1) = sc * (zT*zT.t());
            }
            // 10. zvi0-zuit
            if (nZvi0 > 0 && nZuit > 0) {
                arma::subview<double> zTg = ws.v1Scratch.submat(0,0,nZvi0-1,0);
                zTg = zvi0.row(tt).t();
                arma::subview<double> zTu = ws.v2Scratch.submat(0,0,nZuit-1,0);
                zTu = zuit.row(tt).t();
                double sc = 0.5*sigv0*(dLamdSigV0(tt)*dLu + dSigdSigV0(tt)*dSu);
                H.submat(offU,offG,offU+nZuit-1,offG+nZvi0-1) = sc * (zTu*zTg.t());
                H.submat(offG,offU,offG+nZvi0-1,offU+nZuit-1) = sc * (zTg*zTu.t());
            }
            // 11. zvi0-zvit
            if (nZvi0 > 0 && nZvit > 0) {
                arma::subview<double> zTg = ws.v1Scratch.submat(0,0,nZvi0-1,0);
                zTg = zvi0.row(tt).t();
                arma::subview<double> zTv = ws.v2Scratch.submat(0,0,nZvit-1,0);
                zTv = zvit.row(tt).t();
                double sc = 0.5*sigv0*(dLamdSigV0(tt)*dLv + dSigdSigV0(tt)*dSv);
                H.submat(offV,offG,offV+nZvit-1,offG+nZvi0-1) = sc * (zTv*zTg.t());
                H.submat(offG,offV,offG+nZvi0-1,offV+nZvit-1) = sc * (zTg*zTv.t());
            }
            // 12. zmuit-zmuit (NEW)
            if (nZmuit > 0) {
                arma::subview<double> zT = ws.v1Scratch.submat(0,0,nZmuit-1,0);
                zT = zmuit.row(tt).t();
                double sc = f_mumu(tt);
                H.submat(offM,offM,offM+nZmuit-1,offM+nZmuit-1) = sc * (zT*zT.t());
            }

            // 13. zmuit-zuit (NEW)
            if (nZmuit > 0 && nZuit > 0) {
                arma::subview<double> zTm = ws.v1Scratch.submat(0,0,nZmuit-1,0);
                zTm = zmuit.row(tt).t();
                arma::subview<double> zTu = ws.v2Scratch.submat(0,0,nZuit-1,0);
                zTu = zuit.row(tt).t();
                double sc = dMudSig(tt)*dSu + dMudLam(tt)*dLu;
                H.submat(offM,offU,offM+nZmuit-1,offU+nZuit-1) = sc * (zTm*zTu.t());
                H.submat(offU,offM,offU+nZuit-1,offM+nZmuit-1) = sc * (zTu*zTm.t());
            }
            // 14. zmuit-zvit (NEW)
            if (nZmuit > 0 && nZvit > 0) {
                arma::subview<double> zTm = ws.v1Scratch.submat(0,0,nZmuit-1,0);
                zTm = zmuit.row(tt).t();
                arma::subview<double> zTv = ws.v2Scratch.submat(0,0,nZvit-1,0);
                zTv = zvit.row(tt).t();
                double sc = dMudSig(tt)*dSv + dMudLam(tt)*dLv;
                H.submat(offM,offV,offM+nZmuit-1,offV+nZvit-1) = sc * (zTm*zTv.t());
                H.submat(offV,offM,offV+nZvit-1,offM+nZmuit-1) = sc * (zTv*zTm.t());
            }
            // 15. zmuit-zvi0 (NEW): s_mu_v0 = s*(1 + Db)*vi0*sigv0/(2*sig2)
            if (nZmuit > 0 && nZvi0 > 0) {
                arma::subview<double> zTm = ws.v1Scratch.submat(0,0,nZmuit-1,0);
                zTm = zmuit.row(tt).t();
                arma::subview<double> zTg = ws.v2Scratch.submat(0,0,nZvi0-1,0);
                zTg = zvi0.row(tt).t();
                double sc = this->s * (1.0 + Db(tt)) * vi0Norm * sigv0 / (2.0 * sig2);
                H.submat(offM,offG,offM+nZmuit-1,offG+nZvi0-1) = sc * (zTm*zTg.t());
                H.submat(offG,offM,offG+nZvi0-1,offM+nZmuit-1) = sc * (zTg*zTm.t());
            }
            hessSumNt += H;
        } // end time loop

        h_bar_i += (Qir(0, r) * hessSumNt);
    } // end simulation loop

    // =====================================================
    // FINALISE OUTPUTS
    // =====================================================
    if (hessOut) {
        arma::subview<double> hess_comp2 = ws.hessAccC2.submat(arma::span(0,k-1), arma::span(0,k-1));
        hess_comp2.zeros();
        for (int r = 0; r < ns; ++r) {
            hess_comp2 += Qir(0,r) * (
                (gir.row(r) - g_bar_i).t() * (gir.row(r) - g_bar_i)
            );
        }
        *hessOut = h_bar_i + hess_comp2;
    }
}

// =========================================================
//                      EXPLICIT INSTANTIATIONS
// =========================================================
// weightForDensForPanel
template void ESASfaTreBase::weightFromDensForPanel<arma::dmat>(const arma::Base<double, arma::dmat>&, arma::dmat&) const;
template void ESASfaTreBase::weightFromDensForPanel<arma::subview<double>>(const arma::Base<double, arma::subview<double>>&, arma::dmat&) const;

// non-contiguous first, for when copies are made
template void ESASfaTreBase::internalAnalyticJacHess<
    arma::dmat,     // TY
    arma::dmat,     // TX
    arma::dmat,     // TZmuit
    arma::dmat,     // TZuit
    arma::dmat,     // TZvit
    arma::dmat,     // TZui0
    arma::dmat     // TZvi0
>(
    const unsigned int,
    const ESASfaModelType mT,
    const arma::dcolvec&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const std::optional<arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::dmat&,
    const std::optional<arma::dmat>&,
    const arma::Base<double, arma::subview<double>>&,
    arma::dmat*,
    arma::dmat*
) const;
// contigous data - using subviews
template void ESASfaTreBase::internalAnalyticJacHess<
    arma::subview<double>,     // TY
    arma::subview<double>,     // TX
    arma::subview<double>,     // TZmuit
    arma::subview<double>,     // TZuit
    arma::subview<double>,     // TZvit
    arma::subview<double>,     // TZui0
    arma::subview<double>     // TZvi0
>(
    const unsigned int,
    const ESASfaModelType mT,
    const arma::dcolvec&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const std::optional<arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    const std::optional<arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    arma::dmat*,
    arma::dmat*
) const;
// this one is when TZui0 is really empty, but with default template set to arma::dmat
template void ESASfaTreBase::internalAnalyticJacHess<
    arma::subview<double>,     // TY
    arma::subview<double>,     // TX
    arma::subview<double>,     // TZmuit
    arma::subview<double>,     // TZuit
    arma::subview<double>,     // TZvit
    arma::dmat,     // TZui0
    arma::subview<double>     // TZvi0
>(
    const unsigned int,
    const ESASfaModelType mT,
    const arma::dcolvec&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const std::optional<arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    const std::optional<arma::dmat>&,
    const arma::Base<double, arma::subview<double>>&,
    arma::dmat*,
    arma::dmat*
) const;
// internalAnalyticJacHess - mixed types for LCM (dmat y/x, subview z)
template void ESASfaTreBase::internalAnalyticJacHess<
    arma::dmat,            // TY
    arma::dmat,            // TX
    arma::subview<double>, // TZmuit
    arma::subview<double>, // TZuit
    arma::subview<double>, // TZvit
    arma::dmat,            // TZui0
    arma::subview<double>  // TZvi0
>(
    const unsigned int,
    const ESASfaModelType mT,
    const arma::dcolvec&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const std::optional<arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    const std::optional<arma::dmat>&,
    const arma::Base<double, arma::subview<double>>&,
    arma::dmat*,
    arma::dmat*
) const;

// internalAnalyticJacHessTN - non-contiguous (dmat) variant
template void ESASfaTreBase::internalAnalyticJacHessTN<
    arma::dmat,            // TY
    arma::dmat,            // TX
    arma::dmat,            // TZmuit
    arma::dmat,            // TZuit
    arma::dmat,            // TZvit
    arma::dmat             // TZvi0
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::dmat&,
    const arma::Base<double, arma::subview<double>>&,
    arma::dmat*,
    arma::dmat*
) const;

// internalAnalyticJacHessTN - contiguous (subview) variant
template void ESASfaTreBase::internalAnalyticJacHessTN<
    arma::subview<double>, // TY
    arma::subview<double>, // TX
    arma::subview<double>, // TZmuit
    arma::subview<double>, // TZuit
    arma::subview<double>, // TZvit
    arma::subview<double>  // TZvi0
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    const arma::Base<double, arma::subview<double>>&,
    arma::dmat*,
    arma::dmat*
) const;

// internalAnalyticJacHessTN - mixed types for LCM (dmat y/x, subview z)
template void ESASfaTreBase::internalAnalyticJacHessTN<
    arma::dmat,            // TY
    arma::dmat,            // TX
    arma::subview<double>, // TZmuit
    arma::subview<double>, // TZuit
    arma::subview<double>, // TZvit
    arma::subview<double>  // TZvi0
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    const arma::Base<double, arma::subview<double>>&,
    arma::dmat*,
    arma::dmat*
) const;
