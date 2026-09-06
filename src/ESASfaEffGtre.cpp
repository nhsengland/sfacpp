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

#include "efficiencies/ESASfaEffGtre.hpp"
#include "data/ESADataPanel.hpp"
#include "math/HaltonSeq.hpp"
#include "utils/ThreadContext.hpp"
#include "thread_cache/ESASfaEffGtreTC.hpp"
#include "math/ghkestim.hpp"
#include "math/GHK.hpp"
#include "math/primes.hpp"

/// Constructor
ESASfaEffGtre::ESASfaEffGtre(
    const std::shared_ptr<ESADataBase>& dataObjPtr,
    const double s
) : dataObjPtr(dataObjPtr), s(s), mT(dataObjPtr->getModelType()) {
    // double check the data object
    if (!dynamic_cast<ESADataPanel*>(dataObjPtr.get())){
        throw std::invalid_argument("data object is not of type ESADataPanel (panel)");
    }
};

/// Overall efficiency score calculation - this execute the gtrePanelEfficiency method on each panel
std::unique_ptr<ESASfaEffScores> ESASfaEffGtre::efficiencyScores(
    const arma::dcolvec& par,
    const int nsim,
    const int haltonStart,
    const int seed,
    const bool threaded
){
    // dereference ptr to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    int maxT = dataObj.getMaxT();
    ESASfaModelFamily mF = ESAEnums::getModelFamily(this->mT);
    // halton draw for ghk simulation - this should be nsim x macT
    int hDrawSize = maxT + 1;
    // const arma::dmat haltonDraw = halton::halton<arma::Mat<double>>(nsim, hDrawSize, haltonStart, false, 7, true, true);
    // arma::dmat hseq = HaltonSeq::generate(haltonBase, (nsim * hDrawSize), 1000, true, seed, false);
    // hseq = arma::reshape(hseq, nsim, hDrawSize);
    // draw independent halton sequences
    if (hDrawSize >= 100008) throw std::runtime_error("Not enough prime numbers available");
    arma::dmat hseq(nsim, hDrawSize);
    for (int j = 0; j < hDrawSize; j++) {
        int currBase = my100008Primes[haltonStart + j];
        hseq.col(j) = HaltonSeq::generate(currBase, nsim, 1000, true, seed, false);
    }
    // parameter extraction in main thread
    arma::dcolvec b_x = dataObj.paramX(par);
    // these return optionals - if not retun an empty colvec
    arma::dcolvec b_zuit = dataObj.paramZuit(par).value_or(arma::dcolvec());
    arma::dcolvec b_zvit = dataObj.paramZvit(par).value_or(arma::dcolvec());
    arma::dcolvec b_zvi0 = dataObj.paramZvi0(par).value_or(arma::dcolvec());
    // // handle GTRE vs TRE
    arma::dcolvec b_zui0;
    if (mF == ESASfaModelFamily::TRE) {
        b_zui0 = arma::dcolvec(1, arma::fill::zeros);
    } else {
        b_zui0 = dataObj.paramZui0(par).value_or(arma::dcolvec());
    }
    auto inner = [
        this, &par, &hseq, &b_x, &b_zuit, &b_zvit, &b_zvi0, &b_zui0, &mF
    ](
        const unsigned int idx,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zui0,
        const auto& zvi0,
        arma::dmat* persistentEff,
        arma::dmat* transientEff
    ) {
        int nT = y.get_ref().n_rows;
        arma::dmat outBuf(nT + 1, 1);
        if (mF == ESASfaModelFamily::TRE) {
            // remember this calls the gtre panel method, so that will affect the size of the output
            this->trePanelEfficiency(
                idx, par, y, x, zuit.value(), zvit.value(), zvi0.value(), hseq, outBuf,
                b_x, b_zuit, b_zvit, b_zvi0, b_zui0
            );
        } else if (mF == ESASfaModelFamily::GTRE) {
            if (!zui0) throw std::invalid_argument("missing 'zui0'");
            this->gtrePanelEfficiency(
                idx, par, y, x, zuit.value(), zvit.value(), zui0.value(), zvi0.value(), hseq, outBuf,
                b_x, b_zuit, b_zvit, b_zvi0, b_zui0
            );
        } else {
            throw std::runtime_error("unexpected model type");
        }
        // out is a this is a nT + 1 column;
        // first element is the persistent efficiency
        if (persistentEff) {
            // repeat it for nT observations, just to help ordering down the line
            *persistentEff = arma::dmat(nT, 1, arma::fill::value(outBuf(0, 0)));
        }
        // remainer of the column vector is the transient efficiency
        if (transientEff) {
            *transientEff = outBuf.rows(arma::span(1, nT));
        }
    };
    arma::dmat persist, transient;
    if (threaded) {
        dataObj.panelCallableThreaded(inner, &persist, &transient, false, false);
    } else {
        dataObj.panelCallable(inner, &persist, &transient, false, false);
    }
    ESASfaEffScores scores;
    scores.persistent = std::make_optional<arma::dmat>(persist);
    scores.transient = std::make_optional<arma::dmat>(transient);
    // since its quite large, flush the TLS
    ThreadContext* ctx = getContext();
    if (ctx->gtreEffPanel) ctx->gtreEffPanel.reset();
    return std::make_unique<ESASfaEffScores>(scores);
}


/// Calculate the efficiency score for the TRE model for a given panel
template <typename TX, typename TY, typename TZuit, typename TZvit, typename TZvi0>
void ESASfaEffGtre::trePanelEfficiency(
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
) const
{
    ThreadContext* ctx = getContext();
    if (!ctx->treEffPanel) {
        ctx->treEffPanel = std::make_unique<thread_cache_eff_gtre::WSTrePanelEfficiency>();
    }
    thread_cache_eff_gtre::WSTrePanelEfficiency& ws = *ctx->treEffPanel;
    int nT = yIn.get_ref().n_rows;
    ws.ensureSize(nT);
    ws.zui0Dummy.zeros();
    arma::subview<double> zui0 = ws.zui0Dummy.rows(arma::span(0, nT - 1));
    // call GTRE method
    this->gtrePanelEfficiency(idx, par, yIn, xIn, zuitIn, zvitIn, zui0, zvi0In, haltonDraw, out, b_x, b_zuit, b_zvit, b_zvi0, b_zui0);
}

/// Calculate the efficiency score for the GTRE model for a given panel- this is the main method for the efficiency score calculation itself
template <typename TY, typename TX, typename TZuit, typename TZvit, typename TZui0, typename TZvi0>
void ESASfaEffGtre::gtrePanelEfficiency(
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
) const {
    // persistent buffer storage per thread
    // NOTE: this is different from the others, in that all matricies have to be the
    // correct size - we dont use subviews. This was because all the matrix algebra and operations
    // were causing an issue with SIMD optimization in -O3 (i think). For unbalanced panels, there
    // will therefore be a small slowdown as memory is allocated, whereas for balanced panels this
    // will perform at its optimium.
    ThreadContext* ctx = getContext();
    if (!ctx->gtreEffPanel) {
        ctx->gtreEffPanel = std::make_unique<thread_cache_eff_gtre::WSGtrePanelEfficiency>();
    }
    thread_cache_eff_gtre::WSGtrePanelEfficiency& ws = *ctx->gtreEffPanel;
    // dereference ptr to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    bool isTRE = (ESAEnums::getModelFamily(this->mT) == ESASfaModelFamily::TRE);
    // get various sizes...
    // get the number of periods for this panel
    unsigned int nT = yIn.get_ref().n_rows;
    // int maxT = dataObj.getMaxT();
    int nX = dataObj.getNX(), nZuit = dataObj.getNZuit(), nZvit = dataObj.getNZvit();
    int nZui0 = dataObj.getNZui0(), nZvi0 = dataObj.getNZvi0();
    if (nT == 0) return;
    // make sure correct size matricies in scratchpad
    ws.ensureSize(nT, nX, nZuit, nZvit, nZui0, nZvi0);
    // copy into the aligned buffer
    ws.y = yIn.get_ref();
    ws.x = xIn.get_ref();
    ws.zuit = zuitIn.get_ref();
    ws.zvit = zvitIn.get_ref();
    ws.zui0 = zui0In.get_ref();
    ws.zvi0 = zvi0In.get_ref();
    // view for the destination buffer (remember +1 - for the persistent inefficiency)
    ws.giveUp.fill(std::numeric_limits<double>::quiet_NaN());
    // 
    ws.from.fill(-std::numeric_limits<double>::infinity());
    // calculate xb
    ws.xb = ws.x * b_x;
    // sigma2uit (variance of the time-varying inefficency)
    // ws.sigma2uit = esautils::processSig2Term(b_zuit, ws.zuit);
    ws.sigma2uit = arma::exp(ws.zuit * b_zuit);
    // sigma2vit (variance of the stochastic noise component)
    // ws.sigma2vit = esautils::processSig2Term(b_zvit, ws.zvit);
    ws.sigma2vit = arma::exp(ws.zvit * b_zvit);
    // sigma2ui0 (variance of the time-invariant inefficency)
    // to handle the case where we are estimating the efficiences for TRE, set sigma2ui0 to inperceptibly zero
    // if (dataObj.getDataStructure() != ESASfaDataStructure::GTRE){
    if (ESAEnums::getModelFamily(dataObj.getModelType()) != ESASfaModelFamily::GTRE){
        ws.sigma2ui0 = arma::dmat(nT, 1, arma::fill::value(std::numeric_limits<double>::epsilon()));
    } else {
        // sigma2ui0 = dlib::exp(zui0 * b_zui0);
        // ws.sigma2ui0 = esautils::processSig2Term(b_zui0, ws.zui0);
        ws.sigma2ui0 = arma::exp(ws.zui0 * b_zui0);
    }
    // again, since all the should be the same, should we just select the first element
    double sigma2ui0_val = ws.sigma2ui0(0, 0);
    // sigma2vi0 (variance of the time-invariant stochastic noise component)
    // dlib::matrix<double> sigma2vi0 = dlib::exp(zvi0 * b_zvi0);
    // ws.sigma2vi0 = esautils::processSig2Term(b_zvi0, ws.zvi0);
    ws.sigma2vi0 = arma::exp(ws.zvi0 * b_zvi0);
    // since all the values should be the same, should we just select the first element
    double sigma2vi0_val = ws.sigma2vi0(0, 0);
    // calculate eps (residual)
    ws.eps = ws.y - ws.xb;
    // halton draws
    // arma::dmat draws = halton::halton<arma::Mat<double>>(nsim, (nT + 1), haltonStart, false, seed, true, true);
    // Colombi et al 2014 propose the following to estimate the persistent and time-varying inefficiencies
    // E(exp(t'u_i|y_i)) = \Phi_bar_Ti+1(R_i.r_i - \Lambda_i.t, \Lambda_i) / \Phi_bar_Ti+1(R_i.r_i, \Lambda_i) x exp(0.5t'\Lambda_i.t - t'R_i.r_i)
    // This should be in principle, similar to JLMS (Jondrow et al 1982)
    // ---- A ----
    // so A = [1_{T_i} I_{T_i}] which is a 1_{T_i} column vector followed by an identity matrix
    // arma::dmat A(nT, nT + 1);
    // set the first column to 1
    // A.col(0) = arma::dcolvec(nT, arma::fill::ones);
    ws.A.col(0).ones();
    // set the rest of the columns to the identity matrix
    // A.submat(arma::span(0, nT - 1), arma::span(1, nT)) = arma::dmat(nT, nT, arma::fill::eye);
    ws.A.submat(arma::span(0, nT - 1), arma::span(1, nT)).eye();
    // adjust for cost or production function (s needs to be -1 for cost, +1 for production)
    // A = -1 * s * A;
    ws.A *= (-1.0 * s);
    // ---- V_i ----
    // the diagonal elements of V_i are [sigma2ui0, sigma2uit]
    // std::vector<arma::dmat> vStacks = {sigma2ui0_val, sigma2uit};
    // arma::dmat vSigs = esautils::stackMatricies<double>(vStacks, true);
    ws.vSigs(0) = sigma2ui0_val;
    ws.vSigs.submat(arma::span(1, nT), arma::span(0, 0)) = ws.sigma2uit;
    // arma::dmat V = vSigs.diag();
    // arma::dmat V = arma::dmat(nT + 1, nT + 1, arma::fill::zeros);
    ws.V.zeros();
    ws.V.diag() = arma::clamp(ws.vSigs, 1e-10, arma::datum::inf);
    ws.Vinv = arma::inv(ws.V);
    // ---- Sigma_i ----
    // Sigma_i = sigma2vit.I_{T_i} + sigma2vi0.1_{T_i}.1_{T_i}'
    ws.sigP1.eye();
    ws.sigP1.diag() = ws.sigma2vit;
    ws.sigP2.fill(sigma2vi0_val);
    ws.Sigma = ws.sigP1 + ws.sigP2;
    ws.Sigma.diag() += 1e-10;
    ws.SigmaInv = arma::inv(ws.Sigma);
    /// ---- Lambda_i ----
    // two possible expressions for this
    // 1. Lambda_i = (V^-1+ A' * Sigma^-1 * A)^-1
    // 2. Lambda_i = V - V * A'(Sigma + A * V * A')^-1 * A * V
    bool lamIntSuccess = true;
    if (!isTRE) {
        try {
            ws.lamInternal = ws.Vinv + ws.A.t() * ws.SigmaInv * ws.A;
            ws.Lambda = arma::inv(ws.lamInternal);
        } catch (const std::exception& e)
        {
            lamIntSuccess = false;
        }
    }
    // TRE path (or fallback): Woodbury form, more robust for small V
    arma::dmat SigAVAt;
    if (!lamIntSuccess || isTRE) {
        SigAVAt = ws.Sigma + ws.A * ws.V * ws.A.t();
        // regularize if near-singular
        SigAVAt.diag() += 1e-10;
        try {
            ws.Lambda = ws.V - ws.V * ws.A.t() * arma::inv(SigAVAt) * ws.A * ws.V;
        } catch (const std::exception& e)
        {
            out = ws.giveUp;
            return;
        }
    }

    // ---- R_i ----
    // 1. Lambda_i * A' * Sigma^-1
    // 2. V * A' * (Sigma + A * V * A')^-1
    bool riSuccess = true;
    if (!isTRE) {
        try {
            ws.R = ws.Lambda * ws.A.t() * ws.SigmaInv;
        } catch (const std::exception& e)
        {
            riSuccess = false;
        }
    }
    if (!riSuccess || isTRE) {
        if (SigAVAt.empty()) {
            SigAVAt = ws.Sigma + ws.A * ws.V * ws.A.t();
            SigAVAt.diag() += 1e-10;
        }
        try {
            ws.R = ws.V * ws.A.t() * arma::inv(SigAVAt);
        } catch (const std::exception& e)
        {
            out = ws.giveUp;
            return;
        }
    }
    // ---- t ----
    // t is an identity matrix of size (T_i + 1)
    // arma::dmat t(nT + 1, nT + 1, arma::fill::eye);
    ws.t.eye();
    // ---- E(exp(t'u_i|y_i)) ----
    // define the lower bounds for the GHK estimatation
    // arma::dmat from(nT + 1, 1, arma::fill::value(-std::numeric_limits<double>::infinity()));
    // already filled from in the workspace since it never changes.
    // cholesky decomposition of Lambda_i (nT + 1) x (nT + 1)
    ws.Lambda = arma::symmatu(ws.Lambda);
    try {
        ws.LambdaChol = arma::chol(ws.Lambda, "lower");
    } catch (...) {
        // Cholesky failed — Lambda is near-singular (e.g. boundary variance estimate).
        // clamp eigenvalues to a floor and reconstruct.
        constexpr double eigFloor = 1e-8;
        arma::vec eigval;
        arma::mat eigvec;
        if (arma::eig_sym(eigval, eigvec, ws.Lambda)) {
            eigval = arma::clamp(eigval, eigFloor, eigval.max());
            ws.Lambda = eigvec * arma::diagmat(eigval) * eigvec.t();
            ws.Lambda = arma::symmatu(ws.Lambda);
            try {
                ws.LambdaChol = arma::chol(ws.Lambda, "lower");
            } catch (...) {
                out = ws.giveUp;
                return;
            }
        } else {
            out = ws.giveUp;
            return;
        }
    }
    // the denominator (the CDF part) - R_i.r_i
    // Rr is (nT + 1) x 1
    ws.Rr = ws.R * ws.eps;
    // view for the halton draw, should be (nsim x nT + 1)
    int nsim = haltonDraw.n_rows;
    arma::subview<double> drawView = haltonDraw.submat(arma::span(0, nsim - 1), arma::span(0, nT));
    // ghk estimation of \Phi_bar_Ti+1(R_i.r_i, \Lambda_i)
    // double phiBarDenom = esamath::ghk_estim(ws.LambdaChol, ws.from, ws.Rr, drawView, true, nsim);
    double logPhiBarDenom = ghk::estim(ws.LambdaChol, ws.from, ws.Rr, drawView);
    // next part operates in a loop of the different time periods; -t  is a row of the identity matrix t
    // matrix to store results in
    // arma::dmat expTu_i(nT + 1, 1);
    for (int tt = 0; tt < (nT + 1); ++tt){
        // note this looks slightly different from the R version, but should produce same result; i think
        // its to do with how subsetting in R differs perhaps
        // extract -t
        ws.tRow = ws.t.row(tt).t();
        // arma::dmat tRow = t.row(tt).t();
        // arma::dmat tRowTrans = tRow.t();
        // calculate the xponent part of the expression (0.5t'\Lambda_i.t - t'R_i.r_i)
        ws.tLamt = ws.tRow.t() * ws.Lambda * ws.tRow;
        ws.tRr = ws.tRow.t() * ws.Rr;
        // ws.expPart = arma::exp(0.5 * ws.tLamt - ws.tRr);
        double term1 = 0.5 * ws.tLamt(0, 0);
        double term2 = ws.tRr(0, 0);
        double logExpPart = term1 - term2;
        // arma::dmat tLamt = tRowTrans * Lambda * tRow;
        // arma::dmat tRr = tRowTrans * Rr;
        // arma::dmat expPart = arma::exp(0.5 * tLamt - tRr);
        // // calculate the numerator (the CDF part) - \Phi_bar_Ti+1(R_i.r_i - \Lambda_i.t, \Lambda_i)
        // phiBarNumTo is (nT + 1) x 1
        // arma::dmat phiBarNumTo = Rr - (Lambda * tRow);
        ws.phiBarNumTo = ws.Rr - (ws.Lambda * ws.tRow);
        // double phiBarNum = esamath::ghk_estim(ws.LambdaChol, ws.from, ws.phiBarNumTo, drawView, true, nsim);
        double logPhiBarNum = ghk::estim(ws.LambdaChol, ws.from, ws.phiBarNumTo, drawView);
        // // calculate the final expression
        // 1x1
        // arma::dmat expTu_it = (phiBarNum / phiBarDenom) * expPart;
        // REPLACED 2025-12-21
        // ws.expTu_it = (phiBarNum / phiBarDenom) * ws.expPart;
        // ws.expTu_i.submat(arma::span(tt, tt), arma::span(0, 0)) = ws.expTu_it;
        double logEff = logPhiBarNum - logPhiBarDenom + logExpPart;
        ws.expTu_it = std::exp(logEff);
        if (ws.expTu_it(0,0) > 1.0) ws.expTu_it = 1.0;
        if (ws.expTu_it(0,0) < 0.0) ws.expTu_it = 0.0;
        ws.expTu_i.submat(arma::span(tt, tt), arma::span(0, 0)) = ws.expTu_it;
    }
    // set relevant part in output buffer (nT + 1) x 1
    out = ws.expTu_i;
    // out = expTu_i;
}

// ====================================================
//              TEMPLATE INSTANTISATION
// ====================================================
template void ESASfaEffGtre::gtrePanelEfficiency<
    arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat
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
    arma::dmat&,
    const arma::dcolvec&,
    const arma::dcolvec&,
    const arma::dcolvec&,
    const arma::dcolvec&,
    const arma::dcolvec&
) const;
template void ESASfaEffGtre::gtrePanelEfficiency<
    arma::subview<double>,
    arma::subview<double>,
    arma::subview<double>,
    arma::subview<double>,
    arma::subview<double>,
    arma::subview<double>
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
    arma::dmat&,
    const arma::dcolvec&,
    const arma::dcolvec&,
    const arma::dcolvec&,
    const arma::dcolvec&,
    const arma::dcolvec&
) const;
template void ESASfaEffGtre::trePanelEfficiency<
    arma::dmat, arma::dmat, arma::dmat, arma::dmat, arma::dmat
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::Base<double, arma::dmat>&,
    const arma::dmat&,
    arma::dmat&,
    const arma::dcolvec&,
    const arma::dcolvec&,
    const arma::dcolvec&,
    const arma::dcolvec&,
    const arma::dcolvec&
) const;
template void ESASfaEffGtre::trePanelEfficiency<
    arma::subview<double>,
    arma::subview<double>,
    arma::subview<double>,
    arma::subview<double>,
    arma::subview<double>
>(
    const unsigned int,
    const arma::dcolvec&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::Base<double, arma::subview<double>>&,
    const arma::dmat&,
    arma::dmat&,
    const arma::dcolvec&,
    const arma::dcolvec&,
    const arma::dcolvec&,
    const arma::dcolvec&,
    const arma::dcolvec&
) const;