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

#include "sfa/ESASfaLcTreEM.hpp"
#include "utils/lcmutils.hpp"
#include "data/ESADataLCM.hpp"
#include "math/esamath.hpp"
#include "math/GaussHermite.hpp"
#include "utils/log/logs.hpp"
#include "utils/esautils.hpp"
#include "optim/ESAGlobalOptimParams.hpp"

#ifdef WITHDLIB
#include "optim/DlibWrapper.hpp"
#include "utils/dlib2arma.h"
#include <dlib/optimization.h>
#endif

// ============================================================
// ESASfaClassMStepObj — thin ESASfaBase wrapper for the M-step
// ============================================================

ESASfaClassMStepObj::ESASfaClassMStepObj(
    std::shared_ptr<ESASfaLcTre> ghqModel,
    unsigned int classIdx,
    arma::dvec tau_c,
    arma::drowvec ghqLogWeights,
    ESASfaLcTre::FrozenAGHQNodes frozen
) : ESASfaBase(ghqModel->getLcmDataObj(), 1.0),
    ghqModel_(ghqModel),
    classIdx_(classIdx),
    tau_c_(std::move(tau_c)),
    ghqLogWeights_(std::move(ghqLogWeights)),
    frozen_(std::move(frozen))
{
    ESADataPanelLCM& lcmData = *ghqModel_->getLcmDataObj();
    perClassParams_ = lcmData.getNX() + lcmData.getNZmuit()
                    + lcmData.getNZuit() + lcmData.getNZvit() + lcmData.getNZvi0();
}

double ESASfaClassMStepObj::operator()(
    const arma::dcolvec& classParams,
    const bool /*exceptNotFinite*/
) const
{
    return ghqModel_->weightedClassLLAndGradHess(
        classIdx_, classParams, tau_c_, ghqLogWeights_, frozen_,
        nullptr, nullptr);
}

void ESASfaClassMStepObj::gradHess(
    const arma::dcolvec& classParams,
    const bool /*exceptNotFinite*/,
    arma::dmat* gradOut,
    arma::dmat* hessOut,
    arma::dmat* /*jacOut*/
) const
{
    // delegates to weightedClassLLAndGradHess which computes the analytical
    // (not BHHH) Hessian via the inlined internalAnalyticJacHess loop.
    ghqModel_->weightedClassLLAndGradHess(
        classIdx_, classParams, tau_c_, ghqLogWeights_, frozen_,
        gradOut, hessOut);
}

arma::dcolvec ESASfaClassMStepObj::startingValues() const
{
    return arma::zeros<arma::dcolvec>(perClassParams_);
}

double ESASfaClassMStepObj::getN() const
{
    return static_cast<double>(ghqModel_->getLcmDataObj()->getNids());
}


// ============================================================
// ESASfaLcTreEM
// ============================================================

ESASfaLcTreEM::ESASfaLcTreEM(
    std::shared_ptr<ESASfaLcTre> srcModel,
    const ESAOptimParams& params,
    unsigned int printLevel
) : nQuadPts_(params.em_nquad_pts),
    maxIter_(params.em_max_iter),
    segMaxIter_(params.em_seg_max_iter),
    classMaxIter_(params.em_class_max_iter),
    tol_(params.em_tol),
    segTol_(params.em_seg_tol),
    printLevel_(printLevel),
    optimParams_(params)
{
    ghq::validateNQuadPts(nQuadPts_);

    ESADataPanelLCM& lcmData = *srcModel->getLcmDataObj();
    nClasses_ = lcmData.getNClasses();
    nSeg_     = lcmData.getNSeg();
    // Build the GHQ draws matrix: (nFirms x nQuadPts), stores sqrt(2)*x_k per row.
    unsigned int nFirms = static_cast<unsigned int>(lcmData.getNids());
    arma::dmat ghqDrawsMat = ghq::buildDrawsMatrix(nFirms, nQuadPts_);
    auto ghqDrawsPtr = std::make_shared<arma::dmat>(std::move(ghqDrawsMat));
    // GHQ log-weights (1 x nQuadPts)
    ghqLogWeights_ = ghq::logWeightsRow(nQuadPts_);
    // construct the GHQ model: same data pointers as srcModel, but nsim = nQuadPts
    // and the Halton draws slot holds our GHQ draws matrix.
    auto lcmDataPtr  = srcModel->getLcmDataObj();
    auto helperDataPtr = std::dynamic_pointer_cast<ESADataPanel>(srcModel->getDataObj());
    if (!helperDataPtr)
        throw std::runtime_error("ESASfaLcTreEM: could not retrieve helper ESADataPanel");

    int prodCost = static_cast<int>(srcModel->getProdCost());
    ghqModel_ = std::make_shared<ESASfaLcTre>(
        lcmDataPtr,
        helperDataPtr,
        prodCost,
        static_cast<int>(nQuadPts_), // nsim = nQuadPts
        1234, // seed (unused — draws are injected)
        ghqDrawsPtr
    );
}

// ---- E-step ----

arma::dmat ESASfaLcTreEM::eStep(const arma::dcolvec& params) const
{
    return ghqModel_->computePosteriorsGHQ(params, ghqLogWeights_);
}

// ---- segmentation M-step: NR on multinomial logit ----
// via Greene 2012 textbook
// maximize: Q(γ) = Σ_i Σ_c τ_ic log π_ic(γ)
// gradient (w.r.t. α_c, c=0..C-2):   ∇_c Q = Σ_i (τ_ic - π_ic) z_i
// Hessian  (w.r.t. α_c, α_d):        H_cd   = -Σ_i π_ic(δ_cd - π_id) z_i z_i'
arma::dcolvec ESASfaLcTreEM::mStepSeg(
    const arma::dcolvec& segParams,
    const arma::dmat& tau
) const
{
    ESADataPanelLCM& lcmData = *ghqModel_->getLcmDataObj();
    unsigned int nFirms = static_cast<unsigned int>(lcmData.getNids());
    unsigned int C = nClasses_;
    unsigned int nSeg = nSeg_;
    int totalSegParams = static_cast<int>((C - 1) * nSeg);

    if (totalSegParams == 0) return segParams;

    bool threaded = ESAGlobalOptimParams::GetInstance()->optimThreaded;
    arma::dcolvec gamma = segParams;

    for (unsigned int iter = 0; iter < segMaxIter_; ++iter) {
        std::vector<arma::dmat> gradVec(nFirms);
        std::vector<arma::dmat> hessVec(nFirms);
        // lambda function for the dataCallable method
        auto inner = [&](const unsigned int idx,
                         const auto& seg,
                         const auto& /*y*/, const auto& /*x*/,
                         const auto& /*zmuit*/, const auto& /*zuit*/,
                         const auto& /*zvit*/, const auto& /*zvi0*/,
                         const auto& /*transition*/) {
            arma::rowvec z_i = arma::conv_to<arma::rowvec>::from(seg.row(0));
            // calculate the class probabilities [the multinominal logit]
            arma::dcolvec pi_i = lcmutils::computeClassProbs(z_i, gamma, C);
            // store gradient & hessian
            arma::dmat g_i(totalSegParams, 1, arma::fill::zeros);
            arma::dmat h_i(totalSegParams, totalSegParams, arma::fill::zeros);
            // calculate gradient
            for (unsigned int c = 0; c < C - 1; ++c) {
                double diff = tau(idx, c) - pi_i(c);
                int offset = static_cast<int>(c * nSeg);
                g_i.rows(offset, offset + nSeg - 1) += diff * z_i.t();
            }
            // calculate hessian matrix - C-1 x C-1 in shape
            arma::dcolvec zz = z_i.t();
            arma::dmat zzT = zz * z_i;
            for (unsigned int c = 0; c < C - 1; ++c) {
                // outer loop 
                int rc = static_cast<int>(c * nSeg);
                for (unsigned int d = 0; d < C - 1; ++d) {
                    // inner loop
                    int rd = static_cast<int>(d * nSeg);
                    // check if a diagonal element
                    double isDiag = (c == d) ? 1.0 : 0.0;
                    double coef = -pi_i(c) * (isDiag - pi_i(d));
                    h_i.submat(rc, rd, rc + nSeg - 1, rd + nSeg - 1) += coef * zzT;
                }
            }
            // move into vector of gradients & hessians
            gradVec[idx] = std::move(g_i);
            hessVec[idx] = std::move(h_i);
            arma::dmat dummy(1, 1, arma::fill::zeros);
            return dummy;
        };
        lcmData.dataCallable(inner, nullptr, nullptr, false, false, threaded);
        // full gradient & hessian across sample
        arma::dcolvec grad = arma::vectorise(esautils::sumMatricies<double>(gradVec));
        arma::dmat hess = esautils::sumMatricies<double>(hessVec);
        // Levenberg-Marquardt modification to prevent div 0 errors
        arma::dmat hessReg = hess - 1e-8 * arma::eye(totalSegParams, totalSegParams);
        arma::dcolvec step;
        bool ok = arma::solve(step, hessReg, grad, arma::solve_opts::no_approx);
        if (!ok) {
            step = grad * 0.01;
        }
        gamma -= step;
        // check for convergence (e.g., grad = 0)
        double gnorm = arma::norm(grad, 2);
        if (gnorm < segTol_) break;
    }
    return gamma;
}

// ---- Class M-step: dlib trust-region with analytical Hessian ----

arma::dcolvec ESASfaLcTreEM::mStepClass(
    unsigned int c,
    const arma::dcolvec& classParams,
    const arma::dmat& tau,
    const arma::dcolvec& fullParams
) const
{
#if defined(WITHDLIB)
    arma::dvec tau_c = tau.col(c);
    // freeze the AGHQ nodes using θ_old (fullParams) — must not be recomputed from θ_proposal
    // inside the TR loop, otherwise the gradient ignores ∂(log σ_i*)/∂θ_c and dlib
    // will see a gradient inconsistent with the objective.
    ESASfaLcTre::FrozenAGHQNodes frozen = ghqModel_->computeFrozenAGHQ(c, fullParams, ghqLogWeights_);
    // instantiate the m-step object
    auto classObj = std::make_shared<ESASfaClassMStepObj>(
        ghqModel_, c, tau_c, ghqLogWeights_, std::move(frozen)
    );
    // optimization parameters
    ESAOptimParams classOptimParams = optimParams_;
    classOptimParams.maxit = classMaxIter_;
    // instantiate the dlib wrapper which handles all the printing for the iterations
    DlibWrapper fn(classObj, HessianCalcMethod::ANALYTICAL,
                   printLevel_ >= 2 ? 1u : 0u,
                   true, 0, false, static_cast<int>(classMaxIter_));
    DlibWrapper::column_vector p(classParams.n_rows);
    for (arma::uword i = 0; i < classParams.n_rows; ++i) p(i) = classParams(i);
    // try the m-step optimization using dlib's Trusted Region solver
    try {
        auto stopStrategy = dlib::gradient_norm_stop_strategy(classOptimParams.grad_err_tol);
        dlib::find_max_trust_region(stopStrategy, fn, p, classOptimParams.tr_radius);
    } catch (const std::exception& e) {
        if (printLevel_ >= 1)
            ESALogger::logger()->warn("mStepClass c={}: dlib TR exception: {}", c, e.what());
        return classParams;
    }
    // extract the result
    arma::dcolvec result(classParams.n_rows);
    for (arma::uword i = 0; i < classParams.n_rows; ++i) result(i) = p(i);
    return result;
#else
    throw std::runtime_error("ESASfaLcTreEM::mStepClass requires dlib (WITHDLIB)");
#endif
}

// ---- Observed log-likelihood ----

double ESASfaLcTreEM::observedLL(const arma::dcolvec& params) const
{
    return ghqModel_->computeObservedLLGHQ(params, ghqLogWeights_);
}

// ---- Main EM loop ----

EMResult ESASfaLcTreEM::run(const arma::dcolvec& startVals) const
{
    // derefence ptr to the LCM Data class
    ESADataPanelLCM& lcmData = *ghqModel_->getLcmDataObj();
    unsigned int C = nClasses_;
    unsigned int nSeg = nSeg_;
    int totalSegParams = static_cast<int>((C - 1) * nSeg);
    // get the starting values
    arma::dcolvec params = startVals;
    // vector to store all the log-likelihood scores per iteration
    arma::dvec llHist(maxIter_ + 1, arma::fill::zeros);
    // calculate the prev log-likelihood score based on starting values
    double prevLL = observedLL(params);
    llHist(0) = prevLL;
    if (printLevel_ >= 1) {
        ESALogger::logger()->info("EM init: ll = {:.6f}", prevLL);
    }
    bool converged = false;
    unsigned int iter = 0;
    for (iter = 0; iter < maxIter_; ++iter) {
        // ---- E-step ----
        arma::dmat tau = eStep(params);  // (nFirms x nClasses)
        // ---- M-step: segmentation ----
        if (totalSegParams > 0) {
            arma::dcolvec segParams = lcmData.paramSeg(params);
            arma::dcolvec newSegParams = mStepSeg(segParams, tau);
            params.rows(0, totalSegParams - 1) = newSegParams;
        }
        // ---- M-step: per-class frontier parameters ----
        int classParamStart = totalSegParams;
        int nX = lcmData.getNX();
        int nZmuit = lcmData.getNZmuit();
        int nZuit = lcmData.getNZuit();
        int nZvit = lcmData.getNZvit();
        int nZvi0 = lcmData.getNZvi0();
        int perClass = nX + nZmuit + nZuit + nZvit + nZvi0;
        // for each latent class, use TR to find the new proposed class parameters (theta_c)
        for (unsigned int c = 0; c < C; ++c) {
            int offset = classParamStart + static_cast<int>(c) * perClass;
            arma::dcolvec classParams = params.rows(offset, offset + perClass - 1);
            arma::dcolvec newClassParams = mStepClass(c, classParams, tau, params);
            params.rows(offset, offset + perClass - 1) = newClassParams;
        }
        // ---- convergence check ----
        double curLL = observedLL(params);
        llHist(iter + 1) = curLL;
        // print some information on current loop iter
        if (printLevel_ >= 1) {
            ESALogger::logger()->info("EM iter {:>3}: ll = {:.6f}  Δll = {:.2e}", iter + 1, curLL, curLL - prevLL);
        }
        // check for convergency if the difference in LL scores between this and previous iteration
        // is less than the tolerance
        if (std::abs(curLL - prevLL) < tol_) {
            converged = true;
            prevLL = curLL;
            ++iter;
            break;
        }
        if (curLL < prevLL - 1e-6 && printLevel_ >= 1) {
            ESALogger::logger()->warn("EM iter {}: LL decreased by {:.2e} — check M-step convergence", iter + 1, prevLL - curLL);
        }
        prevLL = curLL;
    }
    // if didn't converge, log this iin the output
    if (!converged && printLevel_ >= 1) {
        ESALogger::logger()->warn(
            "EM did not converge after {} iterations (|ΔLL|={:.2e} > tol={:.2e})",
            iter, std::abs(observedLL(params) - prevLL), tol_
        );
    }
    // build the result object and return
    EMResult result;
    result.params = params;
    result.logLike = prevLL;
    result.nIter = static_cast<int>(iter);
    result.converged = converged;
    result.llHistory = llHist.rows(0, iter);
    return result;
}
