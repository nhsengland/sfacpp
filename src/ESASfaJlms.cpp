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
#include <optional>
#include "efficiencies/ESASfaJlms.hpp"
#include "data/ESADataPanel.hpp"
#include "utils/enums.hpp"
#include "utils/esautils.hpp"
#include "math/esamath.hpp"
#include "math/esandist.hpp"
#include "math/HaltonSeq.hpp"
#include "math/primes.hpp"

/// Constructor
ESASfaJlms::ESASfaJlms(std::shared_ptr<ESADataBase>& dataObjPtr, const double s)
    : dataObjPtr(dataObjPtr), s(s)
{
    // check data object is panel
    if (!dynamic_cast<ESADataPanel*>(dataObjPtr.get())){
        throw std::invalid_argument("data object is not of type ESADataPanel (panel)");
    }
}

/// Calculate the ineffiency score (conditional on eps) based on Jondrow et al (1982)
arma::dmat ESASfaJlms::ineffPredJlms(
    const arma::dcolvec& par,
    const int nsim,
    const int haltonStart,
    const int seed
)
{
    // dereference pointer 
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    // model type
    ESASfaModelType mT = dataObj.getModelType();
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    std::optional<arma::dmat> draws_opt = std::nullopt;
    if (mF == ESASfaModelFamily::TRE) {
        unsigned int nFirms = static_cast<unsigned int>(dataObj.getNids());
        arma::dmat draws(nFirms, nsim);
        for (int j = 0; j < nsim; j++) {
            int currBase = my100008Primes[haltonStart + j];
            draws.col(j) = HaltonSeq::generate(currBase, nFirms, 1000, true, seed, false);
        }
        draws_opt = std::make_optional(std::move(draws));
    }
    std::function<arma::dmat(
        const unsigned int,
        const arma::dcolvec&,
        const arma::dmat&,
        const std::optional<arma::dmat>,
        const std::optional<arma::dmat>,
        const std::optional<arma::dmat>,
        const std::optional<arma::dmat>,
        const std::optional<arma::dmat>
    )> inner = [this, &par, &nsim, &draws_opt, &mT, &mF](
        const unsigned int idx,
        const arma::dcolvec& y,
        const arma::dmat& x,
        const std::optional<arma::dmat>&zmuit,
        const std::optional<arma::dmat>&zuit,
        const std::optional<arma::dmat>&zvit,
        const std::optional<arma::dmat>&zui0,
        const std::optional<arma::dmat>&zvi0
    ){
        if (mF == ESASfaModelFamily::TFE) {
            // true fixed effects — not yet implemented
        } else if (mF == ESASfaModelFamily::TRE) {
            if (!draws_opt) throw std::runtime_error("missing 'draws_opt'");
            if (!zuit || !zvit || !zvi0) throw std::invalid_argument("missing one of 'zuit', 'zvit', 'zvi0'");
            if (mT == ESASfaModelType::TRE_HNORM_ZUIT) {
                return ESASfaJlms::ineffPredJlmsTrePanelHalfNormal(
                    idx, par, y, x, zuit.value(), zvit.value(), zvi0.value(), this->s, nsim, draws_opt.value()
                );
            } else if (mT == ESASfaModelType::TRE_TNORM_ZUIT) {
                if (!zmuit) throw std::invalid_argument("missing 'zmuit'");
                return ESASfaJlms::ineffPredJlmsTrePanelTruncNormal(
                    idx, par, y, x, zuit.value(), zvit.value(), zvi0.value(), zmuit.value(), this->s, nsim, draws_opt.value()
                );
            }
        }
        throw std::runtime_error("Model type not recognised '" + ESAEnums::strForModelType(mT) + "'");
    };
    arma::dmat out = dataObj.panelCallable(inner);
    return out;
}

/// Calculate the efficiency score (conditional on eps) based on Jondrow et al (1982)
arma::dmat ESASfaJlms::effPredJlms(
    const arma::dcolvec& par,
    const int nsim,
    const int haltonStart,
    const int seed
)
{
    // dereference pointer
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    ESASfaModelType mT = dataObj.getModelType();
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    ESASfaModelDistribution mD = ESAEnums::getDistribution(mT);
    // don't apply to GTRE
    if (mF == ESASfaModelFamily::GTRE){
        throw std::invalid_argument("This method is not applicable to the GTRE model");
    }
    std::optional<arma::dmat> draws_opt = std::nullopt;
    if (mF == ESASfaModelFamily::TRE) {
        unsigned int nFirms = static_cast<unsigned int>(dataObj.getNids());
        arma::dmat draws(nFirms, nsim);
        for (int j = 0; j < nsim; j++) {
            int currBase = my100008Primes[haltonStart + j];
            draws.col(j) = HaltonSeq::generate(currBase, nFirms, 1000, true, seed, false);
        }
        draws_opt = std::make_optional(std::move(draws));
    }
    // lambda function which calls the relevant function to process each panel
    std::function<arma::dmat(
        const unsigned int,
        const arma::dcolvec&,
        const arma::dmat&,
        const std::optional<arma::dmat>,
        const std::optional<arma::dmat>,
        const std::optional<arma::dmat>,
        const std::optional<arma::dmat>,
        const std::optional<arma::dmat>
    )> inner = [this, &par, &nsim, &draws_opt, &mT, &mF, &mD](
        const unsigned int idx,
        const arma::dcolvec& y,
        const arma::dmat& x,
        const std::optional<arma::dmat>&zmuit,
        const std::optional<arma::dmat>&zuit,
        const std::optional<arma::dmat>&zvit,
        const std::optional<arma::dmat>&zui0,
        const std::optional<arma::dmat>&zvi0
    ){
        // for tre; need zuit, zvit, zvi0
        if (mF == ESASfaModelFamily::TRE) {
            if (!zuit || !zvit || !zvi0) throw std::invalid_argument("'zuit', 'zvit', 'zvi0' required for tre");
        }
        // for tfe; only need zuit, zvit
        if (mF == ESASfaModelFamily::TFE) {
            if (!zuit || !zvit) throw std::invalid_argument("'zuit', 'zvit' required for tfe");
        }
        // check distribution - if truncated normal, then need the zmuit
        if (mD == ESASfaModelDistribution::TNORM) {
            if (!zmuit) throw std::invalid_argument("'zmuit' required for truncated normal distribution");
        }
        arma::dmat out;
        if (mT == ESASfaModelType::TRE_HNORM_ZUIT){
            out = this->effPredJlmsTrePanelHalfNormal(
                idx, par, y, x, zuit.value(), zvit.value(), zvi0.value(), this->s, nsim, draws_opt.value()
            );
        } else if (mT == ESASfaModelType::TRE_TNORM_ZUIT){
            out = this->effPredJlmsTrePanelTruncNormal(
                idx, par, y, x, zuit.value(), zvit.value(), zvi0.value(), zmuit.value(), this->s, nsim, draws_opt.value()
            );
        } else {
            throw std::runtime_error("Model type not recognised '" + ESAEnums::strForModelType(mT) + "'");
        }
        return out;
    };
    arma::dmat out = dataObj.panelCallable(inner);
    return out;
}

/// Inefficiency prediction - True Random Effects with Half Normal distribution
arma::dmat ESASfaJlms::ineffPredJlmsTrePanelHalfNormal(
    const unsigned int idx,
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& zuit,
    const arma::dmat& zvit,
    const arma::dmat& zvi0,
    const int s,
    const int nsim,
    const arma::dmat& draws
){
    // vector of zeros for mu 
    arma::dmat mu = arma::dmat(y.n_rows, 1, arma::fill::zeros);
    return ESASfaJlms::ineffPredJlmsTrePanelTruncNormal(idx, par, y, x, zuit, zvit, zvi0, mu, s, nsim, draws);
}

/// Inefficiency prediction - True Random Effects with Truncated Normal distribution
arma::dmat ESASfaJlms::ineffPredJlmsTrePanelTruncNormal(
    const unsigned int idx,
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& zuit,
    const arma::dmat& zvit,
    const arma::dmat& zvi0,
    const arma::dmat& muuit,
    const int s,
    const int nsim,
    const arma::dmat& draws
){
    // dereference pointer to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    /// check that the draws is the appropriate length
    if (draws.n_rows < idx){
        throw std::invalid_argument("draws must be of length ident at minimum");
    }
    if (draws.n_cols != nsim){
        throw std::invalid_argument("draws must have the same number of columns as nsim");
    }
    // check s is either -1 or 1
    if (s != -1 && s != 1){
        throw std::invalid_argument("s must be either -1 or 1");
    }
    // extract the coefficients
    arma::dcolvec b_x = dataObj.paramX(par);
    if (!dataObj.paramZuit(par)) throw std::invalid_argument("could not get 'b_zuit'");
    arma::dcolvec b_zuit = dataObj.paramZuit(par).value();
    if (!dataObj.paramZvit(par)) throw std::invalid_argument("could not get 'b_zvit'");
    arma::dcolvec b_zvit = dataObj.paramZvit(par).value();
    if (!dataObj.paramZvi0(par)) throw std::invalid_argument("could not get 'b_zvi0'");
    arma::dcolvec b_zvi0 = dataObj.paramZvi0(par).value();
    std::optional<arma::dcolvec> b_muuit = dataObj.paramZmuit(par);

    // calculate xb
    arma::dmat xb = x * b_x;
    // when mu is not provided, set it to zero
    arma::dmat mu;
    if (!b_muuit){
        mu = muuit;
    } else {
        mu = muuit * b_muuit.value();
    }
    // sigma2uit - variance of the time-varying inefficiency
    arma::dmat sigma2uit = esautils::processSig2Term(b_zuit, zuit);
    // sigma2vit - variance of the stochastic noise component
    arma::dmat sigma2vit = esautils::processSig2Term(b_zvit, zvit);
    // sigma2vi0 - variance of the firm effect
    arma::dmat sigma2vi0 = esautils::processSig2Term(b_zvi0, zvi0);
    arma::dmat sigmavi0 = arma::sqrt(sigma2vi0);
    // calculate sigma2
    arma::dmat sigma2 = sigma2uit + sigma2vit;
    // calculate sigma2star
    arma::dmat sigma2star = (sigma2vit % sigma2uit) / sigma2;
    // calculate sigmastar
    arma::dmat sigmastar = arma::sqrt(sigma2star);
    // create repetitions to the number of simulations
    arma::dmat y_r = esautils::repeatColVecAsCols<double>(y, nsim);
    arma::dmat mu_r = esautils::repeatColVecAsCols<double>(mu, nsim);
    arma::dmat xb_r = esautils::repeatColVecAsCols<double>(xb, nsim);
    arma::dmat sigma2vit_r = esautils::repeatColVecAsCols<double>(sigma2vit, nsim);
    arma::dmat sigma2uit_r = esautils::repeatColVecAsCols<double>(sigma2uit, nsim);
    arma::dmat sigma2_r = esautils::repeatColVecAsCols<double>(sigma2, nsim);
    arma::dmat sigmastar_r = esautils::repeatColVecAsCols<double>(sigmastar, nsim);
    // extract the halton draw - and map to the normal distribution using the inverse of the normal CDF
    arma::dmat draw = draws.submat(arma::span(idx, idx), arma::span(0, nsim - 1));
    // arma::dmat w_i = esamath::ppf_internal(draw, 0.0, 1.0);
    arma::dmat w_i = esandist::ppf(draw, 0.0, 1.0);
    // calculate a_i, which should be a nT x nsim matrix
    arma::dmat a_i = sigmavi0 * w_i;
    // calculate eps_i - y_i - a_i - xb_r
    arma::dmat eps_r = y_r - a_i - xb_r;
    // calculate mustar (mu*sig2v - eps*sig2u) / sig2
    arma::dmat mu_star_numr = (mu_r % sigma2vit_r) - s * (eps_r % sigma2uit_r);
    arma::dmat mu_star = mu_star_numr / sigma2_r;
    // calculate mustar / sigstar
    arma::dmat mustar_div_sigstar = mu_star / sigmastar_r;
    // calculate pdf( mustar / sigstar)
    arma::dmat pdf_mustar_div_sigstar = arma::normpdf(mustar_div_sigstar, 0.0, 1.0);
    // calculate cdf( mustar / sigstar)
    arma::dmat cdf_mustar_div_sigstar = arma::normcdf(mustar_div_sigstar, 0.0, 1.0);
    // calculate pdf / cdf
    arma::dmat pdf_div_cdf = pdf_mustar_div_sigstar / cdf_mustar_div_sigstar;
    // calculate the E(u|eps) = sigstar * (pdf / cdf) + mustar
    arma::dmat u = (sigmastar_r % pdf_div_cdf) + mu_star;
    // calculate the mean [columnwise]
    arma::dmat u_mean = esautils::matrixMeans(u, true);
    return u_mean;
}

/// Calculate efficiency score - True Random Effects with Half Normal distribution
arma::dmat ESASfaJlms::effPredJlmsTrePanelHalfNormal(
    const unsigned int idx,
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& zuit,
    const arma::dmat& zvit,
    const arma::dmat& zvi0,
    const int s,
    const int nsim,
    const arma::dmat& draws
)
{
    // vector of zeros for mu 
    arma::dmat mu = arma::dmat(y.n_rows, 1, arma::fill::zeros);
    return ESASfaJlms::effPredJlmsTrePanelTruncNormal(idx, par, y, x, zuit, zvit, zvi0, mu, s, nsim, draws);
}

/// Calculate efficiency score - True Random Effects with Truncated Normal distribution
arma::dmat ESASfaJlms::effPredJlmsTrePanelTruncNormal(
    const unsigned int idx,
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& zuit,
    const arma::dmat& zvit,
    const arma::dmat& zvi0,
    const arma::dmat& muuit,
    const int s,
    const int nsim,
    const arma::dmat& draws
)
{
    /// dereference pointer to data object
    ESADataPanel& dataObj = (ESADataPanel&)*dataObjPtr;
    /// check that the draws is the appropriate length
    if (draws.n_rows < idx){
        throw std::invalid_argument("draws must be of length ident at minimum");
    }
    if (draws.n_cols != nsim){
        throw std::invalid_argument("draws must have the same number of columns as nsim");
    }
    // check s is either -1 or 1
    if (s != -1 && s != 1){
        throw std::invalid_argument("s must be either -1 or 1");
    }
    // extract the coefficients
    arma::dcolvec b_x = dataObj.paramX(par);
    if (!dataObj.paramZuit(par)) throw std::invalid_argument("missing 'zuit'");
    arma::dcolvec b_zuit = dataObj.paramZuit(par).value();
    if (!dataObj.paramZvit(par)) throw std::invalid_argument("missing 'zvit'");
    arma::dcolvec b_zvit = dataObj.paramZvit(par).value();
    if (!dataObj.paramZvi0(par)) throw std::invalid_argument("missing 'zvi0'");
    arma::dcolvec b_zvi0 = dataObj.paramZvi0(par).value();
    std::optional<arma::dcolvec> b_muuit = dataObj.paramZmuit(par);

    // calculate xb
    arma::dmat xb = x * b_x;
    // when mu is not provided, set it to zero
    arma::dmat mu;
    if (!b_muuit){
        mu = muuit;
    } else {
        mu = muuit * b_muuit.value();
    }
    // sigma2uit - variance of the time-varying inefficiency
    arma::dmat sigma2uit = esautils::processSig2Term(b_zuit, zuit);
    // sigma2vit - variance of the stochastic noise component
    arma::dmat sigma2vit = esautils::processSig2Term(b_zvit, zvit);
    // sigma2vi0 - variance of the firm effect
    arma::dmat sigma2vi0 = esautils::processSig2Term(b_zvi0, zvi0);
    arma::dmat sigmavi0 = arma::sqrt(sigma2vi0);
    // calculate sigma2
    arma::dmat sigma2 = sigma2uit + sigma2vit;
    // calculate sigma2star
    arma::dmat sigma2star = (sigma2vit % sigma2uit) / sigma2;
    // calculate sigmastar
    arma::dmat sigmastar = arma::sqrt(sigma2star);
    // create repetitions to the number of simulations
    arma::dmat y_r = esautils::repeatColVecAsCols<double>(y, nsim);
    arma::dmat mu_r = esautils::repeatColVecAsCols<double>(mu, nsim);
    arma::dmat xb_r = esautils::repeatColVecAsCols<double>(xb, nsim);
    arma::dmat sigma2vit_r = esautils::repeatColVecAsCols<double>(sigma2vit, nsim);
    arma::dmat sigma2uit_r = esautils::repeatColVecAsCols<double>(sigma2uit, nsim);
    arma::dmat sigma2_r = esautils::repeatColVecAsCols<double>(sigma2, nsim);
    arma::dmat sigmastar_r = esautils::repeatColVecAsCols<double>(sigmastar, nsim);
    arma::dmat sigma2star_r = esautils::repeatColVecAsCols<double>(sigma2star, nsim);
    // extract the halton draw - and map to the normal distribution using the inverse of the normal CDF
    arma::dmat draw = draws.submat(arma::span(idx, idx), arma::span(0, nsim - 1));
    // arma::dmat w_i = esamath::ppf_internal(draw, 0.0, 1.0);
    arma::dmat w_i = esandist::ppf(draw, 0.0, 1.0);
    // calculate a_i, which should be a nT x nsim matrix
    arma::dmat a_i = sigmavi0 * w_i;
    // calculate eps_i - y_i - a_i - xb_r
    arma::dmat eps_r = y_r - a_i - xb_r;
    // calculate mustar (mu*sig2v - eps*sig2u) / sig2
    arma::dmat mu_star_numr = (mu_r % sigma2vit_r) - s * (eps_r % sigma2uit_r);
    arma::dmat mu_star = mu_star_numr / sigma2_r;
    // calculate mustar / sigstar
    arma::dmat mustar_div_sigstar = mu_star / sigmastar_r;
    // exponential part of things exp(mu_star + 0.5*sig2star)
    arma::dmat exp_comp = arma::exp(-mu_star + 0.5 * sigma2star_r);
    // component 2 - numerator
    // cdf( (mustar/sigstar) - sigstar )
    arma::dmat comp2_numer_in = mustar_div_sigstar - sigmastar_r;
    arma::dmat comp2_numer = arma::normcdf(comp2_numer_in, 0.0, 1.0);
    // component 2 - denominator
    // cdf( (mustar/sigstar) )
    arma::dmat comp2_denom = arma::normcdf(mustar_div_sigstar, 0.0, 1.0);
    // calculate component 2
    arma::dmat comp2 = comp2_numer / comp2_denom;
    // calculate the E(exp(-u)|eps) = exp_comp * comp2
    arma::dmat exp_neg_u = exp_comp % comp2;
    // calculate the mean [columnwise]
    arma::dmat exp_neg_u_mean = esautils::matrixMeans(exp_neg_u, true);
    return exp_neg_u_mean;
}

/// Calculate the inefficiency score (conditional on eps) based on Jondrow et al. (1982) for TFE
arma::dmat ESASfaJlms::ineffPredJlmsTfeHalfNormal(
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& zuit,
    const arma::dmat& zvit,
    const int s
)
{
    // vector of zeros for mu 
    arma::dmat mu = arma::dmat(y.n_rows, 1, arma::fill::zeros);
    return ESASfaJlms::ineffPredJlmsTfeTruncNormal(par, y, x, zuit, zvit, mu, s);
}

/// Calculate the inefficiency score (conditional on eps) based on Jondrow et al. (1982) for TFE
arma::dmat ESASfaJlms::ineffPredJlmsTfeTruncNormal(
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& zuit,
    const arma::dmat& zvit,
    const arma::dmat& muuit,
    const int s
)
{
    return arma::mat();
}

/// Calculate the efficiency score (conditional on eps) based on Jondrow et al. (1982) for TFE
arma::dmat ESASfaJlms::effPredJlmsTfeHalfNormal(
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& zuit,
    const arma::dmat& zvit,
    const int s
)
{
    // vector of zeros for mu 
    arma::dmat mu = arma::dmat(y.n_rows, 1, arma::fill::zeros);
    return ESASfaJlms::effPredJlmsTfeTruncNormal(par, y, x, zuit, zvit, mu, s);
}

/// Calculate the efficiency score (conditional on eps) based on Jondrow et al. (1982) for TFE
arma::dmat ESASfaJlms::effPredJlmsTfeTruncNormal(
    const arma::dcolvec& par,
    const arma::dmat& y,
    const arma::dmat& x,
    const arma::dmat& zuit,
    const arma::dmat& zvit,
    const arma::dmat& muuit,
    const int s
)
{
    throw std::runtime_error("Not implemented yet");
}