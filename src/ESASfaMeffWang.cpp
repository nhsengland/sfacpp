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

// File: ESASfaMeffWang.cpp
// Author: Edmund Haacke
// Date Created: 2024-12-28
// Description:
//    Marginal effects ala Wang (2002)

#include <optional>
#include "marginaleffects/ESASfaMeffWang.hpp"
#include "utils/enums.hpp"
#include "utils/esautils.hpp"
#include "math/esamath.hpp"
#include "math/esandist.hpp"

// ESASfaMeffWang::~ESASfaMeffWang() {
//     // destructor
// }

/// Calculate the marginal effects of the inefficiency on the determinant
ESASfaMeffReturn ESASfaMeffWang::marginalEffects(const arma::dcolvec& par, const ESASfaModelTerms& modelTerms) const {
    // these marginal effects from Wang (2002) only include parameters that enter through mu_it and sigma2_uit
    // get all of the possible terms in both mu and zuit
    std::vector<std::string> outColNames = modelTerms.getAllCleanedNamesInLocation(
        std::vector<ESASfaModelTermLocation>{
            ESASfaModelTermLocation::MUUIT,
            ESASfaModelTermLocation::ZUIT
        },
        true
    );
    // actual column names to use for output
    std::vector<std::string> accOutNames = outColNames;
    int nobs = dataObjPtr->getNobs();
    // model enums
    ESASfaModelType mT = dataObjPtr->getModelType();
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    ESASfaModelDistribution mD = ESAEnums::getDistribution(mT);
    // ESASfaData& dataObj = (ESASfaData&)*dataObjPtr;
    arma::dmat margeffs;
    // TFE model is currently the only one to use the new data class
    if (mF == ESASfaModelFamily::TFE){
        // dereference ptr to ESADataPanel class
        ESADataPanel& dataObjPan = (ESADataPanel&)*dataObjPtr;
        // check whether can access b_zuit
        std::optional<arma::dcolvec> b_zuit = dataObjPan.paramZuit(par);
        std::optional<arma::dcolvec> b_muuit = std::nullopt;
        // if truncated normal, mu would be parameterised
        if (mD == ESASfaModelDistribution::TNORM){
            b_muuit = dataObjPan.paramZmuit(par);
        }
        // different callback functions too
        margeffs = dataObjPan.dataCallable(
            [this, &b_zuit, &b_muuit, &modelTerms, &outColNames, &nobs, &mD](
                const auto& y, 
                const auto& x,
                const auto& zmuit,
                const auto& zuit,
                const auto& zvit,
                const auto& zui0,
                const auto& zvi0
            ){
                if (mD == ESASfaModelDistribution::TNORM && !zmuit){
                    throw std::invalid_argument("zmuit is required for truncated normal models");
                }
                if (!zuit){
                    throw std::invalid_argument("zuit is required for all models");
                }
                return this->calculateWangMeff(
                    b_muuit,
                    b_zuit,
                    zmuit,
                    zuit,
                    outColNames,
                    modelTerms,
                    nobs,
                    std::make_optional(ESASfaModelTermLocation::MUUIT),
                    ESASfaModelTermLocation::ZUIT
                );
            }
        );
    } else {
        // dereference ptr to ESASfaData class
        ESADataPanel& dataObjOrig = (ESADataPanel&)*dataObjPtr;
        std::optional<arma::dcolvec> b_zuit = dataObjOrig.paramZuit(par);
        std::optional<arma::dcolvec> b_muuit = std::nullopt;
        if (mD == ESASfaModelDistribution::TNORM) {
            b_muuit = dataObjOrig.paramZmuit(par);
        }
        std::optional<arma::dcolvec> b_zui0 = std::nullopt;
        if (mF == ESASfaModelFamily::GTRE) {
            b_zui0 = dataObjOrig.paramZui0(par);
        }
        // calllback on the data to calculate the marginal effects - these are not individual panel specific
        margeffs = dataObjOrig.dataCallable(
            [this, &b_zuit, &b_muuit, &b_zui0, &modelTerms, &outColNames, &nobs, &mF, &accOutNames](
                const auto& y,
                const auto& x,
                const auto& zmuit,
                const auto& zuit,
                const auto& zvit,
                const auto& zui0,
                const auto& zvi0
            ){
                arma::dmat out;
                if (mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE) {
                    // both the TRE & GTRE, calculate for zuit
                    out = this->calculateWangMeff(
                        b_muuit,
                        b_zuit,
                        zmuit,
                        zuit,
                        outColNames,
                        modelTerms,
                        nobs,
                        std::make_optional(ESASfaModelTermLocation::MUUIT),
                        ESASfaModelTermLocation::ZUIT
                    );
                    // prefix the actual out names with zuit
                    for (auto& s : accOutNames) s = "Zuit_" + s;
                } else {
                    throw std::runtime_error("unsupported model in marginal effects");
                }
                // if GTRE model, also calculate for ZUI0
                if (mF == ESASfaModelFamily::GTRE) {
                    std::vector<std::string> zui0Names = modelTerms.getAllCleanedNamesInLocation(ESASfaModelTermLocation::ZUI0, true);
                    if (!zui0Names.empty()) {
                        // if its not empty, proceed to calculate
                        arma::dmat zui0Meff = this->calculateWangMeff(
                            std::nullopt, // no term for mean
                            b_zui0, // coefficent for zui0
                            std::optional<arma::dmat>(std::nullopt), // no data for mean
                            zui0, // zui0 determinants
                            zui0Names, // column names
                            modelTerms,
                            nobs,
                            std::nullopt, // no location for mu
                            ESASfaModelTermLocation::ZUI0
                        );
                        // combine this with the return matrix
                        out.insert_cols(out.n_cols, zui0Meff);
                        // prefix the elements
                        std::vector<std::string> accOutZui0(zui0Names.size());
                        for (int k = 0; k < zui0Names.size(); k++) accOutZui0[k] = "Zui0_" + zui0Names[k];
                        // also, copy over the column names, into the overall column name vector
                        accOutNames.insert(accOutNames.end(), accOutZui0.begin(), accOutZui0.end());
                    }
                }
                return out;
            }
        );
    }
    // create output struct
    ESASfaMeffReturn ret;
    ret.marginalEffects = margeffs;
    ret.columnNames = accOutNames;
    return ret;
}

template <typename TMu, typename TZu>
arma::dmat ESASfaMeffWang::calculateWangMeff(
    const std::optional<arma::dcolvec>& bmu,
    const std::optional<arma::dcolvec>& bzu,
    const std::optional<TMu>& muIn,
    const std::optional<TZu>& zuIn,
    const std::vector<std::string>& outColNames,
    const ESASfaModelTerms& modelTerms,
    const int nobs,
    const std::optional<ESASfaModelTermLocation> locMean,
    const ESASfaModelTermLocation locZu
) const
{
    // create the output matrix for the marginal effects
    arma::dmat margEffs(nobs, outColNames.size(), arma::fill::zeros);
    // calculate variance component (sigma2uit, sigma2ui0)
    arma::dmat sigma2, sigma;
    if (!bzu || !zuIn) {
        // no variance determinants - the marginal effect is zero
        return margEffs;
    } else {
        sigma2 = esautils::processSig2Term(bzu.value(), zuIn.value());
        sigma = arma::sqrt(sigma2);
    }
    // calculate mu component
    arma::dmat mu;
    if (!bmu || !muIn) {
        // fill mu with zeros
        mu = arma::dmat(nobs, 1, arma::fill::zeros);
    } else {
        mu = muIn.value() * bzu.value();
    }
    // calculate lambda
    arma::dmat Lambda = mu / sigma;
    // pdf and cdf of lambda
    arma::dmat pdfLambda = arma::normpdf(Lambda, 0.0, 1.0);
    arma::dmat cdfLambda = arma::normcdf(Lambda, 0.0, 1.0);
    arma::dmat alpha = pdfLambda / cdfLambda;
    // multipliers
    arma::dmat muMulti = 1.0 - (Lambda % alpha) - arma::pow(alpha, 2);
    arma::dmat zuMultiInner = ((1.0 + arma::pow(Lambda, 2)) % alpha) + (Lambda % arma::pow(alpha, 2));
    arma::dmat zuMulti = (sigma / 2.0) % zuMultiInner;
    // iterate thru each element in the output column names
    for (size_t i = 0; i < outColNames.size(); i++) {
        arma::subview<double> margEffi = margEffs.col(i);
        // check if exists
        if (locMean.has_value() && bmu.has_value()) {
            // find terms in mu
            std::vector<ModelTerm> muTerms = modelTerms.getTermsForCleanedNameInLocation(outColNames[i], locMean.value());
            if (!muTerms.empty()) {
                double muCoef = bmu.value()(muTerms[0].indexPosInLocation);
                margEffi += (muMulti * muCoef);
            }
        }
        // check if exists in zuit
        if (bzu.has_value()) {
            std::vector<ModelTerm> zuTerms = modelTerms.getTermsForCleanedNameInLocation(outColNames[i], locZu);
            if (!zuTerms.empty()) {
                double zuCoef = bzu.value()(zuTerms[0].indexPosInLocation);
                margEffi += (zuMulti * zuCoef);
            }
        }
    }
    return margEffs;
}

template arma::dmat ESASfaMeffWang::calculateWangMeff<arma::dmat, arma::dmat>(
    const std::optional<arma::dcolvec>&,
    const std::optional<arma::dcolvec>&,
    const std::optional<arma::dmat>&,
    const std::optional<arma::dmat>&,
    const std::vector<std::string>&,
    const ESASfaModelTerms&,
    const int,
    const std::optional<ESASfaModelTermLocation>,
    const ESASfaModelTermLocation
) const;
template arma::dmat ESASfaMeffWang::calculateWangMeff<arma::subview<double>, arma::subview<double>>(
    const std::optional<arma::dcolvec>&,
    const std::optional<arma::dcolvec>&,
    const std::optional<arma::subview<double>>&,
    const std::optional<arma::subview<double>>&,
    const std::vector<std::string>&,
    const ESASfaModelTerms&,
    const int,
    const std::optional<ESASfaModelTermLocation>,
    const ESASfaModelTermLocation
) const;

// arma::dmat ESASfaMeffWang::calculateWangMeff(
//     const std::optional<arma::dcolvec>& bZmuit,
//     const std::optional<arma::dcolvec>& bZuit,
//     const std::optional<arma::dmat>& zmuit,
//     const std::optional<arma::dmat>& zuit,
//     const std::vector<std::string>& outColNames,
//     const ESASfaModelTerms& modelTerms,
//     const int nobs
// ) const
// {
//     int nrow = nobs;
//     // create the output matrix for the marginal effects
//     arma::dmat margEffs(nrow, outColNames.size(), arma::fill::zeros);
//     // calculate sigma2_uit
//     arma::dmat sigma2_uit, sigma_uit;
//     if (!bZuit || !zuit){
//         sigma2_uit = arma::dmat(nrow, 1, arma::fill::zeros);
//         sigma_uit = arma::dmat(nrow, 1, arma::fill::zeros);
//     } else {
//         // sigma2_uit = esautils::processSig2Term(b_zuit, zuit);
//         sigma2_uit = esautils::processSig2Term(bZuit.value(), zuit.value());
//         sigma_uit = arma::sqrt(sigma2_uit);
//     }
//     // calculate mu_it
//     arma::dmat mu;
//     if (!bZmuit || !zmuit){
//         mu = arma::dmat(nrow, 1, arma::fill::zeros);
//     } else {
//         mu = zmuit.value() * bZmuit.value();
//     }
//     // calculate Lambda (mu_it / sigma2_uit)
//     arma::dmat Lambda = mu / sigma_uit;
//     // calculate pdf and cdf of lambda
//     arma::dmat pdfLambda = arma::normpdf(Lambda, 0.0, 1.0);
//     arma::dmat cdfLambda = arma::normcdf(Lambda, 0.0, 1.0);
//     // calculate pdf(Lambda) / cdf(Lambda)
//     arma::dmat pdfDivCdfLambda = pdfLambda / cdfLambda;
//     arma::dmat pdfDivCdfLambdaSq = arma::pow(pdfDivCdfLambda, 2);
//     // calculate the multiplictor for covariates in mu
//     // (1 - Lambda * pdfDivCdfLambda - (pdfLambda / cdfLambda)^2)
//     arma::dmat muMulti = 1 - (Lambda % pdfDivCdfLambda) - pdfDivCdfLambdaSq;
//     // calculate the multiplicator for covariates in zuit
//     // (sigma_uit / 2) * [ (1 + Lambda^2) * pdfDivCdfLambda + Lambda * pdfLambda^2 ]
//     arma::dmat zuitMultiInner = ((1 + arma::pow(Lambda, 2)) % pdfDivCdfLambda ) + ( Lambda % pdfDivCdfLambdaSq );
//     arma::dmat zuitMulti = (sigma_uit / 2) % zuitMultiInner;
//     // iterate thru each element in the output column names
//     for (size_t i = 0; i < outColNames.size(); ++i){
//         arma::dmat margEffi = arma::dmat(nrow, 1, arma::fill::zeros);
//         // find this term in mu - if it exists
//         std::vector<ModelTerm> muTerms = modelTerms.getTermsForCleanedNameInLocation(outColNames[i], ESASfaModelTermLocation::MUUIT);
//         // if it does exist - then multiply by muMulti
//         if (muTerms.size() > 0 && bZmuit){
//             // get the term - just the first one - there should only be one tbh
//             ModelTerm muTerm = muTerms[0];
//             // this contains the index position of where the associated estimated cofficient is
//             double muCoeff = bZmuit.value()(muTerm.indexPosInLocation);
//             margEffi = margEffi + (muMulti * muCoeff);
//         }
//         // find this term in zuit - if it exists
//         std::vector<ModelTerm> zuitTerms = modelTerms.getTermsForCleanedNameInLocation(outColNames[i], ESASfaModelTermLocation::ZUIT);
//         // if it does exist - then multiply by zuitMulti
//         if (zuitTerms.size() > 0 && bZuit){
//             // get the term - just the first one - there should only be one tbh
//             ModelTerm zuitTerm = zuitTerms[0];
//             // this contains the index position of where the associated estimated cofficient is
//             double zuitCoeff = bZuit.value()(zuitTerm.indexPosInLocation);
//             margEffi = margEffi + (zuitMulti * zuitCoeff);
//         }
//         // now set the column in the results matrix
//         margEffs.col(i) = margEffi;
//     }
//     // return the results matrix
//     return margEffs;
// }