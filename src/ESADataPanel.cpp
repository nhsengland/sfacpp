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
 * @file ESADataPanel.cpp
 * @brief Class to hold panel data - updated version
 * @date 2025-02-22
 * @author Edmund Haacke
 */

#include <string>
#include "data/ESADataPanel.hpp"

namespace {
    bool check_ok(const arma::Col<int>* c){
        if (c == nullptr) throw std::runtime_error("Missing columnvector");
        return true;
    }
}

// ---- constructors ----
// default constructor
ESADataPanel::ESADataPanel() :
    ESADataBase(),
    idVec(),
    timeVec(),
    zmuit(std::nullopt),
    zuit(std::nullopt),
    zvit(std::nullopt),
    zui0(std::nullopt),
    zvi0(std::nullopt),
    nZmuit(0),
    nZuit(0),
    nZvit(0),
    nZui0(0),
    nZvi0(0),
    uniqueIds()
{
    this->nobs = 0;
    this->nids = 0;
    this->npanels = 0;
    this->balanced = true;
};

// main constructor
ESADataPanel::ESADataPanel(
    const arma::dcolvec* y,
    const arma::dmat* x,
    const arma::Col<int>* idVec,
    const arma::Col<int>* timeVec,
    const ESASfaModelType modelType,
    const arma::dmat* zmuit,
    const arma::dmat* zuit,
    const arma::dmat* zvit,
    const arma::dmat* zui0,
    const arma::dmat* zvi0,
    const bool arraysContiguous
) : ESADataBase(y, x, modelType, arraysContiguous),
    idVec(check_ok(idVec) ? *idVec : arma::Col<int>()),
    timeVec(check_ok(timeVec) ? *timeVec : arma::Col<int>()),
    zmuit(esautils::makeZeroCopyStrictView(zmuit)),
    zuit(esautils::makeZeroCopyStrictView(zuit)),
    zvit(esautils::makeZeroCopyStrictView(zvit)),
    zui0(esautils::makeZeroCopyStrictView(zui0)),
    zvi0(esautils::makeZeroCopyStrictView(zvi0)),
    nZmuit(this->findZmuit()),
    nZuit(this->findZuit()),
    nZvit(this->findZvit()),
    nZui0(this->findZui0()),
    nZvi0(this->findZvi0()),
    uniqueIds(esautils::uniqueValsInColVec<int>(*idVec))
    // uniqueIds(arma::unique(idVec))
{
    this->nobs = y->n_rows;
    this->nids = esautils::uniqueValsInColVec<int>(*idVec).n_rows;
    this->npanels = esautils::uniqueValsInColVec<int>(*timeVec).n_rows;
    // this->nids = this->uniqueIds.n_rows;
    // arma::Col<int> uniqTime = arma::unique(timeVec);
    // this->npanels = uniqTime.n_rows;
    this->balanced = (nobs == nids * npanels);
    // run some checks on what was pass thru vs what should've been passed thru
    if (x->n_rows != nobs || idVec->n_rows != nobs || timeVec->n_rows != nobs){
        throw std::invalid_argument("All matrices must have the same number of rows");
    }
    if (this->zmuit){
        if (this->zmuit->n_rows != this->nobs) throw std::invalid_argument("zmuit must have the same number of rows as y");
    }
    if (this->zuit){
        if (this->zuit->n_rows != this->nobs) throw std::invalid_argument("zuit must have the same number of rows as y");
    }
    if (this->zvit){
        if (this->zvit->n_rows != this->nobs) throw std::invalid_argument("zvit must have the same number of rows as y");
    }
    if (this->zui0){
        if (this->zui0->n_rows != this->nobs) throw std::invalid_argument("zui0 must have the same number of rows as y");
    }
    if (this->zvi0){
        if (this->zvi0->n_rows != this->nobs) throw std::invalid_argument("zvi0 must have the same number of rows as y");
    }
    // precalculate index positions for each panel - if contiguous, then start/end positons, otherwise indicies
    if (arraysContiguous) {
        std::vector<std::pair<arma::uword, arma::uword>> contigStartEnd;
        contigStartEnd.reserve(this->nids);
        if (this->idVec.n_elem > 0) {
            arma::uword currStart = 0;
            int currId = this->idVec(0);
            for (arma::uword i = 1; i < this->idVec.n_elem; i++) {
                if (this->idVec(i) != currId) {
                    contigStartEnd.push_back(std::make_pair(currStart, i - 1));
                    currStart = i;
                    currId = this->idVec(i);
                }
            }
            // for the final group
            contigStartEnd.push_back(std::make_pair(currStart, this->idVec.n_elem - 1));
        } else {
            throw std::runtime_error("got no elements for firm identifiers");
        }
        this->firmIdRows = std::move(contigStartEnd);
    } else {
        std::vector<arma::uvec> noncontigInds(this->nids);
        for (int i = 0; i < this->nids; i++) {
            noncontigInds[i] = arma::find(this->idVec == this->uniqueIds(i));
        }
        this->firmIdRows = std::move(noncontigInds);
    }
    // find the maxT in the dataset
    calculateMaxT(this->firmIdRows, arraysContiguous);
    // finally, check whether the time-invariant components are indeed time-invariant
    checkTimeInvariance(1e-8);
}

// ---- Private methods ----

// helper to return variant for subset
FirmIdsVar ESADataPanel::getFirmIdRowsForSubset(const arma::Col<int>& subsetIdents) const
{
    std::vector<arma::uvec> noncontigs(subsetIdents.n_rows);
    std::vector<std::pair<arma::uword, arma::uword>> contigs;
    contigs.reserve(subsetIdents.n_rows);
    for (size_t i = 0; i < subsetIdents.n_rows; i++) {
        if (this->arraysContiguous) {
            // contiguous arrays
            arma::uword currStart = 0;
            bool hasStarted = false;
            int currId = subsetIdents(i);
            for (arma::uword j = 0; j < this->idVec.n_elem; j++) {
                if (currId == this->idVec(j) && !hasStarted) {
                    currStart = j;
                    hasStarted = true;
                }
                if (this->idVec(j) != currId || (j == this->idVec.n_elem - 1 && hasStarted)) {
                    contigs.push_back(std::make_pair(currStart, j - 1));
                }
            }
        } else {
            // non-contiguous
            std::vector<arma::uvec> noncontigInds(this->nids);
            noncontigs[i] = arma::find(this->idVec == subsetIdents(i));
        }
    }
    if (this->arraysContiguous) {
        return contigs;
    }
    // otherwise return non-contiguous vector of indicies
    return noncontigs;
}

// helper to calculate maxT
void ESADataPanel::calculateMaxT(const FirmIdsVar& precalc, bool isContig) {
    this->maxT = 0;
    this->minT = this->nobs;
    if (isContig) {
        // contiguous - used ranges
        FirmIdsRange contigs = std::get<FirmIdsRange>(precalc);
        // iterate thru
        for (size_t i = 0; i < this->nids; i++) {
            int currSize = contigs[i].second - contigs[i].first + 1;
            if (currSize > this->maxT) {
                this->maxT = currSize;
            }
            // handle minT
            if (currSize < this->minT) {
                this->minT = currSize;
            }
        }
    } else {
        // vector of indicies, so can just get the size
        FirmIdsInds noncontigs = std::get<FirmIdsInds>(precalc);
        for (size_t i = 0; i < this->nids; i++) {
            int currSize = noncontigs[i].n_elem;
            if (currSize > this->maxT) {
                this->maxT = currSize;
            }
            // handle minT
            if (currSize < this->minT) {
                this->minT = currSize;
            }
        }
    }
}

// helper to create optional views w/o copying
template <typename T>
std::optional<arma::subview<double>> ESADataPanel::makeView(const std::optional<T>& m, arma::uword start, arma::uword end) const
{
    if (m) return std::make_optional(m->rows(start, end));
    return std::nullopt;
}
// explicit template instantisation
template std::optional<arma::subview<double>> ESADataPanel::makeView<arma::dmat>(const std::optional<arma::dmat>&, arma::uword, arma::uword) const;

// helper to copy for non-contiguous
template <typename T>
std::optional<arma::dmat> ESADataPanel::makeCopy(const std::optional<T>& m, const arma::uvec& inds) const
{
    if (m) return std::make_optional(m->rows(inds));
    return std::nullopt;
}
// explicit template instantisation
template std::optional<arma::dmat> ESADataPanel::makeCopy<arma::dmat>(const std::optional<arma::dmat>&, const arma::uvec&) const;

std::optional<arma::dmat> ESADataPanel::validateZmuit(const std::optional<arma::dmat>& m, const ESASfaModelType mT) const 
{
    // get the model distribution
    ESASfaModelDistribution mD = ESAEnums::getDistribution(mT);
    // check if the model distribution is truncated normal
    if (mD == ESASfaModelDistribution::TNORM){
        // check if the matrix is present
        if (!m){
            throw std::invalid_argument("zmuit must be present for truncated normal distribution");
        }
        // return m;
        return esautils::makeZeroCopyStrictView(m);
    }
    // otherwise delete anything that was passed thru
    return std::nullopt;
}

std::optional<arma::dmat> ESADataPanel::validateZuit(
    const std::optional<arma::dmat>& m,
    const ESASfaModelType mT
) const
{
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    // check if the model distribution is truncated normal
    if (mF == ESASfaModelFamily::TFE || mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE){
        // check if the matrix is present
        if (!m){
            throw std::invalid_argument("zuit must be present for TFE, TRE, GTRE models");
        }
        // return m;
        return esautils::makeZeroCopyStrictView(m);
    }
    // otherwise delete anything that was passed thru
    return std::nullopt;   
}

std::optional<arma::dmat> ESADataPanel::validateZvit(
    const std::optional<arma::dmat>& m,
    const ESASfaModelType mT
) const
{
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    // TFE, TRE, GTRE should have this
    if (mF == ESASfaModelFamily::TFE || mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE){
        // check if the matrix is present
        if (!m){
            throw std::invalid_argument("zvit must be present for TFE, TRE or GTRE models");
        }
        // return m;
        return esautils::makeZeroCopyStrictView(m);
    }
    // otherwise delete anything that was passed thru
    return std::nullopt;
}

std::optional<arma::dmat> ESADataPanel::validateZui0(
    const std::optional<arma::dmat>& m,
    const ESASfaModelType mT
) const
{
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    // Only GTRE should have this term
    if (mF == ESASfaModelFamily::GTRE){
        // check if the matrix is present
        if (!m){
            throw std::invalid_argument("zui0 must be present for GTRE models");
        }
        // return m;
        return esautils::makeZeroCopyStrictView(m);
    }
    // otherwise delete anything that was passed thru
    return std::nullopt;
}

std::optional<arma::dmat> ESADataPanel::validateZvi0(
    const std::optional<arma::dmat>& m,
    const ESASfaModelType mT
) const
{
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    // Only TRE and GTRE should have this term
    if (mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE){
        // check if the matrix is present
        if (!m){
            throw std::invalid_argument("zvi0 must be present for TRE, GTRE models");
        }
        // return m;
        return esautils::makeZeroCopyStrictView(m);
    }
    // otherwise delete anything that was passed thru
    return std::nullopt;
}

unsigned int ESADataPanel::findZmuit()
{
    // check the zmuit attribute
    if (this->zmuit){
        return this->zmuit->n_cols;
    }
    return 0;
}

unsigned int ESADataPanel::findZuit()
{
    // check the zuit attribute
    if (this->zuit){
        return this->zuit->n_cols;
    }
    return 0;
}

unsigned int ESADataPanel::findZvit()
{
    // check the zvit attribute
    if (this->zvit){
        return this->zvit->n_cols;
    }
    return 0;
}

unsigned int ESADataPanel::findZui0()
{
    // check the zui0 attribute
    if (this->zui0){
        return this->zui0->n_cols;
    }
    return 0;
}

unsigned int ESADataPanel::findZvi0()
{
    // check the zvi0 attribute
    if (this->zvi0){
        return this->zvi0->n_cols;
    }
    return 0;
}

std::pair<int, int> ESADataPanel::getXRange(const arma::dcolvec& params) const
{
    // check the shape of the parameters
    ESADataPanel::checkParamsShape(params);
    return std::make_pair(0, nX - 1);
}

std::optional<std::pair<int, int>> ESADataPanel::getZmuitRange(const arma::dcolvec& params) const
{
    // check the shape of the parameters
    ESADataPanel::checkParamsShape(params);
    if (nZmuit == 0){
        return std::nullopt;
    }
    std::pair<int, int> res = std::make_pair(nX, nX + nZmuit - 1);
    return std::make_optional<std::pair<int, int>>(std::move(res));
}

std::optional<std::pair<int, int>> ESADataPanel::getZuitRange(const arma::dcolvec& params) const
{
    // check the shape of the parameters
    ESADataPanel::checkParamsShape(params);
    if (nZuit == 0){
        return std::nullopt;
    }
    std::pair<int, int> res = std::make_pair(nX + nZmuit, nX + nZmuit + nZuit - 1);
    return std::make_optional<std::pair<int, int>>(std::move(res));
}

std::optional<std::pair<int, int>> ESADataPanel::getZvitRange(const arma::dcolvec& params) const
{
    // check the shape of the parameters
    ESADataPanel::checkParamsShape(params);
    if (nZvit == 0){
        return std::nullopt;
    }
    std::pair<int, int> res = std::make_pair(nX + nZmuit + nZuit, nX + nZmuit + nZuit + nZvit - 1);
    return std::make_optional<std::pair<int, int>>(std::move(res));
}

std::optional<std::pair<int, int>> ESADataPanel::getZui0Range(const arma::dcolvec& params) const
{
    // check the shape of the parameters
    ESADataPanel::checkParamsShape(params);
    if (nZui0 == 0){
        return std::nullopt;
    }
    std::pair<int, int> res = std::make_pair(nX + nZmuit + nZuit + nZvit + nZvi0, nX + nZmuit + nZuit + nZvit + nZvi0 + nZui0 - 1);
    return std::make_optional<std::pair<int, int>>(std::move(res));
}

std::optional<std::pair<int, int>> ESADataPanel::getZvi0Range(const arma::dcolvec& params) const
{
    // check the shape of the parameters
    ESADataPanel::checkParamsShape(params);
    if (nZvi0 == 0){
        return std::nullopt;
    }
    std::pair<int, int> res = std::make_pair(nX + nZmuit + nZuit + nZvit, nX + nZmuit + nZuit + nZvit + nZvi0 - 1);
    return std::make_optional<std::pair<int, int>>(std::move(res));
}

void ESADataPanel::checkParamsShape(const arma::dcolvec& params) const
{
    // check if the parameters are a column vector
    if (params.n_cols != 1){
        throw std::invalid_argument("params must be a column vector");
    }
    // check if the parameters have the correct number of rows
    if (params.n_rows != nX + nZmuit + nZuit + nZvit + nZui0 + nZvi0){
        throw std::invalid_argument("params must have the correct number of rows");
    }
}

arma::dcolvec ESADataPanel::paramX(const arma::dcolvec& params) const
{
    // check the shape of the parameters
    std::pair<int, int> row_range = ESADataPanel::getXRange(params);
    arma::dcolvec r = params.rows(row_range.first, row_range.second);
    return r;
}

std::optional<arma::dcolvec> ESADataPanel::paramZmuit(const arma::dcolvec& params) const
{
    // check the shape of the parameters
    std::optional<std::pair<int, int>> row_range = ESADataPanel::getZmuitRange(params);
    if (!row_range){
        return std::nullopt;
    }
    arma::dcolvec res = params.rows(row_range->first, row_range->second);
    return std::make_optional<arma::dcolvec>(std::move(res));
}

std::optional<arma::dcolvec> ESADataPanel::paramZuit(const arma::dcolvec& params) const
{
    // check the shape of the parameters
    std::optional<std::pair<int, int>> row_range = ESADataPanel::getZuitRange(params);
    if (!row_range){
        return std::nullopt;
    }
    arma::dcolvec res = params.rows(row_range->first, row_range->second);
    return std::make_optional<arma::dcolvec>(std::move(res));
}

std::optional<arma::dcolvec> ESADataPanel::paramZvit(const arma::dcolvec& params) const
{
    // check the shape of the parameters
    std::optional<std::pair<int, int>> row_range = ESADataPanel::getZvitRange(params);
    if (!row_range){
        return std::nullopt;
    }
    arma::dcolvec res = params.rows(row_range->first, row_range->second);
    return std::make_optional<arma::dcolvec>(std::move(res));
}

std::optional<arma::dcolvec> ESADataPanel::paramZui0(const arma::dcolvec& params) const
{
    // check the shape of the parameters
    std::optional<std::pair<int, int>> row_range = ESADataPanel::getZui0Range(params);
    if (!row_range){
        return std::nullopt;
    }
    arma::dcolvec res = params.rows(row_range->first, row_range->second);
    return std::make_optional<arma::dcolvec>(std::move(res));
}

std::optional<arma::dcolvec> ESADataPanel::paramZvi0(const arma::dcolvec& params) const
{
    // check the shape of the parameters
    std::optional<std::pair<int, int>> row_range = ESADataPanel::getZvi0Range(params);
    if (!row_range){
        return std::nullopt;
    }
    arma::dcolvec res = params.rows(row_range->first, row_range->second);
    return std::make_optional<arma::dcolvec>(std::move(res));
}

unsigned int ESADataPanel::nParams() const
{
    return nX + nZmuit + nZuit + nZvit + nZui0 + nZvi0;
}

// 
void ESADataPanel::checkTimeInvariance(const double tol) const
{
    auto checkMatrix = [&tol](const auto& mtxIn) {
        const auto& mtx = mtxIn.get_ref();
        // calculate the group means
        arma::dmat means = arma::mean(mtx);
        int ok = 0;
        // loop thru each row, and check approximately the same
        for (int r = 0; r < mtx.n_rows; r++) {
            if (!arma::approx_equal(mtx.row(r), means, "absdiff", tol)) {
                ok = 1;
                break;
            }
        }
        return ok;
    };
    // loop thru each panel
    this->panelCallableSum([this, &checkMatrix](
        const unsigned int idx,
        const auto& y,
        const auto& x,
        const auto& zmuit,
        const auto& zuit,
        const auto& zvit,
        const auto& zui0,
        const auto& zvi0
    ) {
        int panID = (idx < this->idVec.size()) ? this->idVec[idx] : -1;
        // check for zui0
        if (this->getNZui0() > 0 && zui0.has_value()) {
            int zui0Status = checkMatrix(zui0.value());
            if (zui0Status != 0) {
                throw std::invalid_argument(
                    "Panel ID " + std::to_string(panID) +
                    " appears not to be constant over time for zui0."
                );
            }
        }
        // check for zvi0
        if (this->getNZvi0() > 0 && zvi0.has_value()) {
            int zvi0Status = checkMatrix(zvi0.value());
            if (zvi0Status != 0) {
                throw std::invalid_argument(
                    "Panel ID " + std::to_string(panID) +
                    " appears not to be constant over time for zvi0."
                );
            }
        }
        return 0.0;
    });
}