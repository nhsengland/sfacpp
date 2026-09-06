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
 * @file ESADataLCM.cpp
 * @brief Class implementation partly, for class to hold data for latent class models
 * @date 2026-05-06
 * @author Edmund Haacke
 */

#include <string>
#include "data/ESADataLCM.hpp"

namespace {
    std::optional<arma::Col<int>> extractUniqIds(const arma::Col<int>* idVec)
    {
        if (idVec == nullptr) return std::nullopt;
        return esautils::uniqueValsInColVec<int>(*idVec);
    }
}

// ---- constructors ----
/**
 * @brief default constructor
 */
ESADataPanelLCM::ESADataPanelLCM() :
    ESADataBase(),
    idVec(std::nullopt),
    timeVec(std::nullopt),
    seg(),
    zmuit(std::nullopt),
    zuit(std::nullopt),
    zvit(std::nullopt),
    zvi0(std::nullopt),
    transition(std::nullopt),
    nSeg(0),
    nZmuit(0),
    nZuit(0),
    nZvit(0),
    nZvi0(0),
    nLc(0),
    nTransition(0),
    isPanel(false),
    nids(0),
    npanels(0),
    balanced(true),
    uniqueIds(std::nullopt),
    maxT(0),
    minT(0),
    firmIdRows(std::nullopt),
    isMarkov(false)
{
    this->nobs = 0;
}

/**
 * @brief main constructor
 */
ESADataPanelLCM::ESADataPanelLCM(
    const arma::dmat* seg,
    const arma::dcolvec* y,
    const arma::dmat* x,
    const arma::Col<int>* idVec,
    const arma::Col<int>* timeVec,
    const ESASfaModelType modelType,
    const arma::dmat* zmuit,
    const arma::dmat* zuit,
    const arma::dmat* zvit,
    const arma::dmat* zvi0,
    const unsigned int nClasses,
    const arma::dmat* transition,
    const bool arraysContiguous
) : ESADataBase(y, x, modelType, arraysContiguous),
    idVec(idVec ? std::make_optional(*idVec) : std::nullopt),
    timeVec(timeVec ? std::make_optional(*timeVec) : std::nullopt),
    seg(seg ? *seg : arma::dmat()),
    zmuit(zmuit ? std::make_optional(*zmuit) : std::nullopt),
    zuit(zuit ? std::make_optional(*zuit) : std::nullopt),
    zvit(zvit ? std::make_optional(*zvit) : std::nullopt),
    zvi0(zvi0 ? std::make_optional(*zvi0) : std::nullopt),
    transition(transition ? std::make_optional(*transition) : std::nullopt),
    nSeg(this->findSeg()),
    nZmuit(this->findZmuit()),
    nZuit(this->findZuit()),
    nZvit(this->findZvit()),
    nZvi0(this->findZvi0()),
    nLc(nClasses),
    nTransition(this->findNCols(const_cast<arma::dmat*>(transition))),
    isPanel(false),
    nids(0),
    npanels(0),
    balanced(false),
    uniqueIds(extractUniqIds(idVec)),
    maxT(0),
    minT(0),
    firmIdRows(std::nullopt),
    isMarkov(false)
{
    // check provided model type
    ESASfaModelFamily mF = ESAEnums::getModelFamily(modelType);
    if ((mF != ESASfaModelFamily::LC_X) && (mF != ESASfaModelFamily::LC_TRE) && (mF != ESASfaModelFamily::LM_TRE)) {
        throw std::invalid_argument("invalid class for " + ESAEnums::strForModelType(modelType));
    }
    isMarkov = (mF == ESASfaModelFamily::LM_TRE);
    // summary stats
    this->nobs = y->n_rows;
    if ((idVec == nullptr && timeVec != nullptr) || (idVec != nullptr && timeVec == nullptr)) {
        throw std::runtime_error("either no idVec & timeVec OR both idVec & timeVec should be provided");
    }
    this->isPanel = (idVec != nullptr && timeVec != nullptr);
    if (this->isPanel && this->idVec.has_value() && this->timeVec.has_value()) {
        // calculate number if IDs / vecs provided
        this->nids = esautils::uniqueValsInColVec<int>(this->idVec.value()).n_rows;
        this->npanels = esautils::uniqueValsInColVec<int>(this->timeVec.value()).n_rows;
        this->balanced = (this->nobs == (this->nids * this->npanels));
        // check dims of the id; time vectors
        if (this->idVec.value().n_rows != this->nobs || this->timeVec.value().n_rows != this->nobs) {
            throw std::invalid_argument("idvec & timevec is not same length as nobs.");
        }
    } else {
        this->nids = 1;
        this->npanels = 1;
        this->balanced = true;
    }
    // run some checks on what was passed thru
    if (x->n_rows != this->nobs) throw std::invalid_argument("'x' is not same length as nobs.");
    if (this->seg.n_rows != this->nobs) throw std::invalid_argument("'seg' is not same length as nobs.");
    if (this->zmuit){
        if (this->zmuit->n_rows != this->nobs) throw std::invalid_argument("zmuit must have the same number of rows as y");
    }
    if (this->zuit){
        if (this->zuit->n_rows != this->nobs) throw std::invalid_argument("zuit must have the same number of rows as y");
    }
    if (this->zvit){
        if (this->zvit->n_rows != this->nobs) throw std::invalid_argument("zvit must have the same number of rows as y");
    }
    if (this->zvi0){
        if (this->zvi0->n_rows != this->nobs) throw std::invalid_argument("zvi0 must have the same number of rows as y");
    }
    // pre-calculate index positions for each panel. if in cross-sectional mode, then
    // just return the index position of 0 to the end of the panel
    if (!this->isPanel) {
        this->firmIdRows = std::vector<std::pair<arma::uword, arma::uword>>{{0, (arma::uword)(this->nobs - 1)}};
        this->maxT = 0;
        this->minT = 0;
    } else {
        if (arraysContiguous) {
            std::vector<std::pair<arma::uword, arma::uword>> contigStartEnd;
            contigStartEnd.reserve(this->nids);
            if (this->idVec.has_value() && this->idVec.value().n_elem > 0) {
                arma::uword currStart = 0;
                int currId = this->idVec.value()(0);
                for (arma::uword i = 1; i < this->idVec.value().n_elem; i++) {
                    if (this->idVec.value()(i) != currId) {
                        contigStartEnd.push_back(std::make_pair(currStart, i - 1));
                        currStart = i;
                        currId = this->idVec.value()(i);
                    }
                }
                contigStartEnd.push_back(std::make_pair(currStart, this->idVec.value().n_elem - 1));
            } else {
                throw std::runtime_error("got no elements for firm identifiers");
            }
            this->firmIdRows = std::move(contigStartEnd);
        } else {
            // non continuous rows
            std::vector<arma::uvec> noncontigInds(this->nids);
            if (this->idVec.has_value() && this->uniqueIds.has_value()) {
                for (int i = 0; i < this->nids; i++) {
                    noncontigInds[i] = arma::find(this->idVec.value() == this->uniqueIds.value()(i));
                }
            }
            this->firmIdRows = std::move(noncontigInds);
        }
        // calculate maxT in the dataset
        if (this->firmIdRows.has_value()) {
            calculateMaxT(this->firmIdRows.value(), arraysContiguous);
        }
        // check time-invariant components are indeed time-invariant
        checkTimeInvariance(1e-8);
    }
}

// helper method to calculate maximum and minimum T
void ESADataPanelLCM::calculateMaxT(const FirmIdsVar& precalc, bool isContig)
{
    this->maxT = 0;
    this->minT = this->nobs;
    if (isContig) {
        // use ranges
        FirmIdsRange contigs = std::get<FirmIdsRange>(precalc);
        // iterate through
        for (size_t i = 0; i < this->nids; i++) {
            int currSize = contigs[i].second - contigs[i].first + 1;
            if (currSize > this->maxT) this->maxT = currSize;
            // minT
            if (currSize < this->minT) this->minT = currSize;
        }
    } else {
        // vector of indicies - just get the size
        FirmIdsInds noncontigs = std::get<FirmIdsInds>(precalc);
        for (size_t i = 0; i < this->nids; i++) {
            int currSize = noncontigs[i].n_elem;
            if (currSize > this->maxT) this->maxT = currSize;
            if (currSize < this->minT) this->minT = currSize;
        }
    }
}

// check time-invariance for the panel [and segmenting variables if not Markov]
void ESADataPanelLCM::checkTimeInvariance(const double tol) const
{
    // not yet implemented
    // - seg (segmenting variables) are constant over time within each panel
    // - zvi0 (firm effects) are constant over time within each panel?
    // 
}