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

#include <regex>
#include <algorithm>
#include "model/ESASfaModelTerms.hpp"

// default constructor
ESASfaModelTerms::ESASfaModelTerms() :
    termsX(std::nullopt),
    termsZmuit(std::nullopt),
    termsZuit(std::nullopt),
    termsZvit(std::nullopt),
    termsZui0(std::nullopt),
    termsZvi0(std::nullopt),
    modelType(ESASfaModelType::MODEL_UNKNOWN),
    prefixZuit(std::string{"Zuit"}),
    prefixZvit(std::string{"Zvit"}),
    prefixZui0(std::string{"Zui0"}),
    prefixZvi0(std::string{"Zvi0"}),
    prefixMuuit(std::string{"mu"}),
    seperator(std::string{"_"})
{
    
}

/// Constructor
ESASfaModelTerms::ESASfaModelTerms(
    const std::optional<std::vector<std::string>>& termsX,
    const std::optional<std::vector<std::string>>& termsZmuit,
    const std::optional<std::vector<std::string>>& termsZuit,
    const std::optional<std::vector<std::string>>& termsZvit,
    const std::optional<std::vector<std::string>>& termsZui0,
    const std::optional<std::vector<std::string>>& termsZvi0,
    const ESASfaModelType modelType,
    const std::string prefixZuit,
    const std::string prefixZvit,
    const std::string prefixZui0,
    const std::string prefixZvi0,
    const std::string prefixMuuit,
    const std::string seperator
) : termsX(termsX),
    termsZmuit(termsZmuit),
    termsZuit(termsZuit),
    termsZvit(termsZvit),
    termsZui0(termsZui0),
    termsZvi0(termsZvi0),
    modelType(modelType),
    prefixZuit(prefixZuit),
    prefixZvit(prefixZvit),
    prefixZui0(prefixZui0),
    prefixZvi0(prefixZvi0),
    prefixMuuit(prefixMuuit),
    seperator(seperator)
{
    ESASfaModelDistribution mD = ESAEnums::getDistribution(modelType);
    ESASfaModelFamily mF = ESAEnums::getModelFamily(modelType);
    // check how many terms to reserve 
    int totalSize = 0;
    if (termsX) totalSize += termsX->size();
    if (termsZmuit) totalSize += termsZmuit->size();
    if (termsZuit) totalSize += termsZuit->size();
    if (termsZvit) totalSize += termsZvit->size();
    if (termsZui0) totalSize += termsZui0->size();
    if (termsZvi0) totalSize += termsZvi0->size();
    modelTerms.reserve(totalSize);
    if (termsX){
        std::vector<ModelTerm> xTms = ESASfaModelTerms::createTermsForLocation(termsX.value(), ESASfaModelTermLocation::X);
        modelTerms.insert(modelTerms.end(), xTms.begin(), xTms.end());
    }
    if (termsZmuit && (mD == ESASfaModelDistribution::TNORM)){
        std::vector<ModelTerm> muuitTms = ESASfaModelTerms::createTermsForLocation(termsZmuit.value(), ESASfaModelTermLocation::MUUIT, prefixMuuit, seperator);
        modelTerms.insert(modelTerms.end(), muuitTms.begin(), muuitTms.end());
    }
    bool okZuit = (mF == ESASfaModelFamily::TFE || mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE || mF == ESASfaModelFamily::CROSS || mF == ESASfaModelFamily::LC_TRE);
    if (termsZuit && okZuit){
        std::vector<ModelTerm> zuitTms = ESASfaModelTerms::createTermsForLocation(termsZuit.value(), ESASfaModelTermLocation::ZUIT, prefixZuit, seperator);
        modelTerms.insert(modelTerms.end(), zuitTms.begin(), zuitTms.end());
    }
    bool okZvit = (mF == ESASfaModelFamily::TFE || mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE || mF == ESASfaModelFamily::CROSS || mF == ESASfaModelFamily::LC_TRE);
    if (termsZvit && okZvit){
        std::vector<ModelTerm> zvitTms = ESASfaModelTerms::createTermsForLocation(termsZvit.value(), ESASfaModelTermLocation::ZVIT, prefixZvit, seperator);
        modelTerms.insert(modelTerms.end(), zvitTms.begin(), zvitTms.end());
    }
    bool okZvi0 = (mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE || mF == ESASfaModelFamily::LC_TRE);
    if (termsZvi0 && okZvi0){
        std::vector<ModelTerm> zvi0Tms = ESASfaModelTerms::createTermsForLocation(termsZvi0.value(), ESASfaModelTermLocation::ZVI0, prefixZvi0, seperator);
        modelTerms.insert(modelTerms.end(), zvi0Tms.begin(), zvi0Tms.end());
    }
    bool okZui0 = (mF == ESASfaModelFamily::GTRE);
    if (termsZui0 && okZui0){
        std::vector<ModelTerm> zui0Tms = ESASfaModelTerms::createTermsForLocation(termsZui0.value(), ESASfaModelTermLocation::ZUI0, prefixZui0, seperator);
        modelTerms.insert(modelTerms.end(), zui0Tms.begin(), zui0Tms.end());
    }
};

/// Create vector of ModeTerm structs for terms in given location
std::vector<ModelTerm> ESASfaModelTerms::createTermsForLocation(
    const std::vector<std::string>& terms,
    const ESASfaModelTermLocation& location,
    const std::string& prefix,
    const std::string& seperator
){
    std::vector<ModelTerm> termsVec(terms.size());
    for (unsigned int i = 0; i < terms.size(); ++i) {
        ModelTerm term;
        term.originalTerm = terms[i];
        // create cleaned term
        term.cleanedTerm = std::regex_replace(terms[i], std::regex(prefix + seperator), "");
        term.location = location;
        term.indexPosInLocation = i;
        termsVec[i] = term;
    }
    return termsVec;
}
std::vector<ModelTerm> ESASfaModelTerms::createTermsForLocation(
    const std::vector<std::string>& terms,
    const ESASfaModelTermLocation& location
){
    std::vector<ModelTerm> termsVec(terms.size());
    for (unsigned int i = 0; i < terms.size(); ++i){
        ModelTerm term;
        term.originalTerm = terms[i];
        term.cleanedTerm = terms[i];
        term.location = location;
        term.indexPosInLocation = i;
        termsVec[i] = term;
    }
    return termsVec;
}

std::vector<std::string> ESASfaModelTerms::allTerms() const
{
    std::vector<std::string> out(modelTerms.size());
    for (unsigned int i = 0; i < modelTerms.size(); ++i){
        out[i] = modelTerms[i].originalTerm;
    }
    return out;
}

/// Get model term structs for a location
std::vector<ModelTerm> ESASfaModelTerms::getTermsForLocation(const ESASfaModelTermLocation& location) const {
    std::vector<ModelTerm> terms;
    for (unsigned int i = 0; i < modelTerms.size(); ++i) {
        if (modelTerms[i].location == location) {
            terms.push_back(modelTerms[i]);
        }
    }
    return terms;
}

/// Get model term structs for terms in X
std::vector<ModelTerm> ESASfaModelTerms::getXTerms() const {
    return getTermsForLocation(ESASfaModelTermLocation::X);
}

/// Get model term structs for terms in Zuit
std::vector<ModelTerm> ESASfaModelTerms::getZuitTerms() const {
    return getTermsForLocation(ESASfaModelTermLocation::ZUIT);
}

/// Get model term structs for terms in Zvit
std::vector<ModelTerm> ESASfaModelTerms::getZvitTerms() const {
    return getTermsForLocation(ESASfaModelTermLocation::ZVIT);
}

/// Get model term structs for terms in Zui0
std::vector<ModelTerm> ESASfaModelTerms::getZui0Terms() const {
    return getTermsForLocation(ESASfaModelTermLocation::ZUI0);
}

/// Get model term structs for terms in Zvi0
std::vector<ModelTerm> ESASfaModelTerms::getZvi0Terms() const {
    return getTermsForLocation(ESASfaModelTermLocation::ZVI0);
}

/// Get model term structs for terms in Muuit
std::vector<ModelTerm> ESASfaModelTerms::getMuuitTerms() const {
    return getTermsForLocation(ESASfaModelTermLocation::MUUIT);
}

/// get all other terms anywhere that match a given cleaned name
std::vector<ModelTerm> ESASfaModelTerms::getTermsForCleanedName(const std::string& cleanedName) const {
    std::vector<ModelTerm> terms;
    for (unsigned int i = 0; i < modelTerms.size(); ++i) {
        // skip loop depending where term is located, and the model type
        if (!ESAEnums::isModelTypeAndLocationCompatible(modelType, modelTerms[i].location)) {
            continue;
        }
        if (modelTerms[i].cleanedTerm == cleanedName) {
            terms.push_back(modelTerms[i]);
        }
    }
    return terms;
}

/// get all other terms anywhere that match a given cleaned name only in composed error term
std::vector<ModelTerm> ESASfaModelTerms::getTermsForCleanedNameInComposedError(const std::string& cleanedName) const {
    std::vector<ModelTerm> terms;
    for (unsigned int i = 0; i < modelTerms.size(); ++i) {
        // skip loop depending where term is located, and the model type
        if (!ESAEnums::isModelTypeAndLocationCompatible(modelType, modelTerms[i].location)) {
            continue;
        }
        if (modelTerms[i].cleanedTerm == cleanedName && (
            modelTerms[i].location == ESASfaModelTermLocation::ZUIT ||
            modelTerms[i].location == ESASfaModelTermLocation::ZVIT ||
            modelTerms[i].location == ESASfaModelTermLocation::ZUI0 ||
            modelTerms[i].location == ESASfaModelTermLocation::ZVI0 ||
            modelTerms[i].location == ESASfaModelTermLocation::MUUIT
        )) {
            terms.push_back(modelTerms[i]);
        }
    }
    return terms;
}

/// get all other terms anywhere that match a given cleaned name only in a given location
std::vector<ModelTerm> ESASfaModelTerms::getTermsForCleanedNameInLocation(const std::string& cleanedName, const ESASfaModelTermLocation& location) const {
    std::vector<ModelTerm> terms;
    for (unsigned int i = 0; i < modelTerms.size(); ++i) {
        // skip loop depending where term is located, and the model type
        if (!ESAEnums::isModelTypeAndLocationCompatible(modelType, modelTerms[i].location)) {
            continue;
        }
        if (modelTerms[i].cleanedTerm == cleanedName && modelTerms[i].location == location) {
            terms.push_back(modelTerms[i]);
        }
    }
    return terms;
}

/// 
std::vector<std::string> ESASfaModelTerms::getAllCleanedNamesInLocation(const std::vector<ESASfaModelTermLocation>& location, const bool excIntercept) const {
    std::vector<std::string> cleanedNames;
    for (unsigned int i = 0; i < modelTerms.size(); ++i) {
        for (unsigned int j = 0; j < location.size(); ++j) {
            bool isIntercept = (excIntercept == false) ? false : (modelTerms[i].cleanedTerm == "Intercept" || modelTerms[i].cleanedTerm == "cons");
            if (modelTerms[i].location == location[j] && !isIntercept) {
                cleanedNames.push_back(modelTerms[i].cleanedTerm);
            }
        }
    }
    std::sort(cleanedNames.begin(), cleanedNames.end());
    auto l = std::unique(cleanedNames.begin(), cleanedNames.end());
    cleanedNames.erase(l, cleanedNames.end());
    return cleanedNames;
}

///
std::vector<std::string> ESASfaModelTerms::getAllCleanedNamesInLocation(const ESASfaModelTermLocation& location, const bool excIntercept) const {
    std::vector<std::string> cleanedNames;
    for (unsigned int i = 0; i < modelTerms.size(); ++i) {
        bool isIntercept = (excIntercept == false) ? false : (modelTerms[i].cleanedTerm == "Intercept" || modelTerms[i].cleanedTerm == "cons");
        if (modelTerms[i].location == location && !isIntercept) {
            cleanedNames.push_back(modelTerms[i].cleanedTerm);
        }
    }
    return cleanedNames;
}
