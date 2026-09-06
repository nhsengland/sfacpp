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

#ifndef ESA_SFA_MODEL_TERMS_HPP
#define ESA_SFA_MODEL_TERMS_HPP

#include <vector>
#include <string>
#include <optional>
#include "utils/enums.hpp"


typedef struct {
    std::string originalTerm;
    std::string cleanedTerm;
    ESASfaModelTermLocation location;
    unsigned int indexPosInLocation;
} ModelTerm;

class ESASfaModelTerms {

public:
    // default constructor
    ESASfaModelTerms();

    /// Constructors
    ESASfaModelTerms(
        const std::optional<std::vector<std::string>>& termsX,
        const std::optional<std::vector<std::string>>& termsZmuit,
        const std::optional<std::vector<std::string>>& termsZuit,
        const std::optional<std::vector<std::string>>& termsZvit,
        const std::optional<std::vector<std::string>>& termsZui0,
        const std::optional<std::vector<std::string>>& termsZvi0,
        const ESASfaModelType modelType,
        const std::string prefixZuit = "Zuit",
        const std::string prefixZvit = "Zvit",
        const std::string prefixZui0 = "Zui0",
        const std::string prefixZvi0 = "Zvi0",
        const std::string prefixMuuit = "mu",
        const std::string seperator = "_"
    );
    
    /// @brief get a vector of ModelTerm structs for the terms in the X
    /// @return vector of ModelTerm structs
    std::vector<ModelTerm> getXTerms() const;
    
    /// @brief get a vector of ModelTerm structs for the terms in the Y
    /// @return vector of ModelTerm structs
    std::vector<ModelTerm> getYTerms() const;

    /// @brief get a vector of ModelTerm structs for the terms in the Zuit
    /// @return vector of ModelTerm structs
    std::vector<ModelTerm> getZuitTerms() const;

    /// @brief get a vector of ModelTerm structs for the terms in the Zvit
    /// @return vector of ModelTerm structs
    std::vector<ModelTerm> getZvitTerms() const;

    /// @brief get a vector of ModelTerm structs for the terms in the Zui0
    /// @return vector of ModelTerm structs
    std::vector<ModelTerm> getZui0Terms() const;

    /// @brief get a vector of ModelTerm structs for the terms in the Zvi0
    /// @return vector of ModelTerm structs
    std::vector<ModelTerm> getZvi0Terms() const;
    
    /// @brief get a vector of ModelTerm structs for the terms in the Muuit
    /// @return vector of ModelTerm structs
    std::vector<ModelTerm> getMuuitTerms() const;

    /// @brief get a vector of all other variables that match a given cleaned name
    /// @param cleanedName
    /// @return vector of ModelTerm structs
    std::vector<ModelTerm> getTermsForCleanedName(const std::string& cleanedName) const;

    /// @brief get a vector of all other variables that match a given cleaned name only in composed error term
    /// @param cleanedName
    /// @return vector of ModelTerm structs
    std::vector<ModelTerm> getTermsForCleanedNameInComposedError(const std::string& cleanedName) const;

    std::vector<ModelTerm> getTermsForCleanedNameInLocation(const std::string& cleanedName, const ESASfaModelTermLocation& location) const;
    std::vector<std::string> getAllCleanedNamesInLocation(const std::vector<ESASfaModelTermLocation>& location, const bool excIntercept = false) const;
    std::vector<std::string> getAllCleanedNamesInLocation(const ESASfaModelTermLocation& location, const bool excIntercept = false) const;

    /// @brief get all of the terms for the given structure
    std::vector<std::string> allTerms() const;

private:
    const std::optional<std::vector<std::string>> termsX;
    const std::optional<std::vector<std::string>> termsZmuit;
    const std::optional<std::vector<std::string>> termsZuit;
    const std::optional<std::vector<std::string>> termsZvit;
    const std::optional<std::vector<std::string>> termsZui0;
    const std::optional<std::vector<std::string>> termsZvi0;
    const ESASfaModelType modelType;
    const std::string prefixZuit;
    const std::string prefixZvit;
    const std::string prefixZui0;
    const std::string prefixZvi0;
    const std::string prefixMuuit;
    const std::string seperator;
    std::vector<ModelTerm> modelTerms;

    /// @brief create a vector of ModelTerm structs for the terms in a given location
    /// @param 
    std::vector<ModelTerm> createTermsForLocation(
        const std::vector<std::string>& terms,
        const ESASfaModelTermLocation& location,
        const std::string& prefix,
        const std::string& sep
    );
    std::vector<ModelTerm> createTermsForLocation(
        const std::vector<std::string>& terms,
        const ESASfaModelTermLocation& location
    );

    /// @brief get a vector of ModelTerm structs for the terms in a given location
    /// @param location
    /// @return vector of ModelTerm structs    
    std::vector<ModelTerm> getTermsForLocation(const ESASfaModelTermLocation& location) const;

};

#endif // ESA_SFA_MODEL_TERMS_HPP