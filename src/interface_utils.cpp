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


#include <cmath>
#include <spdlog/fmt/fmt.h>
#include "interface/interface_utils.hpp"
#include "data/ESADataCross.hpp"
#include "data/ESADataPanel.hpp"
#include "utils/enums.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"


std::shared_ptr<ESADataBase> interface::createDataObject(
    const arma::dcolvec* y,
    const arma::dmat* x,
    const arma::dmat* zmuit,
    const arma::dmat* zuit,
    const arma::dmat* zvit,
    const arma::dmat* zui0,
    const arma::dmat* zvi0,
    const arma::Col<int>* idVec,
    const arma::Col<int>* timeVec,
    const ESASfaModelType& mT
)
{
    // pointer to the underlying data object
    std::shared_ptr<ESADataBase> dataObjPtr = nullptr;
    // enums for mdoel family, and distribution
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    ESASfaModelDistribution mD = ESAEnums::getDistribution(mT);
    // create data object - depends on model type / family
    // start with true fixed effects
    if (mF == ESASfaModelFamily::TFE) {
        ESADataPanel dataObj(
            y, // y
            x, // x
            idVec, // firm identifiers
            timeVec, // time identifies
            mT, // model type
            ((mD != ESASfaModelDistribution::TNORM) ? nullptr : zmuit), // trunc norm mean
            zuit, // time-varying inefficiency
            zvit, // stochastic noise
            nullptr, // time-invariant inefficiency
            nullptr // firm effect
        );
        dataObjPtr = std::make_shared<ESADataPanel>(dataObj);
    } else if (mF == ESASfaModelFamily::TRE) {
        // true random effects model
        ESADataPanel dataObj(
            y, // y
            x, // x
            idVec, // firm identifiers
            timeVec, // time identifies
            mT, // model type
            ((mD != ESASfaModelDistribution::TNORM) ? nullptr : zmuit), // trunc norm mean
            zuit, // time-varying inefficiency
            zvit, // stochastic noise
            nullptr, // time-invariant inefficiency
            zvi0 // firm effect
        );
        dataObjPtr = std::make_shared<ESADataPanel>(dataObj);
    } else if (mF == ESASfaModelFamily::GTRE) {
        // generalized true random effects model
        ESADataPanel dataObj(
            y, // y
            x, // x
            idVec, // firm identifiers
            timeVec, // time identifies
            mT, // model type
            ((mD != ESASfaModelDistribution::TNORM) ? nullptr : zmuit), // trunc norm mean
            zuit, // time-varying inefficiency
            zvit, // stochastic noise
            zui0, // time-invariant inefficiency
            zvi0 // firm effect
        );
        dataObjPtr = std::make_shared<ESADataPanel>(dataObj);
    } else if (mF == ESASfaModelFamily::CROSS) {
        // cross-sectional [pooled] model
        ESADataCross dataObj(
            y, // y,
            x, // x,
            ((mD != ESASfaModelDistribution::TNORM) ? nullptr : zmuit), // trunc norm mean
            zuit, // inefficiency componenent
            zvit, // stochastic noise component
            mT // model type
        );
        dataObjPtr = std::make_shared<ESADataCross>(dataObj);
    } else {
        throw std::invalid_argument("Model family not recognized");
    }
    // check dataobjptr not null
    if (dataObjPtr == nullptr) {
        throw std::runtime_error("Data object iis null - something went wrong somehwere");
    }
    return dataObjPtr;
}

int findIntCntForDbl(const double& v)
{
    double absV = std::abs(v);
    bool isNeg = (v < 0.0);
    int cnt = 1;
    if (absV >= 10) cnt = 2;
    else if (absV >= 100.0) cnt = 3;
    else if (absV >= 1000.0) cnt = 4;
    else if (absV >= 10000.0) cnt = 5;
    else if (absV >= 100000.0) cnt = 6;
    else if (absV >= 1000000.0) cnt = 7;
    else if (absV >= 10000000.0) cnt = 8;
    else if (absV >= 100000000.0) cnt = 9;
    else if (absV >= 1000000000.0) cnt = 10;
    return cnt + ((isNeg == true) ? 1 : 0);
}

std::string repeat(const std::string& str, size_t n) {
    if (n == 0) return "";
    
    // 1. Pre-allocate memory (Performance critical)
    std::string result;
    result.reserve(str.length() * n);

    // 2. Append n times
    for (size_t i = 0; i < n; ++i) {
        result += str;
    }
    return result;
}

std::string padStrSidesMiddle(const std::string& str, char s, int width)
{
    if (str.size() >= width) return str;
    int start = floor( width / 2.0 - (str.size() / 2.0) );
    std::string out = std::string(start, s) + str + std::string((width - start - str.size()), s);
    return out;
}

// Crash-proof center padding
std::string padCenter(const std::string& str, int width) {
    if (static_cast<int>(str.size()) >= width) return str;
    int totalPad = width - static_cast<int>(str.size());
    int left = totalPad / 2;
    int right = totalPad - left; 
    return std::string(left, ' ') + str + std::string(right, ' ');
}

// Helper to determine integer digits in a double (for alignment)
int getIntDigits(double v) {
    double absV = std::abs(v);
    if (absV < 10.0) return (v < 0) ? 2 : 1;
    return (int)log10(absV) + 1 + ((v < 0) ? 1 : 0);
}

int getIntDigits(int v){
    int absV = std::abs(v);
    if (absV < 10) return (v < 0) ? 2 : 1;
    return (int)log10((double)absV) + 1 + ((v < 0) ? 1 : 0);
}

std::string buildGridLine(
    const std::vector<int>& colWidths, 
    int padding, 
    const std::string& wallL, 
    const std::string& wallR,
    const std::string& lineChar,
    const std::vector<std::string>& intersections
) {
    std::string line = wallL;
    int leftDash = padding / 2;
    int rightDash = padding - leftDash - 1;
    if (rightDash < 0) rightDash = 0;

    std::string padL = repeat(lineChar, leftDash);
    std::string padR = repeat(lineChar, rightDash);

    for (size_t i = 0; i < colWidths.size(); ++i) {
        // Draw the horizontal line for the column content
        line += repeat(lineChar, colWidths[i]);
        
        // If this is not the last column, draw the intersection
        if (i < colWidths.size() - 1) {
            std::string junction = (i < intersections.size()) ? intersections[i] : "┼";
            line += padL + junction + padR;
        }
    }
    line += wallR;
    return line;
}

void interface::printModelOutput(
    const std::shared_ptr<ESADataBase>& dataObjPtr,
    const arma::dmat& modelSummary,
    const std::vector<std::string>& modelSummaryTerms,
    const ESASigmaParams& sigParams,
    const int nsim,
    const double llscore,
    const double gnorm,
    const bool clusteredSE,
    const HaltonSettings& hsetting,
    const double confInt,
    const int decimalPlaces,
    const int consoleWidth,
    const std::optional<std::string>& idColName,
    const std::optional<std::string>& timeColName
)
{
    printModelOutput(
        dataObjPtr->getModelType(),
        (int)dataObjPtr->getNobs(),
        (int)dataObjPtr->getNids(),
        (int)dataObjPtr->getMaxT(),
        (int)dataObjPtr->getMinT(),
        modelSummary,
        modelSummaryTerms,
        sigParams,
        nsim,
        llscore,
        gnorm,
        clusteredSE,
        hsetting,
        confInt,
        decimalPlaces,
        consoleWidth,
        idColName,
        timeColName
    );
}

void interface::printModelOutput(
    const ESASfaModelType mT,
    const int nobs,
    const int nids,
    const int maxT,
    const int minT,
    const arma::dmat& modelSummary,
    const std::vector<std::string>& modelSummaryTerms,
    const ESASigmaParams& sigParams,
    const int nsim,
    const double llscore,
    const double gnorm,
    const bool clusteredSE,
    const HaltonSettings& hsetting,
    const double confInt,
    const int decimalPlaces,
    const int consoleWidth,
    const std::optional<std::string>& idColName,
    const std::optional<std::string>& timeColName
)
{
    // create a new empty logger for the model output, which uses the same sink as the main logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("raw");
    if (logger == nullptr) {
        // get the logger (platform independent)
        const auto lgr = ESALogger::logger();
        // get the sinks associated with that logger
        std::vector<std::shared_ptr<spdlog::sinks::sink>>& sinks = lgr->sinks();
        logger = std::make_shared<spdlog::logger>("raw", sinks.begin(), sinks.end());
    }
    logger->set_pattern("%v");
    // ---- constants ----
    const ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    // const ESASfaModelDistribution mD = ESAEnums::getDistribution(mT);
    // build up extra params
    std::string sigmau = "E(σ_uit)", sigmav = "E(σ_vit)", sigmau0 = "E(σ_ui0)", sigmav0 = "E(σ_vi0)";
    std::string lam = "E(λ)", lam0 = "E(λ_0)", Lam = "E(Λ)";
    std::map<std::string, double> extraParams;
    if (mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE) {
        extraParams = std::map<std::string, double>{
            {sigmau, sigParams.s_uit},
            {sigmav, sigParams.s_vit},
            {sigmav0, sigParams.s_vi0},
            {lam, sigParams.lambda}
        };
    }
    if (mF == ESASfaModelFamily::GTRE) {
        extraParams[sigmau0] = sigParams.s_ui0;
        extraParams[lam0] = sigParams.lambda_0;
        extraParams[Lam] = sigParams.BigLambda;
    }
    // constants
    int dp = (decimalPlaces >= 0) ? decimalPlaces : 4;
    // ────────────────────────────────────────────
    //          SETUP FOR PRE-REGRESSION INFO
    // ────────────────────────────────────────────
    logger->info("\n\n");
    double avgT = nobs/nids;
    // first create a box holding information on nobs, ngroups,
    // obs per group: avg, max
    std::string nobsStr = "num obs", ngrpsStr = "num groups", obsAvgStr = "obs per group: avg";
    std::string obsMinStr = "               min", obsMaxStr = "               max";
    int maxInfoTxtLen = 18, longestNum = std::max(getIntDigits(nobs), std::max(getIntDigits(nids), getIntDigits(maxT)));
    // total box width
    // number is the longest int + 2 points for the dp and 0, and then another buffer one
    int infoNumWidth = longestNum + 3;
    int infoInternal = maxInfoTxtLen + infoNumWidth + 4;
    int infoTotal = infoInternal + 2 + 2;
    // build the title
    std::vector<std::string> titleVec;
    if (mF == ESASfaModelFamily::GTRE) {
        titleVec = {"Generalized", "true", "random", "effects"};
    } else if (mF == ESASfaModelFamily::TRE) {
        titleVec = {"True", "random", "effects"};
    } else if (mF == ESASfaModelFamily::TFE) {
        titleVec = {"True", "fixed", "effects"};
    } else if (mF == ESASfaModelFamily::CROSS) {
        titleVec = {"Cross", "sectional"};
    } else if (mF == ESASfaModelFamily::LC_TRE) {
        titleVec = {"Latent", "class", "true", "random", "effects"};
    }
    // amount of space to print title.
    int titleSpace = consoleWidth - infoTotal;
    std::vector<std::string> titleRows;
    std::string currLine = "";
    bool startNewLine = false;
    for (const auto& s : titleVec) {
        // check if should create a new line, if so, reset temp variables
        if (startNewLine) {
            titleRows.push_back(currLine);
            currLine = "";
            startNewLine = false;
        }
        // check if the string will fit in (with a space)
        int rem = titleSpace - currLine.length() - s.length() - 1;
        if (rem > 0) {
            // there is enough space to fit the word on
            currLine += " " + s;
        } else if (currLine.size() == 0 && rem < 0) {
            // literally no space at all, truncate to what does fit
            currLine = s.substr(0, titleSpace);
            // start new line on the next iteration
            startNewLine = true;
        } else {
            // pad the remaining space, and flag that should start a new line on
            // the next iteration
            currLine += repeat(" ", (titleSpace - currLine.length()));
            startNewLine = true;
        }
    }
    if (currLine.size() > 0) titleRows.push_back(currLine);
    // add the id & time col if provided (and an appropriate panel model)
    if (mF == ESASfaModelFamily::GTRE || mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::TFE) {
        titleRows.push_back("");
        if (idColName.has_value()) {
            // check if we need to truncate the id column
            std::string idStr = " Group var: ";
            if (titleSpace - idStr.length() - idColName.value().length() < 0) {
                // should truncate it
                int trunclen = titleSpace - idStr.length() - 1;
                idStr += idColName.value().substr(0, std::min((int)idColName.value().length(), trunclen));
            } else {
                idStr += idColName.value();
            }
            titleRows.push_back(idStr);
        }
        if (timeColName.has_value()) {
            std::string timeStr = " Time var: ";
            if (titleSpace - timeStr.length() - timeColName.value().length() < 0) {
                int trunclen = titleSpace - timeStr.length() - 1;
                timeStr += timeColName.value().substr(0, std::min((int)timeColName.value().length(), trunclen));
            } else {
                timeStr += timeColName.value();
            }
            titleRows.push_back(timeStr);
        }
    }
    std::string boxTopLine = "┌" + repeat("─", infoInternal) + "┐";
    std::string boxBotLine = "└" + repeat("─", infoInternal) + "┘";
    std::string titleBlankSpace = repeat(" ", titleSpace);
    std::vector<std::string> infoTxt = {nobsStr, ngrpsStr, obsAvgStr, obsMinStr, obsMaxStr};
    std::vector<double> infoVals = {(double)nobs, (double)nids, (double)avgT, (double)minT, (double)maxT};
    // iterate and create the header
    for (int i = 0; i < 7; i++) {
        // check if there is a row avaiable in the titleRows
        std::string fullTitleStr = (i < titleRows.size()) ? titleRows[i] : titleBlankSpace;
        // final check for padding
        if (fullTitleStr.length() < titleSpace) {
            fullTitleStr += repeat(" ", (titleSpace - fullTitleStr.length()));
        }
        if (i == 0) logger->info("{} {}", fullTitleStr, boxTopLine);
        else if (i == 6) logger->info("{} {}", fullTitleStr, boxBotLine);
        else {
            logger->info("{} │ {:<{}} = {:>{}.1f}│", fullTitleStr, infoTxt[i - 1], maxInfoTxtLen, infoVals[i - 1], infoNumWidth);
        }
    }
    // check if anything remains in the title str
    if (titleRows.size() > 7) {
        for (int i = 7; i < titleRows.size(); i++) {
            logger->info("{}", titleRows[i]);
        }
    }

    // add some blank lines
    logger->info("\n");
    std::string logMeth = (mF == ESASfaModelFamily::GTRE || mF == ESASfaModelFamily::TRE) ? "Log simulated-likelihood" : "Log likelihood";
    logger->info(" {} = {:.10f}\n", logMeth, llscore);
    // print information on the halton draws
    if (mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE) {
        // both tre & gtre have halton draws for vi0
        logger->info(" Number of Halton draws for vi0 = {}", nsim);
        // build the string
        std::string vi0HaltStr = "";
        if (hsetting.scrambled) vi0HaltStr += "scrambled";
        if (hsetting.shuffle) vi0HaltStr += (vi0HaltStr != "") ? "shuffled" : " and shuffled";
        if (vi0HaltStr != "") logger->info(" Halton draw for vi0 is {}", vi0HaltStr);
        logger->info(" Halton draw for vi0 has base = {}", hsetting.base);
    }
    if (mF == ESASfaModelFamily::GTRE) {
        // gtre has halton draws for ui0 as well
        logger->info(" Number of Halton draws for vi0 = {}", nsim);
        // build the string
        std::string ui0HaltStr = "";
        if (hsetting.scrambled) ui0HaltStr += "scrambled";
        if (hsetting.shuffle) ui0HaltStr += (ui0HaltStr != "") ? "shuffled" : " and shuffled";
        if (ui0HaltStr != "") logger->info(" Halton draw for ui0 is {}", ui0HaltStr);
        logger->info(" Halton draw for ui0 has base = {}", hsetting.ui0Base);
    }
    if (mF == ESASfaModelFamily::TRE || mF == ESASfaModelFamily::GTRE){
        logger->info(" Halton burn-in = {}", hsetting.burnin);
    }
    if (clusteredSE){
        std::string clusStr = "(std. err adjusted for " + std::to_string(int(nids)) + " clusters in ";
        if (idColName.has_value()) {
            clusStr += idColName.value();
        } else {
            clusStr += "group var";
        }
        clusStr += ")";
        logger->info("{:>{}}", clusStr, consoleWidth);
    }

    // ────────────────────────────────────────────
    //          SETUP FOR REGRESSION TABLE
    // ────────────────────────────────────────────
    int dpZ = 2; 
    int dpP = 3; 
    bool showCI = (modelSummary.n_cols >= 6);
    int ciGap = 3; // Fixed width for the gap between CI Lower and Upper

    // 2. Scan for Column Widths
    int wEst = 8, wSE = 8, wZ = 6, wP = 6, wCI = 10;
    
    for (size_t r = 0; r < modelSummary.n_rows; ++r) {
        wEst = std::max(wEst, getIntDigits(modelSummary(r, 0)) + 1 + dp);
        wSE = std::max(wSE, getIntDigits(modelSummary(r, 1)) + 1 + dp);
        wZ = std::max(wZ, getIntDigits(modelSummary(r, 2)) + 1 + dpZ);
        wP = std::max(wP, getIntDigits(modelSummary(r, 3)) + 1 + dpP);
        if (showCI) {
            wCI = std::max(wCI, getIntDigits(modelSummary(r, 4)) + 1 + dp);
            wCI = std::max(wCI, getIntDigits(modelSummary(r, 5)) + 1 + dp);
        }
    }
    
    // Header Buffers
    wEst = std::max(wEst, (int)std::string("Estimate").length());
    wSE  = std::max(wSE,  (int)std::string("Std. Err").length());
    wZ   = std::max(wZ,   (int)std::string("z-stat").length());
    wP   = std::max(wP,   (int)std::string("P>|z|").length());

    // 3. Layout Calculation
    int minColPad = 3;
    int maxColPad = 6; 
    
    // Width of all DATA columns (excluding variable name)
    // Est + SE + Z + P + (CI_L + CI_Gap + CI_R)
    int fixedContentWidth = wEst + wSE + wZ + wP + (showCI ? (wCI * 2 + ciGap) : 0);
    
    // Separators: Var|Est|SE|Z|P(|CI) -> 4 separators normally, 5 if CI
    int numSeparators = showCI ? 5 : 4; 
    
    int maxVarLen = 10; 
    for(const auto& name : modelSummaryTerms) maxVarLen = std::max(maxVarLen, (int)name.length());

    // Visual width of walls: "┃ " is 2 cols, " ┃" is 2 cols. Total 4.
    int wallOverhead = 4; 
    int requiredWidthPlain = maxVarLen + fixedContentWidth + (numSeparators * minColPad) + wallOverhead;
    // whether or not to use box mode, or standard
    bool useBox = (consoleWidth >= requiredWidthPlain);
    //
    int varWidth = maxVarLen;
    int padding = minColPad;
    // 
    if (useBox) {
        // distribute extra space
        int remaining = consoleWidth - (maxVarLen + fixedContentWidth + wallOverhead);
        // increase padding up to the limit (if there is space)
        padding = remaining / numSeparators;
        if (padding > maxColPad) padding = maxColPad;
        if (padding < minColPad) padding = minColPad;
        // give remaining space to the variable name
        int usedSoFar = maxVarLen + fixedContentWidth + (numSeparators * padding) + wallOverhead;
        if (consoleWidth > usedSoFar) {
            varWidth += (consoleWidth - usedSoFar);
        }
    } else {
        // tight mode: shrink variable column
        wallOverhead = 0;
        varWidth = consoleWidth - (fixedContentWidth + (numSeparators * minColPad));
        if (varWidth < 10) varWidth = 10; 
    }
    // construct formatting strings
    // Sep visual width = padding.
    // Box Mode: "  │  "
    // Plain Mode: "     "
    std::string sep;
    if (useBox) {
        int leftSp = padding / 2;
        int rightSp = padding - leftSp - 1; 
        // Ensure non-negative
        if (rightSp < 0) rightSp = 0; 
        sep = std::string(leftSp, ' ') + "│" + std::string(rightSp, ' ');
    } else {
        sep = std::string(padding, ' ');
    }
    // the walls for the left and right
    std::string wallL = useBox ? "┃ " : "";
    std::string wallR = useBox ? " ┃" : "";
    // VISUAL width of walls (for math)
    int visWallL = useBox ? 2 : 0; 
    int visWallR = useBox ? 2 : 0;
    // confidence interval header spans two columns
    int wCIFull = showCI ? (wCI * 2 + ciGap) : 0;
    std::string ciSep = std::string(ciGap, ' ');
    // wall characters
    std::string wallCharL = useBox ? "┠" : "├"; 
    std::string wallCharR = useBox ? "┨" : "┤";
    // intersection patterns
    // standard pattern for table row
    std::vector<std::string> xStandard;
    for(int i=0; i<numSeparators; ++i) xStandard.push_back("┼");
    // pattern for full x row [e.g., account for the CI]
    std::vector<std::string> xFullRow;
    if(showCI) xFullRow = {"┼", "┼", "┼", "┼", "┼", "─"};
    else       xFullRow = {"┼", "┼", "┼", "┼"};
    // pattern for sigma / extraParams divider
    std::vector<std::string> xSigmaDiv;
    if(showCI) xSigmaDiv = {"┼", "┴", "┴", "┴", "┴", "─"}; 
    else       xSigmaDiv = {"┼", "┴", "┴", "┴"};

    // ----------------------------------------------------
    //          PRINT REGRESSION TABLE
    // ----------------------------------------------------
    // print outline
    if (useBox) logger->info("┏" + repeat("━", consoleWidth - 2) + "┓");
    else logger->info(repeat("━", consoleWidth));
    // print header
    logger->info("{}{}{}{}{}{}{}{}{}{}{}{}", 
        wallL, 
        padCenter("Variable", varWidth), sep, 
        padCenter("Estimate", wEst), sep, 
        padCenter("Std. Err", wSE), sep, 
        padCenter("z", wZ), sep, 
        padCenter("P>|z|", wP), 
        showCI ? (sep + padCenter("[" + std::to_string((int)(confInt*100)) + "% CI]", wCIFull)) : "", 
        wallR
    );
    // seperator between the header and the regression outputs
    if (useBox) logger->info("┃" + repeat("─", consoleWidth - 2) + "┃");
    else logger->info(repeat("─", consoleWidth));
    // draw header seperator (all crosses)
    // std::vector<int> currentWidths = {varWidth, wEst, wSE, wZ, wP};
    // if(showCI) { currentWidths.push_back(wCI); currentWidths.push_back(wCI); }
    // std::vector<int> lineWidths = {varWidth, wEst, wSE, wZ, wP};
    // if(showCI) {
    //     lineWidths.push_back(wCI);
    //     lineWidths.push_back(wCI);
    // }
    // std::string ciGapLine = repeat("─", ciGap);
    // if(showCI) xFullRow[5] = ciGapLine; 
    // if(showCI) xSigmaDiv[5] = ciGapLine;
    // logger->info(buildGridLine(lineWidths, padding, wallCharL + " ", " " + wallCharR, "─", xFullRow));
    // print regression table rows
    // create flags for when z terms first appear
    bool hadZuit = false, hadZvit = false, hadZui0 = false, hadZvi0 = false;
    for (size_t r = 0; r < modelSummary.n_rows; ++r) {
        std::string vName = modelSummaryTerms[r];
        bool printDivider = false;
        if (vName.rfind("Zuit_", 0) == 0 && !hadZuit) {
            printDivider = true;
            hadZuit = true;
        } else if (vName.rfind("Zvit_", 0) == 0 && !hadZvit) {
            printDivider = true;
            hadZvit = true;
        } else if (vName.rfind("Zui0_", 0) == 0 && !hadZui0) {
            printDivider = true;
            hadZui0 = true;
        } else if (vName.rfind("Zvi0_", 0) == 0 && !hadZvi0) {
            printDivider = true;
            hadZvi0 = true;
        }
        // truncate the varname if necessary
        if ((int)vName.size() > varWidth) vName = vName.substr(0, varWidth);
        // divider for Z terms
        if (printDivider) {
             if (useBox) logger->info("┃" + repeat("─", consoleWidth - 2) + "┃");
             else logger->info(repeat("─", consoleWidth));
        }
        // print each row
        if (showCI) {
            logger->info("{}{:>{}}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}", 
                wallL,
                vName, varWidth, 
                sep, modelSummary(r, 0), wEst, dp,
                sep, modelSummary(r, 1), wSE, dp,
                sep, modelSummary(r, 2), wZ, dpZ,
                sep, modelSummary(r, 3), wP, dpP,
                sep, modelSummary(r, 4), wCI, dp,
                ciSep, // Internal fixed gap
                modelSummary(r, 5), wCI, dp,
                wallR
            );
        } else {
            logger->info("{}{:>{}}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}", 
                wallL,
                vName, varWidth, 
                sep, modelSummary(r, 0), wEst, dp,
                sep, modelSummary(r, 1), wSE, dp,
                sep, modelSummary(r, 2), wZ, dpZ,
                sep, modelSummary(r, 3), wP, dpP,
                wallR
            );
        }
    }

    // 7. Print Extra Parameters (Bottom Section)
    if (useBox) logger->info("┃" + repeat("─", consoleWidth - 2) + "┃");
    else logger->info(repeat("━", consoleWidth));

    for (const auto& [name, val] : extraParams) {
        std::string vName = name;
        if ((int)vName.size() > varWidth) vName = vName.substr(0, varWidth);

        // --- ROBUST FILLER CALCULATION ---
        // We calculate exactly how much visual space we have used so far on this line
        // Used = WallL + Var + Sep + Est + WallR
        int currentVisualWidth = visWallL + varWidth + padding + wEst + visWallR;
        
        // Filler is whatever remains to reach consoleWidth
        int fillerLen = consoleWidth - currentVisualWidth;
        if (fillerLen < 0) fillerLen = 0; // Safety

        std::string filler(fillerLen, ' ');
        
        logger->info("{}{:>{}}{}{:>{}.{}f}{}{}", 
            wallL, 
            vName, varWidth, 
            sep, val, wEst, dp,
            filler,
            wallR
        );
    }

    if (useBox) logger->info("┗" + repeat("━", consoleWidth - 2) + "┛");
    else logger->info(repeat("━", consoleWidth));
}

void interface::printLcmOutput(
    const int nobs,
    const int nids,
    const int maxT,
    const int minT,
    const arma::dmat& modelSummary,
    const std::vector<std::string>& modelSummaryTerms,
    const std::vector<std::map<std::string, double>>& sigmasPerClass,
    const int nClasses,
    const int nsim,
    const double llscore,
    const double gnorm,
    const bool clusteredSE,
    const HaltonSettings& hsetting,
    const bool useGhq,
    const double confInt,
    const int decimalPlaces,
    const int consoleWidth,
    const std::optional<std::string>& idColName,
    const std::optional<std::string>& timeColName
)
{
    // ── logger ──────────────────────────────────────────────────
    std::shared_ptr<spdlog::logger> logger = spdlog::get("raw");
    if (logger == nullptr) {
        const auto lgr = ESALogger::logger();
        std::vector<std::shared_ptr<spdlog::sinks::sink>>& sinks = lgr->sinks();
        logger = std::make_shared<spdlog::logger>("raw", sinks.begin(), sinks.end());
    }
    logger->set_pattern("%v");

    // ── partition terms into seg and per-class ───────────────────
    // Seg terms:   "alpha_C:name"   (no "Class_" prefix)
    // Class terms: "Class_C:name"
    const std::string classPrefix = "Class_";
    std::vector<arma::uword> segIdx;
    std::map<int, std::vector<arma::uword>> classIdx;
    for (arma::uword i = 0; i < (arma::uword)modelSummaryTerms.size(); ++i) {
        const std::string& t = modelSummaryTerms[i];
        if (t.rfind(classPrefix, 0) == 0) {
            std::string rest = t.substr(classPrefix.size());
            std::size_t colon = rest.find(':');
            if (colon != std::string::npos) {
                try {
                    int c = std::stoi(rest.substr(0, colon));
                    classIdx[c].push_back(i);
                } catch (...) {
                    segIdx.push_back(i);
                }
            } else {
                segIdx.push_back(i);
            }
        } else {
            segIdx.push_back(i);
        }
    }

    // ── compute column widths across ALL rows for consistent layout ─
    int dp   = (decimalPlaces >= 0) ? decimalPlaces : 4;
    int dpZ  = 2;
    int dpP  = 3;
    bool showCI = (modelSummary.n_cols >= 6);
    int ciGap   = 3;
    int wEst = 8, wSE = 8, wZ = 6, wP = 6, wCI = 10;
    int maxVarLen = 10;
    for (arma::uword r = 0; r < modelSummary.n_rows; ++r) {
        wEst = std::max(wEst, getIntDigits(modelSummary(r, 0)) + 1 + dp);
        wSE  = std::max(wSE,  getIntDigits(modelSummary(r, 1)) + 1 + dp);
        wZ   = std::max(wZ,   getIntDigits(modelSummary(r, 2)) + 1 + dpZ);
        wP   = std::max(wP,   getIntDigits(modelSummary(r, 3)) + 1 + dpP);
        if (showCI) {
            wCI = std::max(wCI, getIntDigits(modelSummary(r, 4)) + 1 + dp);
            wCI = std::max(wCI, getIntDigits(modelSummary(r, 5)) + 1 + dp);
        }
    }
    wEst = std::max(wEst, (int)std::string("Estimate").size());
    wSE  = std::max(wSE,  (int)std::string("Std. Err").size());
    wZ   = std::max(wZ,   (int)std::string("z").size());
    wP   = std::max(wP,   (int)std::string("P>|z|").size());
    // strip Class_C: prefix for varWidth scan
    for (arma::uword i = 0; i < (arma::uword)modelSummaryTerms.size(); ++i) {
        const std::string& t = modelSummaryTerms[i];
        std::string display = t;
        if (t.rfind(classPrefix, 0) == 0) {
            std::size_t colon = t.find(':');
            if (colon != std::string::npos) display = t.substr(colon + 1);
        }
        maxVarLen = std::max(maxVarLen, (int)display.size());
    }

    int fixedContentWidth = wEst + wSE + wZ + wP + (showCI ? (wCI * 2 + ciGap) : 0);
    int numSeparators     = showCI ? 5 : 4;
    int wallOverhead      = 4;
    int minColPad         = 3;
    int maxColPad         = 6;
    int requiredWidthPlain = maxVarLen + fixedContentWidth + (numSeparators * minColPad) + wallOverhead;
    bool useBox = (consoleWidth >= requiredWidthPlain);

    int varWidth = maxVarLen;
    int padding  = minColPad;
    if (useBox) {
        int remaining = consoleWidth - (maxVarLen + fixedContentWidth + wallOverhead);
        padding = remaining / numSeparators;
        if (padding > maxColPad) padding = maxColPad;
        if (padding < minColPad) padding = minColPad;
        int usedSoFar = maxVarLen + fixedContentWidth + (numSeparators * padding) + wallOverhead;
        if (consoleWidth > usedSoFar) varWidth += (consoleWidth - usedSoFar);
    } else {
        wallOverhead = 0;
        varWidth = consoleWidth - (fixedContentWidth + (numSeparators * minColPad));
        if (varWidth < 10) varWidth = 10;
    }

    std::string sep;
    if (useBox) {
        int leftSp  = padding / 2;
        int rightSp = padding - leftSp - 1;
        if (rightSp < 0) rightSp = 0;
        sep = std::string(leftSp, ' ') + "│" + std::string(rightSp, ' ');
    } else {
        sep = std::string(padding, ' ');
    }
    std::string wallL    = useBox ? "┃ " : "";
    std::string wallR    = useBox ? " ┃" : "";
    int visWallL         = useBox ? 2 : 0;
    int visWallR         = useBox ? 2 : 0;
    int wCIFull          = showCI ? (wCI * 2 + ciGap) : 0;
    std::string ciSep    = std::string(ciGap, ' ');
    std::string ciConf   = "[" + std::to_string((int)(confInt * 100)) + "% CI]";
    // ── helper: print one row ────────────────────────────────────
    auto printRow = [&](const std::string& name, arma::uword r) {
        std::string vName = name;
        if ((int)vName.size() > varWidth) vName = vName.substr(0, varWidth);
        if (showCI) {
            logger->info("{}{:>{}}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}",
                wallL, vName, varWidth,
                sep, modelSummary(r, 0), wEst, dp,
                sep, modelSummary(r, 1), wSE,  dp,
                sep, modelSummary(r, 2), wZ,   dpZ,
                sep, modelSummary(r, 3), wP,   dpP,
                sep, modelSummary(r, 4), wCI,  dp,
                ciSep, modelSummary(r, 5), wCI, dp,
                wallR);
        } else {
            logger->info("{}{:>{}}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}",
                wallL, vName, varWidth,
                sep, modelSummary(r, 0), wEst, dp,
                sep, modelSummary(r, 1), wSE,  dp,
                sep, modelSummary(r, 2), wZ,   dpZ,
                sep, modelSummary(r, 3), wP,   dpP,
                wallR);
        }
    };
    // ── helper: print column header line ────────────────────────
    auto printHeader = [&]() {
        logger->info("{}{}{}{}{}{}{}{}{}{}{}{}",
            wallL,
            padCenter("Variable",  varWidth), sep,
            padCenter("Estimate",  wEst),     sep,
            padCenter("Std. Err",  wSE),      sep,
            padCenter("z",         wZ),        sep,
            padCenter("P>|z|",     wP),
            showCI ? (sep + padCenter(ciConf, wCIFull)) : "",
            wallR);
    };
    // ── helper: print divider line ───────────────────────────────
    auto printDivLine = [&]() {
        if (useBox) logger->info("┃" + repeat("─", consoleWidth - 2) + "┃");
        else        logger->info(repeat("─", consoleWidth));
    };
    // ── helper: print extra (sigma) rows for a class ────────────
    auto printExtras = [&](const std::map<std::string, double>& extras) {
        if (extras.empty()) return;
        printDivLine();
        for (const auto& [name, val] : extras) {
            std::string vName = name;
            if ((int)vName.size() > varWidth) vName = vName.substr(0, varWidth);
            int currentVisualWidth = visWallL + varWidth + padding + wEst + visWallR;
            int fillerLen = consoleWidth - currentVisualWidth;
            if (fillerLen < 0) fillerLen = 0;
            logger->info("{}{:>{}}{}{:>{}.{}f}{}{}",
                wallL, vName, varWidth,
                sep, val, wEst, dp,
                std::string(fillerLen, ' '), wallR);
        }
    };
    // ── header block (title + stats box + LL line) ──────────────
    logger->info("\n\n");
    double avgT = (nids > 0) ? (double)nobs / (double)nids : 0.0;
    std::string nobsStr = "num obs",         ngrpsStr  = "num groups";
    std::string obsAvgStr = "obs per group: avg";
    std::string obsMinStr = "               min", obsMaxStr = "               max";
    int maxInfoTxtLen = 18;
    int longestNum = std::max(getIntDigits(nobs), std::max(getIntDigits(nids), getIntDigits(maxT)));
    int infoNumWidth = longestNum + 3;
    int infoInternal = maxInfoTxtLen + infoNumWidth + 4;
    int infoTotal = infoInternal + 2 + 2;
    int titleSpace = consoleWidth - infoTotal;
    // model name
    std::vector<std::string> titleVec = {"Latent", "class", "true", "random", "effects"};
    std::vector<std::string> titleRows;
    std::string currLine = "";
    bool startNewLine = false;
    // iterate thru the words in the title vector and print on either current, or a subsequent line
    for (const auto& s : titleVec) {
        if (startNewLine) { 
            titleRows.push_back(currLine); currLine = ""; startNewLine = false;
        }
        int rem = titleSpace - (int)currLine.size() - (int)s.size() - 1;
        if (rem > 0) {
            currLine += " " + s;
        } else if (currLine.empty() && rem < 0) {
            currLine = s.substr(0, titleSpace);
            startNewLine = true;
        } else {
            currLine += repeat(" ", titleSpace - (int)currLine.size());
            startNewLine = true;
        }
    }
    if (!currLine.empty()) titleRows.push_back(currLine);
    // group var / time var lines
    titleRows.push_back("");
    if (idColName.has_value()) {
        std::string s = " Group var: " + idColName.value();
        if ((int)s.size() > titleSpace) s = s.substr(0, titleSpace);
        titleRows.push_back(s);
    }
    if (timeColName.has_value()) {
        std::string s = " Time var: " + timeColName.value();
        if ((int)s.size() > titleSpace) s = s.substr(0, titleSpace);
        titleRows.push_back(s);
    }
    // setup the box
    std::string boxTop = "┌" + repeat("─", infoInternal) + "┐";
    std::string boxBot = "└" + repeat("─", infoInternal) + "┘";
    std::string blankTitle = repeat(" ", titleSpace);
    std::vector<std::string> infoTxt = {nobsStr, ngrpsStr, obsAvgStr, obsMinStr, obsMaxStr};
    std::vector<double> infoVals = {(double)nobs, (double)nids, avgT, (double)minT, (double)maxT};
    for (int i = 0; i < 7; ++i) {
        std::string tRow = (i < (int)titleRows.size()) ? titleRows[i] : blankTitle;
        if ((int)tRow.size() < titleSpace) tRow += repeat(" ", titleSpace - (int)tRow.size());
        if      (i == 0) logger->info("{} {}", tRow, boxTop);
        else if (i == 6) logger->info("{} {}", tRow, boxBot);
        else             logger->info("{} │ {:<{}} = {:>{}.1f}│", tRow, infoTxt[i-1], maxInfoTxtLen, infoVals[i-1], infoNumWidth);
    }
    if ((int)titleRows.size() > 7) {
        for (int i = 7; i < (int)titleRows.size(); ++i) logger->info("{}", titleRows[i]);
    }
    logger->info("\n");
    logger->info(" Log likelihood = {:.10f}\n", llscore);
    if (!useGhq) {
        logger->info(" Number of Halton draws = {}", nsim);
        std::string haltStr;
        if (hsetting.scrambled) haltStr += "scrambled";
        if (hsetting.shuffle)   haltStr += (haltStr.empty() ? "" : " and ") + std::string("shuffled");
        if (!haltStr.empty()) logger->info(" Halton draw is {}", haltStr);
        logger->info(" Halton base = {}", hsetting.base);
        logger->info(" Halton burn-in = {}", hsetting.burnin);
    }
    if (clusteredSE) {
        std::string cs = "(std. err adjusted for " + std::to_string(nids) + " clusters in ";
        cs += idColName.value_or("group var");
        cs += ")";
        logger->info("{:>{}}", cs, consoleWidth);
    }
    // ── open the single outer box ────────────────────────────────
    if (useBox) logger->info("┏" + repeat("━", consoleWidth - 2) + "┓");
    else        logger->info(repeat("━", consoleWidth));
    printHeader();
    printDivLine();
    // ── segmentation parameters ──────────────────────────────────
    for (arma::uword i : segIdx) {
        printRow(modelSummaryTerms[i], i);
    }
    // ── per-class parameters ─────────────────────────────────────
    for (int c = 0; c < nClasses; ++c) {
        auto it = classIdx.find(c);
        if (it == classIdx.end()) continue;
        const auto& rows = it->second;
        // class label divider
        std::string labelLine = " Class " + std::to_string(c) + " ";
        int dashLen = consoleWidth - 3 - (int)labelLine.size();
        if (dashLen < 0) dashLen = 0;
        if (useBox) {
            logger->info("┠─{}{}┨", labelLine, repeat("─", dashLen));
        } else {
            logger->info("─{}{}", labelLine, repeat("─", dashLen));
        }
        // printDivLine();
        // rows for this class — strip "Class_C:" prefix, fire Zuit_/Zvit_/etc dividers
        bool hadZuit = false, hadZvit = false, hadZui0 = false, hadZvi0 = false;
        const std::string prefix = classPrefix + std::to_string(c) + ":";
        for (arma::uword ri : rows) {
            const std::string& full = modelSummaryTerms[ri];
            std::string display = (full.rfind(prefix, 0) == 0) ? full.substr(prefix.size()) : full;
            bool divider = false;
            // if      (display.rfind("Zuit_", 0) == 0 && !hadZuit) { divider = true; hadZuit = true; }
            // else if (display.rfind("Zvit_", 0) == 0 && !hadZvit) { divider = true; hadZvit = true; }
            // else if (display.rfind("Zui0_", 0) == 0 && !hadZui0) { divider = true; hadZui0 = true; }
            // else if (display.rfind("Zvi0_", 0) == 0 && !hadZvi0) { divider = true; hadZvi0 = true; }
            if (divider) printDivLine();
            printRow(display, ri);
        }
        // per-class sigmas
        std::map<std::string, double> extras;
        if (c < (int)sigmasPerClass.size()) extras = sigmasPerClass[c];
        printExtras(extras);
    }
    // ── close the box ────────────────────────────────────────────
    if (useBox) {
        logger->info("┗" + repeat("━", consoleWidth - 2) + "┛");
    } else {
        logger->info(repeat("━", consoleWidth));
    }
    ESALogger::logger()->sinks().clear();
    spdlog::set_default_logger(nullptr);
}

void interface::printModelTable(
    const std::string& label,
    const arma::dmat& modelSummary,
    const std::vector<std::string>& modelSummaryTerms,
    const std::map<std::string, double>& extraParams,
    const double confInt,
    const int decimalPlaces,
    const int consoleWidth
)
{
    std::shared_ptr<spdlog::logger> logger = spdlog::get("raw");
    if (logger == nullptr) {
        const auto lgr = ESALogger::logger();
        std::vector<std::shared_ptr<spdlog::sinks::sink>>& sinks = lgr->sinks();
        logger = std::make_shared<spdlog::logger>("raw", sinks.begin(), sinks.end());
    }
    logger->set_pattern("%v");
    // minimum number of decimal places
    int dp = (decimalPlaces >= 0) ? decimalPlaces : 4;
    int dpZ = 2;
    int dpP = 3;
    bool showCI = (modelSummary.n_cols >= 6);
    int ciGap = 3;
    // default widths for columns
    int wEst = 8, wSE = 8, wZ = 6, wP = 6, wCI = 10;
    for (size_t r = 0; r < modelSummary.n_rows; ++r) {
        wEst = std::max(wEst, getIntDigits(modelSummary(r, 0)) + 1 + dp);
        wSE = std::max(wSE, getIntDigits(modelSummary(r, 1)) + 1 + dp);
        wZ = std::max(wZ, getIntDigits(modelSummary(r, 2)) + 1 + dpZ);
        wP = std::max(wP, getIntDigits(modelSummary(r, 3)) + 1 + dpP);
        if (showCI) {
            wCI = std::max(wCI, getIntDigits(modelSummary(r, 4)) + 1 + dp);
            wCI = std::max(wCI, getIntDigits(modelSummary(r, 5)) + 1 + dp);
        }
    }
    // calc length of column names
    wEst = std::max(wEst, (int)std::string("Estimate").length());
    wSE = std::max(wSE, (int)std::string("Std. Err").length());
    wZ = std::max(wZ, (int)std::string("z-stat").length());
    wP = std::max(wP, (int)std::string("P>|z|").length());

    int minColPad = 3, maxColPad = 6;
    int fixedContentWidth = wEst + wSE + wZ + wP + (showCI ? (wCI * 2 + ciGap) : 0);
    int numSeparators = showCI ? 5 : 4;

    int maxVarLen = 10;
    for (const auto& name : modelSummaryTerms) maxVarLen = std::max(maxVarLen, (int)name.length());
    if (!label.empty()) maxVarLen = std::max(maxVarLen, (int)label.length());

    int wallOverhead = 4;
    int requiredWidthPlain = maxVarLen + fixedContentWidth + (numSeparators * minColPad) + wallOverhead;
    bool useBox = (consoleWidth >= requiredWidthPlain);

    int varWidth = maxVarLen;
    int padding = minColPad;
    if (useBox) {
        int remaining = consoleWidth - (maxVarLen + fixedContentWidth + wallOverhead);
        padding = remaining / numSeparators;
        if (padding > maxColPad) padding = maxColPad;
        if (padding < minColPad) padding = minColPad;
        int usedSoFar = maxVarLen + fixedContentWidth + (numSeparators * padding) + wallOverhead;
        if (consoleWidth > usedSoFar) varWidth += (consoleWidth - usedSoFar);
    } else {
        wallOverhead = 0;
        varWidth = consoleWidth - (fixedContentWidth + (numSeparators * minColPad));
        if (varWidth < 10) varWidth = 10;
    }

    std::string sep;
    if (useBox) {
        int leftSp = padding / 2;
        int rightSp = padding - leftSp - 1;
        if (rightSp < 0) rightSp = 0;
        sep = std::string(leftSp, ' ') + "│" + std::string(rightSp, ' ');
    } else {
        sep = std::string(padding, ' ');
    }
    std::string wallL = useBox ? "┃ " : "";
    std::string wallR = useBox ? " ┃" : "";
    int visWallL = useBox ? 2 : 0;
    int visWallR = useBox ? 2 : 0;
    int wCIFull = showCI ? (wCI * 2 + ciGap) : 0;
    std::string ciSep = std::string(ciGap, ' ');

    // label row (e.g. "── Class 0 ──────────")
    if (!label.empty()) {
        std::string labelLine = " " + label + " ";
        int dashLen = consoleWidth - 2 - (int)labelLine.length();
        if (dashLen < 0) dashLen = 0;
        if (useBox) {
            logger->info("┠─{}{}┨", labelLine, repeat("─", dashLen));
        } else {
            logger->info("─{}{}",   labelLine, repeat("─", dashLen));
        }
    }
    // column header
    if (useBox) {
        logger->info("┏" + repeat("━", consoleWidth - 2) + "┓");
    } else {
        logger->info(repeat("━", consoleWidth));
    }
    // print the column names
    logger->info("{}{}{}{}{}{}{}{}{}{}{}{}",
        wallL,
        padCenter("Variable", varWidth), sep,
        padCenter("Estimate", wEst), sep,
        padCenter("Std. Err", wSE), sep,
        padCenter("z", wZ), sep,
        padCenter("P>|z|", wP),
        showCI ? (sep + padCenter("[" + std::to_string((int)(confInt*100)) + "% CI]", wCIFull)) : "",
        wallR
    );
    if (useBox) {
        logger->info("┃" + repeat("─", consoleWidth - 2) + "┃");
    } else {
        logger->info(repeat("─", consoleWidth));
    }
    // rows
    bool hadZuit = false, hadZvit = false, hadZui0 = false, hadZvi0 = false;
    for (size_t r = 0; r < modelSummary.n_rows; ++r) {
        std::string vName = modelSummaryTerms[r];
        bool printDivider = false;
        if (vName.rfind("Zuit_", 0) == 0 && !hadZuit) { printDivider = true; hadZuit = true; }
        else if (vName.rfind("Zvit_", 0) == 0 && !hadZvit) { printDivider = true; hadZvit = true; }
        else if (vName.rfind("Zui0_", 0) == 0 && !hadZui0) { printDivider = true; hadZui0 = true; }
        else if (vName.rfind("Zvi0_", 0) == 0 && !hadZvi0) { printDivider = true; hadZvi0 = true; }
        if ((int)vName.size() > varWidth) vName = vName.substr(0, varWidth);
        if (printDivider) {
            if (useBox) logger->info("┃" + repeat("─", consoleWidth - 2) + "┃");
            else        logger->info(repeat("─", consoleWidth));
        }
        if (showCI) {
            logger->info("{}{:>{}}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}",
                wallL, vName, varWidth,
                sep, modelSummary(r, 0), wEst, dp,
                sep, modelSummary(r, 1), wSE,  dp,
                sep, modelSummary(r, 2), wZ,   dpZ,
                sep, modelSummary(r, 3), wP,   dpP,
                sep, modelSummary(r, 4), wCI,  dp,
                ciSep, modelSummary(r, 5), wCI, dp,
                wallR
            );
        } else {
            logger->info("{}{:>{}}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}{:>{}.{}f}{}",
                wallL, vName, varWidth,
                sep, modelSummary(r, 0), wEst, dp,
                sep, modelSummary(r, 1), wSE,  dp,
                sep, modelSummary(r, 2), wZ,   dpZ,
                sep, modelSummary(r, 3), wP,   dpP,
                wallR
            );
        }
    }

    // extra params (sigmas etc.)
    if (useBox) logger->info("┃" + repeat("─", consoleWidth - 2) + "┃");
    else        logger->info(repeat("━", consoleWidth));
    for (const auto& [name, val] : extraParams) {
        std::string vName = name;
        if ((int)vName.size() > varWidth) vName = vName.substr(0, varWidth);
        int currentVisualWidth = visWallL + varWidth + padding + wEst + visWallR;
        int fillerLen = consoleWidth - currentVisualWidth;
        if (fillerLen < 0) fillerLen = 0;
        std::string filler(fillerLen, ' ');
        logger->info("{}{:>{}}{}{:>{}.{}f}{}{}",
            wallL, vName, varWidth,
            sep, val, wEst, dp,
            filler, wallR
        );
    }
    if (useBox) logger->info("┗" + repeat("━", consoleWidth - 2) + "┛");
    else        logger->info(repeat("━", consoleWidth));
}

void interface::printLrTest(const postestimation::LogLikeRatioTest& lrTest, const int dp)
{
    // create a new empty logger for the model output, which uses the same sink as the main logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("raw");
    if (logger == nullptr) {
        // get the main logger (platform independent)
        const auto lgr = ESALogger::logger();
        // get the sinks associated with that logger
        std::vector<std::shared_ptr<spdlog::sinks::sink>>& sinks = lgr->sinks();
        // setup the raw logger
        logger = std::make_shared<spdlog::logger>("raw", sinks.begin(), sinks.end());
    }
    logger->set_pattern("%v");
    int totalOther = 6;
    // calculate the required internal width
    std::vector<std::string> rowNames = {"Statistic (D)", "Degrees of Freedom", "P-value"};
    int maxRN = 0;
    for (const auto& r : rowNames) maxRN = std::max(maxRN, (int)r.length());
    // calculate max space needed for the numbers
    int maxNumSpace = getIntDigits(lrTest.stat) + 1 + dp;
    maxNumSpace = std::max(maxNumSpace, (getIntDigits(lrTest.pval) + 1 + dp));
    maxNumSpace = std::max(maxNumSpace, (lrTest.df));
    // total internal space needed so far
    int internalSpace = maxRN + totalOther + maxNumSpace;

    std::string h0 = "H0: Prefer nested model";
    std::string h1 = "H1: Prefer complex model";
    std::string msg = lrTest.pval < 0.05 ? "Reject H0 at 5%" : "Failed to reject H0 at 5%";

    int maxInfo = std::max((int)h0.length(), (int)h1.length());
    maxInfo = std::max(maxInfo, (int)msg.length());
    // check that we dont need to add some extra to include the message strings
    if ((maxInfo + 2) > internalSpace) {
        // add some extra space to maxRn
        maxRN += (internalSpace - (maxInfo + 2));
        internalSpace = maxRN + totalOther + maxNumSpace;
    }
    int spacelesspad = (internalSpace - 2);
    logger->info("┌{}┐", repeat("─", internalSpace));
    logger->info("│{:^{}}│", "LR Test", internalSpace);
    logger->info("│{}│", repeat(" ", internalSpace));
    logger->info("│ {:^{}} │", h0, spacelesspad);
    logger->info("│ {:^{}} │", h1, spacelesspad);
    logger->info("│{}│", repeat("─", internalSpace));
    logger->info("│ {:<{}} =  {:>{}.{}f} │", rowNames[0], maxRN, lrTest.stat, maxNumSpace, dp);
    logger->info("│ {:<{}} =  {:>{}d} │", rowNames[1], maxRN, lrTest.df, maxNumSpace);
    logger->info("│ {:<{}} =  {:>{}.{}f} │", rowNames[2], maxRN, lrTest.pval, maxNumSpace, dp);
    logger->info("│{}│", repeat("─", internalSpace));
    logger->info("│ {:^{}} │", msg, spacelesspad);
    logger->info("└{}┘", repeat("─", internalSpace));
}

HaltonSettings interface::haltonSettingsForOpts(
    const int haltonBase,
    const int haltonBurnin,
    const int haltonUi0Base,
    const bool scrambledHalton,
    const bool shuffledHalton
)
{
    HaltonSettings hsetting;
    hsetting.base = haltonBase;
    hsetting.start = 7;
    hsetting.useBase = true;
    hsetting.burnin = haltonBurnin;
    hsetting.ui0Base = haltonUi0Base;
    hsetting.ui0Start = 8;
    hsetting.ui0UseBase = true;
    hsetting.scrambled = scrambledHalton;
    hsetting.shuffle = shuffledHalton;
    return hsetting;
}

ESASigmaParams interface::reconstructSigmaParams(
    const std::optional<double> s_uit,
    const std::optional<double> s_vit,
    const std::optional<double> s_vi0,
    const std::optional<double> lambda,
    const std::optional<double> s_ui0,
    const std::optional<double> lambda_0,
    const std::optional<double> BigLambda
)
{
    ESASigmaParams sig;
    if (s_uit) sig.s_uit = s_uit.value();
    if (s_vit) sig.s_vit = s_vit.value();
    if (s_vi0) sig.s_vi0 = s_vi0.value();
    if (lambda) sig.lambda = lambda.value();
    if (s_ui0) sig.s_ui0 = s_ui0.value();
    if (lambda_0) sig.lambda_0 = lambda_0.value();
    if (BigLambda) sig.BigLambda = BigLambda.value();
    return sig;
}

/// Given a column vector of log-likelihood scores, filter for valid, and sort descendingly
std::tuple<arma::dcolvec, arma::uvec> interface::sortMapLogLikeSearches(const arma::dcolvec& llScores)
{
    // finite values only
    arma::uvec validIdxs = arma::find_finite(llScores);
    // extract valid scores
    arma::dcolvec cleanScores = llScores.elem(validIdxs);
    // sort descendingly
    arma::uvec sortOrder = arma::sort_index(cleanScores, "descend");
    // apply sorting order
    arma::dcolvec sortedScores = cleanScores.elem(sortOrder);
    arma::uvec origIdxSorted = validIdxs.elem(sortOrder);
    return std::make_tuple(sortedScores, origIdxSorted);
}

/// Print a summary table of the top searches, to the raw log
void interface::printSearches(
    const arma::dcolvec& logLikeScores, 
    const arma::dmat& params,
    const int topN,
    const int consoleWidth,
    const int decimalPlaces
)
{
    // create a new empty logger for the model output, which uses the same sink as the main logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("raw");
    if (logger == nullptr) {
        // get the logger (platform independent)
        const auto lgr = ESALogger::logger();
        // get the sinks associated with that logger
        std::vector<std::shared_ptr<spdlog::sinks::sink>>& sinks = lgr->sinks();
        logger = std::make_shared<spdlog::logger>("raw", sinks.begin(), sinks.end());
    }
    logger->set_pattern("%v");
    // sort, and order loglikelihood scores
    auto [sortedLL, origIdx] = sortMapLogLikeSearches(logLikeScores);
    int maxIt = std::min((int)sortedLL.n_rows, topN);
    int maxDigiCnt = getIntDigits(maxIt);
    arma::uvec useOldIdx = origIdx.rows(arma::span(0, maxIt - 1));
    // select the subset columns based on the original index positions
    arma::dmat toUseParams = params.cols(useOldIdx);
    // for each one, find the maximum value, (e.g., per row); build each line to print
    std::vector<int> charNeeded(toUseParams.n_rows);
    for (int i = 0; i < toUseParams.n_rows; i++) {
        double maxVal = toUseParams.row(i).max();
        charNeeded[i] = getIntDigits(maxVal) + 1 + decimalPlaces;
    }
    double maxLLScore = sortedLL.max();
    int llCharNeeded = getIntDigits(maxLLScore) + 1 + decimalPlaces+2;
    std::vector<std::vector<std::string>> rowBatches;
    std::vector<std::string> titleBatches;
    for (int i = 0; i < maxIt; i++) {
        double lli = sortedLL(i);
        std::string currLine = fmt::format("{:>{}}. {:>{}.{}f} ", (i+1), maxDigiCnt, lli, llCharNeeded, decimalPlaces+2);
        std::string currTitle = "";
        int currLineCnt = currLine.length();
        int batchCnt = 0;
        // allocate the initial batch
        if (rowBatches.size() < 1) {
            rowBatches.resize(1);
            rowBatches[0] = std::vector<std::string>(maxIt);
        }
        // allocate the initial title
        if (i == 0 && titleBatches.size() == 0) {
            titleBatches.resize(1);
            currTitle = fmt::format("{:^{}}{:^{}}", "", (maxDigiCnt+1), "(ll)", (llCharNeeded+2));
        }
        // iterate thru the number of parameters
        for (int j = 0; j < toUseParams.n_rows; j++) {
            int idxPosColUse = origIdx(i);
            // check if there is space to add it to this batch
            int numCharNeeded = charNeeded[j];
            int totalCharNeeded = numCharNeeded + 4;
            if ((currLine.length() + totalCharNeeded) > consoleWidth) {
                // need to start a new batch
                // finish this line
                rowBatches[batchCnt][i] = currLine;
                // also finish the title if we're on the first pass
                if (i == 0){
                    titleBatches[batchCnt] = currTitle;
                    currTitle = fmt::format("{:^{}}{:^{}}", "", (maxDigiCnt+1), "(ll)", (llCharNeeded+2));;
                }
                currLine = fmt::format("{:>{}}. {:>{}.{}f} ", (i+1), maxDigiCnt, lli, llCharNeeded, decimalPlaces+2);
                // check if the rowBatch has this dimension for the future
                batchCnt++;
                if (rowBatches.size() < (batchCnt + 1)) {
                    rowBatches.resize(batchCnt + 1);
                    rowBatches[batchCnt] = std::vector<std::string>(maxIt);
                }
                // also finish the title
                if (i == 0 && titleBatches.size() < (batchCnt + 1)) {
                    titleBatches.resize(batchCnt + 1);
                }
            }
            // add to the existing string
            double paramVal = params(j, idxPosColUse);
            currLine += fmt::format("│ {:^{}.{}f} ", paramVal, numCharNeeded, decimalPlaces);
            if (i == 0) {
                currTitle += fmt::format("│{:^{}}", fmt::format("({})", j), numCharNeeded+2);
            }
            if (j == (toUseParams.n_rows - 1)) {
                rowBatches[batchCnt][i] = currLine;
                if (i == 0) {
                    titleBatches[batchCnt] = currTitle;
                }
            }
        }
    }
    logger->info("\nThe highest {} searches are presented alongside their parameter estimates.", maxIt);
    for (size_t i = 0; i < rowBatches.size(); i++) {
        if (i < titleBatches.size()) {
            logger->info(repeat("─", titleBatches[i].length()));
            logger->info(titleBatches[i]);
            logger->info(repeat("─", titleBatches[i].length()));
        }
        for (size_t j = 0; j < rowBatches[i].size(); j++) {
            logger->info(rowBatches[i][j]);
        }
        if (i < rowBatches.size() - 1) {
            logger->info(" (cont)...");
        } else {
            logger->info(repeat("─", titleBatches[i].length()));
        }
    }
}

/// Print elapsed time for searches, how long left to go, to a custom logger
void interface::printSearchesElapsedAndETA(
    const std::shared_ptr<spdlog::logger>& lgr,
    const double& totalMinsTaken,
    const int& i,
    const int& reps
)
{
    // calculate average time taken
    double avgTime = totalMinsTaken/i;
    double forcastedTime = avgTime * (reps - i);
    std::string elapsedUnit = "mins";
    std::string forcastUnit = "mins";
    double totalTakenPrint = totalMinsTaken;
    if (totalTakenPrint >= 60.0) {
        totalTakenPrint /= 60.0;
        elapsedUnit = "hrs";
    }
    if (forcastedTime >= 60.0) {
        forcastedTime /= 60.0;
        forcastUnit = "hrs";
    }
    // build the internal strings
    std::string elapsedStr = fmt::format("Elapsed time is {:.3f} {}", totalTakenPrint, elapsedUnit);
    std::string forecastStr = fmt::format("Estimating {:.3f} {} remaining", forcastedTime, forcastUnit);
    std::string avgTimeStr = fmt::format("({:.3f} mins per iteration)", avgTime);
    // find the maximum string length
    int maxLen = std::max(elapsedStr.length(), std::max(forecastStr.length(), avgTimeStr.length()));
    // print a lil box (ofc) 
    lgr->info("┌{}┐", repeat("─", maxLen + 2));
    lgr->info("│ {:^{}} │", elapsedStr, maxLen);
    lgr->info("│ {:^{}} │", forecastStr, maxLen);
    lgr->info("│ {:^{}} │", avgTimeStr, maxLen);
    lgr->info("└{}┘", repeat("─", maxLen + 2));
}