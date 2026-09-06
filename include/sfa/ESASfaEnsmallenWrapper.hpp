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


#ifndef ESA_SFA_ENSMALLEN_WRAPPER_HPP
#define ESA_SFA_ENSMALLEN_WRAPPER_HPP

#include <memory>
#include "sfa/ESASfaBase.hpp"
#include "data/ESADataPanel.hpp"
#include "math/esamath.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"

#ifdef WITHENSMALLEN
#include <ensmallen.hpp>
#endif // WITHENSMALLEN

class ESASfaEnsmallenWrapper
{

public:
    ESASfaEnsmallenWrapper(
        std::shared_ptr<ESASfaBase> sfaPtr,
        bool analyticalGrad = true,
        const bool threaded = true
    ) : _sfaPtr(sfaPtr), _analyticalGrad(analyticalGrad), _threaded(threaded)
    {

    }

protected:
    std::shared_ptr<ESASfaBase> _sfaPtr;
    bool _analyticalGrad;
    bool _threaded;

    double objectiveFunc(const arma::mat& x)
    {
        double ll = this->_sfaPtr->operator()(x);
        return ll;
    }

    void grad(const arma::mat& x, arma::mat& gradient)
    {
        // arma::dmat gr =  this->_sfaPtr->gradient(x, 1e-8, this->_analyticalGrad);
        // double nobs = this->_sfaPtr->getDataObj()->getNobs();
        // gradient = -(esamath::colSum(gr).t() / nobs);
        // ESALogger::logger()->trace("called in Gradient\ngrad {}", gradient);
        arma::dmat gr;
        this->_sfaPtr->gradHess(
            x,
            1e-8,
            this->_analyticalGrad,
            HessianCalcMethod::ANALYTICAL, // irrelevant
            0, // irrelevant,
            this->_threaded,
            &gr,
            nullptr
        );
        double denom = this->_sfaPtr->getDataObj()->getNids();
        gradient = -(esamath::colSum(gr).t() / denom);
        ESALogger::logger()->trace("called in Gradient\ngrad {}", gradient);
    }

    double objectiveFuncGrad(const arma::mat& x, arma::mat& g)
    {
        // arma::dmat gr = this->_sfaPtr->gradient(x, 1e-8, this->_analyticalGrad);
        arma::dmat gr;
        this->_sfaPtr->gradHess(
            x,
            1e-8,
            this->_analyticalGrad,
            HessianCalcMethod::ANALYTICAL, // irrelevant
            0, // irrelevant,
            this->_threaded,
            &gr,
            nullptr
        );
        double denom = this->_sfaPtr->getDataObj()->getNids();
        arma::dmat grAvg = -(esamath::colSum(gr).t() / denom);
        g = std::move(grAvg);
        double ll = this->_sfaPtr->operator()(x);
        return ll;
    }
};


class ESASfaEnsmallenDifferentiable : public ESASfaEnsmallenWrapper
{
public:
    ESASfaEnsmallenDifferentiable(
        std::shared_ptr<ESASfaBase> sfaPtr,
        bool analyticalGrad = true
    ) : ESASfaEnsmallenWrapper(sfaPtr, analyticalGrad)
    {

    }

    double Evaluate(const arma::mat& x)
    {
        return objectiveFunc(x);
    }

    void Gradient(const arma::mat& x, arma::mat& gradient)
    {
        grad(x, gradient);
    }

    double EvaluateWithGradient(const arma::mat& x, arma::mat& g)
    {
        double ll = objectiveFuncGrad(x, g);
        return ll;
    }
};

class ESASfaEnsmallenArbitary : public ESASfaEnsmallenWrapper
{

public:
    ESASfaEnsmallenArbitary(
        std::shared_ptr<ESASfaBase> sfaPtr,
        bool analyticalGrad = true // dont actually need this - only for template
    ) : ESASfaEnsmallenWrapper(sfaPtr, true)
    {

    }

    double Evaluate(const arma::mat& x)
    {
        double ll = objectiveFunc(x);
        ESALogger::logger()->trace("ensmallen arbitary eval: {}", ll);
        return ll;
    }
};

class ESASfaEnsmallenDifferentiableSeperable : public ESASfaEnsmallenWrapper
{

public:
    ESASfaEnsmallenDifferentiableSeperable(
        std::shared_ptr<ESASfaBase> sfaPtr,
        bool analyticalGrad = true
    ) : ESASfaEnsmallenWrapper(sfaPtr, analyticalGrad)
    {
        // implementation for differentiable seperable only covers panel methods
        // try casting the data object ptr to ESADataPanel
        if (!dynamic_cast<ESADataPanel *>(sfaPtr->getDataObj().get())){
            throw std::invalid_argument("data object is not of type ESADataPanel");
        }
        ESADataPanel& dataObj = (ESADataPanel&)*this->_sfaPtr->getDataObj();
        allIdents = arma::Col<int>(dataObj.getIdVec());
        
    }

    // given parameters x, return the sum of the individual functions e.g.
    // f_i(x) + ... + f_{i + batchSize - 1}{x}. i will always be greater than 0,
    // and i + batchSize will be less than, or equal to the value of NumFunctions().
    double Evaluate(const arma::mat& x, const size_t i, const size_t batchSize)
    {
        if ((i + batchSize) > allIdents.n_rows) throw std::invalid_argument("i + batchSize out of bounds");
        // extract the identifiers at subset 
        arma::Col<int> subset = allIdents.rows(i, i + batchSize);
        // calculate objective function for subset of identifiers
        double ll = this->_sfaPtr->operator()(x, subset);
        return ll;
    }

    // given parameters x, and a matrix g, store the sum of the gradient of individual
    // functions f'_i(x) + ... + f'_{i + batchSize - 1}(x) into g. i will always be greater
    // than 0, and i + batchSize will be less than or equal to the value of NumFunctions()
    void Gradient(const arma::mat& x, const size_t i, arma::mat& g, const size_t batchSize)
    {
        if ((i + batchSize) > allIdents.n_rows) throw std::invalid_argument("i + batchSize out of bounds");
        // extract the identifiers at subset 
        arma::Col<int> subset = allIdents.rows(i, i + batchSize);
        // calculate gradient for subset of identifiers
        // arma::dmat gr = this->_sfaPtr->gradient(x, subset, 1e-8, this->_analyticalGrad);
        arma::dmat gr;
        this->_sfaPtr->gradHess(
            x,
            subset,
            1e-8,
            this->_analyticalGrad,
            HessianCalcMethod::ANALYTICAL, // irrelevant
            0, // irrelevant,
            this->_threaded,
            &gr,
            nullptr
        );
        g = -(esamath::colSum(gr).t() / subset.n_rows);
        // g = -esamath::colSum(gr).t();
    }

    // Shuffle the ordering of the functions f_i(x)
    void Shuffle()
    {
        // shuffle the allIdents vector
        allIdents = arma::shuffle(allIdents);
    }

    // Get the number of functions f_i(x)
    double NumFunctions()
    {
        // number of firms - this method only exists on ESADataPanel, so dereference ptr to underlying
        // data object
        ESADataPanel& dataObj = (ESADataPanel&)*this->_sfaPtr->getDataObj();
        return dataObj.getNids();
    }

    // may be implemented in addition to, or instead of Evaluate(), Gradient();
    // given parameters x, and a matrix g. return the sum of the individual functions 
    // f_i(x) + ... + f_{i + batchSize - 1}(x), and store the sum of the gradient of individual functions
    // f'_i(x) + ... + f'_{i + batchSize - 1}(x) into the provided matrix g. g should have the same size
    // (rows, columns) as x. i will always be greater than 0, and i + batchSize will be less than or equal
    // to the value of NumFunctions().
    double EvaluateWithGradient(const arma::mat& x, const size_t i, arma::mat& g, const size_t batchSize)
    {
        if ((i + batchSize) > allIdents.n_rows) throw std::invalid_argument("i + batchSize out of bounds");
        // extract the identifiers at subset 
        arma::Col<int> subset = allIdents.rows(i, i + batchSize);
        // calculate objective function for subset of identifiers
        double ll = this->_sfaPtr->operator()(x, subset);
        // calculate gradient for subset of identifiers
        // arma::dmat gr = this->_sfaPtr->gradient(x, subset, 1e-8, this->_analyticalGrad);
        arma::dmat gr;
        this->_sfaPtr->gradHess(
            x,
            subset,
            1e-8,
            this->_analyticalGrad,
            HessianCalcMethod::ANALYTICAL, // irrelevant
            0, // irrelevant,
            this->_threaded,
            &gr,
            nullptr
        );
        g = -(esamath::colSum(gr).t() / subset.n_rows);
        // g = -esamath::colSum(gr).t();
        return ll;
    }

private:
    arma::Col<int> allIdents;
};

#endif // ESA_SFA_ENSMALLEN_WRAPPER_HPP