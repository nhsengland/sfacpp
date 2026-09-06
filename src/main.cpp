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

#if defined(LOCAL_TEST_BUILD)

#include <chrono>
#include <string>
#include <vector>
#include <optional>
#include <armadillo>
#include <memory>

// --- sfacpp Headers ----
#include "sfa/HaltonSettings.hpp"
#include "data/ESADataBase.hpp"
#include "sfa/ESASfaBase.hpp"
#include "utils/enums.hpp"
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"
#include "interface/interface_utils.hpp"
#include "optim/ESAGlobalOptimParams.hpp"
#include "model/ESASfaModelTerms.hpp"
#include "marginaleffects/ESASfaMeff.hpp"
#include "marginaleffects/ESASfaMeffWang.hpp"
#include "marginaleffects/ESASfaMeffKumb.hpp"
#include "efficiencies/ESASfaEffGtre.hpp"
#include "efficiencies/ESASfaJlms.hpp"
#include "utils/modelsummary.hpp"
#include "utils/sandwich.hpp"
#include "utils/ThreadContext.hpp"
#include "interface/interface_utils.hpp"
#include "sfa/ESASfaRunner.hpp"
#include "math/llratiotest.hpp"

// --- Arrow Headers ---
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/dataset/api.h>
#include <parquet/arrow/reader.h>

// ---- optim definitions ---
#define OPTIM_USE_OPENMP
#define OPTIM_ENABLE_ARMA_WRAPPERS
// #include "optim.hpp"

// ---- eigen ----
#ifdef WITHEIGEN
#include "Eigen/Core"

#ifdef WITHCPPNUMSOLVERS
#include "optim/CppOptLibWrapper.hpp"
#endif
#endif

// ensmallen
#ifdef WITHENSMALLEN
#include <ensmallen.hpp>
#endif // WITHENSMALLEN

// ---- optimization ---
#include "optim/esaoptimization.hpp"
#include "optim/ESAOptimResult.hpp"
#include "optim/optimparams.hpp"

// temp
#include "utils/finitediff.hpp"


struct ProcessedData {
    arma::ivec firm_ids;
    arma::ivec time_ids;
    arma::mat x_vars;
    arma::vec y_var;
    arma::mat zuit_vars;
    arma::mat zvit_vars;
    arma::mat zvi0_vars;
    arma::mat zui0_vars;
};
namespace {

/**
 * @brief Converts specified columns from an Arrow Table into an Armadillo matrix.
 *
 * This helper function selects columns by name from an Arrow table, then iterates
 * through them, copying their data into a new arma::mat. It handles casting
 * from Arrow's int64 or double types to Armadillo's double type.
 *
 * @param table The shared pointer to the input arrow::Table.
 * @param col_names A vector of strings containing the names of columns to include.
 * @return An arrow::Result containing the resulting arma::mat on success.
 */
arrow::Result<arma::mat> ArrowTableToArmaMat(
    const std::shared_ptr<arrow::Table>& table,
    const std::vector<std::string>& col_names) {
    
    if (col_names.empty()) {
        // Return an empty matrix if no columns are requested.
        return arma::mat();
    }
    std::vector<int> col_idxs;
    std::shared_ptr<arrow::Schema> schema = table->schema();
    for (size_t i = 0; i < col_names.size(); i++){
        int pos = schema->GetFieldIndex(col_names[i]);
        if (pos > -1){
            col_idxs.push_back(pos);
        }
    }
    // Select the desired columns from the main table using Table::SelectColumns.
    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Table> sub_table,
                          table->SelectColumns(col_idxs));

    if (sub_table == nullptr) {
        return arrow::Status::Invalid("Column selection resulted in a null table.");
    }

    int num_rows = sub_table->num_rows();
    int num_cols = sub_table->num_columns();
    arma::mat result_matrix(num_rows, num_cols);

    for (int i = 0; i < num_cols; ++i) {
        auto column = sub_table->column(i);
        int64_t current_row_offset = 0;

        // A column can be chunked; we must iterate through all chunks.
        for (const auto& chunk : column->chunks()) {
            // Use a switch to handle different numerical types from the CSV.
            switch (chunk->type()->id()) {
                case arrow::Type::INT64: {
                    auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
                    for (int64_t j = 0; j < arr->length(); ++j) {
                        result_matrix(current_row_offset + j, i) = static_cast<double>(arr->Value(j));
                    }
                    break;
                }
                case arrow::Type::DOUBLE: {
                    auto arr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                    for (int64_t j = 0; j < arr->length(); ++j) {
                        result_matrix(current_row_offset + j, i) = arr->Value(j);
                    }
                    break;
                }
                default:
                    return arrow::Status::TypeError("Unsupported data type for numerical matrix. Expected Int64 or Double, but got: ", chunk->type()->ToString());
            }
            current_row_offset += chunk->length();
        }
    }
    return result_matrix;
}

/**
 * @brief Converts a column to a numerical representation using dictionary encoding.
 *
 * This function takes a column (string or numeric) and creates a unique integer
 * ID for each unique value. This is perfect for converting firm or time identifiers.
 *
 * @param table The shared pointer to the input arrow::Table.
 * @param col_name The name of the column to encode.
 * @return An arrow::Result containing the resulting arma::ivec on success.
 */
arrow::Result<arma::ivec> DictionaryEncodeColumn(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& col_name)
{
    auto column = table->GetColumnByName(col_name);
    if (!column) {
        return arrow::Status::Invalid("Column '", col_name, "' not found in the table.");
    }

    // 1. Perform dictionary encoding on the whole ChunkedArray.
    // Arrow computes a unified dictionary across all chunks so indices are consistent.
    ARROW_ASSIGN_OR_RAISE(arrow::Datum encoded_datum, arrow::compute::DictionaryEncode(column));
    
    auto chunked_array = encoded_datum.chunked_array();
    
    // 2. Pre-allocate the Armadillo vector to the TOTAL length of the column
    arma::ivec result_vector(chunked_array->length());

    int64_t current_offset = 0;

    // 3. Iterate over every chunk
    for (const auto& chunk : chunked_array->chunks()) {
        auto dict_array = std::static_pointer_cast<arrow::DictionaryArray>(chunk);
        
        // The indices within the dictionary array
        auto indices = std::static_pointer_cast<arrow::Int32Array>(dict_array->indices());
        
        // Fast copy: access raw pointer instead of calling Value(i) in a loop
        const int32_t* raw_indices = indices->raw_values();
        
        // Copy this chunk's data into the correct position in result_vector
        for (int64_t i = 0; i < indices->length(); ++i) {
            result_vector(current_offset + i) = raw_indices[i];
        }
        
        current_offset += indices->length();
    }

    return result_vector;
}

} // namespace

/**
 * @brief Main function to read a parquet, and convert specified columns into Armadillo objects
 * @param parquet_file_path Path to the input parquet file.
 * @param firm_col_name Name of the column for the firm identifiers.
 * @param time_col_name Name of the column for the time identifiers.
 * @param x_col_names Vector of names for 'x' variable columns.
 * @param y_col_names Vector of names for 'y' variable columns.
 * @param zu_col_names Vector of names for 'zu' variable columns.
 * @return An arrow:Result containing the ProcessedData struct on scucess, or a status on failure
 */
arrow::Result<ProcessedData> ProcessParquetToArma(
    const std::string& parquet_file_path,
    const std::string& firm_col_name,
    const std::string& time_col_name,
    const std::vector<std::string>& x_col_names,
    const std::string& y_col_name,
    const std::optional<std::vector<std::string>>& zuit_colnames = std::nullopt,
    const std::optional<std::vector<std::string>>& zvit_colnames = std::nullopt,
    const std::optional<std::vector<std::string>>& zvi0_colnames = std::nullopt,
    const std::optional<std::vector<std::string>>& zui0_colnames = std::nullopt
)
{
    // ---- 1. Read Parquet file using Apache Arrow ----
    // ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(parquet_file_path));
    std::shared_ptr<arrow::io::RandomAccessFile> input;
    ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(parquet_file_path));
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    ARROW_ASSIGN_OR_RAISE(arrow_reader, parquet::arrow::OpenFile(input, pool))

    // Read the file into an Arrow Table
    std::shared_ptr<arrow::Table> table1;
    ARROW_RETURN_NOT_OK(arrow_reader->ReadTable(&table1));
    // arrow::PrettyPrintOptions options{4};
    arrow::PrettyPrint(*table1, {}, &std::cout);

    ProcessedData data;
    // --- 2. Process Firm and Time Identifier Columns ---
    ARROW_ASSIGN_OR_RAISE(data.firm_ids, DictionaryEncodeColumn(table1, firm_col_name));
    ARROW_ASSIGN_OR_RAISE(data.time_ids, DictionaryEncodeColumn(table1, time_col_name));
    
    // --- 3. Create Armadillo Matrices for variable columns ---
    ARROW_ASSIGN_OR_RAISE(data.x_vars, ArrowTableToArmaMat(table1, x_col_names));
    // Process the single 'y' column into a vector
    ARROW_ASSIGN_OR_RAISE(arma::mat y_matrix, ArrowTableToArmaMat(table1, {y_col_name}));
    if (y_matrix.n_cols != 1) {
        return arrow::Status::Invalid("Expected 'y' variable to result in a single column, but got ",
                                    std::to_string(y_matrix.n_cols));
    }
    data.y_var = y_matrix.col(0);
    if (zuit_colnames) {
        ARROW_ASSIGN_OR_RAISE(data.zuit_vars, ArrowTableToArmaMat(table1, zuit_colnames.value()));
    }
    if (zvit_colnames) {
        ARROW_ASSIGN_OR_RAISE(data.zvit_vars, ArrowTableToArmaMat(table1, zvit_colnames.value()));
    }
    if (zvi0_colnames) {
        ARROW_ASSIGN_OR_RAISE(data.zvi0_vars, ArrowTableToArmaMat(table1, zvi0_colnames.value()));
    }
    if (zui0_colnames) {
        ARROW_ASSIGN_OR_RAISE(data.zui0_vars, ArrowTableToArmaMat(table1, zui0_colnames.value()));
    }
    return data;
}


/**
 * @brief Main function to read a CSV and convert specified columns into Armadillo objects.
 * * @param csv_file_path Path to the input CSV file.
 * @param firm_col_name Name of the column for firm identifiers.
 * @param time_col_name Name of the column for time identifiers.
 * @param x_col_names Vector of names for 'x' variable columns.
 * @param y_col_name Name for the 'y' variable column.
 * @param zv_col_names Vector of names for 'zv' variable columns.
 * @return An arrow::Result containing the ProcessedData struct on success, or a Status on failure.
 */
arrow::Result<ProcessedData> ProcessCsvToArma(
    const std::string& csv_file_path,
    const std::string& firm_col_name,
    const std::string& time_col_name,
    const std::vector<std::string>& x_col_names,
    const std::string& y_col_name,
    const std::optional<std::vector<std::string>>& zuit_colnames = std::nullopt,
    const std::optional<std::vector<std::string>>& zvit_colnames = std::nullopt,
    const std::optional<std::vector<std::string>>& zvi0_colnames = std::nullopt,
    const std::optional<std::vector<std::string>>& zui0_colnames = std::nullopt
) {
    
    // --- 1. Read CSV file using Apache Arrow ---
    ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(csv_file_path));
    auto reader_result = arrow::csv::TableReader::Make(
        arrow::io::default_io_context(), infile, arrow::csv::ReadOptions::Defaults(),
        arrow::csv::ParseOptions::Defaults(), arrow::csv::ConvertOptions::Defaults());
    if (!reader_result.ok()) {
        return reader_result.status();
    }
    auto reader = *reader_result;
    
    ARROW_ASSIGN_OR_RAISE(auto table, reader->Read());

    ProcessedData data;

    // --- 2. Process Firm and Time Identifier Columns ---
    ARROW_ASSIGN_OR_RAISE(data.firm_ids, DictionaryEncodeColumn(table, firm_col_name));
    ARROW_ASSIGN_OR_RAISE(data.time_ids, DictionaryEncodeColumn(table, time_col_name));
    
    // --- 3. Create Armadillo Matrices for variable columns ---
    ARROW_ASSIGN_OR_RAISE(data.x_vars, ArrowTableToArmaMat(table, x_col_names));

    // Process the single 'y' column into a vector
    ARROW_ASSIGN_OR_RAISE(arma::mat y_matrix, ArrowTableToArmaMat(table, {y_col_name}));
    if (y_matrix.n_cols != 1) {
        return arrow::Status::Invalid("Expected 'y' variable to result in a single column, but got ",
                                    std::to_string(y_matrix.n_cols));
    }
    data.y_var = y_matrix.col(0);
    
    // ARROW_ASSIGN_OR_RAISE(data.zuit_vars, ArrowTableToArmaMat(table, zv_col_names));
    if (zuit_colnames) {
        ARROW_ASSIGN_OR_RAISE(data.zuit_vars, ArrowTableToArmaMat(table, zuit_colnames.value()));
    }
    if (zvit_colnames) {
        ARROW_ASSIGN_OR_RAISE(data.zvit_vars, ArrowTableToArmaMat(table, zvit_colnames.value()));
    }
    if (zui0_colnames) {
        ARROW_ASSIGN_OR_RAISE(data.zui0_vars, ArrowTableToArmaMat(table, zui0_colnames.value()));
    }
    if (zvi0_colnames) {
        ARROW_ASSIGN_OR_RAISE(data.zvi0_vars, ArrowTableToArmaMat(table, zvi0_colnames.value())); 
    }

    return data;
}


void runModel(
    const ESASfaModelType modelType,
    const std::string optim_algo,
    const std::string& fpath,
    const std::string& firm_col,
    const std::string& time_col,
    const std::string& y_var_col,
    const std::vector<std::string>& x_vars,
    const std::optional<std::vector<std::string>>& zuitVars,
    const std::optional<std::vector<std::string>>& zvitVars,
    const std::optional<std::vector<std::string>>& zvi0Vars,
    const std::optional<std::vector<std::string>>& zui0Vars,
    const std::optional<arma::dcolvec>& startVals
)
{
    arrow::Result<ProcessedData> result = ProcessCsvToArma(
        fpath, firm_col, time_col, x_vars, y_var_col, zuitVars, zvitVars, zvi0Vars, zui0Vars
    );
    if (!result.ok()) {
        std::cerr << "Error processing CSV file: " << result.status().ToString() << std::endl;
        return;
    }
    ProcessedData data = *result;

    std::cout << "Successfully processed the CSV file." << std::endl;
    std::cout << "------------------------------------" << std::endl;
    size_t nrows = data.x_vars.n_rows;
    // create some data matricies
    arma::dmat matZuit = zuitVars.has_value() ? data.zuit_vars : arma::dmat(nrows, 1, arma::fill::ones);
    arma::dmat matZvit = zvitVars.has_value() ? data.zvit_vars : arma::dmat(nrows, 1, arma::fill::ones);
    arma::dmat matZvi0 = zvi0Vars.has_value() ? data.zvi0_vars : arma::dmat(nrows, 1, arma::fill::ones);
    arma::dmat matZui0 = zui0Vars.has_value() ? data.zui0_vars : arma::dmat(nrows, 1, arma::fill::ones);
    // add intercept column
    arma::Col<double> intercept(nrows, 1, arma::fill::ones);
    if (zuitVars.has_value()) matZuit.insert_cols(0, intercept);
    if (zvitVars.has_value()) matZvit.insert_cols(0, intercept);
    if (zvi0Vars.has_value()) matZvi0.insert_cols(0, intercept);
    if (zui0Vars.has_value()) matZui0.insert_cols(0, intercept);
    std::vector<std::string> xTerms = {"cons"};
    xTerms.insert(xTerms.end(), std::begin(x_vars), std::end(x_vars));
    // create terms
    std::vector<std::string> zuitTerms = {"Zuit_cons"}, zvitTerms = {"Zvit_cons"}, zvi0Terms = {"Zvi0_cons"}, zui0Terms = {"Zui0_cons"};
    if (zuitVars.has_value()) zuitTerms.insert(zuitTerms.end(), std::begin(zuitVars.value()), std::end(zuitVars.value()));
    if (zvitVars.has_value()) zvitTerms.insert(zvitTerms.end(), std::begin(zvitVars.value()), std::end(zvitVars.value()));
    if (zvi0Vars.has_value()) zvi0Terms.insert(zvi0Terms.end(), std::begin(zvi0Vars.value()), std::end(zvi0Vars.value()));
    if (zui0Vars.has_value()) zui0Terms.insert(zui0Terms.end(), std::begin(zui0Vars.value()), std::end(zui0Vars.value()));
    // convert firm, id vectors
    arma::Col<int> idVec = arma::conv_to<arma::Col<int>>::from(data.firm_ids);
    arma::Col<int> timeVec = arma::conv_to<arma::Col<int>>::from(data.time_ids);
    arma::dvec y = data.y_var;
    // add intercept column for x
    arma::dmat x = data.x_vars;
    x.insert_cols(0, intercept);
    // ---- constants ----
    
    const double prodCost = 1.0;
    const unsigned int nsim = 1000;
    const int seed = 1234;
    const double conf_int = 0.95;
    const int bsrep = 100;
    const int printLevel = 4;
    const double confidenceLevel = 0.95;
    const bool shouldClusterSE = true;
    const bool calculateEfficiencyScores = false;
    arma::arma_rng::set_seed(seed);
    // model type
    // ESASfaModelType mT = ESASfaModelType::TRE_HNORM_ZUIT;
    ESASfaModelType mT = modelType;
    // ESASfaModelType mT = ESASfaModelType::TFE_HNORM_ZUIT;
    // ----
    ESASfaModelFamily mF = ESAEnums::getModelFamily(mT);
    ESASfaModelDistribution mD = ESAEnums::getDistribution(mT);
    // ModelSolver mainModelSolver = ModelSolver::DLIB_TR;
    ModelSolver mainModelSolver = ESAEnums::getModelSolverForMethodAndLib(optim_algo, "dlib");
    // set to global singleton class
    ESAGlobalOptimParams* globalOptimParams = ESAGlobalOptimParams::GetInstance();
    // set default optim params
    ESAOptimParams mainOptimParams;
    mainOptimParams.maxit = 1500;
    globalOptimParams->mainOptimParams = mainOptimParams;
    globalOptimParams->mainModelSolver = mainModelSolver;
    globalOptimParams->optimThreaded = true;
    // setup the halton settings
    HaltonSettings hsetting = interface::haltonSettingsForOpts(2, 1000, 3, true, false);
    // ---- setup the runner ----
    ESASfaRunner runner(mT, prodCost, seed, nsim, printLevel);
    runner.loadData(
        &y,
        &x,
        nullptr,
        &matZuit,
        &matZvit,
        &matZui0,
        &matZvi0,
        &idVec,
        &timeVec
    );
    // setup the model and its terms
    runner.setupModel(hsetting, xTerms, std::nullopt, zuitTerms, zvitTerms, zui0Terms, zvi0Terms);
    // run optimization
    // set starting values
    // arma::dcolvec sval = {};
    arma::dcolvec sval = runner.modelObjPtr->startingValues();
    if (startVals.has_value()) {
        if (startVals.value().n_rows == sval.n_rows) {
            sval = startVals.value();
        } else {
            ESALogger::logger()->warn("starting values given not correct dims {} but need {}", startVals.value().n_rows, sval.n_rows);
        }
    }
    std::unique_ptr<ESAOptimResult> resultUnk = runner.runOptimization(sval, HessianCalcMethod::ANALYTICAL, 0);
    if (resultUnk == nullptr) {
        ESALogger::logger()->error("Unexpected internal error - nothing was returned from optimziation");
    } else if (resultUnk->getDidConverge()) {
        // successful convergence
        ESAOptimResultSuccess& optimRes = (ESAOptimResultSuccess&)*resultUnk;
        arma::dcolvec coefs = optimRes.getX();
        std::vector<std::string> allTerms = runner.modelTerms->allTerms();
        arma::dmat vcov = optimRes.getVcov();
        double llscore = optimRes.getLogLike();
        // calculated the average hessian and gradient - multiply by that factor to get original hess/grad
        arma::dmat hess = (optimRes.getHessian() * optimRes.getN());
        arma::dmat grad = (optimRes.getGradient() * optimRes.getN());
        // jacobian matrix - gradient at each obs
        arma::dmat jac = optimRes.getGradientIndividual();
        // degrees of freedom 
        int dof = optimRes.degreesFreedom();
        // nobs
        int nobs = optimRes.getNobs();
        // ---- clustered se if desired ----
        arma::dmat vcovSummary;
        if (shouldClusterSE) {
            arma::Col<int> empty_ivec;
            vcovSummary = sandwich::clusteredVcov(vcov, jac, idVec);
        } else {
            vcovSummary = vcov;
        }
        ESASigmaParams sigParams = runner.modelObjPtr->getSigmaParams(coefs);
        arma::dmat msummary = runner.buildSummary(coefs, vcovSummary, confidenceLevel, dof);
        std::unique_ptr<ESASfaEffScores> effs;
        if (calculateEfficiencyScores) {
            // calculate efficiency scores
            effs = runner.estimateEfficiencyScores(
                coefs,
                2000, // number of ghk simulations
                0 // start position for prime numbers, used for halton bases
            );
        }
        std::unique_ptr<ESASfaMeffReturn> meff = nullptr;
        std::unique_ptr<ESASfaMeffCIReturn> meffCIs = nullptr;
        runner.estimateMarginalEffects(
            "wang2002", // method
            coefs, // parameter vector/ coefficients
            false, // whether or not to estimate confidence intervals
            0.95, // conf level
            0,
            meff,
            meffCIs
        );
        // check result
        if (meff == nullptr) {
            ESALogger::logger()->error("Calculation of marginal effects failed");
        } else {
            arma::dmat meffs = meff->marginalEffects;
            std::vector<std::string> meffCnames = meff->columnNames;
            // ESALogger::logger()->info("{}\n{}", meffCnames, meffs);
        }
        // ---- print the output ----
        interface::printModelOutput(
            runner.dataObjPtr, // ptr to data object
            msummary, // model summary
            allTerms, // model terms
            sigParams, // sigma parameters
            nsim,
            llscore,
            optimRes.getGnorm(),
            shouldClusterSE,
            hsetting,
            confidenceLevel, // confint,
            5, // decimal places
            120, // console width
            std::nullopt,
            std::nullopt
        );
    }
}


int main()
{


    // 
    // // std::string fpath = "dev/data/TaiwaneseManufacturingCroppedSorted.csv";
    ESASfaModelType modelType = ESASfaModelType::GTRE_HNORM_ZUIT_ZUI0;
    std::string fpath = "dev/data/simulated_gtre_data.csv";
    std::string firm_col = "id";
    std::string time_col = "time";
    std::string y_var_col = "y";
    std::vector<std::string> x_vars = {"x1", "x2", "x3", "x4"};
    arma::dcolvec start = {0.3, 0.3, 0.2, 0.1, 0.2, -0.1, 0.01, -0.1, -0.1, -0.001};
    std::optional<arma::dcolvec> startvals = std::make_optional(start);
    ESALogger::logger()->info("Model 1");
    // runModel(
    //     modelType,
    //     "tr",
    //     fpath, // path to csv
    //     firm_col, // firm column
    //     time_col, // time column
    //     y_var_col, // yvariable column
    //     x_vars, // x variables,
    //     std::make_optional(std::vector<std::string>{"Z_uit_1"}), // zuit
    //     std::nullopt, // zvit 
    //     std::nullopt, // zvi0
    //     std::nullopt, // zui0
    //     startvals
    // );
    ESALogger::logger()->info("Model 2");
    runModel(
        modelType,
        // "hybrid_bfgs_tr",
        "pso_tr",
        fpath, // path to csv
        firm_col, // firm column
        time_col, // time column
        y_var_col, // yvariable column
        x_vars, // x variables,
        std::make_optional(std::vector<std::string>{"Z_uit_1"}), // zuit
        std::nullopt, // zvit 
        std::nullopt, // zvi0
        std::nullopt, //std::make_optional(std::vector<std::string>{"Z_ui0_1"}), // zui0
        startvals
    );
    ESALogger::logger()->info("Model 3");
    runModel(
        modelType,
        "hybrid_lbfgs_tr",
        fpath, // path to csv
        firm_col, // firm column
        time_col, // time column
        y_var_col, // yvariable column
        x_vars, // x variables,
        std::make_optional(std::vector<std::string>{"Z_uit_1"}), // zuit
        std::nullopt, // zvit 
        std::nullopt, // zvi0
        std::nullopt, //std::make_optional(std::vector<std::string>{"Z_ui0_1"}), // zui0
        startvals
    );
    // ESALogger::logger()->info("Model 4");
    // runModel(
    //     modelType,
    //     "hybrid_lbfgs_newton",
    //     fpath, // path to csv
    //     firm_col, // firm column
    //     time_col, // time column
    //     y_var_col, // yvariable column
    //     x_vars, // x variables,
    //     std::make_optional(std::vector<std::string>{"Z_uit_1"}), // zuit
    //     std::nullopt, // zvit 
    //     std::nullopt, // zvi0
    //     std::nullopt, //std::make_optional(std::vector<std::string>{"Z_ui0_1"}), // zui0
    //     startvals
    // );
    
    // // =======================================================
    // ESASfaModelType modelType = ESASfaModelType::TRE_HNORM_ZUIT;
    // std::string fpath = "dev/data/TaiwaneseManufacturing.csv";
    // std::string firm_col = "id";
    // std::string time_col = "time";
    // std::string y_var_col = "y";
    // std::vector<std::string> x_vars = {"x1", "x2"};
    // ESALogger::logger()->info("Model 1");
    // runModel(
    //     modelType,
    //     "tr",
    //     fpath, // path to csv
    //     firm_col, // firm column
    //     time_col, // time column
    //     y_var_col, // yvariable column
    //     x_vars, // x variables,
    //     std::make_optional(std::vector<std::string>{"z1"}), // zuit
    //     std::nullopt, // zvit 
    //     std::nullopt, // zvi0
    //     std::nullopt // zui0
    // );
    // ESALogger::logger()->info("Model 2");
    // runModel(
    //     modelType,
    //     "hybrid_bfgs_tr",
    //     fpath, // path to csv
    //     firm_col, // firm column
    //     time_col, // time column
    //     y_var_col, // yvariable column
    //     x_vars, // x variables,
    //     std::make_optional(std::vector<std::string>{"z1"}), // zuit
    //     std::nullopt, // zvit 
    //     std::nullopt, // zvi0
    //     std::nullopt // zui0
    // );
    // ESALogger::logger()->info("Model 3");
    // runModel(
    //     modelType,
    //     "hybrid_bfgs_newton",
    //     fpath, // path to csv
    //     firm_col, // firm column
    //     time_col, // time column
    //     y_var_col, // yvariable column
    //     x_vars, // x variables,
    //     std::make_optional(std::vector<std::string>{"z1"}), // zuit
    //     std::nullopt, // zvit 
    //     std::nullopt, // zvi0
    //     std::nullopt // zui0
    // );
    // ESALogger::logger()->info("Model 4");
    // runModel(
    //     modelType,
    //     "hybrid_lbfgs_newton",
    //     fpath, // path to csv
    //     firm_col, // firm column
    //     time_col, // time column
    //     y_var_col, // yvariable column
    //     x_vars, // x variables,
    //     std::make_optional(std::vector<std::string>{"z1"}), // zuit
    //     std::nullopt, // zvit 
    //     std::nullopt, // zvi0
    //     std::nullopt // zui0
    // );

    return 0;
}

#endif