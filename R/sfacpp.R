#' @description 
#' Stochastic Frontier Model Estimation
#' 
#' @param formX (`formula`)
#' @param data (`data.table`)
#' @param model (`SFA_MODEL`)
#' @param formZuit (`formula`)
#' @param formZui0 (`formula`)
#' @param formZvit (`formula`)
#' @param formZvi0 (`formula`)
#' @param theta0 (`numeric(k)`)
#' @export
sfacpp <- function(
    form_x,
    data,
    id_col = NULL,
    time_col = NULL,
    form_zmuit = NULL,
    form_zuit = NULL,
    form_zui0 = NULL,
    form_zvit = NULL,
    form_zvi0 = NULL,
    start = NULL,
    # options w.r.t. SFA
    prod = 1,
    dist = "hnorm",
    model = "tre",
    nsim = 500,
    optim_opts = NULL,
    # hessian_calc = "analytical",
    # hessian_num_approx_accuracy = 3,
    conf_int = 0.95,
    # options w.r.t. marginal effects
    marg_eff = "wang2002",
    estimate_marg_eff = TRUE,
    estimate_marg_eff_ci = FALSE,
    marg_eff_bootstrap_reps = 500,
    # general options
    seed = 1234,
    print_level = 2,
    form_has_x_intercept = TRUE,
    clustered_se = TRUE,
    nthreads = 7,
    calculate_efficiency_scores = FALSE,
    # options related to efficiency scores
    ghk_nsim = 1000,
    # halton draw related - for TRE and GTRE
    halton_base = 2,
    halton_ui0_base = 3,
    halton_burnin = 1000,
    halton_scrambled = TRUE,
    halton_shuffled = FALSE,
    display_decimal_places = 5,
    display_console_width = ifelse(
        options("width") > 150, 150, options("width")
    ),
    optim_method = "tr"
){
    set.seed(seed)
    cl <- match.call()
    dt <- data.table::copy(data)
    if (
        !optim_method %in% c(
            "tr", "hybrid_bfgs_tr", "hybrid_lbfgs_tr", "bfgs_tr", "lbfgs_tr"
        )
    ) {
        stop("'optim_method' must be one of 'tr', 'hybrid_bfgs_tr', or 'hybrid_lbfgs_tr'")
    }
    optim_method_lib <- "dlib"
    # check model is an acceptable type.
    if (!model %in% c("tre", "gtre")) {
        stop("'model' must be one of 'tre', 'gtre'")
    }
    # check whether distribution is acceptable #tnorm
    if (!dist %in% c("hnorm")) {
        stop("'dist' must be one of 'hnorm'")
    }
    if (!marg_eff %in% c("wang2002")) {
        stop("'marginalEffect' must be one of 'wang2002'")
    }
    # check that data is data.table
    if (!data.table::is.data.table(data)) {
        stop("'data' must be a data.table")
    }
    if (model %in% c("tfe", "tre", "gtre")) {
        if (is.null(id_col)) {
            stop("idCol must be specified for 'tfe', 'tre', 'gtre' models")
        }
        # check that idCol is in data
        if (!id_col %in% colnames(data)) {
            stop(paste0("idCol ", id_col, " must be in data"))
        }
        if (is.null(time_col)){
            stop("timeCol must be specified for 'tfe', 'tre', 'gtre' models")
        }
        if (!time_col %in% colnames(data)) {
            stop(paste0("timeCol ", time_col, " must be in data"))
        }
    }
    # check confidence level in bounds
    if (conf_int < 0.0 || conf_int > 1.0) {
        stop("confidenceLevel must be between 0 and 1")
    }
    # check marginalEffect is an acceptable type - boolean
    if (!estimate_marg_eff %in% c(TRUE, FALSE)) {
        stop("'marginalEffect' must be a boolean")
    }
    # check marginalEffectBootstrapReps is a positive
    if (marg_eff_bootstrap_reps < 0) {
        stop("'marginalEffectBootstrapReps' must be a positive integer")
    }
    # check printLevel at least 0
    if (print_level < 0) {
        stop("'printLevel' must be at least 0")
    }
    # check nsim is a positive integer
    if (nsim < 0){
        stop("'nsim' must be a positive integer")
    }
    if (!is.null(optim_opts)) {
        if (!is.list(optim_opts)) stop("'optimOpts' must be a list")
    }
    if (ghk_nsim < 0) stop("ghk_nsim must be postivie interger")
    # process halton draw (main)
    if (is.null(halton_base)) {
        stop("must provide 'halton_base' (prime number)")
    }
    # process halton draw (ui0)
    if ((model == "gtre") && is.null(halton_ui0_base)) {
        stop("must provider 'halton_ui0_base' for 'gtre'")
    }
    # also check that if using the halton base approach, and we are in gtre, that both
    # bases are not the same - otherwise will be colinear
    if ((model == "gtre") && (halton_base == halton_ui0_base)) {
        stop("For 'gtre', if using halton base, they must be different. Try using coprimes e.g. 2 and 3")
    }
    # check halton burnin is positive
    if (halton_burnin < 0) stop("halton burnin must be positive integer")
    if (is.null(halton_shuffled)) stop("'halton_shuffled' must be 'TRUE' or 'FALSE'")
    if (is.null(halton_scrambled)) stop("'halton_scrambled' must be 'TRUE' or 'FALSE'")
    # # if the model is tfe, this is estimated through a DV approach, so check whether
    # # the formula contains this already
    # if (model == "tfe") {
    #     if (!id_col %in% all.vars(form_x)){
    #         form_x <- update(form_x, paste0(". ~ . + as.factor(", id_col, ")"))
    #     }
    # }
    hessian_calc <- "analytical"
    hessian_num_approx_accuracy <- 3
    # instansiate the model data class
    mdlDataObj <- sfacpp::EHInternalModelData$new(
        data = dt,
        f_in = form_x,
        idVar = id_col,
        timeVar = time_col,
        fZmuit_in = form_zmuit,
        fZuit_in = form_zuit,
        fZui0_in = form_zui0,
        fZvit_in = form_zvit,
        fZvi0_in = form_zvi0,
        fInHasIntercept = form_has_x_intercept
    )
    startValsMtrx <- NULL
    if (!is.null(start)) startValsMtrx <- matrix(start, ncol = 1)
    # call C++ to do all the rest of the work
    for_cpp <- mdlDataObj$get_data_for_cpp()
    res <- sfacpp::sfacpp_internal(
        y_ = for_cpp$y,
        allData_ = for_cpp$all_data,
        colCounts_ = for_cpp$col_counts,
        idVec_ = for_cpp$idVec,
        timeVec_ = for_cpp$timeVec,
        startVals_ = startValsMtrx,
        prodCost = prod,
        model = model,
        dist = dist,
        method = optim_method,
        methodLib = optim_method_lib,
        optimOpts = optim_opts,
        termsX_ = mdlDataObj$namesX,
        termsZmuit_ = mdlDataObj$namesZmuit,
        termsZuit_ = mdlDataObj$namesZuit,
        termsZvit_ = mdlDataObj$namesZvit,
        termsZui0_ = mdlDataObj$namesZui0,
        termsZvi0_ = mdlDataObj$namesZvi0,
        nsim = nsim,
        hessianCalc = hessian_calc,
        hessianCalcNumApproxAccuracy = hessian_num_approx_accuracy,
        seed = seed,
        confidenceLevel = conf_int,
        estimateMarginalEffects = estimate_marg_eff,
        estimateMargEffCI = estimate_marg_eff_ci,
        marginalEffectBootstrapReps = marg_eff_bootstrap_reps,
        printLevel = print_level,
        clusteredSE = clustered_se,
        haltonBase = halton_base,
        haltonBurnin = halton_burnin,
        haltonUi0Base = halton_ui0_base,
        scrambledHalton = halton_scrambled,
        shuffledHalton = halton_shuffled,
        nthreads = nthreads,
        calculateEfficiencyScores = calculate_efficiency_scores,
        ghkSimReps = ghk_nsim
    )
    # take matrix/array, attach original order index and sorts
    reorder_matrix <- function(m, return_df = TRUE, drop_idtimecols = FALSE) {
        dt_tmp <- data.table::data.table(m)
        dt_order <- data.table::data.table(sfacpp_internal_orig_order = mdlDataObj$order)
        binds <- list(dt_order)
        if (return_df) binds <- c(binds, mdlDataObj$idTimeCols)
        binds <- c(binds, dt_tmp)
        dt_tmp <- do.call("cbind", binds)
        data.table::setorderv(dt_tmp, "sfacpp_internal_orig_order")
        dt_tmp[, "sfacpp_internal_orig_order" := NULL]
        if (return_df && drop_idtimecols) {
            dt_tmp[, (colnames(mdlDataObj$idTimeCols)) := NULL]
        }
        if (return_df) return(dt_tmp)
        return(as.matrix(dt_tmp))
    }
    # extract the model matrix, and reorder
    dt_mm <- reorder_matrix(mdlDataObj$modelMatrix, TRUE)
    # store in return object
    res[["modelData"]] <- dt_mm
    res[["reorderingIdx"]] <- as.vector(mdlDataObj$order)
    res[["call"]] <- cl
    res[["omit"]] <- mdlDataObj$omit
    # # needed for sandwich::vcovCL when using a formula, and some obs were omitted
    if (!is.null(res$omit)) {
        omitAct <- res$omit
        attr(omitAct, "class") <- "omit"
        res[["na.action"]] <- omitAct
    }
    # # needed for vcovCL/vcovHC - only use prod fn formulae?
    res[["formula"]] <- form_x
    res[["terms"]] <- terms(form_x)
    res[["form_zmuit"]] <- form_zmuit
    res[["form_zuit"]] <- form_zuit
    res[["form_zvit"]] <- form_zvit
    res[["form_zvi0"]] <- form_zvi0
    # # add dimension names to the variance covariance matrix (if present)
    if (!is.null(res[["vcov"]])) {
        if (!is.null(res[["vars"]])) {
            vcovvars <- res[["vars"]]
            colnames(res[["vcov"]]) <- vcovvars
            rownames(res[["vcov"]]) <- vcovvars
        } else {
            rownames(res[["vcov"]]) <- paste0(
                "dim_", seq_len(ncol(res[["vcov"]]))
            )
            colnames(res[["vcov"]]) <- paste0(
                "dim_", seq_len(ncol(res[["vcov"]]))
            )
        }
    }
    # row and column names for model summary
    if (!is.null(res[["modelSummary"]])) {
        if (!is.null(res[["vars"]])) {
            rownames(res[["modelSummary"]]) <- res[["vars"]]
        } else {
            rownames(res[["modelSummary"]]) <- paste0(
                "dim_", seq_len(nrow(res[["modelSummary"]]))
            )
        }
        # column names
        colnames(res[["modelSummary"]]) <- c(
            "estimate",
            "std_err",
            "t_statistic",
            "p_value",
            "ci_lower",
            "ci_upper"
        )
    }
    # ---- inefficiency scores ----
    # reorder inefficiency/efficiency scores so they align up with the original data
    numOrderCols <- ifelse(
        is.null(mdlDataObj$idTimeCols),
        0,
        length(colnames(mdlDataObj$idTimeCols))
    )
    if (
        (!is.null(res[["efficiencyTransient"]])) &&
        (!is.null(res[["efficiencyPersistent"]]))
    ) {
        # reorder
        dt1 <- reorder_matrix(res[["efficiencyTransient"]])
        dt2 <- reorder_matrix(res[["efficiencyPersistent"]])
        colnames(dt1)[numOrderCols + 1] <- "transient_efficiency"
        colnames(dt2)[numOrderCols + 1] <- "persistent_efficiency"
        dt_eff <- NULL
        if (numOrderCols == 0) {
            dt_eff <- cbind(dt1, dt2)
        } else {
            dt_eff <- data.table::merge.data.table(
                x = dt1,
                y = dt2,
                by = colnames(mdlDataObj$idTimeCols),
                all.x = TRUE,
                all.y = TRUE
            )
        }
        # calculate overall inefficiency
        dt_eff[, overall_efficiency := transient_efficiency * persistent_efficiency]
        res[["efficiencies"]] <- dt_eff
        res[["efficiencyTransient"]] <- NULL
        res[["efficiencyPersistent"]] <- NULL
    } else if (!is.null(res[["efficiencyTransient"]])) {
        dt_eff <- reorder_matrix(res[["efficiencyTransient"]])
        colnames(dt_eff)[numOrderCols + 1] <- "transient_efficiency"
        res[["efficiencies"]] <- dt_eff
        res[["efficiencyTransient"]] <- NULL
    } else if (!is.null(res[["efficiencyPersistent"]])) {
        dt_eff <- reorder_matrix(res[["efficiencyPersistent"]])
        colnames(dt_eff)[numOrderCols + 1] <- "persistent_efficiency"
        res[["efficiencies"]] <- dt_eff
        res[["efficiencyPersistent"]] <- NULL
    }
    # ---- marginal effects ----
    # process marginal effects return matricies
    if (
        estimate_marg_eff &&
        !is.null(res[["marginalEffects"]])
    ) {
        if (!is.null(res[["marginalEffectsNames"]])) {
            colnames(res[["marginalEffects"]]) <- res[["marginalEffectsNames"]]
        }
        res[["marginalEffects"]] <- reorder_matrix(res[["marginalEffects"]])
    }
    # same principal for the confidence intervals
    if (estimate_marg_eff_ci && !is.null(res[["marginalEffectsLwrCI"]])) {
        if (!is.null(res[["marginalEffectsNames"]])) {
            colnames(res[["marginalEffectsLwrCI"]]) <- res[["marginalEffectsNames"]]
        }
        res[["marginalEffectsLwrCI"]] <- reorder_matrix(res[["marginalEffectsLwrCI"]])
    }
    if (estimate_marg_eff_ci && !is.null(res[["marginalEffectsUprCI"]])) {
        if (!is.null(res[["marginaLEffectsNames"]])) {
            colnames(res[["marginalEffectsUprCI"]]) <- res[["marginalEffectsNames"]]
        }
        res[["marginalEffectsUprCI"]] <- reorder_matrix(res[["marginalEffectsUprCI"]])
    }
    res[["model"]] <- model
    res[["idCol"]] <- id_col
    res[["timeCol"]] <- time_col
    # sort the score matrix too, since that will be needed for clustered se estfun
    if (!is.null(res[["jacobian"]])) {
        res[["jacobian"]] <- reorder_matrix(res[["jacobian"]], FALSE)
        if (!is.null(res[["vars"]])) {
            colnames(res[["jacobian"]]) <- res[["vars"]]
        }
    }
    res[["dist"]] <- dist
    res[["nsim"]] <- nsim
    res[["clusteredSE"]] <- clustered_se
    res[["haltonBase"]] <- halton_base
    res[["haltonBurnin"]] <- halton_burnin
    res[["haltonUi0Base"]] <- halton_ui0_base
    res[["scrambledHalton"]] <- halton_scrambled
    res[["shuffledHalton"]] <- halton_shuffled
    res[["confInt"]] <- conf_int
    return(res)
}

#' @description 
#' Run some searches for a given model
#' 
#' @export 
sfacpp_searches <- function(
    form_x,
    data,
    id_col = NULL,
    time_col = NULL,
    form_zmuit = NULL,
    form_zuit = NULL,
    form_zui0 = NULL,
    form_zvit = NULL,
    form_zvi0 = NULL,
    start = NULL,
    nsearches = 500,
    maxit = 150,
    slength_frontier = 2.0,
    slength_sigmas = 0.8,
    max_attempt_start_vals = 50,
    # options w.r.t. SFA
    prod = 1,
    dist = "hnorm",
    model = "tre",
    nsim = 500,
    # general options
    seed = 1234,
    print_level = 0,
    nthreads = 7,
    # halton draw related - for TRE and GTRE
    halton_base = 2,
    halton_ui0_base = 3,
    halton_burnin = 1000,
    halton_scrambled = TRUE,
    halton_shuffled = FALSE,
    display_decimal_places = 4,
    display_console_width = ifelse(
        options("width") > 150, 150, options("width")
    ),
    optim_method = "tr"
)
{
    set.seed(seed)
    # cl <- match.call()
    
    dt <- data.table::copy(data)
    optim_method_lib <- "dlib"
    if (!optim_method %in% c("tr", "hybrid_bfgs_tr", "hybrid_lbfgs_tr")) {
        stop("'optim_method' must be one of 'tr', 'hybrid_bfgs_tr', or 'hybrid_lbfgs_tr'")
    }
    # check model is an acceptable type. # tfe, cross
    if (!model %in% c("tre", "gtre")) {
        stop("'model' must be one of 'tre', 'gtre'")
    }
    # check whether distribution is acceptable
    if (!dist %in% c("hnorm")) {
        stop("'dist' must be one of 'hnorm'")
    }
    # check that data is data.table
    if (!data.table::is.data.table(data)) {
        stop("'data' must be a data.table")
    }
    if (model %in% c("tfe", "tre", "gtre")) {
        if (is.null(id_col)) {
            stop("idCol must be specified for 'tfe', 'tre', 'gtre' models")
        }
        # check that idCol is in data
        if (!id_col %in% colnames(data)) {
            stop(paste0("idCol ", id_col, " must be in data"))
        }
        if (is.null(time_col)){
            stop("timeCol must be specified for 'tfe', 'tre', 'gtre' models")
        }
        if (!time_col %in% colnames(data)) {
            stop(paste0("timeCol ", time_col, " must be in data"))
        }
    }
    # check printLevel at least 0
    if (print_level < 0) {
        stop("'printLevel' must be at least 0")
    }
    # check nsim is a positive integer
    if (nsim < 0) {
        stop("'nsim' must be a positive integer")
    }
    # process halton draw (main)
    if (is.null(halton_base)) {
        stop("must provide 'halton_base' (prime number)")
    }
    # process halton draw (ui0)
    if ((model == "gtre") && is.null(halton_ui0_base)) {
        stop("must provider 'halton_ui0_base' for 'gtre'")
    }
    # also check that if using the halton base approach, and we are in gtre, that both
    # bases are not the same - otherwise will be colinear
    if ((model == "gtre") && (halton_base == halton_ui0_base)) {
        stop("For 'gtre', if using halton base, they must be different. Try using coprimes e.g. 2 and 3")
    }
    # check halton burnin is positive
    if (halton_burnin < 0) stop("halton burnin must be positive integer")
    if (is.null(halton_shuffled)) stop("'halton_shuffled' must be 'TRUE' or 'FALSE'")
    if (is.null(halton_scrambled)) stop("'halton_scrambled' must be 'TRUE' or 'FALSE'")
    # # if the model is tfe, this is estimated through a DV approach, so check whether
    # # the formula contains this already
    # if (model == "tfe") {
    #     if (!id_col %in% all.vars(form_x)){
    #         form_x <- update(form_x, paste0(". ~ . + as.factor(", id_col, ")"))
    #     }
    # }
    # instansiate the model data class
    mdlDataObj <- sfacpp::EHInternalModelData$new(
        data = dt,
        f_in = form_x,
        idVar = id_col,
        timeVar = time_col,
        fZmuit_in = form_zmuit,
        fZuit_in = form_zuit,
        fZui0_in = form_zui0,
        fZvit_in = form_zvit,
        fZvi0_in = form_zvi0,
        fInHasIntercept = TRUE
    )
    startValsMtrx <- NULL
    if (!is.null(start)) startValsMtrx <- matrix(start, ncol = 1)
    # call C++ to do all the rest of the work
    for_cpp <- mdlDataObj$get_data_for_cpp()
    res <- sfacpp_internal_searches(
        y_ = for_cpp$y,
        allData_ = for_cpp$all_data,
        colCounts_ = for_cpp$col_counts,
        idVec_ = for_cpp$idVec,
        timeVec_ = for_cpp$timeVec,
        startVals_ = startValsMtrx,
        reps = nsearches,
        maxRepIter = maxit,
        slengthFrontier = slength_frontier,
        slengthSigmas = slength_sigmas,
        maxStartValFindAttempt = max_attempt_start_vals,
        prodCost = prod,
        model = model,
        dist = dist,
        method = optim_method,
        methodLib = optim_method_lib,
        nsim = nsim,
        seed = seed,
        printLevel = print_level,
        nthreads = nthreads,
        haltonBase = halton_base,
        haltonBurnin = halton_burnin,
        haltonUi0Base = halton_ui0_base,
        scrambledHalton = halton_scrambled,
        shuffledHalton = halton_shuffled
    )
    return(res)
}

#' @rdname bread
#' Inverse of the negative Hessian matrix aka the variance covariance matrix
#' @export
bread.sfacpp <- function(x, ...){
    if (is.null(x$vcov)) stop("cannot calculate bread")
    multi <- ifelse(x$model %in% c("tre", "gtre"), x$nfirm, x$nobs)
    return(x$vcov * x$nobs)
}

#' @rdname coef
#' @export
coef.sfacpp <- function(x, ...){
    vals <- as.vector(x$par)
    nms <- as.vector(x$vars)
    return(setNames(vals, nms))
}

#' @rdname vcov
#' @export
vcov.sfacpp <- function(x, ...){
    return(x$vcov)
}

#' @rdname estfun
#' @export
estfun.sfacpp <- function(x, ...){
    # extract matrix
    return(x$jacobian)
}

#' @rdname logLik
#' @export
logLik.sfacpp <- function(x, ...){
    return(x$logLikelihood)
}

#' @rdname df
#' @export
df.sfacpp <- function(x, ...) {
    return(x$degreesFreedom)
}

#' @rdname model.frame
#' @export
model.frame.sfacpp <- function(x, ...) {
    return(x$modelData)
}

#' @rdname na.action
#' @brief Return index position of the ORIGINAL data which were discarded
#' @export
na.action.sfacpp <- function(x, ...) {
    return(x$omit)
}

#' @rdname summary
#' @brief 
#' @export
print.summary.sfacpp <- function(
    x,
    digits = 4,
    console_width = ifelse(
        options("width") > 150, 150, options("width")
    ),
    ...
) {
    if (class(x) != "sfacpp") stop("invalid type")
    safe_get <- function(val, default_val) {
        if (is.null(val)) return(default_val)
        return(val)
    }
    sfacpp_internal_call_print_model_summary(
        model = safe_get(x$model, "unk"),
        dist = safe_get(x$dist, "unk"),
        nobs = as.integer(safe_get(x$nobs, -1L)),
        nids = as.integer(safe_get(x$nids, -1L)),
        maxT = as.integer(safe_get(x$maxT, -1L)),
        minT = as.integer(safe_get(x$minT, -1L)),
        modelSummaryIn = x$modelSummary,
        modelSummaryTerms = x$vars,
        sigmaParams = x$sigmas,
        nsim = as.integer(safe_get(x$nsim, -1L)),
        llscore = as.numeric(safe_get(x$logLikelihood, -9999999.0)),
        gnorm = as.numeric(safe_get(x$gnorm, -9999999)),
        clusteredSE = safe_get(x$clusteredSE, FALSE),
        haltonBase = as.integer(safe_get(x$haltonBase, -1L)),
        haltonBurnin = as.integer(safe_get(x$haltonBurnin, -1L)),
        haltonUi0Base = as.integer(safe_get(x$haltonUi0Base, -1L)),
        scrambledHalton = safe_get(x$scrambledHalton, FALSE),
        shuffledHalton = safe_get(x$shuffledHalton, FALSE),
        confInt = as.numeric(safe_get(x$confInt, -1)),
        decimalPlaces = digits,
        consoleWidth = console_width,
        idColName = x$idCol,
        timeColName = x$timeCol
    )
}

#' @rdname print
#' @brief 
#' @export print.sfacpp
print.sfacpp <- function(x, ...) {
    # call summary function
    print.summary.sfacpp(x)
}

#' @rdname lrtest
#' @brief likelihood ratio test
#' @export
lrtest.sfacpp <- function(x1, x2) {
    if (("sfacpp" != class(x1)) || ("sfacpp" != class(x2))) {
        stop("doesn't appear to be from sfacpp")
    }
    # run the LR test, call c++
    return(
        sfacpp_internal_lrtest(
            ll0 = x1$logLikelihood,
            ll1 = x2$logLikelihood,
            nparam0 = x1$nparam,
            nparam1 = x2$nparam
        )
    )
}