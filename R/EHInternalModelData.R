



#' @import R6
#' @import data.table
#' 
#' @description
#' Base class for processing data.table into model matrix
#' 
#' @exportClass
EHInternalModelData <- R6Class(
  classname = "EHInternalModelData",
  public = list(
    #' @field modelMatrix - combined model matrix containing all predictors
    modelMatrix = NULL,
    #' @field response - response vector (y)
    response = NULL,
    #' @field omitted vector of indicies for rows due to missing / infinite values
    omitted = NULL,
    #' @field complete_cases_orig_order The original row indicies of the complete cases
    complete_cases_orig_order = NULL,
    #' @field complete_cases_new_order The new sorted indicies relevative to the original order
    complete_cases_new_order = NULL,
    #' @field dtIdTimeCols
    dtIdTimeCols = NULL,
    #' @description
    #' Initialize EHInternalModelData object
    #' @param data Input data.frame or data.table
    #' @param fIn Formula string for the frontier
    #' @param idVec Name of the ID column
    #' @param timeVar Name of the Time column
    #' @param fZmuit Formula string for determinants of mu - for truncated mean
    #' @param fZuit Formula string for determinants of time-varying inefficiency
    #' @param fZui0 Formula string for determinants of time-invariant inefficiency
    #' @param fZvit Formula string for determinants of time-varying stochastic noise
    #' @param fZvi0 Formula string for determinants of time-invariant latent firm heterogeneity
    #' @param fInHasIntercept Boolean indicating whether fIn has an intercept
    initialize = function(
      data,
      f_in,
      idVar = NULL,
      timeVar = NULL,
      fZmuit_in = NULL,
      fZuit_in = NULL,
      fZui0_in = NULL,
      fZvit_in = NULL,
      fZvi0_in = NULL,
      fInHasIntercept = TRUE
    ) {
      # convert input formula
      fIn <- private$to_formula(f_in, implies_rhs_only = FALSE)
      fZmuit <- private$to_formula(fZmuit_in)
      fZuit <- private$to_formula(fZuit_in)
      fZvit <- private$to_formula(fZvit_in)
      fZui0 <- private$to_formula(fZui0_in)
      fZvi0 <- private$to_formula(fZvi0_in)
      # convert to data.table for efficient processing
      dt <- data.table::as.data.table(data)
      # collect all unique variables to check for missing values
      all_vars <- unique(c(
        private$get_vars_from_formula(fIn),
        private$get_vars_from_formula(fZmuit),
        private$get_vars_from_formula(fZuit),
        private$get_vars_from_formula(fZui0),
        private$get_vars_from_formula(fZvit),
        private$get_vars_from_formula(fZvi0)
      ))
      if (!is.null(idVar)) all_vars <- c(all_vars, idVar)
      if (!is.null(timeVar)) all_vars <- c(all_vars, timeVar)
      # filter only for columns that exist in the dataframe
      cols_to_check <- intersect(all_vars, names(dt))
      # check for missing values - NA, Inf
      is_complete <- complete.cases(dt[, ..cols_to_check])
      is_finite <- apply(dt[, ..cols_to_check], 1, function(row){
        all(is.finite(unlist(row[sapply(row, is.numeric)])))
      })
      is_valid_row <- is_complete & is_finite
      # store omitted indicies
      self$omitted <- which(!is_valid_row)
      # filter data
      dt_work <- dt[is_valid_row, ..cols_to_check]
      # create integer representation of the original order and store
      dt_work[, sfacpp_internal_orig_order := .I]
      self$complete_cases_orig_order <- dt_work$sfacpp_internal_orig_order
      # sort by id & time
      if (!is.null(idVar) && !is.null(timeVar)) {
        setorderv(dt_work, c(idVar, timeVar))
      }
      self$complete_cases_new_order <- dt_work$sfacpp_internal_orig_order
      # store id & time variables
      if(!is.null(idVar) && !is.null(timeVar)) {
        self$dtIdTimeCols <- dt_work[, c(idVar, timeVar), with = FALSE]
      }
      matrix_parts <- c()
      # Extract LHS for Y, RHS for X
      # Extract Y (Response)
      # model.frame is robust for extraction
      mf <- model.frame(fIn, data = dt_work, na.action = na.pass)
      self$response <- model.response(mf)
      if (is.null(self$response)) {
        stop("Formula fIn must have a response variable (LHS).")
      }
      self$response <- as.matrix(self$response) # Ensure col-vector
      # Build X Matrix (RHS)
      # delete.response handles removing Y from the matrix
      mt <- attr(mf, "terms")
      matX <- model.matrix(mt, mf)
      private$varsX <- colnames(matX)
      # --- Helper to process Z matrices ---
      # Process all Z components
      res_mu <- private$build_mat(fZmuit, dt_work, TRUE, "mu", "cons")
      private$varsZmuit <- res_mu$new_names
      origZmuitVars <- res_mu$orig_terms
      # >> zuit <<
      res_u <- private$build_mat(fZuit, dt_work, TRUE, "Zuit", "cons")
      private$varsZuit <- res_u$new_names
      origZuitVars <- res_u$orig_terms
      # >> zvit <<
      res_v <- private$build_mat(fZvit, dt_work, TRUE, "Zvit", "cons")
      private$varsZvit <- res_v$new_names
      origZvitVars <- res_v$orig_terms
      # >> zui0 <<
      res_u0 <- private$build_mat(fZui0, dt_work, TRUE, "Zui0", "cons")
      private$varsZui0 <- res_u0$new_names
      origZui0Vars <- res_u0$orig_terms
      # >> zvi0 <<
      res_v0 <- private$build_mat(fZvi0, dt_work, TRUE, "Zvi0", "cons")
      private$varsZvi0 <- res_v0$new_names
      origZvi0Vars <- res_v0$orig_terms
      # --- ID and Time Variables ---
      mat_id <- matrix(0, nrow = nrow(dt_work), ncol = 0)
      if (!is.null(idVar)) {
        id_vals <- as.numeric(as.factor(dt_work[[idVar]]))
        mat_id <- as.matrix(id_vals)
      }
      private$id_vec <- mat_id
      mat_time <- matrix(0, nrow = nrow(dt_work), ncol = 0)
      if (!is.null(timeVar)) {
        time_vals <- as.numeric(as.factor(dt_work[[timeVar]]))
        mat_time <- as.matrix(time_vals)
      }
      private$time_vec <- mat_time
      # build model matrix
      self$modelMatrix <- cbind(
        matX,
        res_mu$mat,
        res_u$mat,
        res_v$mat,
        res_u0$mat,
        res_v0$mat
      )
      # ensure matrix
      if (!is.matrix(self$modelMatrix)) {
        self$modelMatrix <- as.matrix(self$modelMatrix)
      }
      # --- Column Names ---
      full_names <- c(
        private$varsX,
        private$varsZmuit,
        private$varsZuit,
        private$varsZvit,
        private$varsZui0,
        private$varsZvi0
      )
      private$modelMatrixNames <- full_names
      colnames(self$modelMatrix) <- full_names
      # --- Term Mapping ---
      private$termsToIdx <- list()
      all_det_vars <- unique(
        c(origZmuitVars, origZuitVars, origZui0Vars, origZvitVars, origZvi0Vars)
      )
      all_det_vars <- setdiff(all_det_vars, "cons")
      get_pos <- function(val, vec) {
        idx <- match(val, vec, nomatch = NA)
        if (is.na(idx)) return(NA) else return(idx)
      }
      for (v in all_det_vars) {
        private$termsToIdx[[v]] <- list(
          zmuit = get_pos(v, origZmuitVars),
          zuit  = get_pos(v, origZuitVars),
          zvit  = get_pos(v, origZvitVars),
          zui0  = get_pos(v, origZui0Vars),
          zvi0  = get_pos(v, origZvi0Vars)
        )
      }
    },
    #' @description
    #' Prepare data for zero-copy c++ transfer
    #' Returns simple list, containing large matrix & dimensions
    get_data_for_cpp = function() {
      list(
        # response vector
        y = as.vector(self$response),
        # design matrix
        all_data = self$modelMatrix,
        # column counts for reconstruction in C++
        col_counts = as.integer(c(
          length(private$varsX),
          length(private$varsZmuit),
          length(private$varsZuit),
          length(private$varsZvit),
          length(private$varsZui0),
          length(private$varsZvi0)
        )),
        idVec = private$id_vec,
        timeVec = private$time_vec
      )
    }
  ),
  active = list(
    modelMatrixColumns = function() private$modelMatrixNames,
    nX = function() length(private$varsX),
    nZmuit = function() length(private$varsZmuit),
    nZuit = function() length(private$varsZuit),
    nZui0 = function() length(private$varsZui0),
    nZvit = function() length(private$varsZvit),
    nZvi0 = function() length(private$varsZvi0),
    # --- Matrix Accessors ---
    # R uses 1-based indexing inclusive: [start:end]
    X = function() {
      if (self$nX == 0) return(NULL)
      self$modelMatrix[, 1:self$nX, drop = FALSE]
    },
    Zmuit = function() {
      if (self$nZmuit == 0) return(NULL)
      start <- self$nX + 1
      end <- start + self$nZmuit - 1
      self$modelMatrix[, start:end, drop = FALSE]
    },
    Zuit = function() {
      if (self$nZuit == 0) return(NULL)
      start <- self$nX + self$nZmuit + 1
      end <- start + self$nZuit - 1
      self$modelMatrix[, start:end, drop = FALSE]
    },
    Zvit = function() {
      if (self$nZvit == 0) return(NULL)
      start <- self$nX + self$nZmuit + self$nZuit + 1
      end <- start + self$nZvit - 1
      self$modelMatrix[, start:end, drop = FALSE]
    },
    Zui0 = function() {
      if (self$nZui0 == 0) return(NULL)
      start <- self$nX + self$nZmuit + self$nZuit + self$nZvit + 1
      end <- start + self$nZui0 - 1
      self$modelMatrix[, start:end, drop = FALSE]
    },
    Zvi0 = function() {
      if (self$nZvi0 == 0) return(NULL)
      start <- self$nX + self$nZmuit + self$nZuit + self$nZvit + self$nZui0 + 1
      end <- start + self$nZvi0 - 1
      self$modelMatrix[, start:end, drop = FALSE]
    },
    y = function() self$response,
    model_matrix = function() self$modelMatrix,
    # --- Name Accessors ---
    namesX = function() private$varsX,
    namesZmuit = function() private$varsZmuit,
    namesZuit = function() private$varsZuit,
    namesZui0 = function() private$varsZui0,
    namesZvit = function() private$varsZvit,
    namesZvi0 = function() private$varsZvi0,
    indexPosOfTerms = function() private$termsToIdx,
    # --- ID/Time Accessors ---
    idVec = function() self$id_vec,
    timeVec = function() self$time_vec,
    idTimeCols = function() self$dtIdTimeCols,
    omit = function() self$omitted,
    order = function() self$complete_cases_new_order,
    design_info = function() private$varsX
  ),
  private = list(
    # Variable name storage
    varsX = NULL,
    varsZmuit = NULL,
    varsZuit = NULL,
    varsZvit = NULL,
    varsZui0 = NULL,
    varsZvi0 = NULL,
    # Matrix column indices
    id_vec = NULL,
    time_vec = NULL,
    # Metadata
    modelMatrixNames = NULL,
    termsToIdx = NULL,
    #' @description
    #' Helper to process formula inputs, ensuring they are formula objects
    to_formula = function(x, implies_rhs_only = TRUE) {
      if (is.null(x) || (is.character(x) && x == "")) return(NULL)
      if (inherits(x, "formula")) return(x)
      # is a string - convert
      s <- as.character(x)
      if (length(s) > 1) s <- paste(s, collapse = " ")
      if (implies_rhs_only && !grepl("~", s)) s <- paste("~", s)
      return(as.formula(s))
    },
    #' @description
    #' Helper to extract variables from formula string
    #' @param formula_str A formula string
    get_vars_from_formula = function(form) {
      if (is.null(form)) return(character(0))
      tryCatch({
          return(all.vars(form))
      },
      error = function(e) {
          return(character(0))
      })
    },
    #' @description
    #' Helper to construct matrix
    #' @param formula_str
    #' @param data_source
    #' @param default_name
    #' @return
    build_mat = function(
      f,
      df,
      default_intercept = TRUE,
      prefix = "",
      default_nm = "cons"
    ) {
      if (is.null(f)) {
        nms <- default_nm
        if (prefix != "") {
          nms <- paste0(prefix, "_", nms)
        }
        return(
          list(
            mat = matrix(1, nrow = nrow(df), ncol = 1),
            new_names = nms,
            orig_terms = nms
          )
        )
      }
      mm <- model.matrix(f, data = df)
      nms <- colnames(mm)
      if (prefix != "") {
        nms <- paste0(prefix, "_", nms)
        nms <- gsub("\\(Intercept\\)", default_nm, nms)
      }
      return(
        list(mat = mm, new_names = nms, orig_terms = colnames(mm))
      )
    }
  )
)