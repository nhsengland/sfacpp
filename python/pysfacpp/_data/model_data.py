from typing import Optional
import pandas as pd
import numpy as np
import patsy


class EHInternalModelData:
    """
    Base class for processing pandas DataFrame into model matrices
    """

    def __init__(
        self,
        data: pd.DataFrame,
        fIn: str,
        idVar: Optional[str] = None,
        timeVar: Optional[str] = None,
        fZmuit: Optional[str] = None,
        fZuit: Optional[str] = None,
        fZui0: Optional[str] = None,
        fZvit: Optional[str] = None,
        fZvi0: Optional[str] = None,
        fInHasIntercept: bool = True
    ) -> None:
        """
        Initialization
        
        :param data: Input pandas DataFrame
        :param fIn: Input formula (string) for the main frontier
        :param idVar: Name of the ID column
        :param timeVar: Name of the Time column
        :param fZmuit: Formula (string) for determinants of mu - truncated mean
        :param fZuit: Formula (string) for determinants of time-varying inefficiency
        :param fZui0: Formula (string) for determinants of time-invariant inefficiency
        :param fZvit: Formula (string) for determinants of time-varying stochastic noise
        :param fZvi0: Formula (string) for determinants of time-invariant unobserved firm heterogeneity
        :param fInHasIntercept: Boolean indicating if fIn has an intercept
        """
        
        # --- Internal Helper for Variable Extraction ---
        def get_vars_from_formula(formula_str):
            if not formula_str:
                return []
            # Patsy's ModelDesc can parse the formula to find variable names
            desc = patsy.ModelDesc.from_formula(formula_str)
            # Extract variable names from both LHS and RHS
            vars_set = set()
            for term in desc.lhs_termlist + desc.rhs_termlist:
                for factor in term.factors:
                    vars_set.add(factor.name())
            return list(vars_set)

        # 1. Collect all variables to check for missing values (Complete Cases)
        all_vars = set(get_vars_from_formula(fIn))
        
        if fZmuit: all_vars.update(get_vars_from_formula(fZmuit))
        if fZuit:  all_vars.update(get_vars_from_formula(fZuit))
        if fZui0:  all_vars.update(get_vars_from_formula(fZui0))
        if fZvit:  all_vars.update(get_vars_from_formula(fZvit))
        if fZvi0:  all_vars.update(get_vars_from_formula(fZvi0))
        if idVar:  all_vars.add(idVar)
        if timeVar: all_vars.add(timeVar)
        
        # Check for missing values in the relevant columns
        # Filter down to only columns that exist in the dataframe to avoid KeyErrors
        cols_to_check = [c for c in all_vars if c in data.columns]
        # is_complete_case = data[cols_to_check].notna().all(axis=1)
        is_complete_case = (
            data[cols_to_check].notna() & 
            ~data[cols_to_check].isin([np.inf, -np.inf])
        ).all(axis=1)
        # which rows were omitted (from the original data)
        # self._omitted = data.loc[~is_complete_case, ['sfacpp_internal_orig_order']].copy()
        self._omitted = np.flatnonzero(~is_complete_case)
        
        # Filter data
        dt_work = data.loc[is_complete_case, cols_to_check].copy()
        # check that there are some rows
        if dt_work.shape[0] == 0:
            raise ValueError("There are no records in the data (once NaNs/Infs/NAs have been removed)")
        # create integer representation of the original order
        dt_work["sfacpp_internal_orig_order"] = np.arange(len(dt_work))
        # Store omitted order info
        self.complete_cases_orig_order = dt_work["sfacpp_internal_orig_order"].copy()
        # sort by id and time (if panel)
        sort_cols = [c for c in [idVar, timeVar] if c is not None]
        if sort_cols:
            dt_work = dt_work.sort_values(by=sort_cols)
        self.complete_cases_new_order = dt_work[['sfacpp_internal_orig_order']].copy()
        # 
        if idVar and timeVar:
            self._dtIdTimeCols = dt_work[[idVar, timeVar]].copy()
        else:
            self._dtIdTimeCols = None
        # helper for formulaes
        def parse_matrix(formula, default_name=None):
            if not formula:
                return np.ones((len(dt_work), 1)), [default_name] if default_name else []
            # Handle intercept removal if needed manually or via formula
            rhs = formula.split("~")[1] if "~" in formula else formula
            # patsy.dmatrix returns a DesignMatrix (numpy subclass)
            mat = patsy.dmatrix(rhs, dt_work, return_type='matrix')
            names = mat.design_info.column_names
            # Extract raw values to avoid carrying patsy metadata overhead
            return mat, names
        # append all matricies, then hstack
        matrix_parts = []
        
        # --- X Variables (Frontier) ---
        # Handle intercept manually if needed
        # In patsy, "-1" or "+0" removes intercept.
        # clean_fIn = fIn
        # if not fInHasIntercept:
        #      # Check if formula already removes intercept, if not append -1
        #      if "-1" not in fIn and "+0" not in fIn:
        #          clean_fIn += " - 1"
        clean_fIn = fIn
        if not fInHasIntercept and "-1" not in fIn and "+0" not in fIn:
             clean_fIn += " - 1"
        if "~" in clean_fIn:
            f_lhs, f_rhs = clean_fIn.split("~", 1)
        else:
            f_lhs, f_rhs = "", clean_fIn
                
        matX = patsy.dmatrix(f_rhs, dt_work, return_type='matrix')
        self._varsX = matX.design_info.column_names
        matrix_parts.append(matX)

        # --- Helper to process Z matrices ---
        def process_z_matrix(f, prefix, default):
            m, n = parse_matrix(f, default)
            names = [f"{prefix}_{x}" for x in n]
            matrix_parts.append(m)
            return names, n # Return new names and orig names

        # Process all Z components
        self._varsZmuit, origZmuitVars = process_z_matrix(fZmuit, "mu", "cons")
        self._varsZuit, origZuitVars = process_z_matrix(fZuit, "Zuit", "cons")
        self._varsZvit, origZvitVars = process_z_matrix(fZvit, "Zvit", "cons")
        self._varsZui0, origZui0Vars = process_z_matrix(fZui0, "Zui0", "cons")
        self._varsZvi0, origZvi0Vars = process_z_matrix(fZvi0, "Zvi0", "cons")
        # --- Response Variable (Y) ---
        if "~" in fIn:
            y_form = fIn.split("~")[0].strip() + " - 1"
            self._response = patsy.dmatrix(y_form, dt_work, return_type='matrix')
            # Ensure Y is F-contiguous
            self._response = np.asfortranarray(self._response)
        else:
            # Fallback if no LHS provided
            self._varY = None
            self._response = None

        # --- ID and Time Variables ---
        self._iposMMId = None
        self._iposMMTime = None
        # keep track of curr column count
        current_col = sum(m.shape[1] for m in matrix_parts)
        # process id variable
        if idVar:
            self._iposMMId = current_col
            # factorize returns (codes, uniques)
            codes, _ = pd.factorize(dt_work[idVar], sort=True)
            # Codes + 1 for 1-based indexing if preferred, or 0. 
            # Using float to match matrix type, C++ will cast back to int.
            id_vec = (codes + 1).astype(float).reshape(-1, 1)
            matrix_parts.append(id_vec)
            current_col += 1
        # process time variable
        if timeVar:
            self._iposMMTime = current_col
            codes, _ = pd.factorize(dt_work[timeVar], sort=True)
            time_vec = (codes + 1).astype(float).reshape(-1, 1)
            matrix_parts.append(time_vec)
        # final concatenation & conversion
        # hstack
        full_matrix = np.hstack(matrix_parts)
        # convert to fortran contiguous (column major - e.g, for armadillo zero-copy)
        self._modelMatrix = np.asfortranarray(full_matrix)
        # cleanup intermediate list and dataframe
        del matrix_parts
        del dt_work
        # ---- model matrix column name mapping ----
        self._modelMatrixNames = []
        self._modelMatrixNames.extend(self._varsX)
        self._modelMatrixNames.extend(self._varsZmuit)
        self._modelMatrixNames.extend(self._varsZuit)
        self._modelMatrixNames.extend(self._varsZvit)
        self._modelMatrixNames.extend(self._varsZui0)
        self._modelMatrixNames.extend(self._varsZvi0)        
        # ID and Time are manually appended as single columns
        if idVar:
            self._modelMatrixNames.append(idVar)
        if timeVar:
            self._modelMatrixNames.append(timeVar)

        # --- Term Mapping ---
        # Create mapping from terms to index positions
        self._termsToIdx = {}
        all_det_vars = set(origZmuitVars + origZuitVars + origZui0Vars + origZvitVars + origZvi0Vars)
        # Exclude Intercept
        all_det_vars.discard("cons") 
        
        for v in all_det_vars:
            vec = {
                "zmuit": self._ipos(v, origZmuitVars),
                "zuit": self._ipos(v, origZuitVars),
                "zvit": self._ipos(v, origZvitVars),
                "zui0": self._ipos(v, origZui0Vars),
                "zvi0": self._ipos(v, origZvi0Vars)
            }
            self._termsToIdx[v] = vec
    # --- Private Helper ---
    def _ipos(self, val, vec):
        try:
            return vec.index(val)
        except ValueError:
            return None

    # --- Properties (Active Bindings) ---
    @property
    def modelMatrixColumns(self):
        """
        Returns the list of column names for the model matrix.
        Correctly accounts for one-hot encoding (patsy expansion)
        and intercept terms.
        """
        return self._modelMatrixNames

    @property
    def nX(self):
        return len(self._varsX)

    @property
    def nZmuit(self):
        return len(self._varsZmuit)

    @property
    def nZuit(self):
        return len(self._varsZuit)

    @property
    def nZui0(self):
        return len(self._varsZui0)
    
    @property
    def nZvit(self):
        return len(self._varsZvit)

    @property
    def nZvi0(self):
        return len(self._varsZvi0)

    # --- Matrix Accessors ---
    # Using slicing logic. R's logic uses cumulative lengths.
    
    @property
    def X(self):
        return self._modelMatrix[:, 0:self.nX]
        # return np.ascontiguousarray(self._modelMatrix[:, 0:self.nX])

    @property
    def Zmuit(self):
        start = self.nX
        end = start + self.nZmuit
        return self._modelMatrix[:, start:end]
        # return np.ascontiguousarray(self._modelMatrix[:, start:end])

    @property
    def Zuit(self):
        start = self.nX + self.nZmuit
        end = start + self.nZuit
        return self._modelMatrix[:, start:end]
        # return np.ascontiguousarray(self._modelMatrix[:, start:end])

    @property
    def Zvit(self):
        start = self.nX + self.nZmuit + self.nZuit
        end = start + self.nZvit
        return self._modelMatrix[:, start:end]
        # return np.ascontiguousarray(self._modelMatrix[:, start:end])

    @property
    def Zui0(self):
        start = self.nX + self.nZmuit + self.nZuit + self.nZvit
        end = start + self.nZui0
        return self._modelMatrix[:, start:end]
        # return np.ascontiguousarray(self._modelMatrix[:, start:end])

    @property
    def Zvi0(self):
        start = self.nX + self.nZmuit + self.nZuit + self.nZvit + self.nZui0
        end = start + self.nZvi0
        return self._modelMatrix[:, start:end]
        # return np.ascontiguousarray(self._modelMatrix[:, start:end])

    @property
    def y(self):
        # ensure y is F-contiguous
        return self._response
        # return np.ascontiguousarray(self._response)

    @property
    def model_matrix(self):
        return self._modelMatrix

    # --- Name Accessors ---

    @property
    def namesX(self):
        return self._varsX

    @property
    def namesZmuit(self):
        return self._varsZmuit

    @property
    def namesZuit(self):
        return self._varsZuit

    @property
    def namesZui0(self):
        return self._varsZui0

    @property
    def namesZvit(self):
        return self._varsZvit

    @property
    def namesZvi0(self):
        return self._varsZvi0
    
    @property
    def indexPosOfTerms(self):
        return self._termsToIdx

    # --- ID/Time Accessors ---
    
    @property
    def idVec(self) -> np.ndarray | None:
        if self._iposMMId is None:
            return None
        # Return as column vector
        return np.asfortranarray(a=self._modelMatrix[:, self._iposMMId].astype(dtype=np.int32))

    @property
    def timeVec(self) -> np.ndarray | None:
        if self._iposMMTime is None:
            return None
        return np.asfortranarray(a=self._modelMatrix[:, self._iposMMTime].astype(dtype=np.int32))

    @property
    def idTimeCols(self):
        return self._dtIdTimeCols

    @property
    def omit(self):
        return self._omitted

    @property
    def order(self):
        return self.complete_cases_new_order
    
    # --- For patsy compatibility ---
    @property
    def design_info(self):
        """
        Mimics R's terms() by returning the DesignInfo object 
        from the X matrix construction if available.
        This allows downstream code to inspect term labels.
        """
        # We need to reconstruct or store the DesignInfo from the X construction
        # Since we just stored the names, we might not have the full object 
        # unless we saved it in __init__. 
        # For basic usage, returning namesX might be sufficient, 
        # but storing the actual patsy DesignInfo in __init__ is better practice.
        return self._varsX