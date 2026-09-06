import math
import shutil
from typing import Optional, Any, Union
import pandas as pd
import numpy as np
from .._base.base import PySfaCppBase
from .._data.model_data import EHInternalModelData
from .._result.tre import PySfaCppResult

try:
    import pysfacpp_internal  # ty:ignore[unresolved-import]
    _DEFAULT_NTHREADS = math.floor(pysfacpp_internal._get_available_threads() * 0.8)
except ImportError:
    pysfacpp_internal = None  # type: ignore[assignment]
    _DEFAULT_NTHREADS = 1

class PySfaCpp(PySfaCppBase):

    def __init__(
        self,
        form_x: str,
        data: pd.DataFrame,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        form_zmuit: Optional[str] = None,
        form_zuit: Optional[str] = None,
        form_zui0: Optional[str] = None,
        form_zvit: Optional[str] = None,
        form_zvi0: Optional[str] = None,
        start: Optional[Union[list[float], np.ndarray]] = None,
        prod: int = 1,
        dist: str = "hnorm",
        model: str = "tre",
        nsim: int = 500,
        optim_opts: Optional[dict[str, Any]] = None,
        hessian_calc: str = "analytical",
        hessian_num_approx_accuracy: int = 3,
        conf_int: float = 0.95,
        marg_eff: str = "wang2002",
        estimate_marg_eff: bool = False,
        estimate_marg_eff_ci: bool = False,
        marg_eff_bootstrap_reps: int = 500,
        seed: int = 1234,
        print_level: int = 2,
        form_has_x_intercept: bool = True,
        clustered_se: bool = True,
        nthreads: int = _DEFAULT_NTHREADS,
        calculate_efficiency_scores: bool = False,
        ghk_sim_reps: int = 2000,
        halton_base: int = 2,
        halton_burnin: int = 1000,
        halton_ui0_base: int = 3,
        halton_scrambled: bool = True,
        halton_shuffled : bool = False,
        should_copy_from_numpy: bool = False,
        display_decimal_places: int = 5,
        display_console_width: Optional[int] = None,
        optim_method: str = "tr"
    ) -> None:
        """
        Docstring for __init__
        
        :param self: Description
        :param form_x: Description
        :type form_x: str
        :param data: Description
        :type data: pd.DataFrame
        :param id_col: Description
        :type id_col: Optional[str]
        :param time_col: Description
        :type time_col: Optional[str]
        :param form_zmuit: Description
        :type form_zmuit: Optional[str]
        :param form_zuit: Description
        :type form_zuit: Optional[str]
        :param form_zui0: Description
        :type form_zui0: Optional[str]
        :param form_zvit: Description
        :type form_zvit: Optional[str]
        :param form_zvi0: Description
        :type form_zvi0: Optional[str]
        :param start: Description
        :type start: Optional[Union[list[float], np.ndarray]]
        :param prod: Description
        :type prod: int
        :param dist: Description
        :type dist: str
        :param model: Description
        :type model: str
        :param nsim: Description
        :type nsim: int
        :param optim_opts: Description
        :type optim_opts: Optional[dict[str, Any]]
        :param hessian_calc: Description
        :type hessian_calc: str
        :param hessian_num_approx_accuracy: Description
        :type hessian_num_approx_accuracy: int
        :param conf_int: Description
        :type conf_int: float
        :param marg_eff: Description
        :type marg_eff: str
        :param estimate_marg_eff: Description
        :type estimate_marg_eff: bool
        :param estimate_marg_eff_ci: Description
        :type estimate_marg_eff_ci: bool
        :param marg_eff_bootstrap_reps: Description
        :type marg_eff_bootstrap_reps: int
        :param seed: Description
        :type seed: int
        :param print_level: Description
        :type print_level: int
        :param start_vals_method: Description
        :type start_vals_method: str
        :param start_vals_method_lib: Description
        :type start_vals_method_lib: str
        :param form_has_x_intercept: Description
        :type form_has_x_intercept: bool
        """
        if optim_method not in ["tr", "hybrid_bfgs_tr", "hybrid_lbfgs_tr", "bfgs_tr", "lbfgs_tr"]:
            raise ValueError(
                "'optim_method' must be 'tr' [trusted region], 'hybrid_bfgs_tr' [bfgs + tr], or 'hybrid_lbfgs_tr' [lfbgs + tr]"
            )
        self.__optim_method: str = optim_method
        self.__optim_method_lib: str = "dlib"
        # check model is an acceptable type
        if not model in ["tre", "gtre"]:
            raise ValueError("'model' must be one of 'tre', 'gtre'")
        self.__model: str = model
        # override clustered_se in the case of cross sectional
        if model == "cross":
            self.__clustered_se = False
        else:
            self.__clustered_se = clustered_se
        # check whether distribution is acceptable
        if not dist in ["hnorm"]:
            raise ValueError("'dist' must be one of 'hnorm'")
        self.__dist: str = dist
        # check whether marginal effect is acceptable
        if not marg_eff in ["wang2002"]:
            raise ValueError("'marg_eff' must be one of 'wang2002'")
        self.__marg_eff: str = marg_eff
        # checks on arguments depending on model type
        if model in ["tfe", "tre", "gtre"]:
            if id_col is None:
                raise ValueError("'idCol' must be specified for 'tfe', 'tre', 'gtre' models")
            # check id col is in the data
            if id_col not in data.columns:
                raise ValueError(f"'idCol' {id_col} must be in 'data'")
            self.__id_col: str = id_col
            if time_col is None:
                raise ValueError("'timeCol' must be specified for 'tfe', 'tre', 'gtre' models")
            # check time col is in the data
            if time_col not in data.columns:
                raise ValueError(f"'timeCol' {time_col} must be in 'data'")
            self.__time_col: str = time_col
        # check confidence interval is in bounds
        if (conf_int < 0.0) or (conf_int > 1.0):
            raise ValueError("'conf_int' must be between 0 and 1")
        self.__conf_int: float = conf_int
        # check marginal effect is acceptable type - boolean
        if not isinstance(estimate_marg_eff, bool):
            raise TypeError("'estimate_marg_eff' must be boolean")
        self.__estimate_marg_eff: bool = estimate_marg_eff
        # check bootstrap replications is positive
        if (marg_eff_bootstrap_reps < 0) or (not isinstance(marg_eff_bootstrap_reps, int)):
            raise ValueError("'marg_eff_bootstrap_reps' must be a positive integer")
        self.__marg_eff_bootstrap_reps: int = marg_eff_bootstrap_reps
        # check print level
        if (print_level < 0) or (not isinstance(print_level, int)):
            raise ValueError("'print_level' must be at least 0, and an integer")
        self.__print_level: int = print_level
        # check number of simulations is a positive integer
        if (nsim < 0) or (not isinstance(nsim, int)):
            raise ValueError("'nsim' must be positive integer")
        self.__nsim: int = nsim
        # optim_opts must be a dictionary
        if not optim_opts is None:
            if not isinstance(optim_opts, dict):
                raise TypeError("'optim_opts' should be a dictionary")
        self.__optim_opts: Optional[dict[str, Any]] = optim_opts
        # check prod is 1 or -1 for prod, cost respectively
        if not ((prod == 1) or (prod == -1)):
            raise ValueError("'prod' should be 1 for prod, -1 for cost")
        self.__prod: int = prod
        # set formulas to attributes
        self.__form_x: str = form_x
        self.__form_zmuit: Optional[str] = form_zmuit
        self.__form_zuit: Optional[str] = form_zuit
        self.__form_zui0: Optional[str] = form_zui0
        self.__form_zvit: Optional[str] = form_zvit
        self.__form_zvi0: Optional[str] = form_zvi0
        self.__hessian_calc: str = hessian_calc
        self.__hessian_num_approx_accuracy: int = hessian_num_approx_accuracy
        self.__seed: int = seed
        self.__estimate_marg_eff_ci: bool = estimate_marg_eff_ci
        # threads
        if nthreads < 1:
            raise ValueError("Must have at least 1 thread")
        self.__nthreads = nthreads
        if ghk_sim_reps < 1:
            raise ValueError("ghk_sim_reps must be a positive integer")
        self.__ghk_sim_reps: int = ghk_sim_reps
        self.__calculate_efficiency_scores: bool = calculate_efficiency_scores
        self.__halton_base: int = halton_base
        self.__halton_burnin: int = halton_burnin
        self.__halton_ui0_base: int = halton_ui0_base
        self.__halton_scrambled: bool = halton_scrambled
        self.__halton_shuffled: bool = halton_shuffled
        self.__should_copy_from_numpy: bool = should_copy_from_numpy
        # # create an integer representation of id and time
        # self.__data["sfacpp_internal_id"] = pd.factorize(self.__data[id_col])[0]
        # self.__data["sfacpp_internal_time"] = pd.factorize(self.__data[time_col])[0]
        # if tre model, check whether formula contains ID already, if not, add it
        if self.__model == "tfe":
            if id_col is None:
                raise ValueError("id col should not be none")
            if id_col not in self.__form_x:
                self.__form_x = f"{self.__form_x} + C({id_col})"
        # model data class
        self.__mdl_data_obj = EHInternalModelData(
            data=data,
            fIn=self.__form_x,
            idVar=id_col,
            timeVar=time_col,
            # idVar="sfacpp_internal_id",
            # timeVar="sfacpp_internal_time",
            fZmuit=self.__form_zmuit,
            fZuit=self.__form_zuit,
            fZui0=self.__form_zui0,
            fZvit=self.__form_zvit,
            fZvi0=self.__form_zvi0,
            fInHasIntercept=form_has_x_intercept
        )
        self.__start_vals_mtrx = None
        if start is not None:
            self.__start_vals_mtrx = np.asfortranarray(np.array(start).reshape(-1, 1))
        self.__display_console_width = self._PySfaCppBase__console_width() if display_console_width is None else display_console_width
        self.__display_decimal_places = display_decimal_places
        return

    def searches(
        self,
        nsearches: int = 500,
        maxit: int = 150,
        slength_frontier: float = 2.0,
        slength_sigmas: float = 0.8,
        max_attempt_start_vals: int = 50,
        seed: int = 1234,
        print_level: int = 0,
        start: Optional[Union[list[float], np.ndarray]] = None,
        digits: int = 4,
        display_console_width: Optional[int] = None,
        optim_method: Optional[str] = None,
        maximum_attempts: int = 100000,
        override_start_val_check: bool = False
    ) -> Optional[dict]:
        """
        Run searches 

        optim_method: Optional[str]
            Optimization method (specific to searches, does not overwrite the initialized value used for .fit())
        """
        if optim_method is None:
            optim_method = self.__optim_method
        if optim_method not in ["tr", "hybrid_bfgs_tr", "hybrid_lbfgs_tr"]:
            raise ValueError("'optim_method' should be one of 'tr', 'hybrid_bfgs_tr', 'hybrid_lbfgs_tr'")
        self._PySfaCppBase__setup_logger()  # ty:ignore[unresolved-attribute]
        start_ft = None
        if start is not None:
            start_ft: np.ndarray = np.asfortranarray(a=np.array(object=start).reshape(-1, 1))
        display_console_width: int = self._PySfaCppBase__console_width() if display_console_width is None else display_console_width
        if maximum_attempts < nsearches:
            raise ValueError("'maximum_attempts' should be bigger than 'nsearches'")
        try:
            res: dict = pysfacpp_internal._pysfacpp_searches(
                self.__mdl_data_obj.y, # y
                self.__mdl_data_obj.X, # x
                self.__mdl_data_obj.Zmuit, # zmuit
                self.__mdl_data_obj.Zuit, # zuit
                self.__mdl_data_obj.Zvit, # zvit
                self.__mdl_data_obj.Zui0, # zui0
                self.__mdl_data_obj.Zvi0, # zvi0
                start_ft, # starting values,
                self.__mdl_data_obj.idVec, # id vector
                self.__mdl_data_obj.timeVec, # time vector
                nsearches, # number of searches to do
                maxit, # max iterations per run
                slength_frontier, # how wide the search should be (frontier)
                slength_sigmas, # how wide the search should be for sigma terms
                max_attempt_start_vals, # how many attempts to find a valid set of starting values per iteration
                self.__prod, # whether prod or cost function
                self.__model, # sfa model to estimate
                self.__dist, # distribution of inefficiency component
                optim_method, # estimation algorithm e.g., bfgs
                self.__optim_method_lib, # library to use for algorithm
                self.__nsim, # number of simulations for RE based models
                seed, # seed
                print_level, # print level
                self.__nthreads, # number of threads to use
                self.__halton_base,
                self.__halton_burnin,
                self.__halton_ui0_base,
                self.__halton_scrambled,
                self.__halton_shuffled,
                self.__should_copy_from_numpy, # whether or not to copy from numpy, or read python memory.
                display_console_width,
                digits,
                maximum_attempts,
                override_start_val_check
            )
            return res
        except Exception as e:
            raise e
    
    def fit(self) -> PySfaCppResult:
        """
        Fit the model
        """
        self._PySfaCppBase__setup_logger()
        # find what the width of the console is
        columns, lines = shutil.get_terminal_size()
        try:
            res: dict[str, Any] = pysfacpp_internal._pysfacpp_internal(
                self.__mdl_data_obj.y, # y
                self.__mdl_data_obj.X, # x
                self.__mdl_data_obj.Zmuit, # zmuit
                self.__mdl_data_obj.Zuit, # zuit
                self.__mdl_data_obj.Zvit, # zvit
                self.__mdl_data_obj.Zui0, # zui0
                self.__mdl_data_obj.Zvi0, # zvi0
                self.__start_vals_mtrx, # starting values
                self.__mdl_data_obj.idVec, # id vector
                self.__mdl_data_obj.timeVec, # time vector
                self.__prod, # whether prod or cost function
                self.__model, # sfa model to estimate
                self.__dist, # distribution of inefficiency component
                self.__optim_method, # estimation algorithm e.g., bfgs
                self.__optim_method_lib, # library to use for algorithm
                self.__marg_eff, # type of marginal effect to estimate
                self.__optim_opts, # optimisation options
                self.__mdl_data_obj.namesX, # x names
                self.__mdl_data_obj.namesZmuit, # zmuit names
                self.__mdl_data_obj.namesZuit, # zuit names
                self.__mdl_data_obj.namesZvit, # zvit names
                self.__mdl_data_obj.namesZui0, # zui0 names
                self.__mdl_data_obj.namesZvi0, # zvi0 names
                self.__nsim, # number of simulations for RE based models
                self.__hessian_calc, # how to calculate hessian matrix
                self.__hessian_num_approx_accuracy, # accuracy for hessian num approx algo
                self.__seed, # seed
                self.__conf_int, # confidence interval
                self.__estimate_marg_eff, # whether to estimate MEs
                self.__estimate_marg_eff_ci, # whether to estimate CIs for MEs
                self.__marg_eff_bootstrap_reps, # bootstrap replications
                self.__print_level, # print level
                self.__clustered_se, # whether or not to cluster SEs
                self.__nthreads, # number of threads to use
                self.__calculate_efficiency_scores, # whether to calculate efficiency scores
                self.__ghk_sim_reps, # number of
                self.__halton_base,
                self.__halton_burnin,
                self.__halton_ui0_base,
                self.__halton_scrambled,
                self.__halton_shuffled,
                self.__should_copy_from_numpy, # whether or not to copy from numpy, or read python memory.
                self.__display_decimal_places,
                self.__display_console_width,
                self.__id_col,
                self.__time_col
            )
        except Exception as e:
            raise e
        

        def reorder_matrix(m, return_df: bool = False, with_id_time_cols = False) -> pd.DataFrame | np.ndarray | None: 
            """
            Takes a matrix/array, attaches the stored original order index, 
            sorts back to original order, and returns the matrix.
            """
            if m is None:
                return None
            dt_tmp = pd.DataFrame(data=m)
            # reset_index(drop=True) ensures we align by row number since dt was sorted previously
            dt_tmp: pd.DataFrame = pd.concat(objs=[dt_tmp, self.__mdl_data_obj.order.reset_index(drop=True)], axis=1)
            if return_df:
                if with_id_time_cols:
                    dt_tmp = pd.concat([
                        self.__mdl_data_obj.idTimeCols.reset_index(drop=True),
                        dt_tmp,
                    ],
                    axis = 1
                )
                dt_tmp: pd.DataFrame = dt_tmp.sort_values(by="sfacpp_internal_orig_order")
                dt_tmp: pd.DataFrame = dt_tmp.drop(columns=["sfacpp_internal_orig_order"])
                return dt_tmp
            dt_tmp: pd.DataFrame = dt_tmp.sort_values(by="sfacpp_internal_orig_order")
            dt_tmp: pd.DataFrame = dt_tmp.drop(columns=["sfacpp_internal_orig_order"])
            return dt_tmp.to_numpy()

        # Extract the model matrix, bind the data order to it, and reorder
        if isinstance(self.__mdl_data_obj.model_matrix, pd.DataFrame):
            mm_df = self.__mdl_data_obj.model_matrix.copy()
        else:
            mm_df = pd.DataFrame(self.__mdl_data_obj.model_matrix)
            
        dt_mm = pd.concat([mm_df, self.__mdl_data_obj.order.reset_index(drop=True)], axis=1)
        dt_mm = dt_mm.sort_values(by="sfacpp_internal_orig_order")
        dt_mm = dt_mm.drop(columns=["sfacpp_internal_orig_order"])

        # Store results in dictionary
        # res["origData"] = self.__data
        res["modelData"] = dt_mm
        res["reorderingIdx"] = self.__mdl_data_obj.order.values
        # res["modelDataObj"] = self.__mdl_data_obj
        
        # Handle omitted observations
        # In Python, we generally pass boolean masks rather than an "omit" class
        if hasattr(self.__mdl_data_obj, 'omit') and self.__mdl_data_obj.omit is not None:
            res["omit"] = self.__mdl_data_obj.omit
            # R's na.action class doesn't have a direct Python equivalent, 
            # but we store the omitted indices for downstream usage.
            res["na_action"] = res["omit"]

        res["formula"] = self.__form_x
        # patsy dmatrix objects have a .design_info attribute which serves a similar role to R's terms()
        if hasattr(self.__mdl_data_obj, 'design_info'):
            res["terms"] = self.__mdl_data_obj.design_info 
        
        res["form_zmuit"] = self.__form_zmuit
        res["form_zuit"] = self.__form_zuit
        res["form_zvit"] = self.__form_zvit
        res["form_zvi0"] = self.__form_zvi0
        res["form_zui0"] = self.__form_zui0

        # Process Variance Covariance Matrix (names)
        if res.get("vcov") is not None:
            vcov = pd.DataFrame(data=res["vcov"])
            if res.get("vars") is not None:
                vcovvars = res["vars"]
                vcov.columns = vcovvars
                vcov.index = vcovvars
            else:
                # Create default names: dim_1, dim_2...
                cols: list[str] = [f"dim_{i+1}" for i in range(vcov.shape[1])]
                vcov.columns = cols
                vcov.index = cols
            res["vcov"] = vcov

        # process the model summarry (names)
        if res.get("modelSummary") is not None:
            ms = pd.DataFrame(data=res["modelSummary"])
            # set index (row names)
            if res.get("vars") is not None:
                ms.index = res["vars"]
            else:
                ms.index = [f"dim_{i+1}" for i in range(ms.shape[0])]
            # set column names for model summary
            ms.columns = [
                "estimate",
                "std_err",
                "t_statistic",
                "p_value",
                "ci_lower",
                "ci_upper"
            ]
            res["modelSummary"] = ms
        # process efficiency scores
        if res.get("efficiencyTransient") is not None:
            eff_t_df: pd.DataFrame = reorder_matrix(
                m=res["efficiencyTransient"],
                return_df=True,
                with_id_time_cols=True
            )  # ty:ignore[invalid-assignment]
            # eff_t_df.columns[0] = "transient_efficiency"
            eff_t_df.columns = list(eff_t_df.columns)[:2] + ["transient_efficiency"]
            # overwrite attribute
            res["efficiencyTransient"] = eff_t_df
        # persisent inefficiency
        if res.get("efficiencyPersistent") is not None:
            eff_p_df: pd.DataFrame = reorder_matrix(
                m=res["efficiencyPersistent"],
                return_df=True,
                with_id_time_cols=True
            )  # ty:ignore[invalid-assignment]
            # eff_p_df.columns[0] = "persistent_efficiency"
            eff_p_df.columns = list(eff_p_df.columns)[:2] + ["persistent_efficiency"]
            # overwrite attribute
            res["efficiencyPersistent"] = eff_p_df

        # Process Marginal Effects
        if self.__estimate_marg_eff and res.get("marginalEffects") is not None and res.get("marginalEffectsNames") is not None:
            me_df: pd.DataFrame = reorder_matrix(
                m=res["marginalEffects"],
                return_df=True,
                with_id_time_cols=True
            )  # ty:ignore[invalid-assignment]
            me_df.columns = list(me_df.columns)[:2] + res["marginalEffectsNames"]
            # clear names as no longer need
            res["marginalEffectsNames"] = None
            # overwrite attribute with new dataframe
            res["marginalEffects"] = me_df

        # Process Marginal Effects Confidence Intervals
        if self.__estimate_marg_eff_ci and res.get("marginalEffectsCICols") is not None:
            mcinms = res["marginalEffectsCICols"]
            res["marginalEffectsCICols"] = None
            
            if res.get("marginalEffectsLwrCI") is not None:
                lwr_df: pd.DataFrame = reorder_matrix(
                    m=res["marginalEffectsLwrCI"],
                    return_df=True,
                    with_id_time_cols=True
                )  # ty:ignore[invalid-assignment]
                lwr_df.columns = list(lwr_df.columns)[:2] + mcinms
                res["marginalEffectsLwrCI"] = lwr_df
                
            if res.get("marginalEffectsUprCI") is not None:
                upr_df: pd.DataFrame = reorder_matrix(
                    m=res["marginalEffectsUprCI"],
                    return_df=True,
                    with_id_time_cols=True
                )  # ty:ignore[invalid-assignment]
                upr_df.columns = list(upr_df.columns)[:2] + mcinms
                res["marginalEffectsUprCI"] = upr_df
        res["modelMatrixCols"] = self.__mdl_data_obj.modelMatrixColumns
        res["model"] = self.__model
        res["dist"] = self.__dist
        res["idCol"] = self.__id_col
        res["timeCol"] = self.__time_col
        res["nsim"] = self.__nsim
        res["clusteredSE"] = self.__clustered_se
        res["haltonBase"] = self.__halton_base
        res["haltonBurnin"] = self.__halton_burnin
        res["haltonUi0Base"] = self.__halton_ui0_base
        res["scrambledHalton"] = self.__halton_scrambled
        res["shuffledHalton"] = self.__halton_shuffled
        res["confInt"] = self.__conf_int
        return PySfaCppResult(**res)
    
    @property
    def modelDataObj(self):
        return self.__mdl_data_obj