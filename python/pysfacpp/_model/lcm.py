import math
import shutil
from typing import Optional, Any, Union
import pandas as pd
import numpy as np
from .._base.base import PySfaCppBase
from .._data.model_data import EHInternalModelData
from .._result.lcm import PySfaCppLcmResult

try:
    import pysfacpp_internal  # ty:ignore[unresolved-import]
    _DEFAULT_NTHREADS = math.floor(pysfacpp_internal._get_available_threads() * 0.8)
except ImportError:
    pysfacpp_internal = None  # type: ignore[assignment]
    _DEFAULT_NTHREADS = 1


class PySfaCppLcm(PySfaCppBase):
    """
    Latent Class TRE model specification and estimation.

    Wraps the C++ LC-TRE backend with a formula-based interface analogous
    to PySfaCpp but supporting multiple latent classes with segmentation.
    """

    def __init__(
        self,
        form_x: str,
        data: pd.DataFrame,
        n_classes: int = 2,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        form_seg: Optional[str] = None,
        form_zmuit: Optional[str] = None,
        form_zuit: Optional[str] = None,
        form_zvit: Optional[str] = None,
        form_zvi0: Optional[str] = None,
        start: Optional[Union[list[float], np.ndarray]] = None,
        prod: int = 1,
        dist: str = "hnorm",
        nsim: int = 500,
        optim_opts: Optional[dict[str, Any]] = None,
        optim_method: str = "em_ghq",
        hessian_calc: str = "analytical",
        conf_int: float = 0.95,
        seed: int = 1234,
        print_level: int = 2,
        form_has_x_intercept: bool = True,
        clustered_se: bool = True,
        nthreads: int = _DEFAULT_NTHREADS,
        halton_base: int = 2,
        halton_burnin: int = 1000,
        halton_ui0_base: int = 3,
        halton_scrambled: bool = True,
        halton_shuffled: bool = False,
        should_copy_from_numpy: bool = False,
        display_decimal_places: int = 5,
        display_console_width: Optional[int] = None,
        efficiency_method: str = "colombi",
    ) -> None:
        if n_classes < 1:
            raise ValueError("'n_classes' must be >= 1 for latent class models")
        _valid_optim_methods = ["tr", "hybrid_bfgs_tr", "hybrid_lbfgs_tr",
                                "bfgs_tr", "lbfgs_tr", "em_ghq"]
        if optim_method not in _valid_optim_methods:
            raise ValueError(
                f"'optim_method' must be one of {_valid_optim_methods}"
            )
        if dist not in ["hnorm"]:
            raise ValueError("'dist' must be one of 'hnorm'")
        if not ((prod == 1) or (prod == -1)):
            raise ValueError("'prod' should be 1 for production, -1 for cost")
        if (conf_int < 0.0) or (conf_int > 1.0):
            raise ValueError("'conf_int' must be between 0 and 1")
        if nthreads < 1:
            raise ValueError("Must have at least 1 thread")
        if id_col is None:
            raise ValueError("'id_col' must be specified for LC-TRE models")
        if id_col not in data.columns:
            raise ValueError(f"'id_col' {id_col} not found in data")
        if time_col is None:
            raise ValueError("'time_col' must be specified for LC-TRE models")
        if time_col not in data.columns:
            raise ValueError(f"'time_col' {time_col} not found in data")

        self.__n_classes: int = n_classes
        self.__optim_method: str = optim_method
        self.__optim_method_lib: str = "em" if optim_method == "em_ghq" else "dlib"
        self.__dist: str = dist
        self.__prod: int = prod
        self.__conf_int: float = conf_int
        self.__nsim: int = nsim
        self.__seed: int = seed
        self.__print_level: int = print_level
        self.__clustered_se: bool = clustered_se
        self.__nthreads: int = nthreads
        self.__hessian_calc: str = hessian_calc
        self.__halton_base: int = halton_base
        self.__halton_burnin: int = halton_burnin
        self.__halton_ui0_base: int = halton_ui0_base
        self.__halton_scrambled: bool = halton_scrambled
        self.__halton_shuffled: bool = halton_shuffled
        self.__should_copy_from_numpy: bool = should_copy_from_numpy
        self.__efficiency_method: str = efficiency_method
        self.__id_col: str = id_col
        self.__time_col: str = time_col
        self.__form_x: str = form_x
        self.__form_seg: Optional[str] = form_seg
        self.__form_zmuit: Optional[str] = form_zmuit
        self.__form_zuit: Optional[str] = form_zuit
        self.__form_zvit: Optional[str] = form_zvit
        self.__form_zvi0: Optional[str] = form_zvi0
        self.__optim_opts: Optional[dict[str, Any]] = optim_opts
        # instantiate the internal model data object
        self.__mdl_data_obj = EHInternalModelData(
            data=data,
            fIn=form_x,
            idVar=id_col,
            timeVar=time_col,
            fZmuit=form_zmuit,
            fZuit=form_zuit,
            fZui0=None,
            fZvit=form_zvit,
            fZvi0=form_zvi0,
            fInHasIntercept=form_has_x_intercept,
        )
        # setup starting values
        self.__start_vals_mtrx: Optional[np.ndarray] = None
        if start is not None:
            self.__start_vals_mtrx: np.ndarray = np.asfortranarray(a=np.array(object=start).reshape(-1, 1))
        # display settings
        self.__display_console_width: int = (
            self._PySfaCppBase__console_width()  # ty:ignore[unresolved-attribute]
            if display_console_width is None else display_console_width
        )
        self.__display_decimal_places: int = display_decimal_places
        # build the segmentaiton matrix
        self.__seg_matrix: np.ndarray = self._build_seg_matrix(data, form_seg)

    def _build_seg_matrix(self, data: pd.DataFrame, form_seg: Optional[str]) -> np.ndarray:
        """Build segmentation variable matrix. Defaults to intercept-only."""
        import patsy

        if form_seg is None:
            n_obs: int = self.__mdl_data_obj.model_matrix.shape[0]
            return np.asfortranarray(a=np.ones(shape=(n_obs, 1)))
        rhs: str = form_seg.split(sep="~")[1] if "~" in form_seg else form_seg
        complete_idx: np.ndarray = ~np.isin(element=np.arange(len(data)), test_elements=self.__mdl_data_obj.omit)
        dt_work: pd.DataFrame = data.loc[complete_idx].sort_values(by=[self.__id_col, self.__time_col])
        mat: np.ndarray = patsy.dmatrix(rhs, dt_work, return_type="matrix")  # ty:ignore[unresolved-attribute]
        return np.asfortranarray(a=np.array(object=mat))

    def fit(self) -> PySfaCppLcmResult:
        """Fit the LC-TRE model."""
        self._PySfaCppBase__setup_logger()  # ty:ignore[unresolved-attribute]
        # call the pybind interface
        try:
            res: dict[str, Any] = pysfacpp_internal._pysfacpp_internal_lcm(  # ty:ignore[unresolved-attribute]
                self.__mdl_data_obj.y,
                self.__mdl_data_obj.X,
                self.__seg_matrix,
                self.__mdl_data_obj.Zmuit if self.__form_zmuit else None,
                self.__mdl_data_obj.Zuit,
                self.__mdl_data_obj.Zvit,
                self.__mdl_data_obj.Zvi0,
                self.__mdl_data_obj.idVec,
                self.__mdl_data_obj.timeVec,
                self.__n_classes,
                self.__start_vals_mtrx,
                self.__prod,
                "lctre",
                self.__dist,
                self.__optim_method,
                self.__optim_method_lib,
                self.__optim_opts,
                self.__mdl_data_obj.namesX,
                self.__mdl_data_obj.namesZmuit if self.__form_zmuit else None,
                self.__mdl_data_obj.namesZuit,
                self.__mdl_data_obj.namesZvit,
                self.__mdl_data_obj.namesZvi0,
                None,  # segmentation term names
                self.__nsim,
                self.__hessian_calc,
                self.__seed,
                self.__conf_int,
                self.__print_level,
                self.__clustered_se,
                self.__nthreads,
                self.__halton_base,
                self.__halton_burnin,
                self.__halton_ui0_base,
                self.__halton_scrambled,
                self.__halton_shuffled,
                self.__should_copy_from_numpy,
                self.__display_decimal_places,
                self.__display_console_width,
                self.__id_col,
                self.__time_col,
                self.__efficiency_method,
            )
        except Exception as e:
            raise e
        # inject display params not returned by C++
        res["confInt"] = self.__conf_int
        res["haltonBase"] = self.__halton_base
        res["haltonBurnin"] = self.__halton_burnin
        res["haltonUi0Base"] = self.__halton_ui0_base
        res["scrambledHalton"] = self.__halton_scrambled
        res["shuffledHalton"] = self.__halton_shuffled
        res["useGhq"] = self.__optim_method == "em_ghq"
        res["nsim"] = self.__nsim
        res["idCol"] = self.__id_col
        res["timeCol"] = self.__time_col
        # compute maxT / minT from the id vector
        id_vec = self.__mdl_data_obj.idVec
        if id_vec is not None:
            counts: np.ndarray = np.bincount(id_vec)[1:]  # ids are 1-based
            res["maxT"] = int(counts.max())
            res["minT"] = int(counts.min())

        result_obj = PySfaCppLcmResult(**res)
        if self.__print_level > 0:
            print(result_obj)
        return result_obj

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
        override_start_val_check: bool = False,
        parallel: bool = True,
    ) -> Optional[dict]:
        """
        Run random perturbations of initial starting values to check for global convergence.

        optim_method: Optional[str]
            Optimization method for searches (does not overwrite the value used for .fit()).
        """
        _valid_search_methods = ["tr", "hybrid_bfgs_tr", "hybrid_lbfgs_tr", "bfgs_tr", "lbfgs_tr", "em_ghq"]
        if optim_method is None:
            optim_method: str = self.__optim_method
        if optim_method not in _valid_search_methods:
            raise ValueError(f"'optim_method' for searches must be one of {_valid_search_methods}")
        if maximum_attempts < nsearches:
            raise ValueError("'maximum_attempts' should be bigger than 'nsearches'")
        self._PySfaCppBase__setup_logger()  # ty:ignore[unresolved-attribute]
        start_ft = None
        if start is not None:
            start_ft: np.ndarray = np.asfortranarray(a=np.array(object=start).reshape(-1, 1))
        display_console_width_val: int = (
            self._PySfaCppBase__console_width()  # ty:ignore[unresolved-attribute]
            if display_console_width is None else display_console_width
        )
        method_lib: str = "em" if optim_method == "em_ghq" else "dlib"
        try:
            res: dict = pysfacpp_internal._pysfacpp_searches_lcm(  # ty:ignore[unresolved-attribute]
                self.__mdl_data_obj.y,
                self.__mdl_data_obj.X,
                self.__seg_matrix,
                self.__mdl_data_obj.Zmuit if self.__form_zmuit else None,
                self.__mdl_data_obj.Zuit,
                self.__mdl_data_obj.Zvit,
                self.__mdl_data_obj.Zvi0,
                self.__mdl_data_obj.idVec,
                self.__mdl_data_obj.timeVec,
                start_ft,
                self.__n_classes,
                nsearches,
                maxit,
                slength_frontier,
                slength_sigmas,
                max_attempt_start_vals,
                self.__prod,
                self.__dist,
                optim_method,
                method_lib,
                self.__nsim,
                seed,
                print_level,
                self.__nthreads,
                self.__halton_base,
                self.__halton_burnin,
                self.__halton_ui0_base,
                self.__halton_scrambled,
                self.__halton_shuffled,
                self.__should_copy_from_numpy,
                display_console_width_val,
                digits,
                maximum_attempts,
                override_start_val_check,
                parallel,
            )
            return res
        except Exception as e:
            raise e

    @property
    def n_classes(self) -> int:
        return self.__n_classes

    @property
    def modelDataObj(self) -> EHInternalModelData:
        return self.__mdl_data_obj
