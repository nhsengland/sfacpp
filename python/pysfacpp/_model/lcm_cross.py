import math
from typing import Optional, Any, Union
import pandas as pd
import numpy as np
from .._base.base import PySfaCppBase
from .._data.model_data import EHInternalModelData
from .._result.lcm_cross import PySfaCppLcmCrossResult

try:
    import pysfacpp_internal  # ty:ignore[unresolved-import]
    _DEFAULT_NTHREADS = math.floor(pysfacpp_internal._get_available_threads() * 0.8)
except ImportError:
    pysfacpp_internal = None  # type: ignore[assignment]
    _DEFAULT_NTHREADS = 1


class PySfaCppLcmCross(PySfaCppBase):
    """
    Latent Class cross-sectional SFA model specification and estimation.

    Implements the finite mixture (latent class) cross-sectional stochastic
    frontier model with half-normal inefficiency distribution.
    """

    def __init__(
        self,
        form_x: str,
        data: pd.DataFrame,
        n_classes: int = 2,
        form_seg: Optional[str] = None,
        form_zuit: Optional[str] = None,
        form_zvit: Optional[str] = None,
        start: Optional[Union[list[float], np.ndarray]] = None,
        prod: int = 1,
        dist: str = "hnorm",
        optim_opts: Optional[dict[str, Any]] = None,
        optim_method: str = "tr",
        hessian_calc: str = "bhhh",
        conf_int: float = 0.95,
        seed: int = 1234,
        print_level: int = 2,
        form_has_x_intercept: bool = True,
        nthreads: int = _DEFAULT_NTHREADS,
        should_copy_from_numpy: bool = False,
        display_decimal_places: int = 5,
        display_console_width: Optional[int] = None,
    ) -> None:
        if n_classes < 2:
            raise ValueError("'n_classes' must be >= 2 for latent class models")
        if optim_method not in ["tr", "hybrid_bfgs_tr", "hybrid_lbfgs_tr", "bfgs_tr", "lbfgs_tr"]:
            raise ValueError(
                "'optim_method' must be one of 'tr', 'hybrid_bfgs_tr', 'hybrid_lbfgs_tr', 'bfgs_tr', 'lbfgs_tr'"
            )
        if dist not in ["hnorm"]:
            raise ValueError("'dist' must be 'hnorm'")
        if not ((prod == 1) or (prod == -1)):
            raise ValueError("'prod' should be 1 for production, -1 for cost")
        if (conf_int < 0.0) or (conf_int > 1.0):
            raise ValueError("'conf_int' must be between 0 and 1")
        if nthreads < 1:
            raise ValueError("Must have at least 1 thread")

        self.__n_classes: int = n_classes
        self.__optim_method: str = optim_method
        self.__optim_method_lib: str = "dlib"
        self.__dist: str = dist
        self.__prod: int = prod
        self.__conf_int: float = conf_int
        self.__seed: int = seed
        self.__print_level: int = print_level
        self.__nthreads: int = nthreads
        self.__hessian_calc: str = hessian_calc
        self.__should_copy_from_numpy: bool = should_copy_from_numpy
        self.__form_x: str = form_x
        self.__form_seg: Optional[str] = form_seg
        self.__form_zuit: Optional[str] = form_zuit
        self.__form_zvit: Optional[str] = form_zvit
        self.__optim_opts: Optional[dict[str, Any]] = optim_opts

        self.__mdl_data_obj = EHInternalModelData(
            data=data,
            fIn=form_x,
            idVar=None,
            timeVar=None,
            fZmuit=None,
            fZuit=form_zuit,
            fZui0=None,
            fZvit=form_zvit,
            fZvi0=None,
            fInHasIntercept=form_has_x_intercept,
        )

        self.__start_vals_mtrx: Optional[np.ndarray] = None
        if start is not None:
            self.__start_vals_mtrx = np.asfortranarray(np.array(start).reshape(-1, 1))
        self.__display_console_width: int = (
            self._PySfaCppBase__console_width() if display_console_width is None else display_console_width
        )
        self.__display_decimal_places: int = display_decimal_places

        self.__seg_matrix: np.ndarray = self._build_seg_matrix(data, form_seg)

    def _build_seg_matrix(self, data: pd.DataFrame, form_seg: Optional[str]) -> np.ndarray:
        """Build segmentation variable matrix. Defaults to intercept-only."""
        import patsy

        if form_seg is None:
            n_obs = self.__mdl_data_obj.model_matrix.shape[0]
            return np.asfortranarray(np.ones((n_obs, 1)))
        rhs = form_seg.split("~")[1] if "~" in form_seg else form_seg
        complete_idx = ~np.isin(np.arange(len(data)), self.__mdl_data_obj.omit)
        dt_work = data.loc[complete_idx].reset_index(drop=True)
        mat = patsy.dmatrix(rhs, dt_work, return_type="matrix")
        return np.asfortranarray(np.array(mat))

    def fit(self) -> PySfaCppLcmCrossResult:
        """Fit the cross-sectional latent class model."""
        self._PySfaCppBase__setup_logger()  # ty:ignore[unresolved-attribute]

        try:
            res: dict[str, Any] = pysfacpp_internal._pysfacpp_internal_lcm_cross(
                self.__mdl_data_obj.y,
                self.__mdl_data_obj.X,
                self.__seg_matrix,
                self.__mdl_data_obj.Zuit,
                self.__mdl_data_obj.Zvit,
                self.__n_classes,
                self.__start_vals_mtrx,
                self.__prod,
                self.__dist,
                self.__optim_method,
                self.__optim_method_lib,
                self.__optim_opts,
                self.__mdl_data_obj.namesX,
                self.__mdl_data_obj.namesZuit,
                self.__mdl_data_obj.namesZvit,
                None,  # segmentation term names
                self.__hessian_calc,
                self.__seed,
                self.__conf_int,
                self.__print_level,
                self.__nthreads,
                self.__should_copy_from_numpy,
                self.__display_decimal_places,
                self.__display_console_width,
            )
        except Exception as e:
            raise e

        return PySfaCppLcmCrossResult(**res)

    @property
    def n_classes(self) -> int:
        return self.__n_classes

    @property
    def modelDataObj(self) -> EHInternalModelData:
        return self.__mdl_data_obj
