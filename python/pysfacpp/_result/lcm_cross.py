from typing import Optional
import pandas as pd
import numpy as np
from .._base.base import PySfaCppBase

try:
    import pysfacpp_internal  # ty:ignore[unresolved-import]
except ImportError:
    pysfacpp_internal = None  # type: ignore[assignment]


class PySfaCppLcmCrossResult(PySfaCppBase):

    def __init__(self, **kwargs) -> None:
        self.__params: Optional[np.ndarray] = kwargs.pop("par", None)
        self.__terms: Optional[list] = kwargs.pop("vars", None)
        self.logLike: Optional[float] = kwargs.pop("logLikelihood", None)
        self.__vcov: Optional[np.ndarray] = kwargs.pop("vcov", None)
        self.__hessian: Optional[np.ndarray] = kwargs.pop("hessian", None)
        self.__gradient: Optional[np.ndarray] = kwargs.pop("gradient", None)
        self.__jacobian: Optional[np.ndarray] = kwargs.pop("jacobian", None)
        self.__nobs: int = kwargs.pop("nobs", -1)
        self.__nClasses: int = kwargs.pop("nClasses", -1)
        self.nparam: Optional[int] = kwargs.pop("nparam", None)
        self.__gnorm: float = kwargs.pop("gnorm", -1.0)
        self.__converged: bool = kwargs.pop("converged", False)
        self.__posteriors: Optional[np.ndarray] = kwargs.pop("posteriors", None)

    @property
    def converged(self) -> bool:
        return self.__converged

    @property
    def params(self) -> Optional[np.ndarray]:
        return self.__params

    @property
    def n_classes(self) -> int:
        return self.__nClasses

    @property
    def nobs(self) -> int:
        return self.__nobs

    @property
    def posteriors(self) -> Optional[pd.DataFrame]:
        if self.__posteriors is None:
            return None
        cols: list[str] = [f"Class_{c}" for c in range(self.__nClasses)]
        return pd.DataFrame(data=self.__posteriors, columns=cols)

    @property
    def vcov(self) -> Optional[np.ndarray]:
        return self.__vcov

    @property
    def model_summary(self) -> Optional[pd.DataFrame]:
        if self.__params is None or self.__vcov is None:
            return None
        se = np.sqrt(np.diag(self.__vcov))
        z = self.__params / se
        terms = self.__terms if self.__terms else [f"par_{i}" for i in range(len(self.__params))]
        return pd.DataFrame({
            "variable": terms,
            "estimate": self.__params,
            "std_error": se,
            "z_value": z,
        })

    def lrtest(self, obj, decimals: float = 5.0) -> dict:
        self._PySfaCppBase__setup_logger()  # ty:ignore[unresolved-attribute]
        if (not hasattr(obj, "nparam")) or (not hasattr(obj, "logLike")):
            raise ValueError(f"Got object {type(obj)}, but it doesnt have the required attributes")
        if (obj.nparam is None) or (obj.logLike is None):
            raise ValueError("for argument 'nparam' or 'logLike' were None")
        if (self.nparam is None) or (self.logLike is None):
            raise ValueError("object did not have 'nparam' or 'logLike'")
        test: dict = pysfacpp_internal._lrtest(self.logLike, obj.logLike, self.nparam, obj.nparam, decimals)
        return test
