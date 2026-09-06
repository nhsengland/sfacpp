from typing import Optional
import scipy.stats
import pandas as pd
import numpy as np
from .._base.base import PySfaCppBase

try:
    import pysfacpp_internal  # ty:ignore[unresolved-import]
except ImportError:
    pysfacpp_internal = None  # type: ignore[assignment]

class PySfaCppLcmResult(PySfaCppBase):

    def __init__(self, **kwargs) -> None:
        self.__params: Optional[np.ndarray] = kwargs.pop("par", None)
        self.__terms: Optional[list] = kwargs.pop("vars", None)
        self.logLike: Optional[float] = kwargs.pop("logLikelihood", None)
        self.__vcov: Optional[np.ndarray] = kwargs.pop("vcov", None)
        self.__hessian: Optional[np.ndarray] = kwargs.pop("hessian", None)
        self.__gradient: Optional[np.ndarray] = kwargs.pop("gradient", None)
        self.__jacobian: Optional[np.ndarray] = kwargs.pop("jacobian", None)
        self.__nobs: int = kwargs.pop("nobs", -1)
        self.__nfirms: int = kwargs.pop("nfirm", -1)
        self.__nClasses: int = kwargs.pop("nClasses", -1)
        self.nparam: Optional[int] = kwargs.pop("nparam", None)
        self.__gnorm: float = kwargs.pop("gnorm", -1.0)
        self.__clustered_se: bool = kwargs.pop("clusteredSE", False)
        self.__converged: bool = kwargs.pop("converged", False)
        self.__posteriors: Optional[np.ndarray] = kwargs.pop("posteriors", None)
        self.__eff_transient: Optional[np.ndarray] = kwargs.pop("efficiencyTransient", None)
        self.__eff_transient_pc: Optional[np.ndarray] = kwargs.pop("efficiencyTransientPerClass", None)
        self.__eff_persistent: Optional[np.ndarray] = kwargs.pop("efficiencyPersistent", None)
        self.__eff_persistent_pc: Optional[np.ndarray] = kwargs.pop("efficiencyPersistentPerClass", None)
        self.__meff: Optional[np.ndarray] = kwargs.pop("marginalEffects", None)
        self.__meff_names: Optional[list] = kwargs.pop("marginalEffectsNames", None)
        self.__meff_per_class: Optional[list] = kwargs.pop("marginalEffectsPerClass", None)
        self.__sigmas_per_class: Optional[list] = kwargs.pop("sigmasPerClass", None)
        # display params — injected by PySfaCppLcm.fit()
        self.__conf_int: float = kwargs.pop("confInt", 0.95)
        self.__halton_base: int = kwargs.pop("haltonBase", 2)
        self.__halton_burnin: int = kwargs.pop("haltonBurnin", 1000)
        self.__halton_ui0_base: int = kwargs.pop("haltonUi0Base", 3)
        self.__halton_scrambled: bool = kwargs.pop("scrambledHalton", True)
        self.__halton_shuffled: bool = kwargs.pop("shuffledHalton", False)
        self.__use_ghq: bool = kwargs.pop("useGhq", False)
        self.__nsim: int = kwargs.pop("nsim", -1)
        self.__maxT: int = kwargs.pop("maxT", -1)
        self.__minT: int = kwargs.pop("minT", -1)
        self.__id_col: Optional[str] = kwargs.pop("idCol", None)
        self.__time_col: Optional[str] = kwargs.pop("timeCol", None)

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
    def posteriors(self) -> Optional[pd.DataFrame]:
        if self.__posteriors is None:
            return None
        cols: list[str] = [f"Class_{c}" for c in range(self.__nClasses)]
        return pd.DataFrame(data=self.__posteriors, columns=cols)

    @property
    def efficiencies(self) -> Optional[pd.DataFrame]:
        if self.__eff_transient is None:
            return None
        df = pd.DataFrame(data={"transient_efficiency": self.__eff_transient.flatten()})
        if self.__eff_persistent is not None:
            df["persistent_efficiency"] = self.__eff_persistent.flatten()
            df["overall_efficiency"] = df["transient_efficiency"] * df["persistent_efficiency"]
        if self.__posteriors is not None:
            df["assigned_class"] = np.argmax(self.__posteriors[
                np.repeat(np.arange(self.__posteriors.shape[0]),
                          self.__eff_transient.shape[0] // self.__posteriors.shape[0])
            ] if self.__eff_transient.shape[0] != self.__posteriors.shape[0]
              else self.__posteriors, axis=1)
        return df

    @property
    def efficiencies_per_class(self) -> Optional[pd.DataFrame]:
        if self.__eff_transient_pc is None:
            return None
        cols: list[str] = [f"Class_{c}" for c in range(self.__nClasses)]
        return pd.DataFrame(data=self.__eff_transient_pc, columns=cols)

    @property
    def marginal_effects(self) -> Optional[pd.DataFrame]:
        if self.__meff is None:
            return None
        cols = self.__meff_names if self.__meff_names else [f"ME_{k}" for k in range(self.__meff.shape[1])]
        return pd.DataFrame(data=self.__meff, columns=cols)

    @property
    def marginal_effects_per_class(self) -> Optional[list]:
        if self.__meff_per_class is None:
            return None
        cols = self.__meff_names if self.__meff_names else None
        result = []
        for c, arr in enumerate(self.__meff_per_class):
            df_cols = cols if cols else [f"ME_{k}" for k in range(arr.shape[1])]
            result.append(pd.DataFrame(data=arr, columns=df_cols))
        return result

    @property
    def sigmas_per_class(self) -> Optional[pd.DataFrame]:
        if self.__sigmas_per_class is None:
            return None
        rows: list[dict[str, int]] = []
        for c, sp in enumerate(self.__sigmas_per_class):
            row: dict[str, int] = {"class": c}
            row.update(sp)
            rows.append(row)
        return pd.DataFrame(data=rows)

    @property
    def aic(self) -> Optional[float]:
        if self.logLike is None or self.nparam is None:
            return None
        return -2.0 * self.logLike + 2.0 * self.nparam

    @property
    def bic(self) -> Optional[float]:
        if self.logLike is None or self.nparam is None or self.__nobs <= 0:
            return None
        return -2.0 * self.logLike + self.nparam * np.log(self.__nobs)

    @property
    def hessian(self) -> Optional[np.ndarray]:
        return self.__hessian

    @property
    def vcov(self) -> Optional[np.ndarray]:
        return self.__vcov

    @property
    def model_summary(self) -> Optional[pd.DataFrame]:
        if self.__params is None or self.__vcov is None:
            return None
        par = self.__params.flatten()
        se = np.sqrt(np.maximum(np.diag(self.__vcov), 0.0))
        z = par / se
        p = 2.0 * (1.0 - scipy.stats.norm.cdf(np.abs(z)))
        alpha = 1.0 - self.__conf_int
        z_crit = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
        ci_lo = par - z_crit * se
        ci_hi = par + z_crit * se
        terms = self.__terms if self.__terms else [f"par_{i}" for i in range(len(par))]
        return pd.DataFrame({
            "variable": terms,
            "estimate": par,
            "std_error": se,
            "z_value": z,
            "p_value": p,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        })

    def __str__(
        self,
        console_width: Optional[int] = None,
        decimal_places: Optional[int] = None,
    ) -> str:
        if self.__params is None or self.__vcov is None or self.__terms is None:
            return f"<PySfaCppLcmResult: converged={self.__converged}, params={'available' if self.__params is not None else 'None'}>"
        if console_width is None:
            console_width = self._PySfaCppBase__console_width()  # ty:ignore[attr-defined]  # ty:ignore[unresolved-attribute]
        if decimal_places is None:
            decimal_places = self._PySfaCppBase__default_decimals()  # ty:ignore[attr-defined]  # ty:ignore[unresolved-attribute]
        self._PySfaCppBase__setup_logger()  # ty:ignore[attr-defined]  # ty:ignore[unresolved-attribute]

        par = self.__params.flatten()
        se = np.sqrt(np.maximum(np.diag(self.__vcov), 0.0))
        z = par / se
        p = 2.0 * (1.0 - scipy.stats.norm.cdf(np.abs(z)))
        alpha = 1.0 - self.__conf_int
        z_crit = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
        ci_lo = par - z_crit * se
        ci_hi = par + z_crit * se
        ms = np.asfortranarray(np.column_stack([par, se, z, p, ci_lo, ci_hi]))

        # build per-class sigma maps with display labels
        name_map = {
            "sigma_uit": "E(σ_uit)",
            "sigma_vit": "E(σ_vit)",
            "sigma_vi0": "E(σ_vi0)",
            "lambda":    "E(λ)",
        }
        sigmas_per_class: list[dict[str, float]] = []
        for c in range(self.__nClasses):
            extras: dict[str, float] = {}
            if self.__sigmas_per_class is not None and c < len(self.__sigmas_per_class):
                sp = self.__sigmas_per_class[c]
                for key, label_str in name_map.items():
                    if key in sp:
                        extras[label_str] = sp[key]
            sigmas_per_class.append(extras)

        pysfacpp_internal._print_lcm_output(
            self.__nobs,
            self.__nfirms,
            self.__maxT,
            self.__minT,
            ms,
            list(self.__terms),
            sigmas_per_class,
            self.__nClasses,
            self.__nsim,
            self.logLike if self.logLike is not None else float("nan"),
            self.__gnorm,
            self.__clustered_se,
            self.__halton_base,
            self.__halton_burnin,
            self.__halton_ui0_base,
            self.__halton_scrambled,
            self.__halton_shuffled,
            self.__use_ghq,
            self.__conf_int,
            decimal_places,
            console_width,
            self.__id_col,
            self.__time_col,
        )
        return ""

    def __repr__(self) -> str:
        return self.__str__()

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

