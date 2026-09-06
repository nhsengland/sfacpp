import warnings
from typing import Optional
import pandas as pd
import numpy as np
from .._base.base import PySfaCppBase

try:
    import pysfacpp_internal  # ty:ignore[unresolved-import]
except ImportError:
    pysfacpp_internal = None  # type: ignore[assignment]


class PySfaCppResult(PySfaCppBase):

    def __init__(self, **kwargs) -> None:
        self.__model: str = kwargs.pop("model", "unknown")
        self.__dist: str = kwargs.pop("dist", "unknown")
        self.__params: Optional[np.ndarray] = kwargs.pop("par", None)
        self.__terms: Optional[list[str]] = kwargs.pop("vars", [])
        self.__vcov: Optional[pd.DataFrame] = kwargs.pop("vcov", None)
        self.logLike: Optional[float] = kwargs.pop("logLikelihood", None)
        self.__hessian: Optional[np.ndarray] = kwargs.pop("hessian", None)
        self.__grad: Optional[np.ndarray] = kwargs.pop("gradient", None)
        self.__jacobian: Optional[np.ndarray] = kwargs.pop("jacobian", None)
        self.__model_summary: Optional[np.ndarray] = kwargs.pop("modelSummary", None)
        self.__dof: Optional[float] = kwargs.pop("degreesFreedom", None)
        self.__nobs: Optional[int] = kwargs.pop("nobs", -1)
        self.__nfirms: Optional[int] = kwargs.pop("nfirm", -1)
        self.__maxT: Optional[int] = kwargs.pop("maxT", -1)
        self.__minT: Optional[int] = kwargs.pop("minT", -1)
        self.__marginalEffects: Optional[pd.DataFrame] = kwargs.pop("marginalEffects", None)
        self.__meffCILwr: Optional[np.ndarray] = kwargs.pop("marginalEffectsLwrCI", None)
        self.__meffCIUpr: Optional[np.ndarray] = kwargs.pop("marginalEffectsUprCI", None)
        # self.origData: Optional[pd.DataFrame] = kwargs.pop("origData", None)
        self.__modelData: Optional[pd.DataFrame] = kwargs.pop("modelData", None)
        self.__reorderingIdx: Optional[np.ndarray] = kwargs.pop("reorderingIdx", None)
        # self.modelDataObj: EHInternalModelData = kwargs.pop("modelDataObj", None)
        self.__omit: Optional[np.ndarray] = kwargs.pop("omit", None)
        self.formula: Optional[str] = kwargs.pop("formula", None)
        self.form_zmuit: Optional[str] = kwargs.pop("form_zmuit", None)
        self.form_zuit: Optional[str] = kwargs.pop("form_zuit", None)
        self.form_zvit: Optional[str] = kwargs.pop("form_zvit", None)
        self.form_zvi0: Optional[str] = kwargs.pop("form_zvi0", None)
        self.form_zui0: Optional[str] = kwargs.pop("form_zui0", None)
        self.__clustered_se: bool = kwargs.pop("clusteredSE", False)
        self.__transient_efficiency: Optional[pd.DataFrame] = kwargs.pop("efficiencyTransient", None)
        self.__persistent_efficiency: Optional[pd.DataFrame] = kwargs.pop("efficiencyPersistent", None)
        self.__modelMatrixCols: Optional[list[str]] = kwargs.pop("modelMatrixCols", None)
        self.__id_col: Optional[str] = kwargs.pop("idCol", None)
        self.__time_col: Optional[str] = kwargs.pop("timeCol", None)
        self.__sigmas: Optional[dict] = kwargs.pop("sigmas", {})
        self.nparam: Optional[int] = kwargs.pop("nparam", None)
        self.__nsim: int = kwargs.pop("nsim", -1)
        self.__gnorm: float = kwargs.pop("gnorm", -1.0)
        self.__halton_base: int = kwargs.pop("haltonBase", -1)
        self.__halton_burnin: int = kwargs.pop("haltonBurnin", -1)
        self.__halton_ui0_base: int = kwargs.pop("haltonUi0Base", -1)
        self.__halton_scrambled: bool = kwargs.pop("scrambledHalton", False)
        self.__halton_shuffled: bool = kwargs.pop("shuffledHalton", False)
        self.__conf_int: float = kwargs.pop("confInt", -1.0)
    
    # conduct a log-likelihood ratio test, compared do another model result
    def lrtest(self, obj, decimals: float = 5.0) -> dict:
        self._PySfaCppBase__setup_logger()  # ty:ignore[unresolved-attribute]
        if (not hasattr(obj, "nparam")) or (not hasattr(obj, "logLike")):
            raise ValueError(f"Got object {type(obj)}, but it doesnt have the required attributes :(")
        if (obj.nparam is None) or (obj.logLike is None):
            raise ValueError("for argument 'nparam' or 'logLike' were None")
        if (self.nparam is None) or (self.logLike is None):
            raise ValueError("object did not have 'nparam' or 'logLike'")
        try:
            test: dict = pysfacpp_internal._lrtest(self.logLike, obj.logLike, self.nparam, obj.nparam, decimals)  # ty:ignore[unresolved-reference]
        except Exception as e:
            raise e
        return test

    def __str__(
        self,
        console_width: Optional[int] = None,
        decimal_places: Optional[int] = None
    ) -> str:
        """

        """
        if console_width is None:
            console_width: int = self._PySfaCppBase__console_width()  # ty:ignore[unresolved-attribute]
        if decimal_places is None:
            decimal_places: int = self._PySfaCppBase__default_decimals()  # ty:ignore[unresolved-attribute]
        if self.__model_summary is None:
            raise ValueError("Unexpected model summary to be none")
        # reconnect the logger
        self._PySfaCppBase__setup_logger()  # ty:ignore[unresolved-attribute]
        
        sigma_uit: float | None = None
        sigma_vit: float | None = None
        sigma_vi0: float | None = None
        sigma_lambda: float | None = None
        sigma_ui0: float | None = None
        sigma_lambda_0: float | None = None
        sigma_biglambda: float | None = None
        if self.__sigmas is not None:
            sigma_uit: float | None = self.__sigmas["sigma_uit"] if "sigma_uit" in self.__sigmas.keys() else None
            sigma_vit: float | None = self.__sigmas["sigma_vit"] if "sigma_vit" in self.__sigmas.keys() else None
            sigma_vi0: float | None = self.__sigmas["sigma_vi0"] if "sigma_vi0" in self.__sigmas.keys() else None
            sigma_lambda: float | None = self.__sigmas["lambda"] if "lambda" in self.__sigmas.keys() else None
            sigma_ui0: float | None = self.__sigmas["sigma_ui0"] if "sigma_ui0" in self.__sigmas.keys() else None
            sigma_lambda_0: float | None = self.__sigmas["lambda_0"] if "lambda_0" in self.__sigmas.keys() else None
            sigma_biglambda: float | None = self.__sigmas["BigLambda"] if "BigLambda" in self.__sigmas.keys() else None

        pysfacpp_internal._print_model_summary(  # ty:ignore[unresolved-reference]
            self.__model, # model type
            self.__dist, # distribution of inefficiency
            self.__nobs, # number of observations
            self.__nfirms, # number of firms
            self.__maxT, # maximum number of panels
            self.__minT, # minimum number of panels
            self.__model_summary, # the model summary
            self.__terms, # term names
            # sigma params
            # s_uit
            sigma_uit,
            # s_vit
            sigma_vit,
            # s_vi0
            sigma_vi0,
            # lambda
            sigma_lambda,
            # s_ui0
            sigma_ui0,
            # lambda_0
            sigma_lambda_0,
            # BigLambda
            sigma_biglambda,
            self.__nsim, # number of simulations/halton draws
            self.logLike, # the log-likelihood score
            self.__gnorm, # gradient norm
            self.__clustered_se, # whether or not clustered standard errors
            # halton parameters
            self.__halton_base, # halton base prime
            self.__halton_burnin, # halton burnin
            self.__halton_ui0_base, # halton base prime for ui0
            self.__halton_scrambled, # whether used a scrambled halton draw
            self.__halton_shuffled, # whether used a shuffled halton draw
            self.__conf_int, # confidence lvl used for the model summary
            decimal_places,
            console_width,
            self.__id_col,
            self.__time_col
        )
        return ""

    @property
    def params(self):
        return self.__params

    @property
    def marginal_effects(self) -> Optional[pd.DataFrame]:
        return self.__marginalEffects

    @property
    def model_summary(self) -> Optional[pd.DataFrame]:
        if self.__model_summary is None:
            return None
        df_ms = self.__model_summary
        df_ms = df_ms.reset_index(names="variable")
        if self.__sigmas is not None:
            # setup sigms
            df_sigmas: pd.DataFrame = pd.DataFrame(
                data={k: [v] for k, v in self.__sigmas.items()}
            ).melt(
                var_name="variable",
                value_name="estimate"
            )
            return pd.concat(
                objs=[df_ms, df_sigmas],
                ignore_index=True,
                axis=0,
                join="outer"
            ) 
        return df_ms[~df_ms["estimate"].isna()]
    
    @property
    def sigmas(self) -> Optional[pd.DataFrame]:
        if self.__sigmas is not None:
            return pd.DataFrame(data={k: [v] for k, v in self.__sigmas.items()})
        return None

    @property
    def model_data(self) -> pd.DataFrame:
        """
        Return the model data as a pandas dataframe
        """
        if self.__modelData is None:
            raise ValueError("modelData attribute not found")
        df: pd.DataFrame = pd.DataFrame(self.__modelData)
        if self.__modelMatrixCols is None:
            return df
        # rename the columns
        df: pd.DataFrame = df.rename(
            mapper={
                df.columns[i]: c
                for i, c in enumerate(self.__modelMatrixCols)
            },
            axis=1
        )
        return df

    @property
    def efficiencies(self) -> Optional[pd.DataFrame]:
        """
        Return efficiency scores"
        """
        if self.__model == "unknown":
            return None
        # for tre, tfe, return transient inefficiency
        if self.__model in ["tre", "tfe"]:
            return self.__transient_efficiency
        # if gtre, tre, tfe, should have the id & time cols
        if self.__model in ["gtre"]:
            # for gtre - combine transient and persisent inefficiency
            if (self.__id_col is None) or (self.__time_col is None):
                # raise a warning that the id & time columns are missing
                warnings.warn(message="id or time col are missing")
                return
            # check both attributes required exist
            if self.__transient_efficiency is None:
                warnings.warn(message="missing transient efficiency table")
                return
            if self.__persistent_efficiency is None:
                warnings.warn(message="missing persistent efficiency table")
                return
            # combine transient and persistent inefficiency
            df_ret: pd.DataFrame = self.__transient_efficiency.merge(
                right=self.__persistent_efficiency,
                on=[self.__id_col, self.__time_col],
                how="left"
            ).reset_index()
            # calculate overall efficiency
            df_ret["overall_efficiency"] = df_ret["transient_efficiency"] * df_ret["persistent_efficiency"]
            return df_ret