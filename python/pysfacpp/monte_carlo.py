"""
Monte Carlo simulation framework for SFA models.

Modular design:
- DGP layer: generate synthetic panel data from known parameters
- Estimation layer: thin wrapper around the C++ backend
- Metrics layer: bias, RMSE, coverage, and other MC diagnostics
- Orchestration layer: run replications, collect results, report
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum


# ─── DGP Configuration ───────────────────────────────────────────────────────


class InefficiencyDist(Enum):
    HALF_NORMAL = "hnorm"
    EXPONENTIAL = "exp"


@dataclass
class ClassSpec:
    """Specification for a single latent class in the DGP."""

    beta: np.ndarray
    ln_sigma_u: float
    ln_sigma_v: float
    ln_sigma_0: float

    @property
    def sigma_u(self) -> float:
        return np.exp(self.ln_sigma_u)

    @property
    def sigma_v(self) -> float:
        return np.exp(self.ln_sigma_v)

    @property
    def sigma_0(self) -> float:
        return np.exp(self.ln_sigma_0)

    @property
    def n_frontier_params(self) -> int:
        return len(self.beta) + 3  # beta + ln_sigma_u + ln_sigma_v + ln_sigma_0


@dataclass
class DGPConfig:
    """
    Configuration for a Latent Class TRE data generating process.

    Parameters
    ----------
    n_firms : int
        Number of firms (cross-sectional units).
    n_periods : int
        Number of time periods per firm.
    classes : list[ClassSpec]
        Per-class specifications. Length determines number of classes.
    delta : np.ndarray, optional
        Segmentation parameters for class membership (multinomial logit).
        Shape: (n_classes - 1,) for intercept-only segmentation.
        Default: zeros (equal class probabilities).
    n_seg : int
        Number of segmentation variables (including intercept). Default 1.
    prod_cost : int
        +1 for production frontier, -1 for cost frontier.
    dist : InefficiencyDist
        Distribution for transient inefficiency.
    seed : int
        Base random seed.
    """

    n_firms: int = 100
    n_periods: int = 5
    classes: list[ClassSpec] = field(default_factory=list)
    delta: Optional[np.ndarray] = None
    n_seg: int = 1
    prod_cost: int = 1
    dist: InefficiencyDist = InefficiencyDist.HALF_NORMAL
    seed: int = 1234

    def __post_init__(self):
        if not self.classes:
            self.classes = [
                ClassSpec(
                    beta=np.array([1.0, 0.5, 0.3]),
                    ln_sigma_u=np.log(0.2),
                    ln_sigma_v=np.log(0.2),
                    ln_sigma_0=np.log(0.1),
                ),
                ClassSpec(
                    beta=np.array([1.5, 0.7, 0.4]),
                    ln_sigma_u=np.log(0.35),
                    ln_sigma_v=np.log(0.2),
                    ln_sigma_0=np.log(0.1),
                ),
            ]
        if self.delta is None:
            C = self.n_classes
            self.delta = np.zeros((C - 1) * self.n_seg)
        else:
            # infer n_seg from the supplied delta if not explicitly set
            C = self.n_classes
            inferred = len(self.delta) // max(C - 1, 1)
            if inferred > self.n_seg:
                self.n_seg = inferred

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_x(self) -> int:
        return len(self.classes[0].beta)

    @property
    def n_obs(self) -> int:
        return self.n_firms * self.n_periods

    def true_param_vector(self) -> np.ndarray:
        """
        Build the full parameter vector matching ESADataPanelLCM layout:
          [ seg_params | class_0_params | class_1_params | ... ]

        Seg block: (C-1)*n_seg elements, class-first then seg-variable:
          [ delta_c0_s0, delta_c0_s1, ..., delta_c1_s0, delta_c1_s1, ... ]

        Per-class block: [ beta | ln_sigma_u | ln_sigma_v | ln_sigma_0 ]
        """
        assert self.delta is not None
        parts = []
        if self.n_classes > 1:
            # reshape to (C-1, n_seg), then flatten row-major → class-first ordering
            parts.append(self.delta.reshape(self.n_classes - 1, self.n_seg).flatten())
        for cs in self.classes:
            parts.append(cs.beta)
            # model estimates log(σ²) = 2·log(σ), so scale accordingly
            parts.append(np.array([2*cs.ln_sigma_u, 2*cs.ln_sigma_v, 2*cs.ln_sigma_0]))
        return np.concatenate(parts)

    def param_names(self) -> list[str]:
        """Human-readable parameter names matching true_param_vector order."""
        names = []
        C = self.n_classes
        if C > 1:
            # class-first, then seg-variable — matches ESADataPanelLCM layout
            for c in range(C - 1):
                for s in range(self.n_seg):
                    names.append(f"delta_{c}_seg{s}")
        for c, cs in enumerate(self.classes):
            for k in range(len(cs.beta)):
                names.append(f"beta_{c}_{k}")
            names.append(f"ln_sigma_u_{c}")
            names.append(f"ln_sigma_v_{c}")
            names.append(f"ln_sigma_0_{c}")
        return names


# ─── DGP Output ──────────────────────────────────────────────────────────────


@dataclass
class DGPData:
    """Generated panel data ready for estimation."""

    y: np.ndarray           # (N*T, 1) output
    x: np.ndarray           # (N*T, nX) frontier regressors
    seg: np.ndarray         # (N*T, n_seg) segmentation variables (obs-level, tiled from seg_firm)
    seg_firm: np.ndarray    # (N, n_seg) firm-level segmentation variables (col 0 = intercept)
    id_vec: np.ndarray      # (N*T,) firm IDs (1-indexed)
    time_vec: np.ndarray    # (N*T,) time periods (1-indexed)
    zuit: np.ndarray        # (N*T, 1) transient ineff. heteroscedasticity
    zvit: np.ndarray        # (N*T, 1) noise heteroscedasticity
    zvi0: np.ndarray        # (N*T, 1) persistent ineff. heteroscedasticity

    # Ground truth
    true_class: np.ndarray      # (N,) firm class assignments
    true_u: np.ndarray          # (N*T,) transient inefficiency draws
    true_w0: np.ndarray         # (N,) persistent random effects
    true_efficiency: np.ndarray # (N*T,) exp(-u_it)
    true_params: np.ndarray     # full param vector
    config: DGPConfig


# ─── DGP Generation ──────────────────────────────────────────────────────────


def generate_data(cfg: DGPConfig) -> DGPData:
    """
    Generate panel data from an LC-TRE DGP.

    Firms are assigned to latent classes via multinomial logit. Within each
    class, data follows a standard TRE specification with half-normal inefficiency.
    """
    rng = np.random.default_rng(cfg.seed)
    N, T, C = cfg.n_firms, cfg.n_periods, cfg.n_classes
    nX = cfg.n_x
    s = cfg.prod_cost
    n_obs = N * T

    # Frontier regressors: column 0 is intercept
    x = np.ones((n_obs, nX))
    if nX > 1:
        x[:, 1:] = rng.standard_normal((n_obs, nX - 1))

    # Firm-level segmentation variables: col 0 = intercept, cols 1+ = concomitant vars
    seg_firm = np.ones((N, cfg.n_seg))
    if cfg.n_seg > 1:
        seg_firm[:, 1:] = rng.standard_normal((N, cfg.n_seg - 1))

    # Tile to obs-level for C++ (same value repeated across each firm's T periods)
    seg = np.repeat(seg_firm, T, axis=0)

    # Class assignment via multinomial logit using firm-level segmentation
    if C > 1:
        assert cfg.delta is not None
        # delta is (C-1)*n_seg, class-first: reshape to (C-1, n_seg)
        delta = cfg.delta.reshape(C - 1, cfg.n_seg)
        # logits: (N, C), last class is reference (logit=0)
        logits = np.zeros((N, C))
        for c in range(C - 1):
            logits[:, c] = seg_firm @ delta[c]  # (N, n_seg) @ (n_seg,) → (N,)
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        true_class = np.array([rng.choice(C, p=probs[i]) for i in range(N)])
    else:
        true_class = np.zeros(N, dtype=int)

    # Generate observations
    y = np.zeros(n_obs)
    true_u = np.zeros(n_obs)
    true_w0 = np.zeros(N)

    for i in range(N):
        c = true_class[i]
        cs = cfg.classes[c]
        w_i0 = rng.normal(0, cs.sigma_0)
        true_w0[i] = w_i0

        for t in range(T):
            idx = i * T + t
            v_it = rng.normal(0, cs.sigma_v)

            if cfg.dist == InefficiencyDist.HALF_NORMAL:
                u_it = abs(rng.normal(0, cs.sigma_u))
            else:
                u_it = rng.exponential(cs.sigma_u)

            true_u[idx] = u_it
            y[idx] = x[idx] @ cs.beta + v_it - s * u_it + w_i0

    true_efficiency = np.exp(-true_u)

    # Build arrays in Fortran order for C++
    id_vec = np.repeat(np.arange(1, N + 1), T).astype(np.int32)
    time_vec = np.tile(np.arange(1, T + 1), N).astype(np.int32)
    zuit = np.ones((n_obs, 1))
    zvit = np.ones((n_obs, 1))
    zvi0 = np.ones((n_obs, 1))

    return DGPData(
        y=np.asfortranarray(y.reshape(-1, 1)),
        x=np.asfortranarray(x),
        seg=np.asfortranarray(seg),
        seg_firm=np.asfortranarray(seg_firm),
        id_vec=np.asfortranarray(id_vec),
        time_vec=np.asfortranarray(time_vec),
        zuit=np.asfortranarray(zuit),
        zvit=np.asfortranarray(zvit),
        zvi0=np.asfortranarray(zvi0),
        true_class=true_class,
        true_u=true_u,
        true_w0=true_w0,
        true_efficiency=true_efficiency,
        true_params=cfg.true_param_vector(),
        config=cfg,
    )


# ─── Estimation ──────────────────────────────────────────────────────────────


@dataclass
class EstimationResult:
    """Result from a single model estimation."""

    converged: bool = False
    params: Optional[np.ndarray] = None
    log_likelihood: Optional[float] = None
    n_params: int = 0
    posteriors: Optional[np.ndarray] = None
    efficiencies: Optional[np.ndarray] = None
    wall_time_seconds: float = 0.0


def estimate_lcm(
    data: DGPData,
    n_classes: int,
    method: str = "pso_tr",
    nsim: int = 100,
    seed: int = 1234,
    nthreads: int = 4,
    print_level: int = 0,
    start: Optional[np.ndarray] = None,
    optim_opts: Optional[dict] = None,
) -> EstimationResult:
    """
    Estimate an LC-TRE model on generated data via the C++ backend.

    Returns an EstimationResult with convergence status and estimates.
    """
    import pysfacpp_internal

    start_arr = None
    if start is not None:
        start_arr = np.asfortranarray(start.reshape(-1, 1))

    t0 = time.perf_counter()
    try:
        res = pysfacpp_internal._pysfacpp_internal_lcm(
            data.y,
            data.x,
            data.seg,
            None,           # zmuit
            data.zuit,
            data.zvit,
            data.zvi0,
            data.id_vec,
            data.time_vec,
            n_classes,
            start_arr,
            data.config.prod_cost,
            "lctre",
            "hnorm",
            method,
            "dlib",
            optim_opts,
            None, None, None, None, None, None,  # terms
            nsim,
            "bhhh",
            seed,
            0.95,
            print_level,
            False,          # clusteredSE
            nthreads,
            2, 1000, 3, True, False, False,  # halton settings
            5, 120, None, None,
        )
    except Exception:
        return EstimationResult(converged=False, wall_time_seconds=time.perf_counter() - t0)

    elapsed = time.perf_counter() - t0
    converged = res.get("converged", False)

    return EstimationResult(
        converged=converged,
        params=res.get("par"),
        log_likelihood=res.get("logLikelihood"),
        n_params=res.get("nparam", 0),
        posteriors=res.get("posteriors"),
        efficiencies=res.get("efficiencyTransient"),
        wall_time_seconds=elapsed,
    )


# ─── Metrics ─────────────────────────────────────────────────────────────────


@dataclass
class MCMetrics:
    """Aggregate Monte Carlo performance metrics."""

    n_reps: int
    n_converged: int
    convergence_rate: float
    mean_wall_time: float = 0.0

    # Per-parameter arrays (length = n_params), None if nothing converged
    bias: Optional[np.ndarray] = None
    rmse: Optional[np.ndarray] = None
    std_dev: Optional[np.ndarray] = None
    mae: Optional[np.ndarray] = None
    coverage_95: Optional[np.ndarray] = None
    median_bias: Optional[np.ndarray] = None
    param_names: Optional[list[str]] = None
    true_params: Optional[np.ndarray] = None

    # Raw estimates matrix for further analysis (n_converged x n_params)
    _estimates: Optional[np.ndarray] = None

    def summary(self) -> pd.DataFrame:
        """Per-parameter summary table."""
        if self.bias is None:
            return pd.DataFrame({"info": ["No converged replications"]})
        data = {
            "true": self.true_params,
            "bias": self.bias,
            "rel_bias": self.bias / np.where(np.abs(self.true_params) > 1e-10, self.true_params, 1.0),
            "rmse": self.rmse,
            "std_dev": self.std_dev,
            "mae": self.mae,
        }
        if self.coverage_95 is not None:
            data["coverage_95"] = self.coverage_95
        df = pd.DataFrame(data)
        if self.param_names:
            df.index = self.param_names
        return df

    def __repr__(self) -> str:
        return (
            f"MCMetrics(n_reps={self.n_reps}, converged={self.n_converged}/{self.n_reps} "
            f"({self.convergence_rate:.0%}), mean_time={self.mean_wall_time:.1f}s)"
        )


def compute_metrics(
    results: list[EstimationResult],
    true_params: np.ndarray,
    param_names: Optional[list[str]] = None,
    se_matrix: Optional[np.ndarray] = None,
) -> MCMetrics:
    """
    Compute Monte Carlo diagnostics from estimation results.

    Parameters
    ----------
    results : list[EstimationResult]
        All replication results (converged and failed).
    true_params : np.ndarray
        True DGP parameter vector.
    param_names : list[str], optional
        Names for each parameter.
    se_matrix : np.ndarray, optional
        (n_converged x n_params) standard errors for coverage calculation.
    """
    converged = [r for r in results if r.converged and r.params is not None]
    n_converged = len(converged)
    n_reps = len(results)
    mean_time = np.mean([r.wall_time_seconds for r in results]) if results else 0.0

    if n_converged == 0:
        return MCMetrics(
            n_reps=n_reps,
            n_converged=0,
            convergence_rate=0.0,
            mean_wall_time=mean_time,
        )

    n_params = len(true_params)
    estimates = np.zeros((n_converged, n_params))
    for i, r in enumerate(converged):
        est = r.params.flatten()
        length = min(len(est), n_params)
        estimates[i, :length] = est[:length]

    mean_est = estimates.mean(axis=0)
    bias = mean_est - true_params
    median_bias = np.median(estimates, axis=0) - true_params
    sq_err = (estimates - true_params[np.newaxis, :]) ** 2
    rmse = np.sqrt(sq_err.mean(axis=0))
    std_dev = estimates.std(axis=0, ddof=1) if n_converged > 1 else np.zeros(n_params)
    mae = np.abs(estimates - true_params[np.newaxis, :]).mean(axis=0)

    # Coverage: proportion of 95% CIs that contain the true value
    coverage = None
    if se_matrix is not None and se_matrix.shape == estimates.shape:
        lower = estimates - 1.96 * se_matrix
        upper = estimates + 1.96 * se_matrix
        covered = (true_params >= lower) & (true_params <= upper)
        coverage = covered.mean(axis=0)

    return MCMetrics(
        n_reps=n_reps,
        n_converged=n_converged,
        convergence_rate=n_converged / n_reps,
        mean_wall_time=mean_time,
        bias=bias,
        rmse=rmse,
        std_dev=std_dev,
        mae=mae,
        median_bias=median_bias,
        coverage_95=coverage,
        param_names=param_names,
        true_params=true_params,
        _estimates=estimates,
    )


# ─── Orchestration ───────────────────────────────────────────────────────────


@dataclass
class MCConfig:
    """Configuration for a Monte Carlo experiment."""

    dgp: DGPConfig
    n_reps: int = 100
    n_classes_estimate: Optional[int] = None  # defaults to DGP truth
    method: str = "pso_tr"
    nsim: int = 100
    nthreads: int = 4
    print_level: int = 0
    optim_opts: Optional[dict] = None
    verbose: bool = True
    progress_callback: Optional[Callable[[int, int, int], None]] = None


def run_monte_carlo(mc: MCConfig) -> MCMetrics:
    """
    Run a full Monte Carlo experiment.

    Generates fresh data per replication (incrementing seed), estimates the
    model, and computes aggregate metrics.

    Parameters
    ----------
    mc : MCConfig
        Full experiment configuration.

    Returns
    -------
    MCMetrics
        Aggregated results with bias, RMSE, coverage, etc.
    """
    n_classes_est = mc.n_classes_estimate or mc.dgp.n_classes
    results: list[EstimationResult] = []

    for rep in range(mc.n_reps):
        if mc.verbose and (rep % max(1, mc.n_reps // 10) == 0):
            n_conv = sum(1 for r in results if r.converged)
            print(f"  [{rep + 1}/{mc.n_reps}] converged: {n_conv}")

        if mc.progress_callback:
            n_conv = sum(1 for r in results if r.converged)
            mc.progress_callback(rep, mc.n_reps, n_conv)

        # Fresh DGP draw with incremented seed
        rep_cfg = DGPConfig(
            n_firms=mc.dgp.n_firms,
            n_periods=mc.dgp.n_periods,
            classes=mc.dgp.classes,
            delta=mc.dgp.delta,
            n_seg=mc.dgp.n_seg,
            prod_cost=mc.dgp.prod_cost,
            dist=mc.dgp.dist,
            seed=mc.dgp.seed + rep,
        )
        data = generate_data(rep_cfg)

        result = estimate_lcm(
            data=data,
            n_classes=n_classes_est,
            method=mc.method,
            nsim=mc.nsim,
            seed=mc.dgp.seed + rep,
            nthreads=mc.nthreads,
            print_level=mc.print_level,
            optim_opts=mc.optim_opts,
        )
        results.append(result)

    true_params = mc.dgp.true_param_vector()
    param_names = mc.dgp.param_names()
    metrics = compute_metrics(results, true_params, param_names)

    if mc.verbose:
        print(f"\n  Done: {metrics.n_converged}/{metrics.n_reps} converged "
              f"({metrics.convergence_rate:.0%}), mean time {metrics.mean_wall_time:.1f}s")
        if metrics.bias is not None:
            print(f"  Mean |bias|: {np.abs(metrics.bias).mean():.4f}, "
                  f"Mean RMSE: {metrics.rmse.mean():.4f}")

    return metrics
