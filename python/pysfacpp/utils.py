import pysfacpp_internal
import numpy as np
import pandas as pd

def available_hardware_thread_count() -> int:
    """
    Return the available hardware thread count
    """
    return pysfacpp_internal._get_available_threads()

def flush_thread_local_storage() -> None:
    """
    Flush the thread local storage (e.g., free up some RAM). This is achieved
    by reset the threads (which automatically destruct the thread_local storage)
    associated with them.
    """
    pysfacpp_internal._flush_tls()

def memory_usage() -> float:
    return pysfacpp_internal._memusage()


def simulate_gtre_panel(
    n_firms: int = 100,
    n_periods: int = 10,
    betas=(0.8, 0.5, 0.3, 0.4),
    intercept: float = 0.7,
    # INPUTS: Coefficients for Log-Variance (gamma)
    # Model assumes: sigma^2 = exp(gamma_0 + gamma_1*Z_1 + ...)
    # If scalar: sigma^2 = exp(value)
    # If list:   sigma^2 = exp(intercept + slope*Z + ...)
    sigma2_uit=-1.386,  # approx ln(0.25)
    sigma2_vit=-0.598,  # approx ln(0.55)
    sigma2_vi0=-0.916,  # approx ln(0.40)
    sigma2_ui0=-0.342,  # approx ln(0.71)
    s: int = 1,         # s=+1 (Production), s=-1 (Cost)
    seed: int = 1234,
    x_dist: str = "normal",
    include_latents: bool = True,
) -> pd.DataFrame:
    """
    Simulate a GTRE panel using Log-Variance parameterization.
    
    The sigma2_* inputs represent the COEFFICIENTS (gamma) outputted by the model,
    where variance sigma^2 = exp(Z * gamma).
    
    Y Generation Logic:
      Production (s=1):  y = xb + v_i0 + v_it - u_i0 - u_it
      Cost (s=-1):       y = xb + v_i0 + v_it + u_i0 + u_it
    """
    if s not in (-1, 1):
        raise ValueError("s must be -1 or +1")
    rng = np.random.default_rng(seed)
    # Dimensions
    N = n_firms
    T = n_periods
    total_obs = N * T
    # Indices
    ids = np.repeat(np.arange(N), T)
    ts  = np.tile(np.arange(T), N)
    # 1. Generate Frontier Covariates (X)
    betas = np.asarray(betas, dtype=float)
    k_x = betas.size
    if x_dist == "normal":
        X = rng.normal(size=(total_obs, k_x))
    elif x_dist == "uniform":
        X = rng.uniform(-1.0, 1.0, size=(total_obs, k_x))
    else:
        raise ValueError("x_dist must be 'normal' or 'uniform'")
    xb = intercept + X @ betas
    # 2. Helper to Process Sigma Parameters (Coefficients -> Std Dev)
    # Stores generated Z variables to add to DataFrame later
    z_storage = {} 
    def get_sigma_vec(param, n_rows, prefix, level_name):
        """
        Parses the input parameter (scalar or list of coeffs).
        Generates Z data if needed.
        Returns: Vector of Standard Deviations (sigma), Vector of Z*gamma (log_var)
        """
        # Ensure param is a list/array
        if np.isscalar(param):
            coeffs = np.array([param], dtype=float)
        else:
            coeffs = np.array(param, dtype=float)
        gamma_0 = coeffs[0]
        gamma_slopes = coeffs[1:]
        n_z = len(gamma_slopes)
        # Calculate Linear Predictor (log_variance)
        if n_z == 0:
            # Homoskedastic: log_var = gamma_0
            log_variance = np.full(n_rows, gamma_0)
        else:
            # Heteroskedastic: log_var = gamma_0 + Z * gamma_slopes
            # Generate Zs (Standard Normal)
            Z = rng.normal(0, 1, size=(n_rows, n_z))
            # Store Zs for DataFrame
            for i in range(n_z):
                col_name = f"Z_{prefix}_{i+1}"
                z_storage[col_name] = (Z[:, i], level_name) # Store value and level (firm/obs)
            log_variance = gamma_0 + Z @ gamma_slopes
        # Transform to Variance and Std Dev
        # Model: sigma^2 = exp(log_variance)
        # StdDev: sigma = sqrt(exp(log_variance)) = exp(0.5 * log_variance)
        variance = np.exp(log_variance)
        sigma = np.sqrt(variance)
        return sigma, variance
    # 3. Calculate Sigmas (Standard Deviations)
    # Persistent (Firm-level, N)
    sig_vi0, var_vi0 = get_sigma_vec(sigma2_vi0, N, "vi0", "firm")
    sig_ui0, var_ui0 = get_sigma_vec(sigma2_ui0, N, "ui0", "firm")
    # Transient (Obs-level, N*T)
    sig_vit, var_vit = get_sigma_vec(sigma2_vit, total_obs, "vit", "obs")
    sig_uit, var_uit = get_sigma_vec(sigma2_uit, total_obs, "uit", "obs")
    # 4. Generate Latent Errors
    # Persistent
    vi0 = rng.normal(0.0, sig_vi0) 
    ui0 = np.abs(rng.normal(0.0, sig_ui0))
    # Expand Persistent to Panel (N -> N*T)
    vi0_long = vi0[ids]
    ui0_long = ui0[ids]
    # Transient
    vit = rng.normal(0.0, sig_vit)
    uit = np.abs(rng.normal(0.0, sig_uit))
    # 5. Construct Y
    # Production (s=1):  y = Frontier + Noise - Inefficiency
    # Cost (s=-1):       y = Frontier + Noise + Inefficiency
    y = xb + vi0_long + vit - s * ui0_long - s * uit
    # 6. Build DataFrame
    df = pd.DataFrame({
        "id": ids,
        "time": ts,
        "y": y,
    })
    # Add X columns
    for j in range(k_x):
        df[f"x{j+1}"] = X[:, j]
    # Add Z columns (if any were generated)
    for col_name, (values, level) in z_storage.items():
        if level == "firm":
            df[col_name] = values[ids] # Expand firm Z to panel
        else:
            df[col_name] = values
    # Add Latents and Truths for Validation
    if include_latents:
        # True Errors
        df["vi0_true"] = vi0_long
        df["ui0_true"] = ui0_long
        df["vit_true"] = vit
        df["uit_true"] = uit
        # True Variances (sigma^2)
        # Expand firm-level variances to panel
        df["true_var_vi0"] = var_vi0[ids]
        df["true_var_ui0"] = var_ui0[ids]
        df["true_var_vit"] = var_vit
        df["true_var_uit"] = var_uit
        # True Standard Deviations
        df["true_sigma_vi0"] = sig_vi0[ids]
        df["true_sigma_ui0"] = sig_ui0[ids]
        df["true_sigma_vit"] = sig_vit
        df["true_sigma_uit"] = sig_uit
        # Composed Error (eps)
        # eps = y - xb - vi0 + s*ui0 = vit - s*uit
        df["eps_true"] = y - xb - vi0_long + s * ui0_long
        # True Time-Varying Efficiency
        df["eff_transient"] = np.exp(-df["uit_true"])
        # True Time-Invariant Efficiency
        df["eff_persistent"] = np.exp(-df["ui0_true"])
        # Overall Efficiency
        df["eff_overall"] = np.exp(-(df["ui0_true"] + df["uit_true"]))
        df["true_ineff_transient_0_1"] = 1.0 - df["eff_transient"]
        df["true_ineff_persistent_0_1"] = 1.0 - df["eff_persistent"]
    return df