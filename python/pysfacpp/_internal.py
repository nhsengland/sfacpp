#' Title
#' Author: Edmund Haacke
#' Date: 2026-01-01
#'

import pandas as pd
import numpy as np
import patsy

def __check_required_keys_for_inputs(d: dict[str, str]) -> bool:
    """Check whether the K, L, M keys are within a dictionary's keys"""
    if not isinstance(d, dict):
        raise TypeError(f"Expected 'dict', but got {type(d)}")
    if not ("K" in d.keys()) and ("L" in d.keys()) and ("M" in d.keys()):
        raise ValueError("Expecting 'dict' to contain 'K', 'L', and 'M' as keys")
    return True

def _check_local_concavity(
    df: pd.DataFrame,
    prod_fn_formula: str,
    coefs: dict[str, float],
    input_cols_uncentered: dict[str, str],
    elasticity_cols: dict[str, str],
    tol: float = 1e-8
) -> pd.DataFrame:
    """
    Check for local concavity e.g., whether the Hessian is negative semi-definite,
    for a Translog production function. Designed only for three inputs - K, L, M
    (capital, labor, and intermediate goods)

    Args:
        df (pd.DataFrame): Input dataframe, containing model variables (centered), and 
            raw variables (e.g., uncentered inputs)
        prod_fn_formula (str): Patsy formula string of the production function used
        coefs (dict[str, str]): Dictionary of model coefficients. The keys must align
            with the Hessian terms (e.g., 'b_kk', b_ll', b_ii', 'b_lk', 'b_li', 'b_ki', ...)
        input_cols_uncentered (dict[str, str]): Mapping of inputs (e.g., 'L', 'K', 'M')
            to the column names in `df`, holding the uncentered (log) inputs
        elasticity_cols: Mapping of inputs (e.g., 'L', 'K', 'M') to column names in `df`
            holding the derived elasticities.
        tol: Tolerance for considering an eigenvalue zero
    
    Returns:
        DataFrame with original index, and a boolean column 'is_concave'
    """
    # ---- initial setup ----
    # check that the keys are in two mapping dictionaries
    __check_required_keys_for_inputs(d=input_cols_uncentered)
    __check_required_keys_for_inputs(d=elasticity_cols)
    # for the coefficients, check all the required terms are present
    coefs_keys_xpt: list[str] = [
        "b_kk", "b_ll", "b_mm", "b_lk", "b_lm", "b_km"
    ]
    for k in coefs_keys_xpt:
        if k not in coefs.keys():
            raise ValueError(f"expect '{k}' in 'coef', if coefficient doesn't exist, set to 0")
    # calculate predicted Y (lnY) using the model formula (based on the centered variables)
    try:
        # create design matrix (if possible)
        x_dmat = patsy.dmatrix(prod_fn_formula, df, return_type="dataframe")
        # filter coefficients to match design matrix columns
        beta_vec: np.ndarray = np.array(object=[coefs[col] for col in x_dmat.columns])
        pred_ln_y: np.ndarray = x_dmat.values @ beta_vec
    except KeyError as e:
        raise KeyError(f"Coefficient missing from model term: {e}")
    except Exception as e:
        raise ValueError(f"Error constructing model matrix from formula: {e}")
    # recover the physical levels (X, Y)
    col_l_raw: str = input_cols_uncentered["L"]
    col_k_raw: str = input_cols_uncentered["K"]
    col_m_raw: str = input_cols_uncentered["M"]
    # calculate exponentials to convert logs (of uncentered) back to levels
    l_lvl: np.ndarray = np.exp(df[col_l_raw].values)
    k_lvl: np.ndarray = np.exp(df[col_k_raw].values)
    m_lvl: np.ndarray = np.exp(df[col_m_raw].values)
    y_lvl: np.ndarray = np.exp(pred_ln_y)
    # extract the elasticities
    eps_l: np.ndarray = df[elasticity_cols["L"]]
    eps_k: np.ndarray = df[elasticity_cols["K"]]
    eps_m: np.ndarray = df[elasticity_cols["M"]]
    # construct hessian elements
    # H_xx = (Y / X^2) * (beta_xx - eps_x + eps_x^2)
    # either (Y / X^2) or (Y / X*Z)
    Q_ll: np.ndarray = y_lvl / (l_lvl ** 2)
    Q_kk: np.ndarray = y_lvl / (k_lvl ** 2)
    Q_mm: np.ndarray = y_lvl / (m_lvl ** 2)
    Q_lk: np.ndarray = y_lvl / (l_lvl * k_lvl)
    Q_lm: np.ndarray = y_lvl / (l_lvl * m_lvl)
    Q_km: np.ndarray = y_lvl / (k_lvl * m_lvl)
    # diagonal elements of the hessian
    H_ll: np.ndarray = Q_ll * (coefs["b_ll"] - eps_l + eps_l**2)
    H_kk: np.ndarray = Q_kk * (coefs["b_kk"] - eps_k + eps_k**2)
    H_mm: np.ndarray = Q_mm * (coefs["b_mm"] - eps_m + eps_m**2)
    # off-diagonal elements of the hessian matrix
    H_lk: np.ndarray = Q_lk * (coefs["b_lk"] + eps_l * eps_k)
    H_lm: np.ndarray = Q_lm * (coefs["b_lm"] + eps_l * eps_m)
    H_km: np.ndarray = Q_km * (coefs["b_km"] + eps_k * eps_m)
    # assemble 3x3 hessian matricies - use a 3d array of (n, 3, 3)
    N: int = len(df)
    hessians: np.ndarray = np.zeros(shape=(N, 3, 3))
    # fill the diagonal elements
    hessians[:, 0, 0] = H_ll
    hessians[:, 1, 1] = H_kk
    hessians[:, 2, 2] = H_mm
    # fill the off-diagonal elements (these are symmetric)
    hessians[:, 0, 1] = H_lk
    hessians[:, 1, 0] = H_lk
    hessians[:, 0, 2] = H_lm
    hessians[:, 2, 0] = H_lm
    hessians[:, 1, 2] = H_km
    hessians[:, 2, 1] = H_km
    # check concavity - negative semi-definate; use eigenvalues
    eigenvalues: np.ndarray = np.linalg.eigvalsh(hessians)
    # find the maximum eigenvalue per observation - check its below the tolerance
    # to confirm that it is negative semi-definite
    is_concave: np.ndarray = np.all(a=eigenvalues <= tol, axis=1)
    # return the result, with the original index
    result_df: pd.DataFrame = pd.DataFrame(index=df.index)
    result_df["is_concave"] = is_concave
    eig_df: pd.DataFrame = pd.DataFrame(
        data=eigenvalues,
        columns=[
            f"eigenvalue_{i}"
            for i in range(0, eigenvalues.shape[1])
        ]
    )
    result_df = pd.concat(
        objs=[
            result_df,
            eig_df
        ],
        axis=1
    )
    return result_df
