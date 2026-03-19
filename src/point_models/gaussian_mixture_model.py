"""Gaussian-mixture AR(p) point-estimation model."""

from __future__ import annotations

import numpy as np

from src.optimizers.em import run_em_zero_mean_gaussian_mixture
from src.point_models.gaussian_model import build_ar_design


def fit_gaussian_mixture_ar(
    y: np.ndarray,
    p: int,
    n_components: int = 2,
    random_state: int = 42,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> dict:
    """Fit AR(p) and then fit Gaussian mixture on residuals using EM."""
    y_target, x = build_ar_design(y, p)
    beta, *_ = np.linalg.lstsq(x, y_target, rcond=None)
    residuals = y_target - x @ beta

    em_res = run_em_zero_mean_gaussian_mixture(
        residuals,
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )

    n = len(residuals)
    # Parameters: AR coeffs + (K-1) weights + K variances
    k = p + (n_components - 1) + n_components
    aic = 2 * k - 2 * em_res["log_likelihood"]
    bic = np.log(n) * k - 2 * em_res["log_likelihood"]

    return {
        "model": "gaussian_mixture",
        "p": p,
        "phi": beta,
        "weights": em_res["weights"],
        "component_variances": em_res["variances"],
        "responsibilities": em_res["responsibilities"],
        "residuals": residuals,
        "log_likelihood": em_res["log_likelihood"],
        "aic": aic,
        "bic": bic,
        "n_obs": n,
        "n_iter": em_res["n_iter"],
    }
