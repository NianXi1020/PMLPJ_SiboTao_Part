"""Gaussian AR(p) point-estimation model."""

from __future__ import annotations

import numpy as np


def build_ar_design(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Build AR target and lag matrix."""
    if p < 1:
        raise ValueError("p must be >= 1")
    t = len(y)
    if t <= p:
        raise ValueError("Series is too short for selected lag order.")

    y_target = y[p:]
    x = np.column_stack([y[p - i - 1 : t - i - 1] for i in range(p)])
    return y_target, x


def fit_gaussian_ar(y: np.ndarray, p: int) -> dict:
    """Fit AR(p) with Gaussian residuals via OLS/MLE."""
    y_target, x = build_ar_design(y, p)
    beta, *_ = np.linalg.lstsq(x, y_target, rcond=None)
    fitted = x @ beta
    residuals = y_target - fitted

    n = len(residuals)
    sigma2 = np.mean(residuals**2)
    loglik = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)

    k = p + 1
    aic = 2 * k - 2 * loglik
    bic = np.log(n) * k - 2 * loglik

    return {
        "model": "gaussian",
        "p": p,
        "phi": beta,
        "sigma2": sigma2,
        "residuals": residuals,
        "fitted": fitted,
        "log_likelihood": loglik,
        "aic": aic,
        "bic": bic,
        "n_obs": n,
    }
