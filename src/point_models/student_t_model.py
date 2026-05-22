"""Student-t AR(p) point-estimation with fixed degrees of freedom."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from src.point_models.gaussian_model import build_ar_design


def _student_t_logpdf(x: np.ndarray, scale: float, nu: float) -> np.ndarray:
    z = x / scale
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * nu) - np.log(scale)
    return c - ((nu + 1) / 2) * np.log1p((z**2) / nu)


def fit_student_t_ar(y: np.ndarray, p: int, nu: float = 5.0, max_iter: int = 100, tol: float = 1e-8) -> dict:
    """Fit AR(p) with fixed-nu Student-t residuals using IRLS."""
    if nu <= 2:
        raise ValueError("nu must be > 2 for finite variance.")

    y_target, x = build_ar_design(y, p)
    beta, *_ = np.linalg.lstsq(x, y_target, rcond=None)
    scale2 = np.var(y_target - x @ beta)

    for _ in range(max_iter):
        resid = y_target - x @ beta
        weights = (nu + 1.0) / (nu + resid**2 / np.clip(scale2, 1e-12, None))

        xw = x * np.sqrt(weights[:, None])
        yw = y_target * np.sqrt(weights)
        beta_new, *_ = np.linalg.lstsq(xw, yw, rcond=None)

        resid_new = y_target - x @ beta_new
        scale2_new = np.mean(weights * resid_new**2)

        if np.max(np.abs(beta_new - beta)) < tol and abs(scale2_new - scale2) < tol:
            beta, scale2 = beta_new, scale2_new
            break
        beta, scale2 = beta_new, scale2_new

    residuals = y_target - x @ beta
    scale = np.sqrt(np.clip(scale2, 1e-12, None))
    ll = _student_t_logpdf(residuals, scale=scale, nu=nu).sum()

    n = len(residuals)
    k = p + 1
    aic = 2 * k - 2 * ll
    bic = np.log(n) * k - 2 * ll

    return {
        "model": "student_t",
        "p": p,
        "nu": nu,
        "phi": beta,
        "scale": scale,
        "residuals": residuals,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "n_obs": n,
    }
