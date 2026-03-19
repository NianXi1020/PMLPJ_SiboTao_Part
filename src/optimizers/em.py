"""EM optimizer for zero-mean Gaussian mixtures."""

from __future__ import annotations

import numpy as np


def run_em_zero_mean_gaussian_mixture(
    residuals: np.ndarray,
    n_components: int = 2,
    max_iter: int = 300,
    tol: float = 1e-6,
    random_state: int = 42,
) -> dict:
    """Fit zero-mean Gaussian mixture to residuals and return EM state."""
    x = residuals.reshape(-1)
    n = x.shape[0]
    rng = np.random.default_rng(random_state)

    weights = np.full(n_components, 1.0 / n_components)
    base_var = np.var(x) if np.var(x) > 1e-8 else 1.0
    variances = base_var * (0.5 + rng.random(n_components))

    ll_history = []
    eps = 1e-12

    for _ in range(max_iter):
        # E-step
        densities = np.column_stack(
            [
                np.exp(-0.5 * x**2 / (v + eps)) / np.sqrt(2.0 * np.pi * (v + eps))
                for v in variances
            ]
        )
        weighted = densities * weights
        denom = np.clip(weighted.sum(axis=1, keepdims=True), eps, None)
        responsibilities = weighted / denom

        # M-step
        nk = responsibilities.sum(axis=0)
        weights = nk / n
        variances = (responsibilities * (x[:, None] ** 2)).sum(axis=0) / np.clip(nk, eps, None)
        variances = np.clip(variances, 1e-8, None)

        # Log-likelihood
        ll = np.log(np.clip((densities * weights).sum(axis=1), eps, None)).sum()
        ll_history.append(ll)

        if len(ll_history) > 1 and abs(ll_history[-1] - ll_history[-2]) < tol:
            break

    return {
        "weights": weights,
        "variances": variances,
        "responsibilities": responsibilities,
        "log_likelihood": ll_history[-1],
        "n_iter": len(ll_history),
    }
