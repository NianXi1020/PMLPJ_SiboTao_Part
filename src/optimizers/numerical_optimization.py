"""Numerical optimization helpers."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


def maximize(
    objective: Callable[[np.ndarray], float],
    x0: Sequence[float],
    bounds=None,
    method: str = "L-BFGS-B",
):
    """Maximize objective by minimizing its negative value."""

    def neg_obj(x: np.ndarray) -> float:
        return -objective(x)

    result = minimize(neg_obj, np.asarray(x0, dtype=float), bounds=bounds, method=method)
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x, -result.fun
