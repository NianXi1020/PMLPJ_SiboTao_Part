"""QQ plot utilities for residual diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_qq_gaussian(residuals: np.ndarray, dataset: str, model_name: str, output_path: Path, figsize=(6, 6), dpi: int = 140) -> None:
    """Save Gaussian QQ plot."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f"{dataset} {model_name} QQ Plot vs Gaussian")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)


def plot_qq_student_t(
    residuals: np.ndarray,
    dataset: str,
    model_name: str,
    nu: float,
    output_path: Path,
    figsize=(6, 6),
    dpi: int = 140,
) -> None:
    """Save Student-t QQ plot with fixed degrees of freedom."""
    probs = (np.arange(1, len(residuals) + 1) - 0.5) / len(residuals)
    theo = stats.t.ppf(probs, df=nu)
    sample = np.sort((residuals - residuals.mean()) / residuals.std())

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(theo, sample, s=10, alpha=0.7)
    mn = min(theo.min(), sample.min())
    mx = max(theo.max(), sample.max())
    ax.plot([mn, mx], [mn, mx], color="black", lw=1)
    ax.set_title(f"{dataset} {model_name} QQ Plot vs Student-t (nu={nu})")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)
