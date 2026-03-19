"""Residual plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm, t


def plot_residual_series(residuals: np.ndarray, dataset: str, model_name: str, output_path: Path, figsize=(12, 4), dpi: int = 140) -> None:
    """Plot residual time series."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(residuals, linewidth=1.0)
    ax.set_title(f"{dataset} {model_name} Residual Series")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)


def plot_residual_hist_with_density(
    residuals: np.ndarray,
    dataset: str,
    model_name: str,
    output_path: Path,
    nu: float | None = None,
    figsize=(8, 4),
    dpi: int = 140,
) -> None:
    """Plot residual histogram with Gaussian and optional Student-t densities."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.histplot(residuals, bins=40, stat="density", color="lightsteelblue", edgecolor="white", ax=ax)

    x = np.linspace(residuals.min(), residuals.max(), 400)
    mu, sigma = np.mean(residuals), np.std(residuals)
    ax.plot(x, norm.pdf(x, loc=mu, scale=sigma), label="Gaussian density", lw=2)

    if nu is not None:
        scale = np.sqrt(np.var(residuals) * (nu - 2) / nu)
        ax.plot(x, t.pdf((x - mu) / scale, df=nu) / scale, label=f"Student-t density (nu={nu})", lw=2)

    ax.set_title(f"{dataset} {model_name} Residual Histogram and Density")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)
