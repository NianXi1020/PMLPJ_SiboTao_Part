"""Time-series plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_log_return_series(df: pd.DataFrame, dataset: str, output_path: Path, figsize=(12, 4), dpi: int = 140) -> None:
    """Plot and save log-return time series."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(df["date"], df["log_return"], linewidth=1.0)
    ax.set_title(f"{dataset} Daily Log Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return (%)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)


def plot_rolling_volatility(
    df: pd.DataFrame,
    dataset: str,
    output_path: Path,
    window: int = 21,
    figsize=(12, 4),
    dpi: int = 140,
) -> None:
    """Plot rolling standard deviation of log returns."""
    rolling_vol = df["log_return"].rolling(window=window).std()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(df["date"], rolling_vol, linewidth=1.2)
    ax.set_title(f"{dataset} Rolling Volatility (window={window})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Std of Log Return")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)
