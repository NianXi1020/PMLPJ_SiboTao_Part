"""Plotting helpers for VAR rolling evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_actual_vs_predicted(detail_df: pd.DataFrame, model_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for series, sub in detail_df.groupby("series"):
        ax.plot(sub["date"], sub["y_true"], label=f"{series} true", alpha=0.7)
        ax.plot(sub["date"], sub["pred_mean"], linestyle="--", label=f"{series} pred", alpha=0.8)
    ax.set_title(f"{model_name} Actual vs Predicted Mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend(ncol=2, fontsize=8)
    _save_fig(output_path)


def plot_residual_series(detail_df: pd.DataFrame, model_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    detail_df = detail_df.copy()
    detail_df["residual"] = detail_df["y_true"] - detail_df["pred_mean"]
    for series, sub in detail_df.groupby("series"):
        ax.plot(sub["date"], sub["residual"], label=series, alpha=0.8)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(f"{model_name} Residual Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.legend()
    _save_fig(output_path)


def plot_residual_hist(detail_df: pd.DataFrame, model_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    residual = detail_df["y_true"] - detail_df["pred_mean"]
    ax.hist(residual, bins=40, alpha=0.7, color="steelblue")
    ax.set_title(f"{model_name} Residual Histogram")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    _save_fig(output_path)


def plot_predictive_distribution(draws: np.ndarray, series_name: str, model_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(draws[:, 0], bins=40, alpha=0.75, color="darkorange")
    ax.set_title(f"{model_name} Predictive Draws ({series_name})")
    ax.set_xlabel("Simulated Return")
    ax.set_ylabel("Count")
    _save_fig(output_path)


def plot_var_violations(detail_df: pd.DataFrame, model_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for series, sub in detail_df.groupby("series"):
        ax.plot(sub["date"], sub["violation"].astype(int), marker="o", linestyle="", label=series, alpha=0.6)
    ax.set_title(f"{model_name} VaR Violations")
    ax.set_xlabel("Date")
    ax.set_ylabel("Violation (1=True)")
    ax.legend()
    _save_fig(output_path)
