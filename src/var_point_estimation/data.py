"""Data preparation helpers for VAR point estimation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_utils.loader import load_daily_csv


DEFAULT_SERIES = ["SPY", "QQQ", "DIA", "IWN"]


def load_var_return_frame(
    data_dir: Path,
    series_names: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load and align 4 return series into one dataframe with date column."""
    names = series_names or DEFAULT_SERIES
    merged: pd.DataFrame | None = None

    for name in names:
        series_df = load_daily_csv(data_dir / f"{name}.csv", start_date=start_date, end_date=end_date)
        sub = series_df[["date", "log_return"]].rename(columns={"log_return": name})
        merged = sub if merged is None else merged.merge(sub, on="date", how="inner")

    if merged is None:
        raise ValueError("No data loaded for VAR frame.")

    merged = merged.sort_values("date").dropna().reset_index(drop=True)
    return merged


def build_var2_design(data: pd.DataFrame, target_cols: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build VAR(2) design matrix and output matrix.

    X_t = [1, Y_{t-1}, Y_{t-2}] with 9 columns for 4 series.
    """
    y_all = data[target_cols].to_numpy(dtype=float)
    if len(y_all) < 3:
        raise ValueError("Need at least 3 rows to build VAR(2) design.")

    y_t = y_all[2:]
    lag1 = y_all[1:-1]
    lag2 = y_all[:-2]
    intercept = np.ones((len(y_t), 1), dtype=float)
    x_t = np.hstack([intercept, lag1, lag2])
    dates = pd.DatetimeIndex(data["date"].iloc[2:])
    return x_t, y_t, dates


def build_last_obs_matrix(data: pd.DataFrame, target_cols: list[str]) -> np.ndarray:
    """Return last two observations in ascending order with shape (2, n_series)."""
    arr = data[target_cols].to_numpy(dtype=float)
    if len(arr) < 2:
        raise ValueError("Need at least 2 rows for one-step forecasting.")
    return arr[-2:]
