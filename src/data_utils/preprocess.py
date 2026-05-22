"""Preprocessing for daily price data."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_change",
    "pct_change",
]


def preprocess_daily_prices(
    df: pd.DataFrame,
    rename_map: Mapping[str, str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Clean dataset and compute daily log returns from close prices."""
    out = df.rename(columns=rename_map).copy()

    missing = set(REQUIRED_COLUMNS) - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns after rename: {sorted(missing)}")

    out = out[REQUIRED_COLUMNS]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume", "price_change", "pct_change"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "close"])
    out = out.drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)

    if start_date is not None:
        out = out[out["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        out = out[out["date"] <= pd.to_datetime(end_date)]

    out["log_return"] = 100.0 * np.log(out["close"]).diff()
    out = out.dropna(subset=["log_return"]).reset_index(drop=True)
    return out
