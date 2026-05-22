"""I/O helpers for exporting tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Save a dataframe to CSV with stable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.8f")
