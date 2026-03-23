"""Output paths and saving utilities for VAR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


MODEL_FOLDERS = ("gaussian", "student_t", "mixture")


@dataclass(frozen=True)
class VAROutputPaths:
    """Holds result directories for one VAR run."""

    timestamp: str
    run_root: Path
    metrics: dict[str, Path]
    figures: dict[str, Path]


def build_run_timestamp() -> str:
    """Generate timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_var_output_paths(results_dir: Path, timestamp: str | None = None) -> VAROutputPaths:
    """Create per-model metrics/figures folders under results/<timestamp>/."""
    run_ts = timestamp or build_run_timestamp()
    run_root = results_dir / run_ts
    metrics = {}
    figures = {}
    for model in MODEL_FOLDERS:
        metrics_dir = run_root / model / "metrics"
        figures_dir = run_root / model / "figures"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        metrics[model] = metrics_dir
        figures[model] = figures_dir
    return VAROutputPaths(timestamp=run_ts, run_root=run_root, metrics=metrics, figures=figures)


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save CSV with stable float format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.8f")
