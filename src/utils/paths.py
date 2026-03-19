"""Project path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "test_data" / "Daily"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PARAMS_DIR = RESULTS_DIR / "fitted_params"
OUTPUT_FAMILIES = ("common", "gaussian", "student_t", "gaussian_mixture")


@dataclass(frozen=True)
class RunOutputPaths:
    """Container of output directories for a single workflow run."""

    timestamp: str
    figures: dict[str, Path]
    tables: dict[str, Path]
    params: dict[str, Path]


def ensure_results_dirs() -> None:
    """Create result output folders if they do not exist."""
    for folder in (FIGURES_DIR, TABLES_DIR, PARAMS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def build_run_timestamp() -> str:
    """Build a single run timestamp reused across all saved artifacts."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_run_output_paths(run_timestamp: str) -> RunOutputPaths:
    """Create timestamped output folders for this run and return path mapping."""
    ensure_results_dirs()
    figures_root = FIGURES_DIR / run_timestamp
    tables_root = TABLES_DIR / run_timestamp
    params_root = PARAMS_DIR / run_timestamp

    figures_paths = {family: figures_root / family for family in OUTPUT_FAMILIES}
    tables_paths = {family: tables_root / family for family in OUTPUT_FAMILIES}
    params_paths = {family: params_root / family for family in OUTPUT_FAMILIES}

    for directory in (
        *figures_paths.values(),
        *tables_paths.values(),
        *params_paths.values(),
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return RunOutputPaths(
        timestamp=run_timestamp,
        figures=figures_paths,
        tables=tables_paths,
        params=params_paths,
    )
