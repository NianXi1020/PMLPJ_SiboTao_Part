"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "test_data" / "Daily"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PARAMS_DIR = RESULTS_DIR / "fitted_params"


def ensure_results_dirs() -> None:
    """Create result output folders if they do not exist."""
    for folder in (FIGURES_DIR, TABLES_DIR, PARAMS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
