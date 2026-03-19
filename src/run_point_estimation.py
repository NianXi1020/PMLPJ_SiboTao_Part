"""Main workflow for point-estimation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_utils.loader import load_daily_csv
from src.plotting.qq_plots import plot_qq_gaussian, plot_qq_student_t
from src.plotting.residual_plots import plot_residual_hist_with_density, plot_residual_series
from src.plotting.time_series_plots import plot_log_return_series, plot_rolling_volatility
from src.point_models.gaussian_mixture_model import fit_gaussian_mixture_ar
from src.point_models.gaussian_model import fit_gaussian_ar
from src.point_models.student_t_model import fit_student_t_ar
from src.utils.io_helpers import save_dataframe
from src.utils.naming import figure_name, table_name
from src.utils.paths import DATA_DIR, build_run_timestamp, prepare_run_output_paths


DATASETS = ["DIA", "IWN", "QQQ", "SPY"]
P_VALUES = [1, 2, 3, 4, 5]


def _phi_record(dataset: str, model: str, p: int, phi: np.ndarray) -> dict:
    record = {"dataset": dataset, "model": model, "p": p}
    record.update({f"phi_{i+1}": val for i, val in enumerate(phi)})
    return record


def run_point_estimation_workflow(
    datasets: Iterable[str] = DATASETS,
    p_values: Iterable[int] = P_VALUES,
    nu: float = 5.0,
    n_components: int = 2,
    rolling_window: int = 21,
    random_state: int = 42,
    start_date: str | None = None,
    end_date: str | None = None,
    run_timestamp: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run all datasets, lag orders, and residual models, then save outputs."""
    active_timestamp = run_timestamp or build_run_timestamp()
    run_paths = prepare_run_output_paths(active_timestamp)

    metrics_rows = []
    params_rows = []

    for dataset in datasets:
        csv_path = Path(DATA_DIR) / f"{dataset}.csv"
        if not csv_path.exists():
            print(f"Skip {dataset}: file not found at {csv_path}")
            continue

        df = load_daily_csv(csv_path, start_date=start_date, end_date=end_date)
        y = df["log_return"].to_numpy()

        plot_log_return_series(
            df,
            dataset,
            run_paths.figures["common"] / figure_name(dataset, "common", "log_return_ts", active_timestamp),
        )
        plot_rolling_volatility(
            df,
            dataset,
            run_paths.figures["common"] / figure_name(dataset, "common", "rolling_volatility", active_timestamp),
            window=rolling_window,
        )

        for p in p_values:
            gaussian = fit_gaussian_ar(y, p=p)
            student = fit_student_t_ar(y, p=p, nu=nu)
            mixture = fit_gaussian_mixture_ar(
                y,
                p=p,
                n_components=n_components,
                random_state=random_state,
            )

            model_results = [gaussian, student, mixture]
            for res in model_results:
                metrics_rows.append(
                    {
                        "dataset": dataset,
                        "model": res["model"],
                        "p": p,
                        "log_likelihood": res["log_likelihood"],
                        "aic": res["aic"],
                        "bic": res["bic"],
                        "n_obs": res["n_obs"],
                        "nu": res.get("nu"),
                    }
                )
                params_rows.append(_phi_record(dataset, res["model"], p, res["phi"]))

            for family_name, res in [
                ("gaussian", gaussian),
                ("student_t", student),
                ("gaussian_mixture", mixture),
            ]:
                plot_residual_series(
                    res["residuals"],
                    dataset,
                    family_name,
                    run_paths.figures[family_name]
                    / figure_name(dataset, family_name, f"p{p}_residual_ts", active_timestamp),
                )
                plot_residual_hist_with_density(
                    res["residuals"],
                    dataset,
                    family_name,
                    run_paths.figures[family_name]
                    / figure_name(dataset, family_name, f"p{p}_residual_hist_density", active_timestamp),
                    nu=nu if family_name == "student_t" else None,
                )
                plot_qq_gaussian(
                    res["residuals"],
                    dataset,
                    family_name,
                    run_paths.figures[family_name]
                    / figure_name(dataset, family_name, f"p{p}_qq_gaussian", active_timestamp),
                )
                plot_qq_student_t(
                    res["residuals"],
                    dataset,
                    family_name,
                    nu=nu,
                    output_path=run_paths.figures[family_name]
                    / figure_name(dataset, family_name, f"p{p}_qq_studentt", active_timestamp),
                )

            if "responsibilities" in mixture:
                resp_df = pd.DataFrame(mixture["responsibilities"], columns=[f"component_{i+1}" for i in range(n_components)])
                resp_df.insert(0, "dataset", dataset)
                resp_df.insert(1, "p", p)
                save_dataframe(
                    resp_df,
                    run_paths.params["gaussian_mixture"]
                    / table_name(dataset, "gaussian_mixture", f"p{p}_responsibilities", active_timestamp),
                )

    metrics_df = pd.DataFrame(metrics_rows)
    params_df = pd.DataFrame(params_rows)

    if not metrics_df.empty:
        save_dataframe(
            metrics_df,
            run_paths.tables["common"] / table_name("all_assets", "common", "model_comparison", active_timestamp),
        )
    if not params_df.empty:
        save_dataframe(
            params_df,
            run_paths.params["common"] / table_name("all_assets", "common", "ar_parameter_summary", active_timestamp),
        )

    for family_name in ("gaussian", "student_t", "gaussian_mixture"):
        family_metrics = metrics_df[metrics_df["model"] == family_name] if not metrics_df.empty else pd.DataFrame()
        family_params = params_df[params_df["model"] == family_name] if not params_df.empty else pd.DataFrame()

        if not family_metrics.empty:
            save_dataframe(
                family_metrics,
                run_paths.tables[family_name] / table_name("all_assets", family_name, "metrics_summary", active_timestamp),
            )
        if not family_params.empty:
            save_dataframe(
                family_params,
                run_paths.params[family_name] / table_name("all_assets", family_name, "ar_parameter_summary", active_timestamp),
            )

    return {
        "run_timestamp": pd.DataFrame({"run_timestamp": [active_timestamp]}),
        "metrics": metrics_df,
        "ar_parameters": params_df,
    }


if __name__ == "__main__":
    outputs = run_point_estimation_workflow()
    print(outputs["metrics"].head())
