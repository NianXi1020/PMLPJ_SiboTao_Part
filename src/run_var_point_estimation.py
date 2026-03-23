"""Main entrypoint for multivariate VAR point-estimation workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import DATA_DIR, RESULTS_DIR
from src.var_point_estimation.data import DEFAULT_SERIES, load_var_return_frame
from src.var_point_estimation.models import GaussianVARPointModel, MixtureVARPointModel, StudentTVARPointModel
from src.var_point_estimation.output import prepare_var_output_paths, save_csv
from src.var_point_estimation.plots import (
    plot_actual_vs_predicted,
    plot_predictive_distribution,
    plot_residual_hist,
    plot_residual_series,
    plot_var_violations,
)
from src.var_point_estimation.rolling_eval import rolling_forecast_evaluation


def run_var_point_estimation_workflow(
    series_names: list[str] | None = None,
    window_size: int = 500,
    step_size: int = 1,
    n_sim: int = 2000,
    alpha: float = 0.05,
    nu: float = 5.0,
    n_components: int = 2,
    timestamp: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run all VAR point-estimation models and save structured outputs."""
    names = series_names or DEFAULT_SERIES
    data = load_var_return_frame(Path(DATA_DIR), names, start_date=start_date, end_date=end_date)
    output_paths = prepare_var_output_paths(Path(RESULTS_DIR), timestamp=timestamp)

    models = {
        "gaussian": GaussianVARPointModel(target_cols=names),
        "student_t": StudentTVARPointModel(target_cols=names, nu=nu),
        "mixture": MixtureVARPointModel(target_cols=names, n_components=n_components),
    }

    model_scores = []
    for model_name, model in models.items():
        summary_df, detail_df, params_df, final_draws = rolling_forecast_evaluation(
            data=data,
            model=model,
            window_size=window_size,
            step_size=step_size,
            n_sim=n_sim,
            alpha=alpha,
        )

        ts = output_paths.timestamp
        save_csv(summary_df, output_paths.metrics[model_name] / f"prediction_summary_{model_name}_{ts}.csv")
        save_csv(detail_df, output_paths.metrics[model_name] / f"prediction_detail_{model_name}_{ts}.csv")
        save_csv(params_df, output_paths.metrics[model_name] / f"model_parameters_{model_name}_{ts}.csv")

        plot_actual_vs_predicted(detail_df, model_name, output_paths.figures[model_name] / f"actual_vs_predicted_{model_name}_{ts}.png")
        plot_residual_series(detail_df, model_name, output_paths.figures[model_name] / f"residual_series_{model_name}_{ts}.png")
        plot_residual_hist(detail_df, model_name, output_paths.figures[model_name] / f"residual_hist_{model_name}_{ts}.png")
        if len(final_draws) > 0:
            plot_predictive_distribution(
                final_draws,
                series_name=names[0],
                model_name=model_name,
                output_path=output_paths.figures[model_name] / f"predictive_dist_{model_name}_{ts}.png",
            )
        plot_var_violations(detail_df, model_name, output_paths.figures[model_name] / f"var_violations_{model_name}_{ts}.png")

        if not summary_df.empty:
            model_scores.append(
                {
                    "model": model_name,
                    "avg_lps": summary_df["lps"].mean(),
                    "n_forecasts": len(summary_df),
                    "timestamp": ts,
                }
            )

    comparison_df = pd.DataFrame(model_scores)
    if not comparison_df.empty:
        save_csv(comparison_df, output_paths.run_root / f"model_comparison_{output_paths.timestamp}.csv")

    return {
        "data": data,
        "comparison": comparison_df,
    }


if __name__ == "__main__":
    outputs = run_var_point_estimation_workflow()
    print(outputs["comparison"])
