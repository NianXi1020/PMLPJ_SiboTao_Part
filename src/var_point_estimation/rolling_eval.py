"""Rolling evaluation for VAR point-estimation models."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.var_point_estimation.data import build_last_obs_matrix
from src.var_point_estimation.models import BaseVARPointModel, predictive_lps


def _to_json_list(x: np.ndarray) -> str:
    return json.dumps([float(v) for v in x], ensure_ascii=False)


def rolling_forecast_evaluation(
    data: pd.DataFrame,
    model: BaseVARPointModel,
    window_size: int = 500,
    step_size: int = 1,
    n_sim: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Run rolling fit/simulate/evaluate for one model.

    Returns summary_df, detail_df, params_df, and final predictive draws.
    """
    target_cols = model.target_cols
    n = len(data)
    rows_summary: list[dict] = []
    rows_detail: list[dict] = []
    last_draws = np.empty((0, len(target_cols)))

    for start in range(0, n - window_size - 1, step_size):
        end = start + window_size
        train = data.iloc[start:end].reset_index(drop=True)
        test_idx = end
        test_row = data.iloc[test_idx]

        model.fit(train)
        last_obs = build_last_obs_matrix(train, target_cols)
        draws = model.simulate_one_step(last_obs, n_sim=n_sim, random_state=random_state + start)
        last_draws = draws

        y_true = test_row[target_cols].to_numpy(dtype=float)
        pred_mean = draws.mean(axis=0)
        pred_std = draws.std(axis=0, ddof=0)
        var_alpha = np.quantile(draws, alpha, axis=0)
        es_alpha = np.array(
            [draws[draws[:, j] <= var_alpha[j], j].mean() if np.any(draws[:, j] <= var_alpha[j]) else var_alpha[j] for j in range(len(target_cols))]
        )
        lps = predictive_lps(y_true, draws)
        violations = y_true < var_alpha

        rows_summary.append(
            {
                "date": test_row["date"],
                "y_true": _to_json_list(y_true),
                "pred_mean": _to_json_list(pred_mean),
                "pred_std": _to_json_list(pred_std),
                "var_alpha": _to_json_list(var_alpha),
                "es_alpha": _to_json_list(es_alpha),
                "lps": lps,
            }
        )

        for j, series in enumerate(target_cols):
            rows_detail.append(
                {
                    "date": test_row["date"],
                    "series": series,
                    "y_true": y_true[j],
                    "pred_mean": pred_mean[j],
                    "pred_std": pred_std[j],
                    "var_alpha": var_alpha[j],
                    "es_alpha": es_alpha[j],
                    "lps": lps,
                    "violation": bool(violations[j]),
                }
            )

    summary_df = pd.DataFrame(rows_summary)
    detail_df = pd.DataFrame(rows_detail)
    params_df = model.summary()
    return summary_df, detail_df, params_df, last_draws
