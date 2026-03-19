# Point Estimation Baseline for Univariate Financial Time Series

This repository provides a clean and extensible point-estimation baseline for univariate AR(p) modeling with:

- Gaussian residuals
- Student-t residuals (fixed degrees of freedom)
- Gaussian-mixture residuals (EM, default `K=2`)

## Project Layout

```text
project_root/
├─ test_data/
│  └─ Daily/                     # Keep this folder; add DIA/IWN/QQQ/SPY CSV files manually
├─ notebooks/
│  └─ point_estimation_workflow.ipynb
├─ src/
│  ├─ data_utils/
│  ├─ point_models/
│  ├─ optimizers/
│  ├─ plotting/
│  └─ utils/
└─ results/
   ├─ figures/
   ├─ tables/
   └─ fitted_params/
```

## Data Requirements

Expected input CSV column names (Chinese source columns are auto-renamed):

- 日期, 开盘价, 最高价, 最低价, 收盘价, 成交量(股), 涨跌额, 涨跌幅(%)

Pipeline behavior:

1. Rename to English columns.
2. Parse and sort by date ascending.
3. Remove duplicate dates.
4. Handle missing values.
5. Compute daily log returns from close prices:
   `r_t = 100 * (log(P_t) - log(P_{t-1}))`.

## Usage

Run the notebook or script:

```bash
python -m src.run_point_estimation
```

The workflow evaluates `p=1..5` for each model and dataset, then saves:

- Figures as PNG files under `results/figures/`
- Summary metrics under `results/tables/model_metrics_summary.csv`
- AR parameters and mixture responsibilities under `results/fitted_params/`

## Notes

- Random operations (mixture initialization) support a reproducible random seed.
- Plot labels, titles, and printed outputs are in English.
- This stage intentionally excludes Bayesian inference and multivariate modeling.
