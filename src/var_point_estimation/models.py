"""Multivariate VAR(2) point-estimation models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.var_point_estimation.data import build_var2_design


_EPS = 1e-8


def _safe_cov(x: np.ndarray) -> np.ndarray:
    cov = np.cov(x, rowvar=False, bias=True)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    return cov + _EPS * np.eye(cov.shape[0])


def _mvn_logpdf(samples: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    dim = samples.shape[1]
    cov_reg = cov + _EPS * np.eye(dim)
    sign, logdet = np.linalg.slogdet(cov_reg)
    if sign <= 0:
        cov_reg = cov_reg + 1e-4 * np.eye(dim)
        sign, logdet = np.linalg.slogdet(cov_reg)
    inv = np.linalg.inv(cov_reg)
    centered = samples - mean
    quad = np.einsum("ti,ij,tj->t", centered, inv, centered)
    return -0.5 * (dim * np.log(2 * np.pi) + logdet + quad)


@dataclass
class BaseVARPointModel:
    """Base API for VAR point-estimation models."""

    target_cols: list[str]
    lag_order: int = 2

    def fit(self, data: pd.DataFrame) -> "BaseVARPointModel":  # pragma: no cover - interface
        raise NotImplementedError

    def forecast_one_step(self, last_obs: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def simulate_one_step(self, last_obs: np.ndarray, n_sim: int = 1000, random_state: int | None = None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def summary(self) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError


class GaussianVARPointModel(BaseVARPointModel):
    """Gaussian VAR(2) estimated with OLS."""

    def __init__(self, target_cols: list[str], lag_order: int = 2) -> None:
        super().__init__(target_cols=target_cols, lag_order=lag_order)
        self.beta_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None

    def fit(self, data: pd.DataFrame) -> "GaussianVARPointModel":
        x, y, _ = build_var2_design(data, self.target_cols)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        self.beta_ = beta
        self.sigma_ = _safe_cov(resid)
        self.residuals_ = resid
        return self

    def _features_from_last_obs(self, last_obs: np.ndarray) -> np.ndarray:
        if last_obs.shape != (2, len(self.target_cols)):
            raise ValueError("last_obs must have shape (2, n_series).")
        return np.concatenate([[1.0], last_obs[-1], last_obs[-2]])

    def forecast_one_step(self, last_obs: np.ndarray) -> np.ndarray:
        x = self._features_from_last_obs(last_obs)
        return x @ self.beta_

    def simulate_one_step(self, last_obs: np.ndarray, n_sim: int = 1000, random_state: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        mean = self.forecast_one_step(last_obs)
        noise = rng.multivariate_normal(np.zeros(len(self.target_cols)), self.sigma_, size=n_sim)
        return mean + noise

    def summary(self) -> pd.DataFrame:
        beta_df = pd.DataFrame(self.beta_, columns=self.target_cols)
        beta_df.insert(0, "parameter", ["intercept", *[f"lag1_{c}" for c in self.target_cols], *[f"lag2_{c}" for c in self.target_cols]])
        beta_long = beta_df.melt(id_vars=["parameter"], var_name="target", value_name="value")

        sigma_df = pd.DataFrame(self.sigma_, index=self.target_cols, columns=self.target_cols)
        sigma_long = sigma_df.reset_index().melt(id_vars=["index"], var_name="target", value_name="value")
        sigma_long = sigma_long.rename(columns={"index": "parameter"})
        sigma_long["parameter"] = sigma_long["parameter"].map(lambda x: f"sigma_{x}")

        return pd.concat([beta_long, sigma_long], ignore_index=True)


class StudentTVARPointModel(GaussianVARPointModel):
    """Student-t VAR(2) estimated by iterative reweighting."""

    def __init__(self, target_cols: list[str], lag_order: int = 2, nu: float = 5.0, max_iter: int = 100, tol: float = 1e-6) -> None:
        super().__init__(target_cols=target_cols, lag_order=lag_order)
        if nu <= 2:
            raise ValueError("nu must be > 2.")
        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol
        self.weights_: np.ndarray | None = None

    def fit(self, data: pd.DataFrame) -> "StudentTVARPointModel":
        x, y, _ = build_var2_design(data, self.target_cols)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        sigma = _safe_cov(resid)

        dim = y.shape[1]
        for _ in range(self.max_iter):
            inv_sigma = np.linalg.inv(sigma)
            maha = np.einsum("ti,ij,tj->t", resid, inv_sigma, resid)
            w = (self.nu + dim) / (self.nu + maha)

            xw = x * np.sqrt(w[:, None])
            yw = y * np.sqrt(w[:, None])
            beta_new, *_ = np.linalg.lstsq(xw, yw, rcond=None)
            resid_new = y - x @ beta_new
            sigma_new = (resid_new.T @ (resid_new * w[:, None])) / len(resid_new)
            sigma_new += _EPS * np.eye(dim)

            if np.max(np.abs(beta_new - beta)) < self.tol and np.max(np.abs(sigma_new - sigma)) < self.tol:
                beta, resid, sigma = beta_new, resid_new, sigma_new
                self.weights_ = w
                break

            beta, resid, sigma = beta_new, resid_new, sigma_new
            self.weights_ = w

        self.beta_ = beta
        self.sigma_ = sigma
        self.residuals_ = resid
        return self

    def simulate_one_step(self, last_obs: np.ndarray, n_sim: int = 1000, random_state: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        mean = self.forecast_one_step(last_obs)
        dim = len(self.target_cols)
        chi2 = rng.chisquare(self.nu, size=n_sim)
        scales = np.sqrt(self.nu / np.maximum(chi2, _EPS))
        z = rng.multivariate_normal(np.zeros(dim), self.sigma_, size=n_sim)
        return mean + z * scales[:, None]

    def summary(self) -> pd.DataFrame:
        out = super().summary()
        out["nu"] = self.nu
        return out


class MixtureVARPointModel(GaussianVARPointModel):
    """Gaussian-mixture innovation VAR(2) with shared coefficient matrix."""

    def __init__(self, target_cols: list[str], lag_order: int = 2, n_components: int = 2, max_iter: int = 100, tol: float = 1e-6) -> None:
        super().__init__(target_cols=target_cols, lag_order=lag_order)
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights_mix_: np.ndarray | None = None
        self.sigmas_mix_: np.ndarray | None = None
        self.responsibilities_: np.ndarray | None = None

    def fit(self, data: pd.DataFrame) -> "MixtureVARPointModel":
        x, y, _ = build_var2_design(data, self.target_cols)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        n, dim = resid.shape

        pi = np.ones(self.n_components) / self.n_components
        base_cov = _safe_cov(resid)
        covs = np.stack([base_cov * (1.0 + 0.2 * k) for k in range(self.n_components)], axis=0)
        resp = np.ones((n, self.n_components)) / self.n_components

        for _ in range(self.max_iter):
            log_prob = np.column_stack(
                [np.log(np.clip(pi[k], _EPS, None)) + _mvn_logpdf(resid, np.zeros(dim), covs[k]) for k in range(self.n_components)]
            )
            log_norm = log_prob - log_prob.max(axis=1, keepdims=True)
            prob = np.exp(log_norm)
            resp_new = prob / np.clip(prob.sum(axis=1, keepdims=True), _EPS, None)

            weights_obs = resp_new.sum(axis=1)
            xw = x * np.sqrt(weights_obs[:, None])
            yw = y * np.sqrt(weights_obs[:, None])
            beta_new, *_ = np.linalg.lstsq(xw, yw, rcond=None)
            resid_new = y - x @ beta_new

            nk = resp_new.sum(axis=0)
            pi_new = nk / n
            covs_new = []
            for k in range(self.n_components):
                weighted = resid_new * resp_new[:, [k]]
                cov_k = (weighted.T @ resid_new) / max(nk[k], _EPS)
                covs_new.append(cov_k + _EPS * np.eye(dim))
            covs_new = np.stack(covs_new, axis=0)

            if (
                np.max(np.abs(beta_new - beta)) < self.tol
                and np.max(np.abs(pi_new - pi)) < self.tol
                and np.max(np.abs(covs_new - covs)) < 1e-4
            ):
                beta, resid, pi, covs, resp = beta_new, resid_new, pi_new, covs_new, resp_new
                break

            beta, resid, pi, covs, resp = beta_new, resid_new, pi_new, covs_new, resp_new

        self.beta_ = beta
        self.sigma_ = _safe_cov(resid)
        self.residuals_ = resid
        self.weights_mix_ = pi
        self.sigmas_mix_ = covs
        self.responsibilities_ = resp
        return self

    def simulate_one_step(self, last_obs: np.ndarray, n_sim: int = 1000, random_state: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        mean = self.forecast_one_step(last_obs)
        comp = rng.choice(self.n_components, size=n_sim, p=self.weights_mix_)
        draws = np.zeros((n_sim, len(self.target_cols)))
        for k in range(self.n_components):
            idx = np.where(comp == k)[0]
            if len(idx) == 0:
                continue
            draws[idx] = rng.multivariate_normal(np.zeros(len(self.target_cols)), self.sigmas_mix_[k], size=len(idx))
        return mean + draws

    def summary(self) -> pd.DataFrame:
        out = super().summary()
        mix_rows = []
        for k, w in enumerate(self.weights_mix_):
            mix_rows.append({"parameter": f"mix_weight_{k+1}", "target": "all", "value": w})
        return pd.concat([out, pd.DataFrame(mix_rows)], ignore_index=True)


def predictive_lps(y_true: np.ndarray, draws: np.ndarray) -> float:
    """Approximate log predictive score from simulated draws."""
    mean = draws.mean(axis=0)
    cov = _safe_cov(draws)
    return float(_mvn_logpdf(y_true[None, :], mean, cov)[0])
