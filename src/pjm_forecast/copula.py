from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy import stats

from pjm_forecast.quantile_surface import QuantileSurface, quantile_surfaces_from_frame


COPULA_EPSILON = 1e-6


@dataclass(frozen=True)
class ScenarioMarginals:
    forecast_day: pd.Timestamp
    ds_index: pd.DatetimeIndex
    surfaces: tuple[QuantileSurface, ...]
    observation: np.ndarray | None = None

    def ppf(self, pseudo_observations: np.ndarray) -> np.ndarray:
        uniforms = np.asarray(pseudo_observations, dtype=float)
        if uniforms.ndim == 1:
            if uniforms.shape[0] != len(self.surfaces):
                raise ValueError("pseudo_observations must match the marginal horizon length.")
            return np.asarray(
                [surface.ppf(uniform) for surface, uniform in zip(self.surfaces, uniforms, strict=True)],
                dtype=float,
            )
        if uniforms.ndim != 2 or uniforms.shape[1] != len(self.surfaces):
            raise ValueError("pseudo_observations must have shape (n_samples, horizon).")
        columns = [surface.ppf(uniforms[:, index]) for index, surface in enumerate(self.surfaces)]
        return np.column_stack(columns)

    def sample_paths(
        self,
        copula: BaseCopula,
        n_samples: int,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        uniforms = copula.sample(n_samples, random_state=random_state)
        return self.ppf(uniforms)


@dataclass(frozen=True)
class QuantileSurfacePanel:
    forecast_days: pd.DatetimeIndex
    horizon_offsets: pd.TimedeltaIndex
    observations: np.ndarray
    surfaces_by_day: tuple[tuple[QuantileSurface, ...], ...]

    def __post_init__(self) -> None:
        if len(self.forecast_days) != len(self.surfaces_by_day):
            raise ValueError("forecast_days and surfaces_by_day must align.")
        if self.observations.shape != (len(self.forecast_days), len(self.horizon_offsets)):
            raise ValueError("observations must have shape (n_days, horizon).")

    @property
    def horizon(self) -> int:
        return len(self.horizon_offsets)

    def pseudo_observations(self) -> np.ndarray:
        rows = []
        for row_index, surfaces in enumerate(self.surfaces_by_day):
            rows.append(
                [
                    float(surface.pit(self.observations[row_index, column_index]))
                    for column_index, surface in enumerate(surfaces)
                ]
            )
        return _clip_unit_interval(np.asarray(rows, dtype=float))

    def marginals_for_day(self, forecast_day: pd.Timestamp | str) -> ScenarioMarginals:
        normalized = pd.Timestamp(forecast_day).normalize()
        for index, candidate in enumerate(self.forecast_days):
            if candidate == normalized:
                return ScenarioMarginals(
                    forecast_day=candidate,
                    ds_index=pd.DatetimeIndex(candidate + self.horizon_offsets),
                    surfaces=self.surfaces_by_day[index],
                    observation=self.observations[index].copy(),
                )
        raise KeyError(f"forecast_day {normalized} is not present in the panel.")


@dataclass(frozen=True)
class BaseCopula:
    correlation: np.ndarray

    def __post_init__(self) -> None:
        correlation = np.asarray(self.correlation, dtype=float)
        if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
            raise ValueError("correlation must be a square matrix.")
        object.__setattr__(self, "correlation", _nearest_correlation(correlation))

    @property
    def dimension(self) -> int:
        return int(self.correlation.shape[0])

    def sample(
        self,
        n_samples: int,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def log_likelihood(self, pseudo_observations: np.ndarray) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class GaussianCopula(BaseCopula):
    @classmethod
    def fit(cls, pseudo_observations: np.ndarray) -> GaussianCopula:
        uniforms = _clip_unit_interval(np.asarray(pseudo_observations, dtype=float))
        latent = stats.norm.ppf(uniforms)
        return cls(correlation=_estimate_correlation(latent))

    def sample(
        self,
        n_samples: int,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        latent = rng.multivariate_normal(np.zeros(self.dimension), self.correlation, size=n_samples)
        return _clip_unit_interval(stats.norm.cdf(latent))

    def log_likelihood(self, pseudo_observations: np.ndarray) -> float:
        uniforms = _clip_unit_interval(np.asarray(pseudo_observations, dtype=float))
        latent = stats.norm.ppf(uniforms)
        mvn = stats.multivariate_normal(mean=np.zeros(self.dimension), cov=self.correlation, allow_singular=True)
        joint = mvn.logpdf(latent)
        marginal = np.sum(stats.norm.logpdf(latent), axis=1)
        return float(np.sum(joint - marginal))


@dataclass(frozen=True)
class StudentTCopula(BaseCopula):
    degrees_of_freedom: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if float(self.degrees_of_freedom) <= 2.0:
            raise ValueError("degrees_of_freedom must be > 2.")
        object.__setattr__(self, "degrees_of_freedom", float(self.degrees_of_freedom))

    @classmethod
    def fit(
        cls,
        pseudo_observations: np.ndarray,
        *,
        dof_grid: list[float] | np.ndarray | None = None,
    ) -> StudentTCopula:
        uniforms = _clip_unit_interval(np.asarray(pseudo_observations, dtype=float))
        candidates = np.asarray(dof_grid if dof_grid is not None else [3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0], dtype=float)
        best_model: StudentTCopula | None = None
        best_score = float("-inf")
        for dof in candidates:
            latent = stats.t.ppf(uniforms, df=float(dof))
            correlation = _estimate_correlation(latent)
            model = cls(correlation=correlation, degrees_of_freedom=float(dof))
            score = model.log_likelihood(uniforms)
            if score > best_score:
                best_model = model
                best_score = score
        if best_model is None:
            raise ValueError("StudentTCopula.fit could not select a valid model.")
        return best_model

    def sample(
        self,
        n_samples: int,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        samples = stats.multivariate_t(
            loc=np.zeros(self.dimension),
            shape=self.correlation,
            df=self.degrees_of_freedom,
        ).rvs(size=n_samples, random_state=rng)
        samples_2d = np.atleast_2d(samples)
        return _clip_unit_interval(stats.t.cdf(samples_2d, df=self.degrees_of_freedom))

    def log_likelihood(self, pseudo_observations: np.ndarray) -> float:
        uniforms = _clip_unit_interval(np.asarray(pseudo_observations, dtype=float))
        latent = stats.t.ppf(uniforms, df=self.degrees_of_freedom)
        joint = stats.multivariate_t(
            loc=np.zeros(self.dimension),
            shape=self.correlation,
            df=self.degrees_of_freedom,
        ).logpdf(latent)
        marginal = np.sum(stats.t.logpdf(latent, df=self.degrees_of_freedom), axis=1)
        return float(np.sum(joint - marginal))


def build_quantile_surface_panel(predictions: pd.DataFrame) -> QuantileSurfacePanel:
    grouped_days: list[pd.Timestamp] = []
    grouped_observations: list[np.ndarray] = []
    grouped_surfaces: list[tuple[QuantileSurface, ...]] = []
    expected_offsets: pd.TimedeltaIndex | None = None

    forecast_days = _forecast_day_series(predictions)
    for forecast_day, day_frame in predictions.groupby(forecast_days, sort=True):
        forecast_day = pd.Timestamp(forecast_day).normalize()
        surfaces = quantile_surfaces_from_frame(day_frame)
        ds_index = pd.DatetimeIndex(sorted(surfaces))
        offsets = pd.TimedeltaIndex(ds_index - forecast_day)
        if expected_offsets is None:
            expected_offsets = offsets
        elif len(offsets) != len(expected_offsets) or not offsets.equals(expected_offsets):
            raise ValueError("prediction frame must contain the same ordered horizon offsets for each forecast day.")
        grouped_days.append(forecast_day)
        grouped_surfaces.append(tuple(surfaces[ds] for ds in ds_index))
        grouped_observations.append(day_frame.groupby("ds", sort=True)["y"].first().reindex(ds_index).to_numpy(dtype=float))

    if expected_offsets is None:
        raise ValueError("prediction frame does not contain any forecast days.")

    return QuantileSurfacePanel(
        forecast_days=pd.DatetimeIndex(grouped_days),
        horizon_offsets=expected_offsets,
        observations=np.vstack(grouped_observations),
        surfaces_by_day=tuple(grouped_surfaces),
    )


def fit_copula_from_predictions(
    predictions: pd.DataFrame,
    *,
    family: str = "student_t",
    dof_grid: list[float] | np.ndarray | None = None,
) -> tuple[BaseCopula, QuantileSurfacePanel]:
    panel = build_quantile_surface_panel(predictions)
    uniforms = panel.pseudo_observations()
    family_name = str(family).lower()
    if family_name == "gaussian":
        return GaussianCopula.fit(uniforms), panel
    if family_name in {"student_t", "t"}:
        return StudentTCopula.fit(uniforms, dof_grid=dof_grid), panel
    raise ValueError(f"Unsupported copula family: {family!r}")


def sample_copula_scenarios(
    copula: BaseCopula,
    marginals: ScenarioMarginals,
    n_samples: int,
    *,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    return marginals.sample_paths(copula, n_samples, random_state=random_state)


def energy_score(observation: np.ndarray, scenarios: np.ndarray) -> float:
    observed = np.asarray(observation, dtype=float)
    sampled = np.asarray(scenarios, dtype=float)
    if sampled.ndim != 2 or sampled.shape[1] != observed.shape[0]:
        raise ValueError("scenarios must have shape (n_samples, dimension).")
    term_1 = np.mean(np.linalg.norm(sampled - observed, axis=1))
    pairwise = sampled[:, None, :] - sampled[None, :, :]
    term_2 = 0.5 * np.mean(np.linalg.norm(pairwise, axis=2))
    return float(term_1 - term_2)


def variogram_score(observation: np.ndarray, scenarios: np.ndarray, *, power: float = 0.5) -> float:
    observed = np.asarray(observation, dtype=float)
    sampled = np.asarray(scenarios, dtype=float)
    if sampled.ndim != 2 or sampled.shape[1] != observed.shape[0]:
        raise ValueError("scenarios must have shape (n_samples, dimension).")
    total = 0.0
    dimension = observed.shape[0]
    for first in range(dimension):
        for second in range(first + 1, dimension):
            observed_diff = abs(observed[first] - observed[second]) ** power
            scenario_diff = np.mean(np.abs(sampled[:, first] - sampled[:, second]) ** power)
            total += float((observed_diff - scenario_diff) ** 2)
    return total


def _clip_unit_interval(values: np.ndarray) -> np.ndarray:
    return np.clip(values, COPULA_EPSILON, 1.0 - COPULA_EPSILON)


def _estimate_correlation(latent: np.ndarray) -> np.ndarray:
    values = np.asarray(latent, dtype=float)
    if values.ndim != 2:
        raise ValueError("latent values must have shape (n_samples, dimension).")
    if values.shape[0] == 1:
        return np.eye(values.shape[1], dtype=float)
    centered = values - np.mean(values, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    std = np.sqrt(np.diag(covariance))
    scale = np.outer(std, std)
    correlation = np.divide(covariance, scale, out=np.zeros_like(covariance), where=scale > 0)
    np.fill_diagonal(correlation, 1.0)
    return _nearest_correlation(correlation)


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    symmetric = (values + values.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, 1e-6, None)
    projected = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    scale = np.sqrt(np.diag(projected))
    normalized = projected / np.outer(scale, scale)
    np.fill_diagonal(normalized, 1.0)
    return normalized


def _forecast_day_series(predictions: pd.DataFrame) -> pd.Series:
    if "metadata" not in predictions.columns:
        return pd.to_datetime(predictions["ds"]).dt.normalize()

    derived: list[pd.Timestamp] = []
    for _, row in predictions.iterrows():
        payload = row.get("metadata")
        if isinstance(payload, str):
            try:
                value = json.loads(payload).get("forecast_day")
            except Exception:
                value = None
            if value is not None:
                derived.append(pd.Timestamp(value).normalize())
                continue
        derived.append(pd.Timestamp(row["ds"]).normalize())
    return pd.Series(derived, index=predictions.index)
