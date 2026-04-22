from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pjm_forecast.copula import (
    StudentTCopula,
    build_quantile_surface_panel,
    energy_score,
    fit_copula_from_predictions,
    sample_copula_scenarios,
    variogram_score,
)
from pjm_forecast.prediction_contract import is_quantile_prediction_frame


def compute_scenario_diagnostics(
    train_predictions: pd.DataFrame | None,
    eval_predictions: pd.DataFrame,
    *,
    family: str = "student_t",
    n_samples: int = 256,
    dof_grid: list[float] | None = None,
    random_seed: int = 7,
) -> dict[str, float | bool | str]:
    diagnostics: dict[str, float | bool | str] = {"has_scenarios": False, "family": str(family).lower()}
    if train_predictions is None or not is_quantile_prediction_frame(train_predictions) or not is_quantile_prediction_frame(eval_predictions):
        diagnostics.update(_empty_scenario_metrics())
        return diagnostics

    copula, train_panel = fit_copula_from_predictions(train_predictions, family=family, dof_grid=dof_grid)
    eval_panel = build_quantile_surface_panel(eval_predictions)
    diagnostics["has_scenarios"] = True
    diagnostics["family"] = "student_t" if isinstance(copula, StudentTCopula) else "gaussian"
    diagnostics["train_days"] = float(len(train_panel.forecast_days))
    diagnostics["eval_days"] = float(len(eval_panel.forecast_days))
    diagnostics["horizon"] = float(eval_panel.horizon)
    diagnostics["n_samples"] = float(n_samples)
    diagnostics["train_log_likelihood"] = float(copula.log_likelihood(train_panel.pseudo_observations()))
    diagnostics["degrees_of_freedom"] = float(copula.degrees_of_freedom) if isinstance(copula, StudentTCopula) else float("nan")

    energy_values: list[float] = []
    variogram_values: list[float] = []
    path_mean_mae_values: list[float] = []
    daily_max_abs_error_values: list[float] = []
    daily_spread_abs_error_values: list[float] = []
    daily_ramp_abs_error_values: list[float] = []
    stacked_scenarios: list[np.ndarray] = []

    for day_index, forecast_day in enumerate(eval_panel.forecast_days):
        marginals = eval_panel.marginals_for_day(forecast_day)
        scenarios = sample_copula_scenarios(
            copula,
            marginals,
            n_samples,
            random_state=int(random_seed) + int(day_index),
        )
        observation = np.asarray(marginals.observation, dtype=float)
        scenario_mean = np.mean(scenarios, axis=0)
        energy_values.append(energy_score(observation, scenarios))
        variogram_values.append(variogram_score(observation, scenarios))
        path_mean_mae_values.append(float(np.mean(np.abs(scenario_mean - observation))))
        daily_max_abs_error_values.append(float(abs(np.mean(np.max(scenarios, axis=1)) - np.max(observation))))
        daily_spread_abs_error_values.append(
            float(abs(np.mean(np.max(scenarios, axis=1) - np.min(scenarios, axis=1)) - (np.max(observation) - np.min(observation))))
        )
        observed_ramp = np.max(np.abs(np.diff(observation)))
        scenario_ramp = np.mean(np.max(np.abs(np.diff(scenarios, axis=1)), axis=1))
        daily_ramp_abs_error_values.append(float(abs(scenario_ramp - observed_ramp)))
        stacked_scenarios.append(scenarios)

    diagnostics["energy_score"] = float(np.mean(energy_values))
    diagnostics["variogram_score"] = float(np.mean(variogram_values))
    diagnostics["path_mean_mae"] = float(np.mean(path_mean_mae_values))
    diagnostics["daily_max_abs_error"] = float(np.mean(daily_max_abs_error_values))
    diagnostics["daily_spread_abs_error"] = float(np.mean(daily_spread_abs_error_values))
    diagnostics["daily_ramp_abs_error"] = float(np.mean(daily_ramp_abs_error_values))
    diagnostics["spearman_corr_mae"] = _spearman_corr_mae(eval_panel.observations, np.vstack(stacked_scenarios))
    return diagnostics


def _empty_scenario_metrics() -> dict[str, float]:
    return {
        "train_days": float("nan"),
        "eval_days": float("nan"),
        "horizon": float("nan"),
        "n_samples": float("nan"),
        "train_log_likelihood": float("nan"),
        "degrees_of_freedom": float("nan"),
        "energy_score": float("nan"),
        "variogram_score": float("nan"),
        "path_mean_mae": float("nan"),
        "daily_max_abs_error": float("nan"),
        "daily_spread_abs_error": float("nan"),
        "daily_ramp_abs_error": float("nan"),
        "spearman_corr_mae": float("nan"),
    }


def _spearman_corr_mae(observations: np.ndarray, scenarios: np.ndarray) -> float:
    observed_frame = pd.DataFrame(np.asarray(observations, dtype=float))
    scenario_frame = pd.DataFrame(np.asarray(scenarios, dtype=float))
    if observed_frame.shape[0] < 2 or scenario_frame.shape[0] < 2:
        return float("nan")
    observed_corr = observed_frame.corr(method="spearman").to_numpy(dtype=float)
    scenario_corr = scenario_frame.corr(method="spearman").to_numpy(dtype=float)
    mask = ~np.eye(observed_corr.shape[0], dtype=bool)
    return float(np.mean(np.abs(observed_corr[mask] - scenario_corr[mask])))
