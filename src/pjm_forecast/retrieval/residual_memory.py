from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json

import numpy as np
import pandas as pd

from pjm_forecast.prepared_data import forecast_day_from_prediction_frame


@dataclass(frozen=True)
class RetrievalConfig:
    history_days: int
    price_weight: float
    load_weight: float
    calendar_weight: float
    top_k: int
    min_gap_days: int
    residual_clip_quantile: float
    horizon: int = 24
    load_columns: tuple[str, ...] = ("system_load_forecast", "zonal_load_forecast")
    calendar_columns: tuple[str, ...] = (
        "is_weekend",
        "is_holiday",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "month_sin",
        "month_cos",
    )
    output_model_name: str = "nbeatsx_rag"


@dataclass(frozen=True)
class RetrievalParams:
    alpha: float
    tau: float
    predicted_volatility_threshold: float | None = None


@dataclass
class _MemoryItem:
    forecast_day: pd.Timestamp
    price_block: np.ndarray
    load_block: np.ndarray
    calendar_block: np.ndarray
    residual: np.ndarray


def _forecast_day_from_prediction(day_df: pd.DataFrame) -> pd.Timestamp:
    return forecast_day_from_prediction_frame(day_df)


def _validate_retrieval_inputs(
    feature_df: pd.DataFrame,
    base_predictions: pd.DataFrame,
    initial_memory_predictions: pd.DataFrame,
    config: RetrievalConfig,
) -> None:
    required_feature_columns = ["ds", *config.load_columns, *config.calendar_columns, "y"]
    missing_feature_columns = [column for column in required_feature_columns if column not in feature_df.columns]
    if missing_feature_columns:
        raise ValueError(f"Feature frame is missing retrieval columns: {missing_feature_columns}")

    for frame_name, frame in [("base_predictions", base_predictions), ("initial_memory_predictions", initial_memory_predictions)]:
        missing_prediction_columns = [column for column in ["ds", "y", "y_pred"] if column not in frame.columns]
        if missing_prediction_columns:
            raise ValueError(f"{frame_name} is missing required prediction columns: {missing_prediction_columns}")


def _history_slice(feature_df: pd.DataFrame, forecast_day: pd.Timestamp, history_days: int) -> pd.DataFrame:
    start = forecast_day - pd.Timedelta(days=history_days)
    end = forecast_day - pd.Timedelta(hours=1)
    history = feature_df.loc[(feature_df["ds"] >= start) & (feature_df["ds"] <= end)].copy()
    expected = history_days * 24
    if len(history) != expected:
        raise ValueError(f"Expected {expected} history rows for {forecast_day}, got {len(history)}.")
    return history


def _future_slice(feature_df: pd.DataFrame, forecast_day: pd.Timestamp, horizon: int) -> pd.DataFrame:
    end = forecast_day + pd.Timedelta(hours=horizon - 1)
    future = feature_df.loc[(feature_df["ds"] >= forecast_day) & (feature_df["ds"] <= end)].copy()
    if len(future) != horizon:
        raise ValueError(f"Expected {horizon} future rows for {forecast_day}, got {len(future)}.")
    return future


def _zscore_vector(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    mean = float(values.mean())
    std = float(values.std())
    if std <= 1e-8:
        std = 1.0
    return (values - mean) / std


def _asinh_q95(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    scale = float(np.quantile(np.abs(values), 0.95))
    if scale <= 1e-8:
        scale = 1.0
    return np.arcsinh(values / scale)


def _build_price_block(feature_df: pd.DataFrame, forecast_day: pd.Timestamp, history_days: int) -> np.ndarray:
    history = _history_slice(feature_df, forecast_day, history_days)
    return _zscore_vector(_asinh_q95(history["y"].to_numpy(dtype=float)))


def _build_load_block(feature_df: pd.DataFrame, forecast_day: pd.Timestamp, config: RetrievalConfig) -> np.ndarray:
    future = _future_slice(feature_df, forecast_day, horizon=config.horizon)
    values = [future[column].to_numpy(dtype=float) for column in config.load_columns]
    return _zscore_vector(np.concatenate(values))


def _build_calendar_block(feature_df: pd.DataFrame, forecast_day: pd.Timestamp, config: RetrievalConfig) -> np.ndarray:
    future = _future_slice(feature_df, forecast_day, horizon=config.horizon)
    row = future.iloc[0]
    return np.array([float(row[column]) for column in config.calendar_columns], dtype=float)


def _build_memory_items(
    feature_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    config: RetrievalConfig,
) -> list[_MemoryItem]:
    items: list[_MemoryItem] = []
    for _, day_df in prediction_df.groupby(prediction_df["ds"].dt.normalize(), sort=True):
        forecast_day = _forecast_day_from_prediction(day_df)
        base_pred = day_df.sort_values("ds")["y_pred"].to_numpy(dtype=float)
        truth = day_df.sort_values("ds")["y"].to_numpy(dtype=float)
        items.append(
            _MemoryItem(
                forecast_day=forecast_day,
                price_block=_build_price_block(feature_df, forecast_day, config.history_days),
                load_block=_build_load_block(feature_df, forecast_day, config),
                calendar_block=_build_calendar_block(feature_df, forecast_day, config),
                residual=truth - base_pred,
            )
        )
    return items


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _weighted_similarity(query: _MemoryItem, candidate: _MemoryItem, config: RetrievalConfig) -> float:
    return (
        config.price_weight * _cosine_similarity(query.price_block, candidate.price_block)
        + config.load_weight * _cosine_similarity(query.load_block, candidate.load_block)
        + config.calendar_weight * _cosine_similarity(query.calendar_block, candidate.calendar_block)
    )


def _select_neighbors(query: _MemoryItem, memory: list[_MemoryItem], config: RetrievalConfig) -> list[tuple[float, _MemoryItem]]:
    scored = [( _weighted_similarity(query, item, config), item) for item in memory if item.forecast_day < query.forecast_day]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected: list[tuple[float, _MemoryItem]] = []
    for score, item in scored:
        if len(selected) >= config.top_k:
            break
        if any(abs((item.forecast_day - chosen.forecast_day).days) < config.min_gap_days for _, chosen in selected):
            continue
        selected.append((score, item))
    return selected


def _softmax_weights(scores: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 1e-8:
        raise ValueError("tau must be positive.")
    shifted = scores / tau
    shifted = shifted - shifted.max()
    weights = np.exp(shifted)
    total = float(weights.sum())
    if total <= 1e-8:
        return np.full_like(weights, fill_value=1.0 / len(weights))
    return weights / total


def _clip_residual(delta: np.ndarray, memory: list[_MemoryItem], quantile: float) -> np.ndarray:
    residual_matrix = np.stack([item.residual for item in memory], axis=0)
    bounds = np.quantile(np.abs(residual_matrix), quantile, axis=0)
    bounds = np.where(bounds <= 1e-8, np.inf, bounds)
    return np.clip(delta, -bounds, bounds)


def _correct_day(
    query: _MemoryItem,
    base_day_df: pd.DataFrame,
    memory: list[_MemoryItem],
    config: RetrievalConfig,
    params: RetrievalParams,
) -> pd.DataFrame:
    if params.predicted_volatility_threshold is not None:
        predicted_volatility = float(base_day_df["y_pred"].astype(float).std(ddof=0))
        if predicted_volatility < params.predicted_volatility_threshold:
            corrected = base_day_df.copy()
            corrected["model"] = config.output_model_name
            return corrected

    neighbors = _select_neighbors(query, memory, config)
    if not neighbors:
        corrected = base_day_df.copy()
        corrected["model"] = config.output_model_name
        return corrected

    scores = np.array([score for score, _ in neighbors], dtype=float)
    weights = _softmax_weights(scores, tau=params.tau)
    residuals = np.stack([item.residual for _, item in neighbors], axis=0)
    delta = np.sum(weights[:, None] * residuals, axis=0)
    delta = _clip_residual(delta, memory, quantile=config.residual_clip_quantile)

    corrected = base_day_df.sort_values("ds").copy()
    corrected["y_pred"] = corrected["y_pred"].to_numpy(dtype=float) + params.alpha * delta
    corrected["model"] = config.output_model_name
    return corrected


def apply_residual_retrieval(
    feature_df: pd.DataFrame,
    base_predictions: pd.DataFrame,
    initial_memory_predictions: pd.DataFrame,
    config: RetrievalConfig,
    params: RetrievalParams,
) -> pd.DataFrame:
    _validate_retrieval_inputs(feature_df, base_predictions, initial_memory_predictions, config)
    memory = _build_memory_items(feature_df, initial_memory_predictions, config=config)
    corrected_days: list[pd.DataFrame] = []
    for _, day_df in base_predictions.groupby(base_predictions["ds"].dt.normalize(), sort=True):
        forecast_day = _forecast_day_from_prediction(day_df)
        query = _MemoryItem(
            forecast_day=forecast_day,
            price_block=_build_price_block(feature_df, forecast_day, config.history_days),
            load_block=_build_load_block(feature_df, forecast_day, config),
            calendar_block=_build_calendar_block(feature_df, forecast_day, config),
            residual=np.zeros(config.horizon, dtype=float),
        )
        corrected = _correct_day(query, day_df, memory, config, params)
        corrected_days.append(corrected)
        memory.extend(_build_memory_items(feature_df, day_df, config=config))
    return pd.concat(corrected_days, axis=0, ignore_index=True)


def tune_retrieval_params(
    feature_df: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    initial_memory_predictions: pd.DataFrame,
    config: RetrievalConfig,
    alpha_grid: list[float],
    tau_grid: list[float],
    volatility_quantile_grid: list[float | None],
) -> tuple[RetrievalParams, dict[str, float]]:
    daily_predicted_volatility = (
        validation_predictions.assign(day=validation_predictions["ds"].dt.normalize())
        .groupby("day")["y_pred"]
        .std(ddof=0)
    )
    best_params: RetrievalParams | None = None
    best_mae = float("inf")
    scores: dict[str, float] = {}
    for alpha, tau, volatility_quantile in product(alpha_grid, tau_grid, volatility_quantile_grid):
        threshold = None
        if volatility_quantile is not None:
            threshold = float(daily_predicted_volatility.quantile(float(volatility_quantile)))
        params = RetrievalParams(
            alpha=float(alpha),
            tau=float(tau),
            predicted_volatility_threshold=threshold,
        )
        corrected = apply_residual_retrieval(
            feature_df=feature_df,
            base_predictions=validation_predictions,
            initial_memory_predictions=initial_memory_predictions,
            config=config,
            params=params,
        )
        mae = float(np.mean(np.abs(corrected["y"].to_numpy(dtype=float) - corrected["y_pred"].to_numpy(dtype=float))))
        gate = "none" if volatility_quantile is None else f"{float(volatility_quantile):.2f}"
        key = f"alpha={alpha:.4f},tau={tau:.4f},vol_q={gate}"
        scores[key] = mae
        if mae < best_mae:
            best_mae = mae
            best_params = params
    if best_params is None:
        raise RuntimeError("Failed to tune retrieval parameters.")
    return best_params, scores
