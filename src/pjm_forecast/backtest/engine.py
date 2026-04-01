from __future__ import annotations
from pathlib import Path
from typing import Callable

import pandas as pd

from pjm_forecast.config import ProjectConfig
from pjm_forecast.model_io import validate_model_prediction_output
from pjm_forecast.models.base import ForecastModel
from pjm_forecast.prepared_data import FeatureSchema, forecast_day_from_prediction_frame, prediction_metadata


def _window_slice(feature_df: pd.DataFrame, forecast_day: pd.Timestamp, window_days: int) -> pd.DataFrame:
    history_end = forecast_day - pd.Timedelta(hours=1)
    window_start = forecast_day - pd.Timedelta(days=window_days)
    mask = (feature_df["ds"] >= window_start) & (feature_df["ds"] <= history_end)
    return feature_df.loc[mask].copy()


def _future_slice(feature_df: pd.DataFrame, forecast_day: pd.Timestamp, horizon: int) -> pd.DataFrame:
    future_end = forecast_day + pd.Timedelta(hours=horizon - 1)
    mask = (feature_df["ds"] >= forecast_day) & (feature_df["ds"] <= future_end)
    future_df = feature_df.loc[mask].copy()
    if len(future_df) != horizon:
        raise ValueError(f"Expected {horizon} rows for future horizon, got {len(future_df)} on {forecast_day}.")
    return future_df


def _retrain_due(forecast_day: pd.Timestamp, retrain_weekday: int, existing_model: object | None) -> bool:
    return existing_model is None or forecast_day.weekday() == retrain_weekday


def _latest_retrain_anchor_day(forecast_days: list[pd.Timestamp], retrain_weekday: int) -> pd.Timestamp | None:
    anchor_day: pd.Timestamp | None = None
    for index, forecast_day in enumerate(forecast_days):
        if index == 0 or forecast_day.weekday() == retrain_weekday:
            anchor_day = forecast_day
    return anchor_day


def _chunk_path(output_path: Path, forecast_day: pd.Timestamp) -> Path:
    chunk_dir = output_path.parent / "chunks" / output_path.stem
    chunk_dir.mkdir(parents=True, exist_ok=True)
    return chunk_dir / f"{forecast_day.strftime('%Y-%m-%d')}.parquet"


def _write_chunk(chunk_path: Path, chunk_df: pd.DataFrame) -> None:
    chunk_df.to_parquet(chunk_path, index=False)


def _load_existing_chunk(chunk_path: Path) -> pd.DataFrame:
    return pd.read_parquet(chunk_path)


def _validate_existing_chunk(
    schema: FeatureSchema,
    chunk_df: pd.DataFrame,
    *,
    forecast_day: pd.Timestamp,
    split_name: str,
    model_name: str,
    seed: int,
    horizon: int,
) -> None:
    schema.validate_prediction_frame(chunk_df, require_metadata=True)
    if len(chunk_df) != horizon:
        raise ValueError(f"Existing chunk for {forecast_day} has {len(chunk_df)} rows; expected {horizon}.")
    if str(chunk_df["split"].iloc[0]) != split_name:
        raise ValueError(f"Existing chunk for {forecast_day} has split={chunk_df['split'].iloc[0]!r}, expected {split_name!r}.")
    if str(chunk_df["model"].iloc[0]) != model_name:
        raise ValueError(f"Existing chunk for {forecast_day} has model={chunk_df['model'].iloc[0]!r}, expected {model_name!r}.")
    if int(chunk_df["seed"].iloc[0]) != seed:
        raise ValueError(f"Existing chunk for {forecast_day} has seed={chunk_df['seed'].iloc[0]!r}, expected {seed!r}.")
    if forecast_day_from_prediction_frame(chunk_df) != forecast_day.normalize():
        raise ValueError(f"Existing chunk metadata does not match forecast day {forecast_day}.")
    expected_ds = list(pd.date_range(forecast_day, periods=horizon, freq="h"))
    actual_ds = list(pd.to_datetime(chunk_df["ds"]))
    if actual_ds != expected_ds:
        raise ValueError(f"Existing chunk timestamps do not match the expected horizon for {forecast_day}.")


def _restore_model_for_resume(
    *,
    feature_df: pd.DataFrame,
    forecast_days: list[pd.Timestamp],
    completed_prefix_len: int,
    window_days: int,
    retrain_weekday: int,
    model_builder: Callable[[], ForecastModel],
) -> ForecastModel | None:
    if completed_prefix_len <= 0 or completed_prefix_len >= len(forecast_days):
        return None

    first_pending_day = forecast_days[completed_prefix_len]
    if first_pending_day.weekday() == retrain_weekday:
        return None

    anchor_day = _latest_retrain_anchor_day(forecast_days[:completed_prefix_len], retrain_weekday)
    if anchor_day is None:
        return None

    history_df = _window_slice(feature_df, forecast_day=anchor_day, window_days=window_days)
    if history_df.empty:
        raise ValueError(f"Empty history window for resumed anchor day {anchor_day}.")

    model = model_builder()
    model.fit(history_df)
    return model


def _finalize_output(output_path: Path, forecast_days: list[pd.Timestamp]) -> pd.DataFrame:
    chunk_frames = []
    for forecast_day in forecast_days:
        chunk_path = _chunk_path(output_path, forecast_day)
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing chunk file for forecast day {forecast_day}: {chunk_path}")
        chunk_frames.append(pd.read_parquet(chunk_path))

    result = pd.concat(chunk_frames, axis=0, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def run_rolling_backtest(
    config: ProjectConfig,
    feature_df: pd.DataFrame,
    split_name: str,
    forecast_days: list[pd.Timestamp],
    model_builder: Callable[[], ForecastModel],
    model_name: str,
    seed: int,
    output_path: Path | None = None,
) -> pd.DataFrame:
    horizon = config.backtest["horizon"]
    window_days = config.backtest["rolling_window_days"]
    retrain_weekday = config.backtest["retrain_weekday"]
    schema = FeatureSchema(config)
    model: ForecastModel | None = None
    predictions = []
    completed_prefix_len = 0

    if output_path is not None:
        missing_chunk_seen = False
        for forecast_day in forecast_days:
            chunk_path = _chunk_path(output_path, forecast_day)
            if not chunk_path.exists():
                missing_chunk_seen = True
                continue
            if missing_chunk_seen:
                raise ValueError("Existing chunk files must form a contiguous prefix of forecast_days for safe resume.")
            chunk_df = _load_existing_chunk(chunk_path)
            _validate_existing_chunk(
                schema,
                chunk_df,
                forecast_day=forecast_day,
                split_name=split_name,
                model_name=model_name,
                seed=seed,
                horizon=horizon,
            )
            predictions.append(chunk_df)
            completed_prefix_len += 1
        model = _restore_model_for_resume(
            feature_df=feature_df,
            forecast_days=forecast_days,
            completed_prefix_len=completed_prefix_len,
            window_days=window_days,
            retrain_weekday=retrain_weekday,
            model_builder=model_builder,
        )

    for index, forecast_day in enumerate(forecast_days):
        if index < completed_prefix_len:
            continue

        history_df = _window_slice(feature_df, forecast_day=forecast_day, window_days=window_days)
        future_df = _future_slice(feature_df, forecast_day=forecast_day, horizon=horizon)

        if history_df.empty:
            raise ValueError(f"Empty history window for forecast day {forecast_day}.")

        if _retrain_due(forecast_day, retrain_weekday=retrain_weekday, existing_model=model):
            model = model_builder()
            model.fit(history_df)

        prediction_df = validate_model_prediction_output(
            model.predict(history_df=history_df, future_df=future_df),
            future_df=future_df,
            model_name=model_name,
        )
        merged = future_df.loc[:, ["ds", "y"]].merge(prediction_df, on="ds", how="left")
        merged["model"] = model_name
        merged["split"] = split_name
        merged["seed"] = seed
        merged["quantile"] = pd.NA
        merged["metadata"] = prediction_metadata(forecast_day)
        schema.validate_prediction_frame(merged, require_metadata=True)
        predictions.append(merged)

        if output_path is not None:
            _write_chunk(_chunk_path(output_path, forecast_day), merged)

    result = pd.concat(predictions, axis=0, ignore_index=True)
    if output_path is not None:
        result = _finalize_output(output_path, forecast_days)
    schema.validate_prediction_frame(result, require_metadata=True)
    return result
