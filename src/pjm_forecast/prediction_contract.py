from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


PREDICTION_COLUMNS = ["ds", "y", "y_pred", "model", "split", "seed", "quantile", "metadata"]
MEDIAN_QUANTILE = 0.5


def quantile_values(frame: pd.DataFrame) -> list[float]:
    if "quantile" not in frame.columns:
        return []
    values = frame["quantile"].dropna()
    if values.empty:
        return []
    return sorted(float(value) for value in values.astype(float).drop_duplicates().tolist())


def is_quantile_prediction_frame(frame: pd.DataFrame) -> bool:
    return bool(quantile_values(frame))


def expected_prediction_rows(horizon: int, quantiles: Iterable[float] | None = None) -> int:
    quantile_list = list(quantiles or [])
    return horizon if not quantile_list else horizon * len(quantile_list)


def validate_expected_quantiles(
    frame: pd.DataFrame,
    expected_quantiles: Iterable[float] | None,
    *,
    subject: str,
) -> None:
    normalized_expected = sorted({float(value) for value in (expected_quantiles or [])})
    if not normalized_expected:
        return

    if not any(np.isclose(value, MEDIAN_QUANTILE) for value in normalized_expected):
        raise ValueError(f"{subject} expected quantile grid must include 0.5.")

    actual_quantiles = quantile_values(frame)
    if not actual_quantiles:
        raise ValueError(
            f"{subject} must contain probabilistic quantiles {normalized_expected}; received point predictions instead."
        )

    missing = [value for value in normalized_expected if not any(np.isclose(value, actual) for actual in actual_quantiles)]
    extra = [value for value in actual_quantiles if not any(np.isclose(value, expected) for expected in normalized_expected)]
    if missing or extra:
        raise ValueError(
            f"{subject} quantile grid does not match expected quantiles. "
            f"missing={missing}, extra={extra}, expected={normalized_expected}, actual={actual_quantiles}"
        )


def point_prediction_view(frame: pd.DataFrame, quantile: float = MEDIAN_QUANTILE) -> pd.DataFrame:
    if not is_quantile_prediction_frame(frame):
        point_view = frame.copy()
        if "quantile" in point_view.columns:
            point_view = point_view.loc[point_view["quantile"].isna()].copy()
        return point_view.sort_values("ds").reset_index(drop=True)

    quantile_mask = np.isclose(frame["quantile"].astype(float), quantile)
    if not quantile_mask.any():
        available = quantile_values(frame)
        raise ValueError(f"Prediction frame does not contain quantile={quantile}. Available quantiles: {available}")
    return frame.loc[quantile_mask].copy().sort_values("ds").reset_index(drop=True)


def enforce_monotonic_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    if not is_quantile_prediction_frame(frame):
        return frame.copy()

    corrected = frame.copy().sort_values(["ds", "quantile"]).reset_index(drop=True)
    corrected["y_pred"] = corrected.groupby("ds", sort=False)["y_pred"].cummax()
    return corrected


def normalize_model_prediction_output(
    prediction_df: pd.DataFrame,
    *,
    future_df: pd.DataFrame,
    model_name: str,
    expected_quantiles: Iterable[float] | None = None,
) -> pd.DataFrame:
    required_columns = ["ds", "y_pred"]
    missing_columns = [column for column in required_columns if column not in prediction_df.columns]
    if missing_columns:
        raise ValueError(f"Model '{model_name}' prediction output is missing required columns: {missing_columns}")

    normalized_columns = ["ds", "y_pred"]
    has_quantile = "quantile" in prediction_df.columns
    if has_quantile:
        normalized_columns.insert(1, "quantile")

    normalized = prediction_df.loc[:, normalized_columns].copy()
    normalized["ds"] = pd.to_datetime(normalized["ds"], utc=False)
    if has_quantile:
        normalized["quantile"] = normalized["quantile"].astype(float)
        if normalized["quantile"].isna().any():
            raise ValueError(f"Model '{model_name}' prediction output contains missing quantile values.")
        normalized = normalized.sort_values(["ds", "quantile"]).reset_index(drop=True)
    else:
        normalized["quantile"] = pd.NA
        normalized = normalized.sort_values("ds").reset_index(drop=True)

    if normalized["y_pred"].isna().any():
        raise ValueError(f"Model '{model_name}' prediction output contains missing y_pred values.")

    _validate_prediction_index(normalized, model_name=model_name, future_df=future_df)
    validate_expected_quantiles(
        normalized,
        expected_quantiles,
        subject=f"Model '{model_name}' prediction output",
    )
    return normalized.loc[:, ["ds", "quantile", "y_pred"]]


def validate_canonical_prediction_frame(
    prediction_df: pd.DataFrame,
    require_metadata: bool = True,
    expected_quantiles: Iterable[float] | None = None,
) -> None:
    missing_columns = [column for column in PREDICTION_COLUMNS if column not in prediction_df.columns]
    if missing_columns:
        raise ValueError(f"prediction frame is missing required columns: {missing_columns}")
    if not is_datetime64_any_dtype(prediction_df["ds"]):
        raise ValueError("prediction frame requires a datetime-like 'ds' column.")
    if prediction_df["ds"].dt.tz is not None:
        raise ValueError("prediction frame must use timezone-naive local timestamps in 'ds'.")
    if prediction_df[["y", "y_pred", "model", "split", "seed"]].isna().any().any():
        missing = prediction_df[["y", "y_pred", "model", "split", "seed"]].isna().sum()
        raise ValueError(f"Prediction frame contains missing required values: {missing.to_dict()}")
    if require_metadata and prediction_df["metadata"].isna().any():
        raise ValueError("Prediction frame metadata column must be populated for every row.")
    for column in ["model", "split", "seed"]:
        if prediction_df[column].nunique(dropna=False) > 1:
            raise ValueError(f"Prediction frame column '{column}' must be constant within a run.")

    validate_expected_quantiles(prediction_df, expected_quantiles, subject="prediction frame")

    if not is_quantile_prediction_frame(prediction_df):
        point_df = prediction_df.copy()
        if not point_df["ds"].is_monotonic_increasing:
            raise ValueError("prediction frame must be sorted by 'ds'.")
        if point_df["ds"].duplicated().any():
            raise ValueError("prediction frame contains duplicate 'ds' timestamps.")
        return

    quantile_df = prediction_df.copy()
    quantile_df["quantile"] = quantile_df["quantile"].astype(float)
    expected = quantile_df.sort_values(["ds", "quantile"]).reset_index(drop=True)
    if not expected.equals(quantile_df.reset_index(drop=True)):
        raise ValueError("prediction frame must be sorted by ['ds', 'quantile'] for probabilistic runs.")
    if quantile_df.duplicated(["ds", "quantile"]).any():
        raise ValueError("prediction frame contains duplicate ['ds', 'quantile'] pairs.")

    ds_grid = None
    for quantile in quantile_values(quantile_df):
        quantile_ds = quantile_df.loc[np.isclose(quantile_df["quantile"], quantile), "ds"].reset_index(drop=True)
        if ds_grid is None:
            ds_grid = quantile_ds
            continue
        if len(quantile_ds) != len(ds_grid) or not quantile_ds.equals(ds_grid):
            raise ValueError("prediction frame must contain the same ordered ds horizon for every quantile.")


def _validate_prediction_index(
    prediction_df: pd.DataFrame,
    *,
    model_name: str,
    future_df: pd.DataFrame,
) -> None:
    expected_ds = future_df["ds"].reset_index(drop=True)
    if prediction_df["ds"].dt.tz is not None:
        raise ValueError(f"Model '{model_name}' prediction output must use timezone-naive local timestamps.")

    if not is_quantile_prediction_frame(prediction_df):
        actual_ds = prediction_df["ds"].reset_index(drop=True)
        if prediction_df["ds"].duplicated().any():
            raise ValueError(f"Model '{model_name}' prediction output contains duplicate ds timestamps.")
        if len(actual_ds) != len(expected_ds) or not actual_ds.equals(expected_ds):
            raise ValueError(
                f"Model '{model_name}' prediction timestamps do not align and must exactly match the requested future horizon."
            )
        return

    if prediction_df.duplicated(["ds", "quantile"]).any():
        raise ValueError(f"Model '{model_name}' prediction output contains duplicate ['ds', 'quantile'] pairs.")
    for quantile in quantile_values(prediction_df):
        quantile_df = prediction_df.loc[np.isclose(prediction_df["quantile"], quantile)].reset_index(drop=True)
        actual_ds = quantile_df["ds"]
        if len(actual_ds) != len(expected_ds) or not actual_ds.equals(expected_ds):
            raise ValueError(
                f"Model '{model_name}' quantile={quantile} timestamps do not align and must exactly match the requested future horizon."
            )
