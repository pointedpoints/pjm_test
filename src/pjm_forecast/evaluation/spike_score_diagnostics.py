from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecast.prediction_contract import point_prediction_view


def compute_spike_score_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "spike_score",
    threshold: float = 0.67,
) -> dict[str, float | bool | int]:
    point_view = point_prediction_view(predictions)
    row_count = int(len(point_view))
    missing_result: dict[str, float | bool | int] = {
        "has_spike_score": False,
        "row_count": row_count,
        "non_null_rate": float("nan"),
        "score_min": float("nan"),
        "score_mean": float("nan"),
        "score_p50": float("nan"),
        "score_p90": float("nan"),
        "score_p95": float("nan"),
        "score_max": float("nan"),
        "regime_threshold": float(threshold),
        "spike_share": float("nan"),
        "normal_count": 0,
        "spike_count": 0,
        "normal_p50_mae": float("nan"),
        "spike_p50_mae": float("nan"),
        "normal_y_mean": float("nan"),
        "spike_y_mean": float("nan"),
    }
    if score_column not in point_view.columns:
        return missing_result

    scores = point_view[score_column].astype(float)
    valid = scores.notna()
    if not valid.any():
        result = dict(missing_result)
        result["has_spike_score"] = True
        result["non_null_rate"] = 0.0
        return result

    point_valid = point_view.loc[valid].copy()
    valid_scores = scores.loc[valid]
    spike_mask = valid_scores >= float(threshold)
    normal_mask = ~spike_mask
    abs_error = (point_valid["y"].astype(float) - point_valid["y_pred"].astype(float)).abs()

    return {
        "has_spike_score": True,
        "row_count": row_count,
        "non_null_rate": float(valid.mean()),
        "score_min": float(valid_scores.min()),
        "score_mean": float(valid_scores.mean()),
        "score_p50": float(valid_scores.quantile(0.50)),
        "score_p90": float(valid_scores.quantile(0.90)),
        "score_p95": float(valid_scores.quantile(0.95)),
        "score_max": float(valid_scores.max()),
        "regime_threshold": float(threshold),
        "spike_share": float(spike_mask.mean()),
        "normal_count": int(normal_mask.sum()),
        "spike_count": int(spike_mask.sum()),
        "normal_p50_mae": _masked_mean(abs_error, normal_mask),
        "spike_p50_mae": _masked_mean(abs_error, spike_mask),
        "normal_y_mean": _masked_mean(point_valid["y"].astype(float), normal_mask),
        "spike_y_mean": _masked_mean(point_valid["y"].astype(float), spike_mask),
    }


def _masked_mean(values: pd.Series, mask: pd.Series) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float(np.mean(values.loc[mask].to_numpy(dtype=float)))
