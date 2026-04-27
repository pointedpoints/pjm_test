from __future__ import annotations

import pandas as pd
import pytest

from pjm_forecast.quantile_postprocess import (
    ALL_GROUP,
    apply_conformal_quantile_calibration,
    apply_median_bias_calibration,
    fit_conformal_quantile_calibration,
    fit_median_bias_calibration,
    parse_interval_coverage_floors,
    postprocess_quantile_predictions,
    symmetric_quantile_pairs,
)


def _frame(
    y_true: list[float],
    q10: list[float],
    q50: list[float],
    q90: list[float],
    *,
    start: str = "2026-01-01 00:00:00",
) -> pd.DataFrame:
    ds = pd.date_range(start, periods=len(y_true), freq="h")
    rows = []
    for index, ts in enumerate(ds):
        for quantile, value in [(0.1, q10[index]), (0.5, q50[index]), (0.9, q90[index])]:
            rows.append(
                {
                    "ds": ts,
                    "y": y_true[index],
                    "y_pred": value,
                    "model": "nbeatsx",
                    "split": "validation",
                    "seed": 7,
                    "quantile": quantile,
                    "metadata": "{}",
                }
            )
    return pd.DataFrame(rows)


def test_symmetric_quantile_pairs_requires_matching_upper_partner() -> None:
    assert symmetric_quantile_pairs([0.1, 0.5, 0.9]) == [(0.1, 0.9)]


def test_parse_interval_coverage_floors_normalizes_string_keys() -> None:
    assert parse_interval_coverage_floors({"0.10-0.90": 0.76}) == {(0.1, 0.9): 0.76}


def test_fit_conformal_quantile_calibration_estimates_interval_adjustment() -> None:
    calibration_frame = _frame(
        y_true=[10.0, 20.0],
        q10=[9.0, 18.0],
        q50=[9.0, 19.0],
        q90=[9.0, 19.0],
    )
    calibration = fit_conformal_quantile_calibration(calibration_frame)
    pair_adjustment = calibration.adjustments[(ALL_GROUP, 0.1, 0.9)]
    assert pair_adjustment.lower_adjustment == 1.0
    assert pair_adjustment.upper_adjustment == 1.0


def test_asymmetric_cqr_fits_independent_tail_adjustments() -> None:
    calibration_frame = _frame(
        y_true=[10.0, 20.0],
        q10=[11.0, 14.0],
        q50=[12.0, 17.0],
        q90=[13.0, 19.5],
    )
    calibration = fit_conformal_quantile_calibration(
        calibration_frame,
        calibration_method="cqr_asymmetric",
    )
    pair_adjustment = calibration.adjustments[(ALL_GROUP, 0.1, 0.9)]
    assert pair_adjustment.lower_adjustment == 1.0
    assert pair_adjustment.upper_adjustment == 0.5


def test_interval_coverage_floor_uses_requested_target_instead_of_nominal() -> None:
    y_true = [float((index + 1) * 10) for index in range(20)]
    lower_scores = [1.0] * 18 + [3.0, 5.0]
    calibration_frame = _frame(
        y_true=y_true,
        q10=[target + score for target, score in zip(y_true, lower_scores, strict=True)],
        q50=[target + score + 1.0 for target, score in zip(y_true, lower_scores, strict=True)],
        q90=[target + score + 2.0 for target, score in zip(y_true, lower_scores, strict=True)],
    )
    nominal = fit_conformal_quantile_calibration(
        calibration_frame,
        calibration_method="cqr_asymmetric",
    )
    floored = fit_conformal_quantile_calibration(
        calibration_frame,
        calibration_method="cqr_asymmetric",
        interval_coverage_floors={(0.1, 0.9): 0.6},
    )
    assert floored.adjustments[(ALL_GROUP, 0.1, 0.9)].lower_adjustment < nominal.adjustments[(ALL_GROUP, 0.1, 0.9)].lower_adjustment


def test_hourly_grouped_cqr_uses_group_specific_adjustments() -> None:
    calibration_frame = _frame(
        y_true=[10.0, 20.0, 10.0, 20.0],
        q10=[11.0, 25.0, 11.0, 21.0],
        q50=[10.0, 20.0, 10.0, 20.0],
        q90=[10.0, 20.0, 10.0, 20.0],
    )
    calibration_frame["ds"] = [
        pd.Timestamp("2026-01-01 00:00:00"),
        pd.Timestamp("2026-01-01 00:00:00"),
        pd.Timestamp("2026-01-01 00:00:00"),
        pd.Timestamp("2026-01-01 01:00:00"),
        pd.Timestamp("2026-01-01 01:00:00"),
        pd.Timestamp("2026-01-01 01:00:00"),
        pd.Timestamp("2026-01-02 00:00:00"),
        pd.Timestamp("2026-01-02 00:00:00"),
        pd.Timestamp("2026-01-02 00:00:00"),
        pd.Timestamp("2026-01-02 01:00:00"),
        pd.Timestamp("2026-01-02 01:00:00"),
        pd.Timestamp("2026-01-02 01:00:00"),
    ]
    calibration = fit_conformal_quantile_calibration(
        calibration_frame,
        calibration_method="cqr_asymmetric",
        group_by="hour",
        min_group_size=2,
    )
    assert calibration.adjustments[(0, 0.1, 0.9)].lower_adjustment == 1.0
    assert calibration.adjustments[(1, 0.1, 0.9)].lower_adjustment == 5.0


def test_hour_x_regime_grouped_cqr_uses_regime_specific_adjustments() -> None:
    calibration_frame = _frame(
        y_true=[10.0, 30.0, 10.0, 30.0],
        q10=[11.0, 40.0, 11.0, 31.0],
        q50=[10.0, 30.0, 10.0, 30.0],
        q90=[10.0, 30.0, 10.0, 30.0],
    )
    calibration_frame["ds"] = [
        pd.Timestamp("2026-01-01 17:00:00"),
        pd.Timestamp("2026-01-01 17:00:00"),
        pd.Timestamp("2026-01-01 17:00:00"),
        pd.Timestamp("2026-01-02 17:00:00"),
        pd.Timestamp("2026-01-02 17:00:00"),
        pd.Timestamp("2026-01-02 17:00:00"),
        pd.Timestamp("2026-01-03 17:00:00"),
        pd.Timestamp("2026-01-03 17:00:00"),
        pd.Timestamp("2026-01-03 17:00:00"),
        pd.Timestamp("2026-01-04 17:00:00"),
        pd.Timestamp("2026-01-04 17:00:00"),
        pd.Timestamp("2026-01-04 17:00:00"),
    ]
    calibration_frame["spike_score"] = [0.2] * 3 + [0.9] * 3 + [0.2] * 3 + [0.9] * 3

    calibration = fit_conformal_quantile_calibration(
        calibration_frame,
        calibration_method="cqr_asymmetric",
        group_by="hour_x_regime",
        min_group_size=2,
        regime_threshold=0.67,
    )

    assert calibration.adjustments[((17, 0), 0.1, 0.9)].lower_adjustment == 1.0
    assert calibration.adjustments[((17, 1), 0.1, 0.9)].lower_adjustment == 10.0


def test_hour_x_regime_median_bias_shifts_all_quantiles_by_group() -> None:
    calibration_frame = _frame(
        y_true=[12.0, 30.0, 14.0, 34.0],
        q10=[7.0, 18.0, 9.0, 22.0],
        q50=[10.0, 20.0, 12.0, 24.0],
        q90=[13.0, 22.0, 15.0, 26.0],
    )
    calibration_frame["ds"] = [
        pd.Timestamp("2026-01-01 17:00:00"),
        pd.Timestamp("2026-01-01 17:00:00"),
        pd.Timestamp("2026-01-01 17:00:00"),
        pd.Timestamp("2026-01-02 17:00:00"),
        pd.Timestamp("2026-01-02 17:00:00"),
        pd.Timestamp("2026-01-02 17:00:00"),
        pd.Timestamp("2026-01-03 17:00:00"),
        pd.Timestamp("2026-01-03 17:00:00"),
        pd.Timestamp("2026-01-03 17:00:00"),
        pd.Timestamp("2026-01-04 17:00:00"),
        pd.Timestamp("2026-01-04 17:00:00"),
        pd.Timestamp("2026-01-04 17:00:00"),
    ]
    calibration_frame["spike_score"] = [0.2] * 3 + [0.9] * 3 + [0.2] * 3 + [0.9] * 3
    calibration = fit_median_bias_calibration(
        calibration_frame,
        group_by="hour_x_regime",
        min_group_size=2,
        regime_threshold=0.5,
    )

    target = _frame(y_true=[10.0, 20.0], q10=[1.0, 2.0], q50=[5.0, 6.0], q90=[9.0, 10.0])
    target["ds"] = [
        pd.Timestamp("2026-01-05 17:00:00"),
        pd.Timestamp("2026-01-05 17:00:00"),
        pd.Timestamp("2026-01-05 17:00:00"),
        pd.Timestamp("2026-01-06 17:00:00"),
        pd.Timestamp("2026-01-06 17:00:00"),
        pd.Timestamp("2026-01-06 17:00:00"),
    ]
    target["spike_score"] = [0.2] * 3 + [0.9] * 3

    corrected = apply_median_bias_calibration(target, calibration)

    assert corrected.sort_values(["ds", "quantile"])["y_pred"].tolist() == [3.0, 7.0, 11.0, 12.0, 16.0, 20.0]


def test_median_bias_uses_global_fallback_for_small_groups_and_clamps_adjustment() -> None:
    calibration_frame = _frame(
        y_true=[20.0, 40.0, 60.0],
        q10=[0.0, 0.0, 0.0],
        q50=[0.0, 0.0, 0.0],
        q90=[0.0, 0.0, 0.0],
    )
    calibration = fit_median_bias_calibration(
        calibration_frame,
        group_by="hour",
        min_group_size=2,
        max_abs_adjustment=10.0,
    )

    assert calibration.adjustments[ALL_GROUP] == 10.0
    assert len(calibration.adjustments) == 1


def test_apply_conformal_quantile_calibration_preserves_monotonicity() -> None:
    frame = _frame(
        y_true=[10.0],
        q10=[11.0],
        q50=[10.5],
        q90=[10.0],
    )
    calibration = fit_conformal_quantile_calibration(
        _frame(
            y_true=[10.0, 20.0],
            q10=[9.0, 18.0],
            q50=[9.0, 19.0],
            q90=[9.0, 19.0],
        )
    )
    corrected = apply_conformal_quantile_calibration(frame, calibration)
    values = corrected.sort_values(["ds", "quantile"])["y_pred"].tolist()
    assert values == [10.0, 10.5, 11.0]


def test_postprocess_quantile_predictions_applies_validation_calibration() -> None:
    calibration_frame = _frame(
        y_true=[10.0, 20.0],
        q10=[9.0, 18.0],
        q50=[9.0, 19.0],
        q90=[9.0, 19.0],
    )
    test_frame = _frame(
        y_true=[12.0],
        q10=[8.0],
        q50=[9.0],
        q90=[10.0],
    )
    corrected = postprocess_quantile_predictions(
        test_frame,
        monotonic=True,
        calibration_frame=calibration_frame,
        calibration_method="cqr",
    )
    values = corrected.sort_values(["ds", "quantile"])["y_pred"].tolist()
    assert values == [7.0, 9.0, 11.0]


def test_postprocess_quantile_predictions_applies_median_bias_before_interval_calibration() -> None:
    calibration_frame = _frame(
        y_true=[12.0, 22.0],
        q10=[10.0, 20.0],
        q50=[10.0, 20.0],
        q90=[10.0, 20.0],
    )
    test_frame = _frame(
        y_true=[12.0],
        q10=[8.0],
        q50=[10.0],
        q90=[12.0],
    )

    corrected = postprocess_quantile_predictions(
        test_frame,
        monotonic=True,
        calibration_frame=calibration_frame,
        calibration_method="cqr_asymmetric",
        median_bias_enabled=True,
        median_bias_max_abs_adjustment=20.0,
    )

    values = corrected.sort_values(["ds", "quantile"])["y_pred"].tolist()
    assert values == [10.0, 12.0, 14.0]


def test_postprocess_quantile_predictions_rejects_unknown_grouping() -> None:
    with pytest.raises(ValueError, match="group_by"):
        postprocess_quantile_predictions(
            _frame(y_true=[10.0], q10=[9.0], q50=[10.0], q90=[11.0]),
            calibration_frame=_frame(y_true=[10.0], q10=[9.0], q50=[10.0], q90=[11.0]),
            calibration_method="cqr_asymmetric",
            calibration_group_by="weekday",
        )
