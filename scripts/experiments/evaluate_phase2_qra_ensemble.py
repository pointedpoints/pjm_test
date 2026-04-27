from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
import yaml

from pjm_forecast.evaluation.metrics import compute_metrics, compute_quantile_diagnostics
from pjm_forecast.evaluation.scenarios import compute_scenario_diagnostics
from pjm_forecast.prediction_contract import enforce_monotonic_quantiles
from pjm_forecast.quantile_postprocess import postprocess_quantile_predictions
from pjm_forecast.quantile_surface import QuantileSurface, quantile_surfaces_from_frame


TARGET_QUANTILES = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995]


@dataclass(frozen=True)
class MemberSpec:
    name: str
    validation_path: Path
    test_path: Path
    interpolation_tail_policy: str = "linear"


def _default_members(root: Path) -> list[MemberSpec]:
    return [
        MemberSpec(
            name="nbeatsx_current",
            validation_path=root / "artifacts_current" / "predictions" / "nbeatsx_validation_seed7.parquet",
            test_path=root / "artifacts_current" / "predictions" / "nbeatsx_test_seed7.parquet",
        ),
        MemberSpec(
            name="nhits_q50w150",
            validation_path=root / "artifacts_tmp" / "nhits_q50_weight_grid" / "predictions" / "nhits_q50w150_validation_seed7.parquet",
            test_path=root / "artifacts_tmp" / "nhits_q50_weight_grid" / "predictions" / "nhits_q50w150_test_seed7.parquet",
        ),
        MemberSpec(
            name="nhits_tail_spike_context",
            validation_path=root
            / "artifacts_tmp"
            / "nhits_tail_grid_weighted_long_spike_context"
            / "predictions"
            / "nhits_tail_grid_weighted_long_validation_seed7.parquet",
            test_path=root
            / "artifacts_tmp"
            / "nhits_tail_grid_weighted_long_spike_context"
            / "predictions"
            / "nhits_tail_grid_weighted_long_test_seed7.parquet",
        ),
    ]


def _load_config(config_path: Path) -> tuple[list[MemberSpec], Path, int]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    member_specs = [
        MemberSpec(
            name=str(item["name"]),
            validation_path=(config_path.parent / str(item["validation_path"])).resolve(),
            test_path=(config_path.parent / str(item["test_path"])).resolve(),
            interpolation_tail_policy=str(item.get("interpolation_tail_policy", "linear")),
        )
        for item in payload["members"]
    ]
    output_dir = (config_path.parent / str(payload["output_dir"])).resolve()
    holdout_days = int(payload.get("validation_holdout_days", 91))
    return member_specs, output_dir, holdout_days


def _load_member_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Prediction frame is empty: {path}")
    return enforce_monotonic_quantiles(frame)


def _attach_context(frame: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    if "spike_score" in frame.columns:
        return frame.copy()
    context_df = context.loc[:, ["ds", "spike_score"]].drop_duplicates("ds")
    merged = frame.merge(context_df, on="ds", how="left")
    if merged["spike_score"].isna().any():
        raise ValueError("Failed to attach spike_score context to prediction frame.")
    return merged


def _surface_grid(frame: pd.DataFrame, *, tail_policy: str) -> pd.DataFrame:
    surfaces = quantile_surfaces_from_frame(frame, tail_policy=tail_policy)
    y_by_ds = frame.groupby("ds", sort=True)["y"].first()
    rows: list[dict[str, float | pd.Timestamp]] = []
    for ds in sorted(surfaces):
        surface = surfaces[ds]
        row: dict[str, float | pd.Timestamp] = {"ds": ds, "y": float(y_by_ds.loc[ds])}
        for quantile in TARGET_QUANTILES:
            row[f"q_{quantile:.3f}"] = float(surface.ppf(quantile))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)


def _split_days(frame: pd.DataFrame, holdout_days: int) -> tuple[pd.Index, pd.Index]:
    forecast_days = pd.Index(pd.to_datetime(frame["ds"]).dt.floor("D").unique()).sort_values()
    if holdout_days <= 0 or holdout_days >= len(forecast_days):
        raise ValueError(f"holdout_days must be in [1, {len(forecast_days) - 1}]")
    return forecast_days[:-holdout_days], forecast_days[-holdout_days:]


def _slice_days(frame: pd.DataFrame, days: pd.Index) -> pd.DataFrame:
    day_series = pd.to_datetime(frame["ds"]).dt.floor("D")
    return frame.loc[day_series.isin(set(days))].copy()


def _aligned_context(reference_frame: pd.DataFrame, aligned_ds: pd.Index) -> pd.DataFrame:
    context = reference_frame.loc[:, ["ds", "y", "spike_score"]].drop_duplicates("ds").sort_values("ds").reset_index(drop=True)
    context = context.loc[context["ds"].isin(set(aligned_ds))].copy().sort_values("ds").reset_index(drop=True)
    if not context["ds"].reset_index(drop=True).equals(pd.Series(aligned_ds).reset_index(drop=True)):
        raise ValueError("Context timestamps do not align across members.")
    return context


def _fit_qra(train_context: pd.DataFrame, train_member_grids: dict[str, pd.DataFrame]) -> dict[float, QuantileRegressor]:
    models: dict[float, QuantileRegressor] = {}
    y_train = train_context["y"].to_numpy(dtype=float)
    for quantile in TARGET_QUANTILES:
        feature_columns = []
        for member_name in train_member_grids:
            feature_columns.append(train_member_grids[member_name][f"q_{quantile:.3f}"].to_numpy(dtype=float))
        X_train = np.column_stack(feature_columns)
        model = QuantileRegressor(quantile=quantile, alpha=1e-6, fit_intercept=True, solver="highs")
        model.fit(X_train, y_train)
        models[quantile] = model
    return models


def _predict_qra(
    *,
    models: dict[float, QuantileRegressor],
    context: pd.DataFrame,
    member_grids: dict[str, pd.DataFrame],
    run_name: str,
    split_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered_members = list(member_grids)
    for quantile in TARGET_QUANTILES:
        feature_columns = []
        for member_name in ordered_members:
            feature_columns.append(member_grids[member_name][f"q_{quantile:.3f}"].to_numpy(dtype=float))
        X = np.column_stack(feature_columns)
        predictions = models[quantile].predict(X)
        for ds, y_true, spike_score, y_pred in zip(
            context["ds"],
            context["y"],
            context["spike_score"],
            predictions,
            strict=True,
        ):
            rows.append(
                {
                    "ds": ds,
                    "y": float(y_true),
                    "spike_score": float(spike_score),
                    "quantile": float(quantile),
                    "y_pred": float(y_pred),
                    "model": run_name,
                    "split": split_name,
                    "seed": 7,
                    "metadata": "{}",
                }
            )
    frame = pd.DataFrame(rows).sort_values(["ds", "quantile"]).reset_index(drop=True)
    return enforce_monotonic_quantiles(frame)


def _evaluate_variants(
    *,
    raw_prediction: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    variant_specs = [
        ("raw_monotonic", {"monotonic": True, "calibration_frame": None}),
        (
            "hour_cqr",
            {
                "monotonic": True,
                "calibration_frame": calibration_frame,
                "calibration_method": "cqr_asymmetric",
                "calibration_group_by": "hour",
                "calibration_min_group_size": 24,
            },
        ),
        (
            "hour_regime_cqr_t50",
            {
                "monotonic": True,
                "calibration_frame": calibration_frame,
                "calibration_method": "cqr_asymmetric",
                "calibration_group_by": "hour_x_regime",
                "calibration_regime_score_column": "spike_score",
                "calibration_regime_threshold": 0.50,
                "calibration_min_group_size": 24,
            },
        ),
        (
            "hour_regime_cqr_t67",
            {
                "monotonic": True,
                "calibration_frame": calibration_frame,
                "calibration_method": "cqr_asymmetric",
                "calibration_group_by": "hour_x_regime",
                "calibration_regime_score_column": "spike_score",
                "calibration_regime_threshold": 0.67,
                "calibration_min_group_size": 24,
            },
        ),
    ]

    rows: list[dict[str, object]] = []
    variant_frames: dict[str, pd.DataFrame] = {}
    for variant_name, params in variant_specs:
        processed = postprocess_quantile_predictions(raw_prediction, **params)
        variant_frames[variant_name] = processed
        metrics = compute_metrics(processed)
        diagnostics = compute_quantile_diagnostics(processed)
        row: dict[str, object] = {"mode": mode, "variant": variant_name}
        row.update(metrics)
        row.update(diagnostics)
        rows.append(row)
    return pd.DataFrame(rows), variant_frames


def _scenario_summary(
    *,
    validation_variants: dict[str, pd.DataFrame],
    test_variants: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, test_frame in test_variants.items():
        diagnostics = compute_scenario_diagnostics(
            validation_variants[variant_name],
            test_frame,
            family="student_t",
            n_samples=256,
            dof_grid=[3.0, 5.0, 7.0, 10.0],
            random_seed=7,
            tail_policy="linear",
        )
        rows.append({"variant": variant_name, **diagnostics})
    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 heterogeneous QRA ensemble.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--config", help="Optional YAML config for the QRA experiment.")
    parser.add_argument("--output-dir", default="artifacts_phase2/qra_ensemble", help="Output directory.")
    parser.add_argument("--validation-holdout-days", type=int, default=91, help="Validation holdout days.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.config:
        members, output_dir, holdout_days = _load_config((root / args.config).resolve())
    else:
        members = _default_members(root)
        output_dir = (root / args.output_dir).resolve()
        holdout_days = int(args.validation_holdout_days)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_frames = {member.name: _load_member_frame(member.validation_path) for member in members}
    test_frames = {member.name: _load_member_frame(member.test_path) for member in members}

    # Use the NHITS q50 branch as the canonical spike-score context carrier.
    validation_frames["nbeatsx_current"] = _attach_context(validation_frames["nbeatsx_current"], validation_frames["nhits_q50w150"])
    test_frames["nbeatsx_current"] = _attach_context(test_frames["nbeatsx_current"], test_frames["nhits_q50w150"])

    validation_grids = {
        member.name: _surface_grid(validation_frames[member.name], tail_policy=member.interpolation_tail_policy) for member in members
    }
    test_grids = {member.name: _surface_grid(test_frames[member.name], tail_policy=member.interpolation_tail_policy) for member in members}

    train_days, holdout_eval_days = _split_days(validation_frames["nhits_q50w150"], holdout_days)
    validation_context_train = _aligned_context(
        _slice_days(validation_frames["nhits_q50w150"], train_days),
        _slice_days(validation_grids["nhits_q50w150"], train_days)["ds"],
    )
    validation_context_holdout = _aligned_context(
        _slice_days(validation_frames["nhits_q50w150"], holdout_eval_days),
        _slice_days(validation_grids["nhits_q50w150"], holdout_eval_days)["ds"],
    )

    train_member_grids = {name: _slice_days(grid, train_days).reset_index(drop=True) for name, grid in validation_grids.items()}
    holdout_member_grids = {
        name: _slice_days(grid, holdout_eval_days).reset_index(drop=True) for name, grid in validation_grids.items()
    }

    holdout_models = _fit_qra(validation_context_train, train_member_grids)
    train_prediction = _predict_qra(
        models=holdout_models,
        context=validation_context_train,
        member_grids=train_member_grids,
        run_name="phase2_qra_ensemble",
        split_name="validation_train",
    )
    holdout_prediction = _predict_qra(
        models=holdout_models,
        context=validation_context_holdout,
        member_grids=holdout_member_grids,
        run_name="phase2_qra_ensemble",
        split_name="validation_holdout",
    )

    validation_summary, _ = _evaluate_variants(
        raw_prediction=holdout_prediction,
        calibration_frame=train_prediction,
        mode="validation_holdout",
    )
    validation_summary = validation_summary.sort_values(["pinball", "mae", "variant"]).reset_index(drop=True)
    validation_summary.to_csv(output_dir / "validation_holdout_summary.csv", index=False)

    full_validation_context = _aligned_context(validation_frames["nhits_q50w150"], validation_grids["nhits_q50w150"]["ds"])
    final_models = _fit_qra(full_validation_context, validation_grids)
    full_validation_prediction = _predict_qra(
        models=final_models,
        context=full_validation_context,
        member_grids=validation_grids,
        run_name="phase2_qra_ensemble",
        split_name="validation",
    )
    test_context = _aligned_context(test_frames["nhits_q50w150"], test_grids["nhits_q50w150"]["ds"])
    test_prediction = _predict_qra(
        models=final_models,
        context=test_context,
        member_grids=test_grids,
        run_name="phase2_qra_ensemble",
        split_name="test",
    )

    test_summary, test_variant_frames = _evaluate_variants(
        raw_prediction=test_prediction,
        calibration_frame=full_validation_prediction,
        mode="test",
    )
    test_summary = test_summary.sort_values(["pinball", "mae", "variant"]).reset_index(drop=True)
    test_summary.to_csv(output_dir / "test_summary.csv", index=False)

    validation_variant_frames = {
        variant: postprocess_quantile_predictions(
            full_validation_prediction,
            monotonic=True,
            calibration_frame=None,
        )
        for variant in test_variant_frames
    }
    validation_variant_frames["hour_cqr"] = postprocess_quantile_predictions(
        full_validation_prediction,
        monotonic=True,
        calibration_frame=full_validation_prediction,
        calibration_method="cqr_asymmetric",
        calibration_group_by="hour",
        calibration_min_group_size=24,
    )
    validation_variant_frames["hour_regime_cqr_t50"] = postprocess_quantile_predictions(
        full_validation_prediction,
        monotonic=True,
        calibration_frame=full_validation_prediction,
        calibration_method="cqr_asymmetric",
        calibration_group_by="hour_x_regime",
        calibration_regime_score_column="spike_score",
        calibration_regime_threshold=0.50,
        calibration_min_group_size=24,
    )
    validation_variant_frames["hour_regime_cqr_t67"] = postprocess_quantile_predictions(
        full_validation_prediction,
        monotonic=True,
        calibration_frame=full_validation_prediction,
        calibration_method="cqr_asymmetric",
        calibration_group_by="hour_x_regime",
        calibration_regime_score_column="spike_score",
        calibration_regime_threshold=0.67,
        calibration_min_group_size=24,
    )

    scenario_summary = _scenario_summary(
        validation_variants=validation_variant_frames,
        test_variants=test_variant_frames,
    )
    scenario_summary.to_csv(output_dir / "test_scenario_summary.csv", index=False)

    full_validation_prediction.to_parquet(output_dir / "validation_raw_qra_prediction.parquet", index=False)
    test_prediction.to_parquet(output_dir / "test_raw_qra_prediction.parquet", index=False)
    for variant_name, frame in test_variant_frames.items():
        frame.to_parquet(output_dir / f"test_{variant_name}.parquet", index=False)


if __name__ == "__main__":
    main()
