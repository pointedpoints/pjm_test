"""
spike_filter_winsor_causal_experiment.py — Detrended residual causal Winsorization experiment.

This experiment keeps the spike_v2 detrending philosophy but replaces interpolation
with a causal upper-tail cap on residuals. It writes a cleaned raw CSV for the
existing pipeline plus audit artifacts for review.

Usage:
    python scripts/experiments/spike_filter_winsor_causal_experiment.py \
        --config configs/pjm_baseline_raw.yaml \
        --source-csv /mnt/d/pjm_remaster/data/raw/PJM_COMED_20210101_20260331_weather_ready.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pjm_forecast.config import load_config
from pjm_forecast.prepared_data import FeatureSchema

DEFAULT_TREND_WINDOW_HOURS = 365 * 24
DEFAULT_QUANTILE_WINDOW_HOURS = 730 * 24
DEFAULT_QUANTILE = 0.97
DEFAULT_OUTPUT_DIR = "data/processed_spike_filtered_v2_winsor_causal"
DEFAULT_OUTPUT_CSV_NAME = "PJM_COMED_20210101_20260331_weather_ready_spike_winsor_causal.csv"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def compute_causal_trend(
    y: pd.Series,
    *,
    trend_window_hours: int = DEFAULT_TREND_WINDOW_HOURS,
) -> pd.Series:
    if trend_window_hours <= 0:
        raise ValueError("trend_window_hours must be positive.")
    return y.astype(float).shift(1).rolling(window=trend_window_hours, min_periods=trend_window_hours).median()


def compute_causal_residual_cap(
    residual: pd.Series,
    *,
    quantile_window_hours: int = DEFAULT_QUANTILE_WINDOW_HOURS,
    quantile: float = DEFAULT_QUANTILE,
) -> pd.Series:
    if quantile_window_hours <= 0:
        raise ValueError("quantile_window_hours must be positive.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1).")
    return residual.astype(float).shift(1).rolling(
        window=quantile_window_hours,
        min_periods=quantile_window_hours,
    ).quantile(quantile)


def apply_causal_winsorization(
    panel_df: pd.DataFrame,
    *,
    trend_window_hours: int = DEFAULT_TREND_WINDOW_HOURS,
    quantile_window_hours: int = DEFAULT_QUANTILE_WINDOW_HOURS,
    quantile: float = DEFAULT_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"ds", "y"}
    missing = required.difference(panel_df.columns)
    if missing:
        raise ValueError(f"panel_df is missing required columns: {sorted(missing)}")

    result = panel_df.copy()
    result["ds"] = pd.to_datetime(result["ds"], utc=False)
    result = result.sort_values("ds").reset_index(drop=True)
    y = pd.to_numeric(result["y"], errors="coerce")
    if y.isna().any():
        raise ValueError("panel_df contains non-numeric y values.")

    trend = compute_causal_trend(y, trend_window_hours=trend_window_hours)
    residual = y - trend
    residual_cap = compute_causal_residual_cap(
        residual,
        quantile_window_hours=quantile_window_hours,
        quantile=quantile,
    )
    is_capped = residual_cap.notna() & residual.gt(residual_cap)
    residual_capped = residual.where(~is_capped, residual_cap)
    y_clean = y.where(~is_capped, trend + residual_cap)
    cap_delta = (y - y_clean).where(is_capped, 0.0).astype(float)

    cleaned = result.copy()
    cleaned["y"] = y_clean.astype(float)

    diagnostics = pd.DataFrame(
        {
            "ds": result["ds"],
            "y": y.astype(float),
            "trend": trend.astype(float),
            "residual": residual.astype(float),
            "residual_cap": residual_cap.astype(float),
            "residual_capped": residual_capped.astype(float),
            "y_clean": y_clean.astype(float),
            "is_capped": is_capped.astype(bool),
            "cap_delta": cap_delta,
        }
    )
    return cleaned, diagnostics


def _default_output_csv_path(config_path: Path) -> Path:
    del config_path
    return _project_root() / "data" / "raw" / DEFAULT_OUTPUT_CSV_NAME


def _default_output_dir(config_path: Path) -> Path:
    del config_path
    return _project_root() / DEFAULT_OUTPUT_DIR


def _summary_payload(
    diagnostics: pd.DataFrame,
    source_csv_path: Path,
    output_csv_path: Path,
    *,
    trend_window_hours: int,
    quantile_window_hours: int,
    quantile: float,
) -> dict[str, float | int | str]:
    capped = diagnostics["is_capped"].astype(bool)
    cap_delta = diagnostics["cap_delta"].astype(float)
    total_rows = int(len(diagnostics))
    capped_count = int(capped.sum())
    return {
        "source_csv": str(source_csv_path),
        "output_csv": str(output_csv_path),
        "rows": total_rows,
        "trend_window_hours": int(trend_window_hours),
        "quantile_window_hours": int(quantile_window_hours),
        "quantile": float(quantile),
        "capped_count": capped_count,
        "capped_share": 0.0 if total_rows == 0 else capped_count / total_rows,
        "mean_cap_delta": 0.0 if capped_count == 0 else float(cap_delta.loc[capped].mean()),
        "max_cap_delta": 0.0 if capped_count == 0 else float(cap_delta.loc[capped].max()),
        "max_y_before": float(diagnostics["y"].max()),
        "max_y_after": float(diagnostics["y_clean"].max()),
    }


def save_outputs(
    *,
    config_path: Path,
    source_csv_path: Path,
    raw_df: pd.DataFrame,
    raw_timestamp_column: str,
    raw_price_column: str,
    cleaned_panel: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_csv_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    output_csv_path = (output_csv_path or _default_output_csv_path(config_path)).resolve()
    output_dir = (output_dir or _default_output_dir(config_path)).resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_by_ds = pd.Series(cleaned_panel["y"].to_numpy(), index=pd.to_datetime(cleaned_panel["ds"], utc=False))
    raw_timestamps = pd.to_datetime(raw_df[raw_timestamp_column], utc=False)
    cleaned_values = raw_timestamps.map(cleaned_by_ds)
    if cleaned_values.isna().any():
        missing = int(cleaned_values.isna().sum())
        raise ValueError(f"Unable to map cleaned values back to raw CSV rows: missing={missing}")

    cleaned_raw = raw_df.copy()
    cleaned_raw[raw_price_column] = cleaned_values.to_numpy(dtype=float)
    cleaned_raw.to_csv(output_csv_path, index=False)

    cleaned_panel.to_parquet(output_dir / "panel_cleaned.parquet", index=False)
    diagnostics.loc[:, ["ds", "trend"]].to_parquet(output_dir / "trend.parquet", index=False)
    diagnostics.loc[:, ["ds", "residual"]].to_parquet(output_dir / "residual.parquet", index=False)
    diagnostics.loc[:, ["ds", "residual_cap"]].to_parquet(output_dir / "residual_cap.parquet", index=False)
    diagnostics.loc[:, ["ds", "is_capped"]].to_parquet(output_dir / "capped_mask.parquet", index=False)
    diagnostics.loc[:, ["ds", "cap_delta"]].to_parquet(output_dir / "cap_delta.parquet", index=False)
    diagnostics.to_parquet(output_dir / "diagnostics.parquet", index=False)
    return output_csv_path, output_dir


def run(
    config_path: str,
    *,
    source_csv: str | None = None,
    output_csv: str | None = None,
    output_dir: str | None = None,
    trend_window_hours: int = DEFAULT_TREND_WINDOW_HOURS,
    quantile_window_hours: int = DEFAULT_QUANTILE_WINDOW_HOURS,
    quantile: float = DEFAULT_QUANTILE,
) -> tuple[Path, Path]:
    config = load_config(config_path)
    source_config = config.without_weather_feature_contracts() if config.weather_enabled else config
    schema = FeatureSchema(source_config)

    source_csv_path = Path(source_csv).resolve() if source_csv else source_config.resolve_path(str(source_config.dataset["local_csv_path"]))
    raw_df = pd.read_csv(source_csv_path)
    raw_df.columns = [column.strip() for column in raw_df.columns]

    panel_df = schema.normalize_panel_frame(raw_df)
    cleaned_panel, diagnostics = apply_causal_winsorization(
        panel_df,
        trend_window_hours=trend_window_hours,
        quantile_window_hours=quantile_window_hours,
        quantile=quantile,
    )

    output_csv_path, output_dir_path = save_outputs(
        config_path=Path(config_path).resolve(),
        source_csv_path=source_csv_path,
        raw_df=raw_df,
        raw_timestamp_column=str(config.dataset["timestamp_col"]),
        raw_price_column=str(config.dataset["price_col"]),
        cleaned_panel=cleaned_panel,
        diagnostics=diagnostics,
        output_csv_path=Path(output_csv).resolve() if output_csv else None,
        output_dir=Path(output_dir).resolve() if output_dir else None,
    )

    summary = _summary_payload(
        diagnostics,
        source_csv_path=source_csv_path,
        output_csv_path=output_csv_path,
        trend_window_hours=trend_window_hours,
        quantile_window_hours=quantile_window_hours,
        quantile=quantile,
    )
    (output_dir_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Source CSV: {source_csv_path}")
    print(f"Cleaned CSV saved: {output_csv_path}")
    print(f"Diagnostics dir: {output_dir_path}")
    print(f"Rows: {summary['rows']}")
    print(f"Capped hours: {summary['capped_count']} ({100 * float(summary['capped_share']):.2f}%)")
    print(f"Mean cap delta: {summary['mean_cap_delta']:.2f}")
    print(f"Max cap delta: {summary['max_cap_delta']:.2f}")
    print(f"Max y before/after: {summary['max_y_before']:.2f} → {summary['max_y_after']:.2f}")
    return output_csv_path, output_dir_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-csv")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--trend-window-hours", type=int, default=DEFAULT_TREND_WINDOW_HOURS)
    parser.add_argument("--quantile-window-hours", type=int, default=DEFAULT_QUANTILE_WINDOW_HOURS)
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    args = parser.parse_args()

    run(
        args.config,
        source_csv=args.source_csv,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        trend_window_hours=args.trend_window_hours,
        quantile_window_hours=args.quantile_window_hours,
        quantile=args.quantile,
    )


if __name__ == "__main__":
    main()
