from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from pjm_forecast.config import load_config
from pjm_forecast.models import build_model
from pjm_forecast.paths import ensure_project_directories


def _decode_mlp_units(value):
    if not isinstance(value, str):
        return value

    match = re.fullmatch(r"(\d+)x(\d+)", value)
    if not match:
        raise ValueError(f"Unsupported mlp_units encoding: {value}")

    width_in = int(match.group(1))
    width_out = int(match.group(2))
    return [[width_in, width_out], [width_in, width_out], [width_in, width_out]]


def _load_best_params(config, hyperparameter_dir: Path) -> None:
    best_params_path = hyperparameter_dir / "nbeatsx_best_params.json"
    if not best_params_path.exists():
        return
    best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
    best_params["mlp_units"] = _decode_mlp_units(best_params.get("mlp_units"))
    config.models["nbeatsx"].update(best_params)


def run_export_nbeatsx_snapshot(config_path: str) -> Path:
    config = load_config(config_path)
    directories = ensure_project_directories(config)
    _load_best_params(config, directories["hyperparameter_dir"])

    feature_df = pd.read_parquet(directories["processed_data_dir"] / "feature_store.parquet")
    window_days = config.backtest["rolling_window_days"]
    history_end = feature_df["ds"].max()
    window_start = history_end - pd.Timedelta(days=window_days) + pd.Timedelta(hours=1)
    history_df = feature_df.loc[(feature_df["ds"] >= window_start) & (feature_df["ds"] <= history_end)].copy()
    if history_df.empty:
        raise ValueError("No history rows found for export window.")

    model = build_model(config, "nbeatsx", seed=config.project["benchmark_seed"])
    model.fit(history_df)

    output_dir = directories["artifact_dir"] / "models" / "nbeatsx_snapshot"
    model.save(output_dir)
    print(f"Exported NBEATSx snapshot to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_export_nbeatsx_snapshot(args.config)


if __name__ == "__main__":
    main()
