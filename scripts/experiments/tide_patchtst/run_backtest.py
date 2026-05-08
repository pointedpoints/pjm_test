#!/usr/bin/env python3
"""
Run TiDE and PatchTST backtest on the PJM COMED dataset.

Usage:
    python scripts/experiments/tide_patchtst/run_backtest.py --model tide --config configs/pjm_tide.yaml
    python scripts/experiments/tide_patchtst/run_backtest.py --model patchtst --config configs/pjm_patchtst.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pjm_forecast.backtest.engine import run_rolling_backtest
from pjm_forecast.config import ProjectConfig, load_config
from pjm_forecast.workspace import Workspace, ArtifactStore
from pjm_forecast.evaluation import compute_metrics


def build_model_from_config(config: ProjectConfig, model_name: str):
    """Build a TiDE or PatchTST model from config."""
    from tide_model import TidePipelineModel
    from patchtst_model import PatchTSTPipelineModel

    model_cfg = dict(config.models[model_name])
    model_type = model_cfg.pop("type")

    if model_type == "tide":
        return TidePipelineModel(
            h=int(model_cfg.get("h", 24)),
            input_size=int(model_cfg.get("input_size", 168)),
            hidden_size=int(model_cfg.get("hidden_size", 256)),
            latent_size=int(model_cfg.get("latent_size", 64)),
            num_encoder_layers=int(model_cfg.get("num_encoder_layers", 2)),
            num_decoder_layers=int(model_cfg.get("num_decoder_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            learning_rate=float(model_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(model_cfg.get("weight_decay", 1e-5)),
            max_epochs=int(model_cfg.get("max_epochs", 100)),
            batch_size=int(model_cfg.get("batch_size", 64)),
            early_stop_patience=int(model_cfg.get("early_stop_patience", 10)),
            quantiles=list(model_cfg.get("quantiles", [0.5])),
            num_future_covariates=int(model_cfg.get("num_future_covariates", 0)),
        )
    elif model_type == "patchtst":
        return PatchTSTPipelineModel(
            h=int(model_cfg.get("h", 24)),
            input_size=int(model_cfg.get("input_size", 168)),
            patch_len=int(model_cfg.get("patch_len", 12)),
            stride=int(model_cfg.get("stride", 12)),
            d_model=int(model_cfg.get("d_model", 128)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            n_layers=int(model_cfg.get("n_layers", 3)),
            d_ff=int(model_cfg.get("d_ff", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            learning_rate=float(model_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(model_cfg.get("weight_decay", 1e-5)),
            max_epochs=int(model_cfg.get("max_epochs", 100)),
            batch_size=int(model_cfg.get("batch_size", 64)),
            early_stop_patience=int(model_cfg.get("early_stop_patience", 10)),
            quantiles=list(model_cfg.get("quantiles", [0.5])),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_single_model(
    workspace: Workspace,
    config: ProjectConfig,
    model_name: str,
    split: str,
    seed: int,
) -> Path:
    """Run backtest for a single model and return prediction path."""
    store = workspace.artifacts
    print(f"[{model_name}] Building model...")

    model = build_model_from_config(config, model_name)

    # Load prepared panel data
    panel_path = store.panel()
    print(f"[{model_name}] Loading panel data from {panel_path}")
    panel_df = pd.read_parquet(panel_path)

    # Run rolling backtest
    print(f"[{model_name}] Starting rolling backtest (split={split})...")
    t0 = time.time()

    prediction_df = run_rolling_backtest(
        panel_df=panel_df,
        config=config,
        model=model,
        model_name=model_name,
        split=split,
        seed=seed,
    )

    elapsed = time.time() - t0
    print(f"[{model_name}] Backtest completed in {elapsed:.1f}s. "
          f"Predictions shape: {prediction_df.shape}")

    # Save predictions
    output_path = store.prediction(model_name, split, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_parquet(output_path, index=False)
    print(f"[{model_name}] Predictions saved to {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TiDE/PatchTST backtest")
    parser.add_argument("--model", required=True, choices=["tide", "patchtst"],
                        help="Model to run")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    workspace = Workspace.open(args.config)
    config = workspace.config
    seed = args.seed or config.project.get("benchmark_seed", 7)

    # Run
    run_single_model(workspace, config, args.model, args.split, seed)

    # Compute metrics if existing baseline metrics exist
    store = workspace.artifacts
    metrics_path = store.metrics(args.split)
    if metrics_path.exists():
        print(f"\nMetrics already exist at {metrics_path}")
        existing = pd.read_csv(metrics_path)
        print("Existing metrics:")
        print(existing.to_string(index=False))
    else:
        print(f"\nNo existing metrics at {metrics_path}. "
              "Run evaluate_and_plot.py after all models complete.")


if __name__ == "__main__":
    main()
