from __future__ import annotations

import argparse
from itertools import combinations

import pandas as pd

from pjm_forecast.config import load_config
from pjm_forecast.evaluation import compute_metrics, dm_test
from pjm_forecast.evaluation.reporting import plot_high_volatility_week, plot_hourly_mae
from pjm_forecast.paths import ensure_project_directories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    directories = ensure_project_directories(config)

    prediction_frames = {}
    metric_rows = []
    for prediction_file in sorted(directories["prediction_dir"].glob(f"*_{args.split}_seed*.parquet")):
        prediction_df = pd.read_parquet(prediction_file)
        model_name = prediction_df["model"].iloc[0]
        seed = int(prediction_df["seed"].iloc[0])
        run_name = f"{model_name}_seed{seed}"
        prediction_frames[run_name] = prediction_df
        metric_rows.append({"run": run_name, "model": model_name, "seed": seed, **compute_metrics(prediction_df)})

    metrics_df = pd.DataFrame(metric_rows).sort_values(["mae", "model", "seed"]).reset_index(drop=True)
    metrics_path = directories["metrics_dir"] / f"{args.split}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    dm_rows = []
    for left_key, right_key in combinations(prediction_frames, 2):
        left = prediction_frames[left_key]
        right = prediction_frames[right_key]
        if not left["ds"].equals(right["ds"]):
            continue
        dm_rows.append(
            {
                "left": left_key,
                "right": right_key,
                **dm_test(
                    y_true=left["y"].to_numpy(),
                    y_pred_a=left["y_pred"].to_numpy(),
                    y_pred_b=right["y_pred"].to_numpy(),
                ),
            }
        )

    pd.DataFrame(dm_rows).to_csv(directories["metrics_dir"] / f"{args.split}_dm.csv", index=False)
    plot_hourly_mae(prediction_frames, directories["plots_dir"] / f"{args.split}_hourly_mae.png")

    best_run = metrics_df.iloc[0]["run"]
    plot_high_volatility_week(prediction_frames[best_run], directories["plots_dir"] / f"{args.split}_high_vol_week.png")
    print(f"Saved metrics and plots under {directories['artifact_dir']}")


if __name__ == "__main__":
    main()

