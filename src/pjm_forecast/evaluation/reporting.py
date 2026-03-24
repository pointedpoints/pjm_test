from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import compute_hourly_mae


def plot_hourly_mae(predictions_by_model: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, prediction_df in predictions_by_model.items():
        hourly_mae = compute_hourly_mae(prediction_df)
        ax.plot(hourly_mae["hour"], hourly_mae["mae"], label=model_name)
    ax.set_title("Hourly MAE")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MAE")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_high_volatility_week(predictions: pd.DataFrame, output_path: Path) -> None:
    frame = predictions.copy()
    frame["day"] = pd.to_datetime(frame["ds"]).dt.normalize()
    daily_vol = frame.groupby("day")["y"].agg(lambda values: float(values.max() - values.min()))
    target_day = daily_vol.sort_values(ascending=False).index[0]
    week_mask = (frame["day"] >= target_day) & (frame["day"] < target_day + pd.Timedelta(days=7))
    week_df = frame.loc[week_mask].copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(week_df["ds"], week_df["y"], label="y_true")
    ax.plot(week_df["ds"], week_df["y_pred"], label="y_pred")
    ax.set_title("High-volatility week")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

