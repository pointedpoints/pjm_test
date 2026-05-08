"""
spike_filter_experiment.py — Detrended spike filtering + replacement experiment.

Phase 1 (local): Apply detrended q98 spike filtering and linear interpolation replacement.
Phase 2 (AutoDL): Train models on cleaned vs full data, compare normal-hour MAE.

Usage:
    python scripts/experiments/spike_filter_experiment.py --config configs/pjm_day_ahead_current_processed.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pjm_forecast.workspace import Workspace


def compute_detrended_spike_threshold(
    y: pd.Series,
    *,
    trend_window_hours: int = 365 * 24,       # 1 year for trend
    quantile_window_hours: int = 730 * 24,     # 2 years for residual quantile
    quantile: float = 0.98,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute spike threshold on detrended residuals.

    1. Rolling median trend (window=1 year).
    2. Residual = y - trend.
    3. Rolling quantile on residuals (window=2 years).
    4. Spike iff residual > residual q98.

    Returns (spike_mask, trend).
    """
    trend = y.rolling(window=trend_window_hours, min_periods=trend_window_hours).median()
    residual = y - trend
    residual_q = residual.rolling(window=quantile_window_hours, min_periods=quantile_window_hours).quantile(quantile)
    spike_mask = residual > residual_q
    spike_mask.loc[residual_q.isna()] = False  # not enough history
    return spike_mask, trend


def linear_interpolation(
    y: pd.Series,
    spike_mask: pd.Series,
) -> pd.Series:
    """
    Replace spike hours via linear interpolation from surrounding non-spike points.

    - Single spike: average of nearest non-spike neighbours on each side.
    - Consecutive spike block: evenly spaced transition from left neighbour to right neighbour.
    - Boundary spikes (start/end): fill with nearest non-spike value.
    """
    y_clean = y.copy()
    non_spike_idx = np.where(~spike_mask)[0]

    if len(non_spike_idx) == 0:
        return y_clean  # all spikes, nothing to interpolate from

    # Find contiguous spike segments
    spike_segments = []  # list of (start, end) inclusive indices
    in_spike = False
    for i in range(len(y)):
        if spike_mask.iloc[i]:
            if not in_spike:
                start = i
                in_spike = True
        else:
            if in_spike:
                spike_segments.append((start, i - 1))
                in_spike = False
    if in_spike:
        spike_segments.append((start, len(y) - 1))

    for seg_start, seg_end in spike_segments:
        # Find left boundary (last non-spike before segment)
        left_candidates = non_spike_idx[non_spike_idx < seg_start]
        if len(left_candidates) > 0:
            left_idx = left_candidates[-1]
            left_val = y_clean.iloc[left_idx]
        else:
            # Boundary at start: fill with first non-spike value
            fill_val = y_clean.iloc[non_spike_idx[0]]
            for j in range(seg_start, seg_end + 1):
                y_clean.iloc[j] = fill_val
            continue

        # Find right boundary (first non-spike after segment)
        right_candidates = non_spike_idx[non_spike_idx > seg_end]
        if len(right_candidates) > 0:
            right_idx = right_candidates[0]
            right_val = y_clean.iloc[right_idx]
        else:
            # Boundary at end: fill with last non-spike value
            fill_val = y_clean.iloc[non_spike_idx[-1]]
            for j in range(seg_start, seg_end + 1):
                y_clean.iloc[j] = fill_val
            continue

        # Linear interpolation across segment
        seg_len = seg_end - seg_start + 1
        for j in range(seg_start, seg_end + 1):
            t = (j - seg_start + 1) / (seg_len + 1)  # fraction from left to right
            y_clean.iloc[j] = left_val + t * (right_val - left_val)

    return y_clean


def mark_and_replace_spikes(
    df: pd.DataFrame,
    *,
    quantile: float = 0.98,
    replacement_method: str = "linear_interpolation",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Mark spike hours using detrended rolling quantile, then replace with normal values.

    Returns (cleaned_df, spike_mask, trend).
    """
    df = df.copy()
    ds = df["ds"]
    y = df["y"]

    # Step 1: detrended spike detection
    spike_mask, trend = compute_detrended_spike_threshold(y, quantile=quantile)

    n_spikes = spike_mask.sum()
    n_total = len(spike_mask)
    print(f"Spike hours: {n_spikes}/{n_total} ({100*n_spikes/n_total:.1f}%)")

    # Diagnostics: consecutive spike lengths
    if n_spikes > 0:
        spike_groups = (spike_mask != spike_mask.shift()).cumsum()
        spike_run_lengths = spike_mask.groupby(spike_groups).sum()
        spike_run_lengths = spike_run_lengths[spike_run_lengths > 0]
        max_run = int(spike_run_lengths.max())
        print(f"  Max consecutive spike length: {max_run} hours")
    else:
        max_run = 0
        print(f"  No spikes detected.")

    # Step 2: replace spike hours
    if replacement_method == "linear_interpolation":
        y_clean = linear_interpolation(y, spike_mask)
    elif replacement_method == "same_hour_weekday_median":
        y_clean = y.copy()
        for idx in np.where(spike_mask)[0]:
            hour = ds.iloc[idx].hour
            weekday = ds.iloc[idx].weekday()
            lookback = []
            for w in range(1, 9):
                candidate = ds.iloc[idx] - pd.Timedelta(weeks=w)
                candidate_idx = df.index[(ds == candidate)]
                if len(candidate_idx) == 1:
                    ci = candidate_idx[0]
                    if not spike_mask.iloc[ci]:
                        lookback.append(y.iloc[ci])
            if lookback:
                y_clean.iloc[idx] = np.median(lookback)
            else:
                y_clean.iloc[idx] = y[~spike_mask].median()
    else:
        raise ValueError(f"Unknown replacement method: {replacement_method}")

    df["y"] = y_clean

    # Diagnostics: normal-hour MAE before/after
    normal = ~spike_mask
    if normal.sum() > 0:
        mae_before = np.abs(y[normal] - y.rolling(168, min_periods=168).mean().shift(1)[normal]).mean()
        mae_after = np.abs(y_clean[normal] - y_clean.rolling(168, min_periods=168).mean().shift(1)[normal]).mean()
        print(f"  Normal-hour MAE (naive 168): {mae_before:.2f} → {mae_after:.2f}")

    return df, spike_mask, trend


def run(config_path: str, **overrides) -> None:
    ws = Workspace.open(config_path)

    # Load panel
    panel_path = ws.directories["processed_data_dir"] / "panel.parquet"
    print(f"Loading panel: {panel_path}")
    panel = pd.read_parquet(panel_path)

    # Parameters with optional CLI overrides
    quantile = overrides.get("quantile", 0.98)
    replacement_method = overrides.get("replacement_method", "linear_interpolation")

    print(f"Parameters: quantile={quantile}, replacement_method={replacement_method}")

    # Filter and replace
    cleaned, spike_mask, trend = mark_and_replace_spikes(
        panel, quantile=quantile, replacement_method=replacement_method,
    )

    # Save outputs to v2 directory
    out_dir = Path("data/processed_spike_filtered_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = out_dir / "panel_cleaned.parquet"
    spike_mask_path = out_dir / "spike_mask.parquet"
    trend_path = out_dir / "trend.parquet"
    cleaned.to_parquet(cleaned_path)
    spike_mask.to_frame("is_spike").to_parquet(spike_mask_path)
    trend.to_frame("trend").to_parquet(trend_path)

    print(f"\nCleaned panel saved: {cleaned_path}")
    print(f"Spike mask saved:    {spike_mask_path}")
    print(f"Trend saved:         {trend_path}")

    # Quick stats
    y_orig = panel["y"]
    y_new = cleaned["y"]
    print(f"\n=== Before vs After ===")
    print(f"  y mean:  {y_orig.mean():.2f} → {y_new.mean():.2f}")
    print(f"  y std:   {y_orig.std():.2f} → {y_new.std():.2f}")
    print(f"  y max:   {y_orig.max():.2f} → {y_new.max():.2f}")
    print(f"\n=== Spike summary ===")
    spike_detail = y_orig[spike_mask].describe()
    print(f"  Spike price range: {spike_detail['min']:.1f} – {spike_detail['max']:.1f}")
    print(f"  Spike price mean:  {spike_detail['mean']:.1f}")
    print(f"  Spike rate: {spike_mask.sum() / len(spike_mask) * 100:.2f}%")
    if spike_mask.sum() > 0:
        spike_groups = (spike_mask != spike_mask.shift()).cumsum()
        spike_run_lengths = spike_mask.groupby(spike_groups).sum()
        spike_run_lengths = spike_run_lengths[spike_run_lengths > 0]
        print(f"  Max consecutive spike length: {int(spike_run_lengths.max())} hours")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--quantile", type=float, default=None)
    parser.add_argument("--replacement-method", type=str, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.quantile is not None:
        overrides["quantile"] = args.quantile
    if args.replacement_method is not None:
        overrides["replacement_method"] = args.replacement_method

    run(args.config, **overrides)


if __name__ == "__main__":
    main()
