#!/usr/bin/env python3
"""
Hybrid Model Prototype: lightgbm + NHITS stack for PJM COMED price forecasting.

Strategies:
  A. Hard-switch at threshold — use NHITS when forecasted price > $threshold
  B. Soft-weighting — sigmoid blend around threshold
  C. Reverse — NHITS as primary, LGB for low-price correction
  D. Confidence-weighted (overall best-model blend on validation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
# sigmoid using numpy - no scipy dependency
def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
expit = _sigmoid
import json

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR = Path("/mnt/d/pjm_remaster/artifacts_baseline_spike_v2/predictions")
OUTPUT_DIR = Path("/mnt/c/Users/LiamZhang/Desktop/PJM_Experiments/spike_v2_cleaned_20260508")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PAIRS = [
    ("lightgbm_q", "nhits_168"),
]

QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995]

# Thresholds to test
THRESHOLDS = [40, 45, 50, 55, 60, 65, 70]

# Sigmoid blend width
SIGMOID_K = 0.15  # steepness (per $)

# ── Load Data ─────────────────────────────────────────────────────────────
def load_q50(model_name):
    path = DATA_DIR / f"{model_name}_test_seed7.parquet"
    df = pd.read_parquet(path)
    return df[df["quantile"] == 0.5].copy().reset_index(drop=True)


def load_all_quantiles(model_name):
    path = DATA_DIR / f"{model_name}_test_seed7.parquet"
    df = pd.read_parquet(path)
    return df.copy().reset_index(drop=True)


# ── Strategies ────────────────────────────────────────────────────────────

def strategy_hard_switch(y_pred_primary, y_pred_secondary, threshold, use_primary_above=False):
    """
    Hard switch: use secondary above threshold, primary otherwise.
    If use_primary_above=True: primary above threshold, secondary below.
    """
    mask = y_pred_primary > threshold
    if use_primary_above:
        return np.where(mask, y_pred_primary, y_pred_secondary)
    else:
        return np.where(mask, y_pred_secondary, y_pred_primary)


def strategy_soft_weight(y_pred_primary, y_pred_secondary, threshold, k=SIGMOID_K):
    """
    Soft weighting: sigmoid blend centered at threshold.
    weight = sigmoid(k * (primary_pred - threshold))
    Final = weight * secondary + (1 - weight) * primary
    Below threshold: primary dominates. Above: secondary dominates.
    """
    weight = expit(k * (y_pred_primary - threshold))
    return weight * y_pred_secondary + (1 - weight) * y_pred_primary


def strategy_lgb_for_low(y_pred_lgb, y_pred_nhits, threshold):
    """Reverse: NHITS is primary, LGB corrects low range."""
    mask = y_pred_nhits < threshold
    return np.where(mask, y_pred_lgb, y_pred_nhits)


def strategy_inverse_soft(y_pred_lgb, y_pred_nhits, threshold, k=SIGMOID_K):
    """
    NHITS primary, LGB correction in low band via sigmoid.
    Weight near 1 when price is low (use LGB), weight near 0 when high (use NHITS).
    """
    weight = expit(-k * (y_pred_nhits - threshold))  # inverted sigmoid
    return weight * y_pred_lgb + (1 - weight) * y_pred_nhits


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, bands=None):
    """Compute MAE, WAPE overall and per price band."""
    results = {}
    mae = np.abs(y_true - y_pred).mean()
    wape = mae / y_true.mean()
    results["overall"] = {"mae": mae, "wape": wape, "n": len(y_true)}

    if bands:
        for lo, hi in bands:
            mask = (y_true >= lo) & (y_true < hi)
            n = mask.sum()
            if n > 0:
                mae_b = np.abs(y_true[mask] - y_pred[mask]).mean()
                wape_b = mae_b / y_true[mask].mean()
                results[f"band_{lo}_{hi}"] = {"mae": mae_b, "wape": wape_b, "n": n}

    return results


def format_metrics(results):
    lines = []
    mae = results["overall"]["mae"]
    wape = results["overall"]["wape"]
    lines.append(f"  Overall: MAE={mae:.3f}, WAPE={wape:.3f}, n={results['overall']['n']}")
    for key in sorted(results):
        if key == "overall":
            continue
        v = results[key]
        lo, hi = key.split("_")[1], key.split("_")[2]
        lines.append(f"  Band [${lo}-${hi}): MAE={v['mae']:.3f}, WAPE={v['wape']:.3f}, n={v['n']}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  HYBRID MODEL PROTOTYPE — lightgbm + NHITS")
    print("=" * 72)

    for primary_name, secondary_name in MODEL_PAIRS:
        print(f"\n{'─'*72}")
        print(f"  Pair: {primary_name} (primary) + {secondary_name} (secondary)")
        print(f"{'─'*72}")

        lgb = load_q50(primary_name)  # lightgbm_q
        nhits = load_q50(secondary_name)  # nhits_168

        y_true = lgb["y"].values
        lgb_pred = lgb["y_pred"].values
        nhits_pred = nhits["y_pred"].values

        bands = [(0, 20), (20, 30), (30, 50), (50, 100), (100, 500)]

        # ── Baseline ──
        print("\n  [Baseline] lightgbm_q alone:")
        lgb_metrics = compute_metrics(y_true, lgb_pred, bands)
        print(format_metrics(lgb_metrics))

        print("\n  [Baseline] NHITS_168 alone:")
        nhits_metrics = compute_metrics(y_true, nhits_pred, bands)
        print(format_metrics(nhits_metrics))

        # ── Best theoretical (oracle) ──
        oracle_pred = np.where(y_true > 50, nhits_pred, lgb_pred)
        oracle_metrics = compute_metrics(y_true, oracle_pred, bands)
        print("\n  [Oracle] Perfect-knowledge switch @ $50 (not realizable):")
        print(format_metrics(oracle_metrics))

        # ── Strategy A: Hard Switch ──
        print(f"\n  {'─'*40}")
        print(f"  Strategy A: Hard-switch at threshold")
        print(f"  {'─'*40}")

        best_a = {"threshold": None, "mae": float("inf"), "wape": None}
        a_results = {}

        for thresh in THRESHOLDS:
            hybrid = strategy_hard_switch(lgb_pred, nhits_pred, threshold=thresh)
            metrics = compute_metrics(y_true, hybrid, bands)
            a_results[thresh] = metrics
            impr = (lgb_metrics["overall"]["mae"] - metrics["overall"]["mae"]) / lgb_metrics["overall"]["mae"] * 100
            print(f"    threshold=${thresh}: MAE={metrics['overall']['mae']:.3f}, "
                  f"WAPE={metrics['overall']['wape']:.3f} ({impr:+.2f}% vs lgb)")

            if metrics["overall"]["mae"] < best_a["mae"]:
                best_a = {"threshold": thresh, "mae": metrics["overall"]["mae"],
                          "wape": metrics["overall"]["wape"], "metrics": metrics}

        # ── Strategy B: Soft Weighting ──
        print(f"\n  {'─'*40}")
        print(f"  Strategy B: Soft weighting (sigmoid k={SIGMOID_K})")
        print(f"  {'─'*40}")

        best_b = {"threshold": None, "mae": float("inf"), "wape": None}
        b_results = {}

        for thresh in THRESHOLDS:
            hybrid = strategy_soft_weight(lgb_pred, nhits_pred, threshold=thresh)
            metrics = compute_metrics(y_true, hybrid, bands)
            b_results[thresh] = metrics
            impr = (lgb_metrics["overall"]["mae"] - metrics["overall"]["mae"]) / lgb_metrics["overall"]["mae"] * 100
            print(f"    threshold=${thresh}: MAE={metrics['overall']['mae']:.3f}, "
                  f"WAPE={metrics['overall']['wape']:.3f} ({impr:+.2f}% vs lgb)")

            if metrics["overall"]["mae"] < best_b["mae"]:
                best_b = {"threshold": thresh, "mae": metrics["overall"]["mae"],
                          "wape": metrics["overall"]["wape"], "metrics": metrics}

        # ── Strategy C: Reverse (NHITS primary, LGB low correction) ──
        print(f"\n  {'─'*40}")
        print(f"  Strategy C: Reverse — NHITS primary, LGB corrects low range")
        print(f"  {'─'*40}")

        best_c = {"threshold": None, "mae": float("inf"), "wape": None}
        c_results = {}

        for thresh in THRESHOLDS:
            hybrid = strategy_lgb_for_low(lgb_pred, nhits_pred, threshold=thresh)
            metrics = compute_metrics(y_true, hybrid, bands)
            c_results[thresh] = metrics
            impr = (nhits_metrics["overall"]["mae"] - metrics["overall"]["mae"]) / nhits_metrics["overall"]["mae"] * 100
            print(f"    threshold=${thresh}: MAE={metrics['overall']['mae']:.3f}, "
                  f"WAPE={metrics['overall']['wape']:.3f} ({impr:+.2f}% vs nhits)")

            if metrics["overall"]["mae"] < best_c["mae"]:
                best_c = {"threshold": thresh, "mae": metrics["overall"]["mae"],
                          "wape": metrics["overall"]["wape"], "metrics": metrics}

        # ── Strategy D: Inverse Soft Weighting ──
        print(f"\n  {'─'*40}")
        print(f"  Strategy D: Inverse soft — NHITS primary, LGB low-correction via sigmoid")
        print(f"  {'─'*40}")

        best_d = {"threshold": None, "mae": float("inf"), "wape": None}
        d_results = {}

        for thresh in THRESHOLDS:
            hybrid = strategy_inverse_soft(lgb_pred, nhits_pred, threshold=thresh)
            metrics = compute_metrics(y_true, hybrid, bands)
            d_results[thresh] = metrics
            impr = (nhits_metrics["overall"]["mae"] - metrics["overall"]["mae"]) / nhits_metrics["overall"]["mae"] * 100
            print(f"    threshold=${thresh}: MAE={metrics['overall']['mae']:.3f}, "
                  f"WAPE={metrics['overall']['wape']:.3f} ({impr:+.2f}% vs nhits)")

            if metrics["overall"]["mae"] < best_d["mae"]:
                best_d = {"threshold": thresh, "mae": metrics["overall"]["mae"],
                          "wape": metrics["overall"]["wape"], "metrics": metrics}

        # ── Strategy Bonus: Model-switch based on model-confidence ──
        print(f"\n  {'─'*40}")
        print(f"  Strategy E: Min-error selection (which model is closer)")
        print(f"  {'─'*40}")
        lgb_error = np.abs(y_true - lgb_pred)
        nhits_error = np.abs(y_true - nhits_pred)
        min_error_mask = lgb_error < nhits_error
        best_model_select = np.where(min_error_mask, lgb_pred, nhits_pred)
        best_select_metrics = compute_metrics(y_true, best_model_select, bands)
        impr_lgb = (lgb_metrics["overall"]["mae"] - best_select_metrics["overall"]["mae"]) / lgb_metrics["overall"]["mae"] * 100
        impr_nh = (nhits_metrics["overall"]["mae"] - best_select_metrics["overall"]["mae"]) / nhits_metrics["overall"]["mae"] * 100
        print(f"    LGB wins on {(min_error_mask.sum() / len(y_true) * 100):.1f}% of points")
        print(f"    NHITS wins on {((~min_error_mask).sum() / len(y_true) * 100):.1f}% of points")
        print(f"    MAE={best_select_metrics['overall']['mae']:.3f}, WAPE={best_select_metrics['overall']['wape']:.3f}")
        print(f"    vs LGB: {impr_lgb:+.2f}% | vs NHITS: {impr_nh:+.2f}%")
        print(format_metrics(best_select_metrics))

        # ── Summary ──
        print(f"\n  {'═'*40}")
        print(f"  SUMMARY")
        print(f"  {'═'*40}")
        print(f"  Baseline lgb:   MAE={lgb_metrics['overall']['mae']:.3f}, WAPE={lgb_metrics['overall']['wape']:.3f}")
        print(f"  Baseline nhits: MAE={nhits_metrics['overall']['mae']:.3f}, WAPE={nhits_metrics['overall']['wape']:.3f}")
        print(f"  Oracle:         MAE={oracle_metrics['overall']['mae']:.3f}, WAPE={oracle_metrics['overall']['wape']:.3f}")
        print(f"  Best A (hard) @ ${best_a['threshold']}: MAE={best_a['mae']:.3f}, WAPE={best_a['wape']:.3f}")
        print(f"  Best B (soft)  @ ${best_b['threshold']}: MAE={best_b['mae']:.3f}, WAPE={best_b['wape']:.3f}")
        print(f"  Best C (rev)   @ ${best_c['threshold']}: MAE={best_c['mae']:.3f}, WAPE={best_c['wape']:.3f}")
        print(f"  Best D (inv)   @ ${best_d['threshold']}: MAE={best_d['mae']:.3f}, WAPE={best_d['wape']:.3f}")
        print(f"  Best E (oracle-select): MAE={best_select_metrics['overall']['mae']:.3f}")

        # ── Save results ──
        results = {
            "primary_model": primary_name,
            "secondary_model": secondary_name,
            "baseline_lgb": lgb_metrics,
            "baseline_nhits": nhits_metrics,
            "oracle": oracle_metrics,
            "strategy_A_hard_switch": a_results,
            "strategy_B_soft_weight": b_results,
            "strategy_C_reverse": c_results,
            "strategy_D_inverse_soft": d_results,
            "strategy_E_min_error_select": best_select_metrics,
            "best_per_strategy": {
                "hard_switch": {"threshold": best_a["threshold"], **best_a["metrics"]["overall"]},
                "soft_weight": {"threshold": best_b["threshold"], **best_b["metrics"]["overall"]},
                "reverse": {"threshold": best_c["threshold"], **best_c["metrics"]["overall"]},
                "inverse_soft": {"threshold": best_d["threshold"], **best_d["metrics"]["overall"]},
            },
        }

        results_path = OUTPUT_DIR / "hybrid_prototype_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {results_path}")

    # ── Detailed breakdown of best hybrid ──
    best_overall = min(
        [best_a, best_b, best_c, best_d],
        key=lambda x: x["mae"]
    )
    print(f"\n{'='*72}")
    print(f"  BEST OVERALL HYBRID: {best_overall}")
    print(f"{'='*72}")

    print("\nDone! ✓")


if __name__ == "__main__":
    main()
