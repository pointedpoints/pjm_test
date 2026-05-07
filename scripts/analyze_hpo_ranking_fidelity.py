import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

pred_dir = Path("/mnt/d/pjm_remaster/artifacts_current/predictions")

models = {
    "seasonal_naive": pred_dir / "seasonal_naive_validation_seed7.parquet",
    "lightgbm_q": pred_dir / "lightgbm_q_validation_seed7.parquet",
    "xgboost_q": pred_dir / "xgboost_q_validation_seed7.parquet",
    "NHITS": pred_dir / "nhits_tail_grid_weighted_main_validation_seed7.parquet",
}

def load_predictions(path):
    df = pd.read_parquet(path)
    df["ds"] = pd.to_datetime(df["ds"])
    # Use point forecast (quantile=0.5) for ranking
    if "quantile" in df.columns and df["quantile"].notna().any():
        median_mask = np.isclose(df["quantile"].astype(float), 0.5)
        if median_mask.any():
            df = df[median_mask]
    # Compute daily metrics
    df["date"] = df["ds"].dt.date
    daily = df.groupby("date").apply(
        lambda g: pd.Series({
            "mae": np.mean(np.abs(g["y"] - g["y_pred"])),
            "rmse": np.sqrt(np.mean((g["y"] - g["y_pred"])**2)),
            "smape": 200 * np.mean(np.abs(g["y"] - g["y_pred"]) / (np.abs(g["y"]) + np.abs(g["y_pred"]))),
            "p90_ape": np.percentile(np.abs(g["y"] - g["y_pred"]) / np.abs(g["y"]), 90) * 100 if (g["y"] != 0).all() else np.nan,
        })
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily

# Load all models
results = {}
for name, path in models.items():
    results[name] = load_predictions(path)
    print(f"{name}: {len(results[name])} days, MAE={results[name]['mae'].mean():.4f}")

# Full 182-day ranking
print("\n" + "="*70)
print("FULL 182-DAY RANKINGS")
print("="*70)

metrics = ["mae", "rmse", "smape"]
full_rankings = {}

for metric in metrics:
    scores = {name: df[metric].mean() for name, df in results.items()}
    ranking = pd.Series(scores).rank()
    full_rankings[metric] = ranking
    print(f"\n{metric.upper()}:")
    for name, rank in ranking.sort_values().items():
        print(f"  #{int(rank)} {name:20s} = {scores[name]:.4f}")

# === TEST 1: Positional strata (early/mid/late) ===
all_dates = sorted(results["seasonal_naive"]["date"].unique())
n = len(all_dates)

strata_sets = [
    all_dates[:14],                  # early ~2 weeks
    all_dates[n//2-7:n//2+7],       # middle ~2 weeks
    all_dates[-14:],                 # late ~2 weeks
]
subset_dates_pos = sorted(set(d for s in strata_sets for d in s))

print(f"\n\nPositional strata: {len(subset_dates_pos)} days (early {len(strata_sets[0])} + mid {len(strata_sets[1])} + late {len(strata_sets[2])})")

print("\n" + "="*70)
print("STRATIFIED SUBSET RANKINGS (positional)")
print("="*70)

for metric in metrics:
    scores = {}
    for name, df in results.items():
        subset = df[df["date"].isin(subset_dates_pos)]
        scores[name] = subset[metric].mean()
    ranking = pd.Series(scores).rank()
    print(f"\n{metric.upper()}:")
    for name, rank in ranking.sort_values().items():
        print(f"  #{int(rank)} {name:20s} = {scores[name]:.4f}")

print("\n" + "="*70)
print("SPEARMAN: Full vs Positional Subset")
print("="*70)

for metric in metrics:
    sub_scores = {}
    for name, df in results.items():
        subset = df[df["date"].isin(subset_dates_pos)]
        sub_scores[name] = subset[metric].mean()
    sub_rank = pd.Series(sub_scores).rank()
    rho, p = spearmanr(full_rankings[metric], sub_rank)
    print(f"  {metric:6s}: \u03c1={rho:.4f} (p={p:.4f})")

# === TEST 2: Seasonal strata (1 week per month) ===
print("\n\n" + "="*70)
print("SEASONAL STRATA: 1 week per available month")
print("="*70)

seasonal_dates = []
for month in sorted(set(d.month for d in all_dates)):
    month_dates = [d for d in all_dates if d.month == month]
    seasonal_dates.extend(sorted(month_dates)[:7])
seasonal_dates = sorted(set(seasonal_dates))

print(f"Seasonal subset: {len(seasonal_dates)} days across {len(set(d.month for d in seasonal_dates))} months")
for month in sorted(set(d.month for d in seasonal_dates)):
    count = sum(1 for d in seasonal_dates if d.month == month)
    print(f"  Month {month}: {count} days")

print("\n" + "="*70)
print("SPEARMAN: Full vs Seasonal Strata")
print("="*70)

for metric in metrics:
    scores = {}
    for name, df in results.items():
        subset = df[df["date"].isin(seasonal_dates)]
        scores[name] = subset[metric].mean()
    sub_rank = pd.Series(scores).rank()
    rho, p = spearmanr(full_rankings[metric], sub_rank)
    print(f"  {metric:6s}: \u03c1={rho:.4f} (p={p:.4f})")
    print(f"    Full:    {full_rankings[metric].sort_values().to_dict()}")
    print(f"    Seasonal:{sub_rank.sort_values().to_dict()}")

# === TEST 3: Random subsets (bootstrap) ===
print("\n\n" + "="*70)
print("BOOTSTRAP: Random 6-week subsets (100 trials)")
print("="*70)

np.random.seed(42)
subset_size = 42  # 6 weeks

for metric in metrics:
    rhos = []
    for _ in range(100):
        sample_dates = sorted(np.random.choice(all_dates, size=subset_size, replace=False))
        scores = {}
        for name, df in results.items():
            subset = df[df["date"].isin(sample_dates)]
            scores[name] = subset[metric].mean()
        sub_rank = pd.Series(scores).rank()
        rho, _ = spearmanr(full_rankings[metric], sub_rank)
        rhos.append(rho)
    print(f"  {metric:6s}: mean \u03c1={np.mean(rhos):.4f} \u00b1 {np.std(rhos):.4f}, min={np.min(rhos):.4f}, max={np.max(rhos):.4f}")
