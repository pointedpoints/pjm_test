import pandas as pd
import numpy as np
from pathlib import Path

pred_dir = Path("/mnt/d/pjm_remaster/artifacts_current/predictions")

def load_full(path):
    df = pd.read_parquet(path)
    df["ds"] = pd.to_datetime(df["ds"])
    # Keep only median quantile
    if "quantile" in df.columns and df["quantile"].notna().any():
        mask = np.isclose(df["quantile"].astype(float), 0.5)
        if mask.any():
            df = df[mask]
    df["hour"] = df["ds"].dt.hour
    df["month"] = df["ds"].dt.month
    df["dow"] = df["ds"].dt.dayofweek
    df["date"] = df["ds"].dt.date
    df["ape"] = np.abs(df["y"] - df["y_pred"]) / np.abs(df["y"]) * 100
    df["error"] = df["y_pred"] - df["y"]  # positive = overpredict
    df["price_bin"] = pd.cut(df["y"], bins=[0, 15, 25, 40, 100, float("inf")],
                             labels=["<$15", "$15-25", "$25-40", "$40-100", ">$100"])
    return df

lightgbm = load_full(pred_dir / "lightgbm_q_validation_seed7.parquet")
nhits = load_full(pred_dir / "nhits_tail_grid_weighted_main_validation_seed7.parquet")

# ============ 1. Overall ============
print("="*70)
print("OVERALL COMPARISON")
print("="*70)
for name, df in [("NHITS", nhits), ("lightgbm_q", lightgbm)]:
    print(f"\n{name}:")
    print(f"  MAE   = {np.mean(np.abs(df['y'] - df['y_pred'])):.4f}")
    print(f"  RMSE  = {np.sqrt(np.mean((df['y'] - df['y_pred'])**2)):.4f}")
    print(f"  SMAPE = {200 * np.mean(np.abs(df['y'] - df['y_pred']) / (np.abs(df['y']) + np.abs(df['y_pred']))):.4f}")
    print(f"  MdAPE = {np.median(df['ape']):.2f}%")
    print(f"  P90 APE = {np.percentile(df['ape'], 90):.2f}%")
    print(f"  Mean bias = {np.mean(df['error']):.4f} (positive=overpredict)")
    print(f"  Bias (abs<1% price) = {np.mean(np.abs(df['error']) < 0.15):.2%} of hours")

# ============ 2. Per price bin ============
print("\n\n" + "="*70)
print("BY PRICE BIN (relative error & bias)")
print("="*70)
for price_bin in ["<$15", "$15-25", "$25-40", "$40-100", ">$100"]:
    lb = lightgbm[lightgbm["price_bin"] == price_bin]
    nh = nhits[nhits["price_bin"] == price_bin]
    if len(lb) == 0:
        continue
    print(f"\n  {price_bin} ({len(lb)}h / {len(nh)}h):")
    print(f"    {'':20s} {'lightgbm':>10s} {'NHITS':>10s} {'diff':>10s}")
    print(f"    {'MAE':20s} {np.mean(np.abs(lb['y'] - lb['y_pred'])):10.4f} "
          f"{np.mean(np.abs(nh['y'] - nh['y_pred'])):10.4f} "
          f"{np.mean(np.abs(nh['y'] - nh['y_pred'])) - np.mean(np.abs(lb['y'] - lb['y_pred'])):+10.4f}")
    print(f"    {'MdAPE':20s} {np.median(lb['ape']):10.2f}% "
          f"{np.median(nh['ape']):10.2f}% ")
    print(f"    {'P90 APE':20s} {np.percentile(lb['ape'], 90):10.2f}% "
          f"{np.percentile(nh['ape'], 90):10.2f}% ")
    print(f"    {'Mean bias':20s} {np.mean(lb['error']):10.4f} "
          f"{np.mean(nh['error']):10.4f} ")

# ============ 3. By hour ============
print("\n\n" + "="*70)
print("BY HOUR OF DAY (MAE)")
print("="*70)
print(f"  {'Hour':6s} {'lightgbm':>10s} {'NHITS':>10s} {'diff':>10s} {'NHITS_worse?':>12s}")
for h in range(24):
    lb = lightgbm[lightgbm["hour"] == h]
    nh = nhits[nhits["hour"] == h]
    lb_mae = np.mean(np.abs(lb['y'] - lb['y_pred']))
    nh_mae = np.mean(np.abs(nh['y'] - nh['y_pred']))
    worse = "❌" if nh_mae > lb_mae + 0.5 else ("✅" if nh_mae < lb_mae - 0.5 else "≈")
    print(f"  {h:3d}:00 {lb_mae:10.4f} {nh_mae:10.4f} {nh_mae - lb_mae:+10.4f} {worse:>12s}")

# ============ 4. By day of week ============
print("\n\n" + "="*70)
print("BY DAY OF WEEK (MAE)")
print("="*70)
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for d in range(7):
    lb = lightgbm[lightgbm["dow"] == d]
    nh = nhits[nhits["dow"] == d]
    lb_mae = np.mean(np.abs(lb['y'] - lb['y_pred']))
    nh_mae = np.mean(np.abs(nh['y'] - nh['y_pred']))
    worse = "❌" if nh_mae > lb_mae + 0.5 else ("✅" if nh_mae < lb_mae - 0.5 else "≈")
    print(f"  {days[d]:5s} {lb_mae:10.4f} {nh_mae:10.4f} {nh_mae - lb_mae:+10.4f} {worse:>12s}")

# ============ 5. Extreme events ============
print("\n\n" + "="*70)
print("EXTREME EVENTS (actual price > $40)")
print("="*70)
for name, df in [("NHITS", nhits), ("lightgbm_q", lightgbm)]:
    extreme = df[df["y"] > 40]
    print(f"\n  {name} ({len(extreme)} extreme hours):")
    print(f"    MAE    = {np.mean(np.abs(extreme['y'] - extreme['y_pred'])):.4f}")
    print(f"    MdAPE  = {np.median(extreme['ape']):.2f}%")
    print(f"    Bias   = {np.mean(extreme['error']):.4f} ({'overpredict' if np.mean(extreme['error']) > 0 else 'underpredict'})")
    print(f"    P90 APE= {np.percentile(extreme['ape'], 90):.2f}%")
    print(f"    Worst 5 APE: {extreme.nlargest(5, 'ape')[['ds', 'y', 'y_pred', 'ape']].to_string(index=False)}")

# ============ 6. Low price regime ============
print("\n\n" + "="*70)
print("LOW PRICE REGIME (actual price < $20)")
print("="*70)
for name, df in [("NHITS", nhits), ("lightgbm_q", lightgbm)]:
    low = df[df["y"] < 20]
    print(f"\n  {name} ({len(low)} hours):")
    print(f"    MAE    = {np.mean(np.abs(low['y'] - low['y_pred'])):.4f}")
    print(f"    MdAPE  = {np.median(low['ape']):.2f}%")
    print(f"    P90 APE= {np.percentile(low['ape'], 90):.2f}%")

# ============ 7. Daily P90 APE ============
print("\n\n" + "="*70)
print("DAILY P90 APE COMPARISON")
print("="*70)
lb_daily = lightgbm.groupby("date")["ape"].apply(lambda g: np.percentile(g, 90))
nh_daily = nhits.groupby("date")["ape"].apply(lambda g: np.percentile(g, 90))
print(f"  lightgbm: mean P90 APE = {lb_daily.mean():.2f}%, median = {lb_daily.median():.2f}%")
print(f"  NHITS:    mean P90 APE = {nh_daily.mean():.2f}%, median = {nh_daily.median():.2f}%")
print(f"  NHITS worse on {np.mean(nh_daily > lb_daily) * 100:.0f}% of days")

# Worst days for NHITS
diff = nh_daily - lb_daily
print(f"\n  Top 5 days where NHITS is much worse than lightgbm:")
for date in diff.nlargest(5).index:
    print(f"    {date.date()}: NHITS P90={nh_daily[date]:.1f}%, lightgbm P90={lb_daily[date]:.1f}%, gap={diff[date]:.1f}pp")
