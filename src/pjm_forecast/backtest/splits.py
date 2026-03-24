from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_split_boundaries(path: Path) -> dict[str, pd.Timestamp]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: pd.Timestamp(value) for key, value in payload.items()}


def get_daily_split_days(feature_df: pd.DataFrame, split_boundaries: dict[str, pd.Timestamp], split_name: str) -> list[pd.Timestamp]:
    days = pd.Index(feature_df["ds"].dt.normalize().drop_duplicates().sort_values())
    if split_name == "validation":
        mask = (days >= split_boundaries["validation_start"]) & (days <= split_boundaries["validation_end"])
        return list(days[mask])
    if split_name == "test":
        mask = (days >= split_boundaries["test_start"]) & (days <= split_boundaries["test_end"])
        return list(days[mask])
    raise ValueError(f"Unsupported split name: {split_name}")

