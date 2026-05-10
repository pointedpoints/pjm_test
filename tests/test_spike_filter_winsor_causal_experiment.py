from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import yaml


MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "experiments" / "spike_filter_winsor_causal_experiment.py"


spec = importlib.util.spec_from_file_location("spike_filter_winsor_causal_experiment", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)


def _frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=len(values), freq="h"),
            "y": values,
        }
    )


def test_causal_winsorization_is_prefix_invariant() -> None:
    prefix = _frame([10.0, 12.0, 11.0, 13.0, 12.0, 14.0])
    full = _frame([10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 15.0, 200.0, 16.0])

    cleaned_prefix, diagnostics_prefix = module.apply_causal_winsorization(
        prefix,
        trend_window_hours=2,
        quantile_window_hours=2,
        quantile=0.9,
    )
    cleaned_full, diagnostics_full = module.apply_causal_winsorization(
        full,
        trend_window_hours=2,
        quantile_window_hours=2,
        quantile=0.9,
    )

    overlap = cleaned_full[cleaned_full["ds"].isin(prefix["ds"])]
    overlap_diag = diagnostics_full[diagnostics_full["ds"].isin(prefix["ds"])]

    pd.testing.assert_frame_equal(cleaned_prefix.reset_index(drop=True), overlap.reset_index(drop=True))
    pd.testing.assert_frame_equal(diagnostics_prefix.reset_index(drop=True), overlap_diag.reset_index(drop=True))



def test_causal_winsorization_caps_upper_residual_without_interpolation() -> None:
    frame = _frame([10.0, 12.0, 11.0, 13.0, 12.0, 100.0])

    cleaned, diagnostics = module.apply_causal_winsorization(
        frame,
        trend_window_hours=2,
        quantile_window_hours=2,
        quantile=0.9,
    )

    last = diagnostics.iloc[-1]
    assert bool(last["is_capped"])
    assert cleaned.iloc[-1]["y"] == last["y_clean"]
    assert last["y_clean"] == last["trend"] + last["residual_cap"]
    assert last["y_clean"] < frame.iloc[-1]["y"]
    assert last["cap_delta"] > 0.0



def test_causal_winsorization_leaves_rows_unchanged_without_enough_history() -> None:
    frame = _frame([10.0, 11.0, 12.0, 50.0])

    cleaned, diagnostics = module.apply_causal_winsorization(
        frame,
        trend_window_hours=4,
        quantile_window_hours=4,
        quantile=0.98,
    )

    assert diagnostics["is_capped"].sum() == 0
    pd.testing.assert_series_equal(cleaned["y"], frame["y"], check_names=False)



def test_default_output_paths_are_repo_relative_even_for_nested_experiment_configs() -> None:
    root = Path(__file__).resolve().parent.parent
    config_path = root / "configs" / "experiments" / "pjm_spike_smoke_v2_winsor_causal.yaml"

    assert module._default_output_csv_path(config_path) == (
        root / "data" / "raw" / "PJM_COMED_20210101_20260331_weather_ready_spike_winsor_causal.csv"
    )
    assert module._default_output_dir(config_path) == (root / "data" / "processed_spike_filtered_v2_winsor_causal")



def test_new_winsor_configs_are_isolated_from_old_v2() -> None:
    root = Path(__file__).resolve().parent.parent
    old_full = yaml.safe_load((root / "configs" / "pjm_baseline_spike_v2.yaml").read_text(encoding="utf-8"))
    new_full = yaml.safe_load(
        (root / "configs" / "experiments" / "pjm_baseline_spike_v2_winsor_causal.yaml").read_text(encoding="utf-8")
    )
    old_smoke = yaml.safe_load((root / "configs" / "pjm_spike_smoke_v2.yaml").read_text(encoding="utf-8"))
    new_smoke = yaml.safe_load(
        (root / "configs" / "experiments" / "pjm_spike_smoke_v2_winsor_causal.yaml").read_text(encoding="utf-8")
    )

    assert old_full["dataset"]["local_csv_path"] != new_full["dataset"]["local_csv_path"]
    assert old_full["project"]["name"] != new_full["project"]["name"]
    assert old_full["project"]["directories"]["artifact_dir"] != new_full["project"]["directories"]["artifact_dir"]
    assert old_smoke["dataset"]["local_csv_path"] != new_smoke["dataset"]["local_csv_path"]
    assert old_smoke["project"]["name"] != new_smoke["project"]["name"]
    assert old_smoke["project"]["directories"]["artifact_dir"] != new_smoke["project"]["directories"]["artifact_dir"]
