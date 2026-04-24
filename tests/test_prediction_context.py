from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from pjm_forecast.config import ProjectConfig, load_config
from pjm_forecast.prediction_context import inject_prediction_context_dir, inject_prediction_context_frame


def _config(tmp_path: Path) -> ProjectConfig:
    base = load_config(Path("configs/pjm_day_ahead_v1.yaml"))
    raw = deepcopy(base.raw)
    raw["project"]["root_override"] = str(tmp_path)
    raw["project"]["directories"]["processed_data_dir"] = "processed"
    raw["models"]["quantile_dummy"] = {"loss_name": "mqloss", "quantiles": [0.1, 0.5, 0.9]}
    return ProjectConfig(raw=raw, path=base.path)


def _prediction_frame(split: str = "test") -> pd.DataFrame:
    ds_values = pd.date_range("2026-01-01 00:00:00", periods=2, freq="h")
    rows = []
    for ds in ds_values:
        for quantile, value in [(0.1, 8.0), (0.5, 10.0), (0.9, 12.0)]:
            rows.append(
                {
                    "ds": ds,
                    "y": 10.0,
                    "y_pred": value,
                    "model": "quantile_dummy",
                    "split": split,
                    "seed": 7,
                    "quantile": quantile,
                    "metadata": "{}",
                }
            )
    return pd.DataFrame(rows)


def test_inject_prediction_context_dir_joins_context_and_validates_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=2, freq="h"),
            "spike_score": [0.2, 0.8],
        }
    ).to_parquet(processed_dir / "feature_store.parquet", index=False)

    source_dir = tmp_path / "source_predictions"
    source_dir.mkdir()
    _prediction_frame("test").to_parquet(source_dir / "quantile_dummy_test_seed7.parquet", index=False)

    output_dir = tmp_path / "context_predictions"
    results = inject_prediction_context_dir(
        config,
        source_prediction_dir=source_dir,
        output_prediction_dir=output_dir,
        context_columns=["spike_score"],
        splits=["test"],
    )

    assert len(results) == 1
    output = pd.read_parquet(output_dir / "quantile_dummy_test_seed7.parquet")
    assert output["spike_score"].tolist() == [0.2, 0.2, 0.2, 0.8, 0.8, 0.8]
    assert list(output.columns[:3]) == ["ds", "y", "spike_score"]


def test_inject_prediction_context_frame_rejects_missing_context_values() -> None:
    prediction_frame = _prediction_frame("test")
    context_frame = pd.DataFrame({"ds": [pd.Timestamp("2026-01-01 00:00:00")], "spike_score": [0.2]})

    with pytest.raises(ValueError, match="missing values"):
        inject_prediction_context_frame(
            prediction_frame,
            context_frame,
            context_columns=["spike_score"],
        )


def test_inject_prediction_context_frame_rejects_invalid_context_columns() -> None:
    prediction_frame = _prediction_frame("test")
    context_frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=2, freq="h"),
            "spike_score": [0.2, 0.8],
        }
    )

    with pytest.raises(ValueError, match="must be unique"):
        inject_prediction_context_frame(
            prediction_frame,
            context_frame,
            context_columns=["spike_score", "spike_score"],
        )

    with pytest.raises(ValueError, match="prediction contract columns"):
        inject_prediction_context_frame(
            prediction_frame,
            context_frame,
            context_columns=["ds"],
        )


def test_inject_prediction_context_frame_rejects_existing_context_without_replace() -> None:
    prediction_frame = _prediction_frame("test")
    prediction_frame["spike_score"] = 0.1
    context_frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=2, freq="h"),
            "spike_score": [0.2, 0.8],
        }
    )

    with pytest.raises(ValueError, match="already contains context columns"):
        inject_prediction_context_frame(
            prediction_frame,
            context_frame,
            context_columns=["spike_score"],
        )

    enriched = inject_prediction_context_frame(
        prediction_frame,
        context_frame,
        context_columns=["spike_score"],
        replace_existing_context=True,
    )
    assert enriched["spike_score"].tolist() == [0.2, 0.2, 0.2, 0.8, 0.8, 0.8]


def test_inject_prediction_context_dir_rejects_feature_store_missing_context_column(tmp_path: Path) -> None:
    config = _config(tmp_path)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=2, freq="h"),
            "other_context": [0.2, 0.8],
        }
    ).to_parquet(processed_dir / "feature_store.parquet", index=False)

    source_dir = tmp_path / "source_predictions"
    source_dir.mkdir()
    _prediction_frame("test").to_parquet(source_dir / "quantile_dummy_test_seed7.parquet", index=False)

    with pytest.raises(ValueError, match="missing context columns"):
        inject_prediction_context_dir(
            config,
            source_prediction_dir=source_dir,
            output_prediction_dir=tmp_path / "context_predictions",
            context_columns=["spike_score"],
            splits=["test"],
        )


def test_inject_prediction_context_dir_rejects_overwrite_by_default(tmp_path: Path) -> None:
    config = _config(tmp_path)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=2, freq="h"),
            "spike_score": [0.2, 0.8],
        }
    ).to_parquet(processed_dir / "feature_store.parquet", index=False)

    source_dir = tmp_path / "source_predictions"
    output_dir = tmp_path / "context_predictions"
    source_dir.mkdir()
    output_dir.mkdir()
    _prediction_frame("test").to_parquet(source_dir / "quantile_dummy_test_seed7.parquet", index=False)
    (output_dir / "quantile_dummy_test_seed7.parquet").write_bytes(b"already here")

    with pytest.raises(FileExistsError, match="already exists"):
        inject_prediction_context_dir(
            config,
            source_prediction_dir=source_dir,
            output_prediction_dir=output_dir,
            context_columns=["spike_score"],
            splits=["test"],
        )


def test_inject_prediction_context_dir_rejects_non_prediction_parquet(tmp_path: Path) -> None:
    config = _config(tmp_path)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=2, freq="h"),
            "spike_score": [0.2, 0.8],
        }
    ).to_parquet(processed_dir / "feature_store.parquet", index=False)

    source_dir = tmp_path / "source_predictions"
    source_dir.mkdir()
    pd.DataFrame({"ds": [pd.Timestamp("2026-01-01 00:00:00")], "value": [1.0]}).to_parquet(
        source_dir / "not_a_prediction.parquet",
        index=False,
    )

    with pytest.raises(ValueError, match="missing required metadata columns"):
        inject_prediction_context_dir(
            config,
            source_prediction_dir=source_dir,
            output_prediction_dir=tmp_path / "context_predictions",
            context_columns=["spike_score"],
            splits=["test"],
        )


def test_inject_prediction_context_dir_rejects_mixed_prediction_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 00:00:00", periods=2, freq="h"),
            "spike_score": [0.2, 0.8],
        }
    ).to_parquet(processed_dir / "feature_store.parquet", index=False)

    source_dir = tmp_path / "source_predictions"
    source_dir.mkdir()
    mixed = _prediction_frame("test")
    mixed.loc[mixed.index[-1], "split"] = "validation"
    mixed.to_parquet(source_dir / "mixed_prediction.parquet", index=False)

    with pytest.raises(ValueError, match="mixed metadata values"):
        inject_prediction_context_dir(
            config,
            source_prediction_dir=source_dir,
            output_prediction_dir=tmp_path / "context_predictions",
            context_columns=["spike_score"],
            splits=["test"],
        )
