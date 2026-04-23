from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from pjm_forecast.config import ProjectConfig
from pjm_forecast.prepared_data import FeatureSchema


@dataclass(frozen=True)
class ContextInjectionResult:
    source_path: Path
    output_path: Path
    rows: int
    model: str
    split: str
    seed: int
    context_columns: tuple[str, ...]


def inject_prediction_context_dir(
    config: ProjectConfig,
    *,
    source_prediction_dir: str | Path,
    output_prediction_dir: str | Path,
    context_columns: Iterable[str],
    splits: Iterable[str],
    models: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    overwrite: bool = False,
    replace_existing_context: bool = False,
) -> list[ContextInjectionResult]:
    source_dir = Path(source_prediction_dir)
    output_dir = Path(output_prediction_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source prediction directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Source prediction path is not a directory: {source_dir}")

    context_column_list = [str(column) for column in context_columns]
    if not context_column_list:
        raise ValueError("At least one context column is required.")

    requested_splits = {str(split) for split in splits}
    if not requested_splits:
        raise ValueError("At least one split is required.")

    requested_models = None if models is None else {str(model) for model in models}
    requested_seeds = None if seeds is None else {int(seed) for seed in seeds}

    schema = FeatureSchema(config)
    context_frame = _load_context_frame(config, context_column_list)

    results: list[ContextInjectionResult] = []
    for source_path in sorted(source_dir.glob("*.parquet")):
        metadata = pd.read_parquet(source_path, columns=["model", "split", "seed"])
        if metadata.empty:
            continue
        model = str(metadata["model"].iloc[0])
        split = str(metadata["split"].iloc[0])
        seed = int(metadata["seed"].iloc[0])
        if split not in requested_splits:
            continue
        if requested_models is not None and model not in requested_models:
            continue
        if requested_seeds is not None and seed not in requested_seeds:
            continue

        prediction_frame = pd.read_parquet(source_path)
        schema.validate_prediction_frame(prediction_frame, require_metadata=False, model_name=model)
        enriched = inject_prediction_context_frame(
            prediction_frame,
            context_frame,
            context_columns=context_column_list,
            replace_existing_context=replace_existing_context,
        )
        schema.validate_prediction_frame(enriched, require_metadata=False, model_name=model)

        output_path = output_dir / source_path.name
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output prediction file already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(output_path, index=False)
        results.append(
            ContextInjectionResult(
                source_path=source_path,
                output_path=output_path,
                rows=len(enriched),
                model=model,
                split=split,
                seed=seed,
                context_columns=tuple(context_column_list),
            )
        )

    if not results:
        raise FileNotFoundError(
            f"No prediction parquet files matched splits={sorted(requested_splits)}, "
            f"models={None if requested_models is None else sorted(requested_models)}, "
            f"seeds={None if requested_seeds is None else sorted(requested_seeds)} in {source_dir}."
        )
    return results


def inject_prediction_context_frame(
    prediction_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    *,
    context_columns: Iterable[str],
    replace_existing_context: bool = False,
) -> pd.DataFrame:
    context_column_list = [str(column) for column in context_columns]
    existing = [column for column in context_column_list if column in prediction_frame.columns]
    if existing and not replace_existing_context:
        raise ValueError(
            f"Prediction frame already contains context columns {existing}. "
            "Use replace_existing_context=True to replace them."
        )

    frame = prediction_frame.copy()
    frame["ds"] = pd.to_datetime(frame["ds"], utc=False)
    if existing:
        frame = frame.drop(columns=existing)

    required_context_columns = ["ds", *context_column_list]
    missing_context_columns = [column for column in required_context_columns if column not in context_frame.columns]
    if missing_context_columns:
        raise ValueError(f"Context frame is missing required columns: {missing_context_columns}")

    context = context_frame.loc[:, required_context_columns].copy()
    context["ds"] = pd.to_datetime(context["ds"], utc=False)
    if context["ds"].duplicated().any():
        duplicates = context.loc[context["ds"].duplicated(), "ds"].head(5).astype(str).tolist()
        raise ValueError(f"Context frame contains duplicate ds timestamps, examples: {duplicates}")

    enriched = frame.merge(context, on="ds", how="left", validate="many_to_one")
    missing_counts = enriched[context_column_list].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(f"Injected context has missing values: {missing_counts.to_dict()}")

    canonical_order = ["ds", "y", *context_column_list, "y_pred", "model", "split", "seed", "quantile", "metadata"]
    ordered_columns = [column for column in canonical_order if column in enriched.columns]
    ordered_columns.extend(column for column in enriched.columns if column not in ordered_columns)
    return enriched.loc[:, ordered_columns]


def _load_context_frame(config: ProjectConfig, context_columns: list[str]) -> pd.DataFrame:
    feature_store_path = config.resolve_path(config.project["directories"]["processed_data_dir"]) / "feature_store.parquet"
    if not feature_store_path.exists():
        raise FileNotFoundError(f"Feature store does not exist: {feature_store_path}")
    columns = ["ds", *context_columns]
    feature_frame = pd.read_parquet(feature_store_path)
    missing = [column for column in columns if column not in feature_frame.columns]
    if missing:
        raise ValueError(f"Feature store {feature_store_path} is missing context columns: {missing}")
    return feature_frame.loc[:, columns].copy()
