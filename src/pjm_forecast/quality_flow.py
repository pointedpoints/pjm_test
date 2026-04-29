from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd


CANONICAL_CANDIDATE = "CANONICAL_CANDIDATE"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
_EVENT_AUDIT_FILES = (
    "overlay_implementation_audit.json",
    "spike_score_audit.json",
    "width_by_regime.csv",
)


def build_quality_gate_summary(
    *,
    split: str,
    metrics_path: Path,
    quantile_diagnostics_path: Path,
    event_audit_dir: Path,
) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    diagnostics = pd.read_csv(quantile_diagnostics_path)
    merged = metrics.merge(diagnostics, on="run", how="left", suffixes=("", "_diagnostics"))
    event_audit = _load_event_audit(event_audit_dir)

    rows = []
    for _, source in merged.iterrows():
        row = {
            "split": split,
            "run": source["run"],
            "model": _coalesce(source.get("model"), source.get("model_diagnostics")),
            "seed": _coalesce(source.get("seed"), source.get("seed_diagnostics")),
            "pinball": _coalesce(source.get("pinball"), source.get("post_pinball")),
            "mae": _coalesce(source.get("mae"), source.get("post_q50_mae")),
            "post_crossing_rate": _coalesce(source.get("post_crossing_rate"), source.get("crossing_rate")),
            "post_q99_exceedance_rate": source.get("post_q99_exceedance_rate"),
            "post_q99_excess_mean": source.get("post_q99_excess_mean"),
            "post_worst_q99_underprediction": source.get("post_worst_q99_underprediction"),
            "post_width_98": source.get("post_width_98"),
            "event_audit_available": bool(event_audit["available"]),
            "spike_score_audit_status": event_audit["spike_score_audit_status"],
            "all_width98_ratio": event_audit["all_width98_ratio"],
            "normal_width98_ratio": event_audit["normal_width98_ratio"],
            "normal_width_status": _normal_width_status(event_audit["normal_width98_ratio"]),
        }
        row["decision"] = _quality_decision(row)
        rows.append(row)

    summary = pd.DataFrame(rows, columns=_summary_columns())
    if "event_audit_available" in summary:
        summary["event_audit_available"] = summary["event_audit_available"].astype(object)
    return summary


def build_run_manifest(
    *,
    split: str,
    config_path: Path,
    artifact_paths: list[Path],
    model_name: str,
    seed: int,
) -> dict[str, object]:
    return {
        "split": split,
        "config_path": str(config_path),
        "model_name": model_name,
        "seed": int(seed),
        "artifacts": _artifact_records(artifact_paths),
    }


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_event_audit(event_audit_dir: Path) -> dict[str, object]:
    available = event_audit_dir.is_dir() and all((event_audit_dir / name).exists() for name in _EVENT_AUDIT_FILES)
    spike_score_audit = _load_spike_score_audit(event_audit_dir / "spike_score_audit.json")
    ratios = _compute_regime_ratios(event_audit_dir / "width_by_regime.csv")
    return {
        "available": available,
        "spike_score_audit_status": spike_score_audit.get("availability_status", "MISSING"),
        "all_width98_ratio": ratios.get("all", float("nan")),
        "normal_width98_ratio": ratios.get("normal", float("nan")),
    }


def _load_spike_score_audit(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_regime_ratios(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    ratios: dict[str, float] = {}
    for _, row in frame.iterrows():
        regime = str(row.get("regime", ""))
        ratio = _row_width98_ratio(row)
        if regime and not pd.isna(ratio):
            ratios[regime] = float(ratio)
    return ratios


def _row_width98_ratio(row: pd.Series) -> float:
    if "width98_ratio" in row:
        return float(row["width98_ratio"])
    before = _coalesce(row.get("before_width_98"), row.get("baseline_width_98"), row.get("raw_width_98"))
    after = _coalesce(row.get("after_width_98"), row.get("candidate_width_98"), row.get("post_width_98"))
    if pd.isna(before) or float(before) == 0.0 or pd.isna(after):
        return float("nan")
    return float(after) / float(before)


def _artifact_records(artifact_paths: list[Path]) -> list[dict[str, object]]:
    return [_artifact_record(path) for path in artifact_paths]


def _artifact_record(path: Path) -> dict[str, object]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "sha256": _sha256(path) if exists and path.is_file() else None,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quality_decision(row: Mapping[str, object]) -> str:
    if not bool(row["event_audit_available"]):
        return REVIEW_REQUIRED
    if row["spike_score_audit_status"] != "PASS":
        return REVIEW_REQUIRED
    if _missing_required_diagnostics(row):
        return REVIEW_REQUIRED
    if float(row["post_crossing_rate"]) != 0.0:
        return REVIEW_REQUIRED
    if float(row["post_q99_exceedance_rate"]) > 0.025:
        return REVIEW_REQUIRED
    return CANONICAL_CANDIDATE


def _missing_required_diagnostics(row: Mapping[str, object]) -> bool:
    required = ("post_crossing_rate", "post_q99_exceedance_rate")
    return any(pd.isna(row.get(name)) for name in required)


def _normal_width_status(normal_width_ratio: object) -> str:
    if pd.isna(normal_width_ratio):
        return "WARN"
    return "PASS" if float(normal_width_ratio) <= 1.05 else "WARN"


def _summary_columns() -> list[str]:
    return [
        "split",
        "run",
        "model",
        "seed",
        "pinball",
        "mae",
        "post_crossing_rate",
        "post_q99_exceedance_rate",
        "post_q99_excess_mean",
        "post_worst_q99_underprediction",
        "post_width_98",
        "event_audit_available",
        "spike_score_audit_status",
        "all_width98_ratio",
        "normal_width98_ratio",
        "normal_width_status",
        "decision",
    ]


def _coalesce(*values: object) -> object:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        return value
    return float("nan")
