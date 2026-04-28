from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

from .workspace import Workspace


STAGE_ORDER = [
    "prepare_data",
    "tune_model",
    "backtest_all_models",
    "evaluate_and_plot",
    "audit_event_risk_overlay",
    "finalize_quality_flow",
    "export_report_assets",
    "export_model_snapshot",
]


def _run_prepare(workspace: Workspace, split: str) -> None:
    del split
    workspace.prepare()


def _run_tune(workspace: Workspace, split: str) -> None:
    del split
    workspace.tune_model()


def _run_backtest(workspace: Workspace, split: str) -> None:
    workspace.backtest(split=split)


def _run_evaluate(workspace: Workspace, split: str) -> None:
    workspace.evaluate(split=split)


def _run_audit_event_risk_overlay(workspace: Workspace, split: str) -> None:
    postprocess = workspace.config.raw.get("report", {}).get("quantile_postprocess", {})
    event_overlay = postprocess.get("event_risk_tail_overlay") if isinstance(postprocess, Mapping) else None
    if not isinstance(event_overlay, Mapping) or not event_overlay.get("enabled", False):
        return
    workspace.audit_event_risk_overlay(split=split)


def _run_export(workspace: Workspace, split: str) -> None:
    workspace.export_report(split=split)


def _run_finalize_quality_flow(workspace: Workspace, split: str) -> None:
    workspace.finalize_quality_flow(split=split)


def _run_export_model_snapshot(workspace: Workspace, split: str) -> None:
    del split
    model_name = str(workspace.config.tuning.get("model_name") or workspace.config.backtest["benchmark_models"][0])
    model_type = str(workspace.config.models.get(model_name, {}).get("type", "")).lower()
    if model_type not in {"nbeatsx", "nhits"}:
        return
    snapshot_name = f"{model_name}_snapshot"
    if workspace.artifacts.snapshot_manifest(snapshot_name).exists():
        return
    workspace.export_model_snapshot(model_name=model_name, snapshot_name=snapshot_name)


STAGE_FUNCTIONS: dict[str, Callable[[Workspace, str], None]] = {
    "prepare_data": _run_prepare,
    "tune_model": _run_tune,
    "backtest_all_models": _run_backtest,
    "evaluate_and_plot": _run_evaluate,
    "audit_event_risk_overlay": _run_audit_event_risk_overlay,
    "export_report_assets": _run_export,
    "finalize_quality_flow": _run_finalize_quality_flow,
    "export_model_snapshot": _run_export_model_snapshot,
}


def run_pipeline(config_path: str, split: str = "test", start_from: str = STAGE_ORDER[0], stop_after: str = STAGE_ORDER[-1]) -> None:
    workspace = Workspace.open(config_path)
    start_index = STAGE_ORDER.index(start_from)
    stop_index = STAGE_ORDER.index(stop_after)
    if start_index > stop_index:
        raise ValueError("start_from must come before or match stop_after.")

    for stage_name in STAGE_ORDER[start_index : stop_index + 1]:
        STAGE_FUNCTIONS[stage_name](workspace, split)
