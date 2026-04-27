from __future__ import annotations

from typing import Callable

from .workspace import Workspace


STAGE_ORDER = [
    "prepare_data",
    "tune_model",
    "backtest_all_models",
    "evaluate_and_plot",
    "export_report_assets",
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


def _run_export(workspace: Workspace, split: str) -> None:
    workspace.export_report(split=split)


STAGE_FUNCTIONS: dict[str, Callable[[Workspace, str], None]] = {
    "prepare_data": _run_prepare,
    "tune_model": _run_tune,
    "backtest_all_models": _run_backtest,
    "evaluate_and_plot": _run_evaluate,
    "export_report_assets": _run_export,
}


def run_pipeline(config_path: str, split: str = "test", start_from: str = STAGE_ORDER[0], stop_after: str = STAGE_ORDER[-1]) -> None:
    workspace = Workspace.open(config_path)
    start_index = STAGE_ORDER.index(start_from)
    stop_index = STAGE_ORDER.index(stop_after)
    if start_index > stop_index:
        raise ValueError("start_from must come before or match stop_after.")

    for stage_name in STAGE_ORDER[start_index : stop_index + 1]:
        STAGE_FUNCTIONS[stage_name](workspace, split)
