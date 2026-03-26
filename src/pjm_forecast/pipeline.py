from __future__ import annotations

from typing import Callable

from scripts.backtest_all_models import run_backtest_all_models
from scripts.evaluate_and_plot import run_evaluate_and_plot
from scripts.export_report_assets import run_export_report_assets
from scripts.prepare_data import run_prepare_data
from scripts.retrieve_nbeatsx import run_retrieve_nbeatsx
from scripts.tune_nbeatsx import run_tune_nbeatsx


STAGE_ORDER = [
    "prepare_data",
    "tune_nbeatsx",
    "backtest_all_models",
    "retrieve_nbeatsx",
    "evaluate_and_plot",
    "export_report_assets",
]


STAGE_FUNCTIONS: dict[str, Callable[..., None]] = {
    "prepare_data": run_prepare_data,
    "tune_nbeatsx": run_tune_nbeatsx,
    "backtest_all_models": run_backtest_all_models,
    "retrieve_nbeatsx": run_retrieve_nbeatsx,
    "evaluate_and_plot": run_evaluate_and_plot,
    "export_report_assets": run_export_report_assets,
}


def run_pipeline(config_path: str, split: str = "test", start_from: str = STAGE_ORDER[0], stop_after: str = STAGE_ORDER[-1]) -> None:
    start_index = STAGE_ORDER.index(start_from)
    stop_index = STAGE_ORDER.index(stop_after)
    if start_index > stop_index:
        raise ValueError("start_from must come before or match stop_after.")

    for stage_name in STAGE_ORDER[start_index : stop_index + 1]:
        stage_fn = STAGE_FUNCTIONS[stage_name]
        if stage_name in {"prepare_data", "tune_nbeatsx"}:
            stage_fn(config_path)
        else:
            stage_fn(config_path, split=split)
