from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    "prepare_data.py",
    "tune_nbeatsx.py",
    "backtest_all_models.py",
    "retrieve_nbeatsx.py",
    "evaluate_and_plot.py",
    "export_report_assets.py",
]


def run_step(script_name: str, config_path: str, split: str) -> None:
    script_path = Path(__file__).with_name(script_name)
    command = [sys.executable, str(script_path), "--config", config_path]
    if script_name in {"backtest_all_models.py", "retrieve_nbeatsx.py", "evaluate_and_plot.py", "export_report_assets.py"}:
        command.extend(["--split", split])

    print(f"\n=== Running {script_name} ===")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full PJM forecasting pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["validation", "test"],
        help="Backtest/evaluation split for downstream stages.",
    )
    parser.add_argument(
        "--start-from",
        default=SCRIPT_ORDER[0],
        choices=SCRIPT_ORDER,
        help="Resume pipeline from a specific stage.",
    )
    parser.add_argument(
        "--stop-after",
        default=SCRIPT_ORDER[-1],
        choices=SCRIPT_ORDER,
        help="Stop pipeline after a specific stage.",
    )
    args = parser.parse_args()

    start_index = SCRIPT_ORDER.index(args.start_from)
    stop_index = SCRIPT_ORDER.index(args.stop_after)
    if start_index > stop_index:
        raise ValueError("--start-from must come before or match --stop-after.")

    for script_name in SCRIPT_ORDER[start_index : stop_index + 1]:
        run_step(script_name=script_name, config_path=args.config, split=args.split)


if __name__ == "__main__":
    main()
