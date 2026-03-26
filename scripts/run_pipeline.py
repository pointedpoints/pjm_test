from __future__ import annotations

import argparse

from pjm_forecast.pipeline import STAGE_ORDER, run_pipeline


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
        default=STAGE_ORDER[0],
        choices=STAGE_ORDER,
        help="Resume pipeline from a specific stage.",
    )
    parser.add_argument(
        "--stop-after",
        default=STAGE_ORDER[-1],
        choices=STAGE_ORDER,
        help="Stop pipeline after a specific stage.",
    )
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        split=args.split,
        start_from=args.start_from,
        stop_after=args.stop_after,
    )


if __name__ == "__main__":
    main()
