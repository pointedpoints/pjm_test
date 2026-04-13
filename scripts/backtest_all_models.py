from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def run_backtest_all_models(config_path: str, split: str = "test") -> None:
    Workspace.open(config_path).backtest(split=split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    run_backtest_all_models(args.config, split=args.split)


if __name__ == "__main__":
    main()
