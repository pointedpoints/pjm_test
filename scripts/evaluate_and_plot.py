from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def run_evaluate_and_plot(config_path: str, split: str = "test") -> None:
    Workspace.open(config_path).evaluate(split=split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    run_evaluate_and_plot(args.config, split=args.split)


if __name__ == "__main__":
    main()
