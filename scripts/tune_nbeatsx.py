from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def run_tune_nbeatsx(config_path: str) -> None:
    Workspace.open(config_path).tune_nbeatsx()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_tune_nbeatsx(args.config)


if __name__ == "__main__":
    main()
