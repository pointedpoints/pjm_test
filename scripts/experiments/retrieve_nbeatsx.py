from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def run_retrieve_nbeatsx(config_path: str, split: str = "test") -> None:
    Workspace.open(config_path).retrieve_nbeatsx(split=split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    run_retrieve_nbeatsx(args.config, split=args.split)


if __name__ == "__main__":
    main()
