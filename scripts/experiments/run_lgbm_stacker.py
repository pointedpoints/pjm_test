from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def run_lgbm_stacker(config_path: str, split: str = "test") -> None:
    Workspace.open(config_path).run_lgbm_stacker(split=split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    run_lgbm_stacker(args.config, split=args.split)


if __name__ == "__main__":
    main()
