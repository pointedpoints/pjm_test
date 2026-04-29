from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def run_finalize_quality_flow(config_path: str, split: str = "test") -> None:
    written = Workspace.open(config_path).finalize_quality_flow(split=split)
    for path in written:
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    run_finalize_quality_flow(args.config, split=args.split)


if __name__ == "__main__":
    main()
