from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def run_export_report_assets(config_path: str, split: str = "test") -> None:
    copied = Workspace.open(config_path).export_report(split=split)
    print(f"Report assets exported: {len(copied)} files.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    run_export_report_assets(args.config, split=args.split)


if __name__ == "__main__":
    main()
