from __future__ import annotations

import argparse
import shutil

from pjm_forecast.config import load_config
from pjm_forecast.paths import ensure_project_directories


def run_export_report_assets(config_path: str, split: str = "test") -> None:
    config = load_config(config_path)
    directories = ensure_project_directories(config)
    report_dir = directories["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    for source in [
        directories["metrics_dir"] / f"{split}_metrics.csv",
        directories["metrics_dir"] / f"{split}_dm.csv",
        directories["plots_dir"] / f"{split}_hourly_mae.png",
        directories["plots_dir"] / f"{split}_high_vol_week.png",
    ]:
        if source.exists():
            shutil.copy2(source, report_dir / source.name)
    print(f"Report assets exported to {report_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    run_export_report_assets(args.config, split=args.split)


if __name__ == "__main__":
    main()
