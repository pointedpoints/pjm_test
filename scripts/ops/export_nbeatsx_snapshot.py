from __future__ import annotations

import argparse

from pjm_forecast.ops import export_nbeatsx_snapshot


def run_export_nbeatsx_snapshot(config_path: str):
    output_dir = export_nbeatsx_snapshot(config_path)
    print(f"Exported NBEATSx snapshot to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_export_nbeatsx_snapshot(args.config)


if __name__ == "__main__":
    main()
