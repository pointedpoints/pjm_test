from __future__ import annotations

import argparse
from pathlib import Path

from pjm_forecast.data.official_weather_ready import build_comed_weather_ready_dataset, save_weather_ready_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/raw/PJM_COMED_20210101_20260331_weather_ready.csv")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    dataset = build_comed_weather_ready_dataset(
        Path(args.raw_dir),
        start=args.start,
        end=args.end,
    )
    output_path = save_weather_ready_dataset(dataset, Path(args.output))
    print(f"Wrote {len(dataset.frame)} rows to {output_path}")
    print(f"Range: {dataset.start} -> {dataset.end}")


if __name__ == "__main__":
    main()
