from __future__ import annotations

import argparse

from pjm_forecast.config import load_config
from pjm_forecast.data import build_split_boundaries, download_dataset_if_needed, load_panel_dataset, save_split_boundaries
from pjm_forecast.features import build_feature_frame, save_feature_frame
from pjm_forecast.paths import ensure_project_directories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    directories = ensure_project_directories(config)

    csv_path = download_dataset_if_needed(config, directories["raw_data_dir"])
    panel_df = load_panel_dataset(config, csv_path)
    feature_df = build_feature_frame(config, panel_df)
    split_boundaries = build_split_boundaries(config, panel_df)

    save_feature_frame(feature_df, directories["processed_data_dir"] / "feature_store.parquet")
    save_split_boundaries(split_boundaries, directories["processed_data_dir"] / "split_boundaries.json")
    panel_df.to_parquet(directories["processed_data_dir"] / "panel.parquet", index=False)
    print(f"Prepared dataset with {len(feature_df)} rows.")


if __name__ == "__main__":
    main()

