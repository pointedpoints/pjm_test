from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def run_prepare_data(config_path: str) -> None:
    workspace = Workspace.open(config_path)
    workspace.prepare()
    feature_df = workspace.feature_frame()
    print(f"Prepared dataset with {len(feature_df)} rows.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_prepare_data(args.config)


if __name__ == "__main__":
    main()
