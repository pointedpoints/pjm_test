from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def run_tune_model(config_path: str) -> None:
    Workspace.open(config_path).tune_model()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the model named by tuning.model_name in the config.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_tune_model(args.config)


if __name__ == "__main__":
    main()
