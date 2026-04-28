from __future__ import annotations

import argparse
from pathlib import Path

from pjm_forecast.ops import export_configured_model_snapshot


def run_export_model_snapshot(
    config_path: str,
    model_name: str | None = None,
    snapshot_name: str | None = None,
) -> Path:
    output_dir = export_configured_model_snapshot(
        config_path,
        model_name=model_name,
        snapshot_name=snapshot_name,
    )
    print(f"Exported model snapshot to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--snapshot-name")
    args = parser.parse_args()
    run_export_model_snapshot(
        args.config,
        model_name=args.model_name,
        snapshot_name=args.snapshot_name,
    )


if __name__ == "__main__":
    main()
