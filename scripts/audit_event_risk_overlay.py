from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def run_audit_event_risk_overlay(config_path: str, split: str = "test") -> None:
    output_dir = Workspace.open(config_path).audit_event_risk_overlay(split=split)
    print(f"Event-risk tail overlay audit written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()
    run_audit_event_risk_overlay(args.config, split=args.split)


if __name__ == "__main__":
    main()
