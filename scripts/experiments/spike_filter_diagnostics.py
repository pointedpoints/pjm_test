from __future__ import annotations

import argparse

from pjm_forecast.workspace import Workspace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    args = parser.parse_args()
    output_path = Workspace.open(args.config).compute_spike_filter_diagnostics(split=args.split)
    print(f"Wrote spike-filter diagnostics to {output_path}")


if __name__ == "__main__":
    main()
