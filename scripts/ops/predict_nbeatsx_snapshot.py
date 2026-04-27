from __future__ import annotations

import argparse

from pjm_forecast.ops import predict_model_snapshot


def run_predict_nbeatsx_snapshot(snapshot_path: str, history_path: str, future_path: str, output_path: str) -> None:
    predict_model_snapshot(snapshot_path, history_path, future_path, output_path)
    print(f"Wrote predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-path", required=True)
    parser.add_argument("--history-path", required=True)
    parser.add_argument("--future-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    run_predict_nbeatsx_snapshot(
        snapshot_path=args.snapshot_path,
        history_path=args.history_path,
        future_path=args.future_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
