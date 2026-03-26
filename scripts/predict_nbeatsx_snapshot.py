from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pjm_forecast.models.nbeatsx import NBEATSxModel


def run_predict_nbeatsx_snapshot(snapshot_path: str, history_path: str, future_path: str, output_path: str) -> None:
    model = NBEATSxModel.load(Path(snapshot_path))
    history_df = pd.read_parquet(history_path)
    future_df = pd.read_parquet(future_path)
    predictions = model.predict(history_df=history_df, future_df=future_df)
    predictions.to_parquet(output_path, index=False)
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
