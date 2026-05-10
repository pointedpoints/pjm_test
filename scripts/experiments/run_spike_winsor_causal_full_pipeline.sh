#!/bin/bash
# run_spike_winsor_causal_full_pipeline.sh — local causal-Winsorization pipeline

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
elif [ -x "/mnt/d/pjm_remaster/.venv/bin/python" ]; then
  PYTHON_BIN="/mnt/d/pjm_remaster/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi
SOURCE_CONFIG="${SOURCE_CONFIG:-$PROJECT_DIR/configs/pjm_baseline_raw.yaml}"
CONFIG_PATH="${1:-$PROJECT_DIR/configs/experiments/pjm_baseline_spike_v2_winsor_causal.yaml}"
SPLIT="${SPLIT:-test}"
SOURCE_CSV="${SOURCE_CSV:-/mnt/d/pjm_remaster/data/raw/PJM_COMED_20210101_20260331_weather_ready.csv}"
OUTPUT_CSV="$PROJECT_DIR/data/raw/PJM_COMED_20210101_20260331_weather_ready_spike_winsor_causal.csv"
OUTPUT_DIR="$PROJECT_DIR/data/processed_spike_filtered_v2_winsor_causal"

mkdir -p "$PROJECT_DIR/data/raw"
export PYTHONPATH="$PROJECT_DIR/src"
RESET_OUTPUTS="${RESET_OUTPUTS:-1}"

if [ "$RESET_OUTPUTS" = "1" ]; then
  mapfile -t RUN_PATHS < <("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys
from pjm_forecast.config import load_config
config = load_config(sys.argv[1])
print(config.resolve_path(str(config.project['directories']['processed_data_dir'])))
print(config.resolve_path(str(config.project['directories']['artifact_dir'])))
PY
  )
  for path in "${RUN_PATHS[@]}"; do
    rm -rf "$path"
  done
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/experiments/spike_filter_winsor_causal_experiment.py" \
  --config "$SOURCE_CONFIG" \
  --source-csv "$SOURCE_CSV" \
  --output-csv "$OUTPUT_CSV" \
  --output-dir "$OUTPUT_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/prepare_data.py" --config "$CONFIG_PATH"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/backtest_all_models.py" --config "$CONFIG_PATH" --split "$SPLIT"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_and_plot.py" --config "$CONFIG_PATH" --split "$SPLIT"
