#!/bin/bash
# run_spike_cleaned_full_pipeline.sh — AutoDL spike-cleaned v2 pipeline
#
# Prerequisites:
#   panel_cleaned_v2.parquet already uploaded to:
#     /root/autodl-tmp/pjm_remaster/data/processed_spike_filtered_v2/
#
# Environment:
#   PYTHONPATH=/root/autodl-tmp/pjm_remaster/src
#   Python: /root/miniconda3/envs/pjm/bin/python
#
# Output: artifacts_spike_cleaned_v2/

set -euo pipefail

PROJECT_DIR=/root/autodl-tmp/pjm_remaster
PYTHON=/root/miniconda3/envs/pjm/bin/python
OUTPUT_DIR=$PROJECT_DIR/artifacts_spike_cleaned_v2
DATA_DIR=$PROJECT_DIR/data/processed_spike_filtered_v2
LOG_FILE=$PROJECT_DIR/pipeline.log

cd "$PROJECT_DIR"
export PYTHONPATH=$PROJECT_DIR/src

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline started" | tee -a "$LOG_FILE"

# ============================================================
# Step 0: Verify input exists
# ============================================================
INPUT_PANEL=$DATA_DIR/panel_cleaned_v2.parquet
if [ ! -f "$INPUT_PANEL" ]; then
    echo "[ERROR] Input panel not found: $INPUT_PANEL" | tee -a "$LOG_FILE"
    exit 1
fi
echo "[0] Input panel found: $INPUT_PANEL" | tee -a "$LOG_FILE"

# ============================================================
# Step 1: Copy panel to processed_current location
# ============================================================
mkdir -p "$PROJECT_DIR/data/processed_current"
cp "$INPUT_PANEL" "$PROJECT_DIR/data/processed_current/panel_spike_filtered_v2.parquet"
echo "[1] Copied panel to data/processed_current/panel_spike_filtered_v2.parquet" | tee -a "$LOG_FILE"

# ============================================================
# Step 2: Create temporary config from multi-model baseline
# ============================================================
TMP_CONFIG=$PROJECT_DIR/configs/pjm_spike_filtered_v2_tmp.yaml
BASE_CONFIG=$PROJECT_DIR/configs/tmp_full_compare_q15_multi.yaml

# Override key fields: project name, artifact paths, weather off
$PYTHON -c "
import yaml
with open('$BASE_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('project', {})['name'] = 'pjm_spike_filtered_v2'
cfg['project']['directories'] = {
    'raw_data_dir': 'data/raw',
    'processed_data_dir': 'data/processed_current',
    'artifact_dir': 'artifacts_spike_cleaned_v2',
    'hyperparameter_dir': 'artifacts_spike_cleaned_v2/hyperparameters',
    'prediction_dir': 'artifacts_spike_cleaned_v2/predictions',
    'metrics_dir': 'artifacts_spike_cleaned_v2/metrics',
    'plots_dir': 'artifacts_spike_cleaned_v2/plots',
    'report_dir': 'artifacts_spike_cleaned_v2/report',
}
cfg['weather'] = {'enabled': False}
cfg['backtest']['rolling_window_days'] = 728
with open('$TMP_CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print('Config written:', '$TMP_CONFIG')
"

echo "[2] Created temp config: $TMP_CONFIG" | tee -a "$LOG_FILE"

# ============================================================
# Step 3: Run prepare_data.py
# ============================================================
echo "[3] Running prepare_data.py..." | tee -a "$LOG_FILE"
$PYTHON scripts/prepare_data.py --config "$TMP_CONFIG" 2>&1 | tee -a "$LOG_FILE"
echo "[3] prepare_data.py done" | tee -a "$LOG_FILE"

# ============================================================
# Step 4: Run backtest_all_models.py
# Models: seasonal_naive lightgbm_q xgboost_q nhits_168 nhits_336 nhits_720 nhits_tail_grid_weighted_main
# ============================================================
echo "[4] Running backtest_all_models.py --split test..." | tee -a "$LOG_FILE"
$PYTHON scripts/backtest_all_models.py --config "$TMP_CONFIG" --split test 2>&1 | tee -a "$LOG_FILE"
echo "[4] backtest_all_models.py done" | tee -a "$LOG_FILE"

# ============================================================
# Step 5: Run evaluate_and_plot.py
# ============================================================
echo "[5] Running evaluate_and_plot.py --split test..." | tee -a "$LOG_FILE"
$PYTHON scripts/evaluate_and_plot.py --config "$TMP_CONFIG" --split test 2>&1 | tee -a "$LOG_FILE"
echo "[5] evaluate_and_plot.py done" | tee -a "$LOG_FILE"

# ============================================================
# Step 6: Run ensemble if script exists
# ============================================================
ENSEMBLE_SCRIPT=$PROJECT_DIR/scripts/experiments/ensemble_predictions.py
if [ -f "$ENSEMBLE_SCRIPT" ]; then
    echo "[6] Running ensemble_predictions.py..." | tee -a "$LOG_FILE"
    $PYTHON "$ENSEMBLE_SCRIPT" \
        --predictions-dir "$OUTPUT_DIR/predictions" \
        --split test 2>&1 | tee -a "$LOG_FILE"
    echo "[6] ensemble_predictions.py done" | tee -a "$LOG_FILE"
else
    echo "[6] Ensemble script not found, skipping integration step." | tee -a "$LOG_FILE"
fi

# ============================================================
# Summary
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline complete" | tee -a "$LOG_FILE"
echo "  Config:   $TMP_CONFIG" | tee -a "$LOG_FILE"
echo "  Artifacts: $OUTPUT_DIR/" | tee -a "$LOG_FILE"
echo "  Log:      $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
