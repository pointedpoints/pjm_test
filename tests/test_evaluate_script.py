from __future__ import annotations

from pathlib import Path

from scripts.evaluate_and_plot import PREDICTION_FILENAME


def test_prediction_filename_pattern_ignores_backup_suffixes() -> None:
    assert PREDICTION_FILENAME.fullmatch(Path("nbeatsx_test_seed7").stem)
    assert PREDICTION_FILENAME.fullmatch("nbeatsx_rag_test_seed7")
    assert PREDICTION_FILENAME.fullmatch("nbeatsx_test_seed7_20260326_pre_ensemble") is None
