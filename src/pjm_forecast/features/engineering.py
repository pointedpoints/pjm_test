from __future__ import annotations

import pandas as pd

from pjm_forecast.config import ProjectConfig
from pjm_forecast.prepared_data import FeatureSchema


def build_feature_frame(config: ProjectConfig, panel_df: pd.DataFrame) -> pd.DataFrame:
    return FeatureSchema(config).build_feature_frame(panel_df)


def nbeatsx_futr_exog_columns(config: ProjectConfig) -> list[str]:
    return FeatureSchema(config).nbeatsx_futr_exog_columns()


def nbeatsx_hist_exog_columns(config: ProjectConfig) -> list[str]:
    return FeatureSchema(config).nbeatsx_hist_exog_columns()
