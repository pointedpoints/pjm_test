from __future__ import annotations

import pandas as pd

from pjm_forecast.models.seasonal_naive import SeasonalNaiveModel


def test_seasonal_naive_uses_requested_lag() -> None:
    model = SeasonalNaiveModel(seasonal_lag_hours=24)
    history = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01 00:00:00", periods=48, freq="h"),
            "y": list(range(48)),
        }
    )
    future = pd.DataFrame({"ds": pd.date_range("2020-01-03 00:00:00", periods=24, freq="h")})
    model.fit(history)
    predictions = model.predict(history_df=history, future_df=future)
    assert predictions["y_pred"].iloc[0] == 24
