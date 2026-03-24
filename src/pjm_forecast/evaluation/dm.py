from __future__ import annotations

import numpy as np
from scipy.stats import t


def dm_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, power: int = 1) -> dict[str, float]:
    error_a = np.abs(y_true - y_pred_a) ** power
    error_b = np.abs(y_true - y_pred_b) ** power
    d = error_a - error_b
    n = len(d)
    if n < 2:
        raise ValueError("DM test requires at least two observations.")

    mean_d = float(np.mean(d))
    variance_d = float(np.var(d, ddof=1))
    if variance_d == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = mean_d / np.sqrt(variance_d / n)
        p_value = float(2 * (1 - t.cdf(abs(statistic), df=n - 1)))
    return {"statistic": statistic, "p_value": p_value}

