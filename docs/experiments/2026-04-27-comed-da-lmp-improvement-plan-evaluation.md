# COMED DA LMP Improvement Plan Evaluation

Input plan: `D:/BaiduNetdiskDownload/comed_da_lmp_improvement_process.md`.

## Verdict

The plan is directionally sound and matches the current project decision to stay
on total COMED DA LMP, keep NHITS as the preferred neural family, and avoid
component-sum forecasting for now.

The highest-value immediate work is infrastructure, not another large training
run:

- fix the forecast issue-time contract,
- document feature availability,
- add regime metrics so P50 and upper-tail trade-offs are visible,
- support horizon-aligned future-known price lags,
- keep `spike_score` as context until a rolling-safe version is proven.

## Scope Applied In This Pass

Implemented the low-risk foundation needed before the next experiment run:

- `docs/protocol/forecast_issue_time.md`
- `docs/protocol/feature_availability_matrix.md`
- `src/pjm_forecast/evaluation/regime_metrics.py`
- evaluator/export support for `{split}_regime_metrics.csv`
- `future_known_lag` derived feature support for candidates such as
  `future_price_lag_168`

## Deferred

The following remain experiment work and should not be promoted without
validation/test evidence:

- rolling-safe `spike_score`,
- CQR threshold grid over `0.50 / 0.60 / 0.67 / 0.75`,
- mild tail-weight/delta grid,
- tail expert or gated blend,
- final spike-basis ablation.
