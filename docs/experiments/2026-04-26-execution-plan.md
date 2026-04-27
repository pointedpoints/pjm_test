# Prediction Quality Execution Plan

## Purpose

This document locks the next execution path for prediction-quality work. Future
changes should follow this plan in order and should not promote side paths into
the mainline unless they pass the gates defined here.

The phase timeboxes in this document are planning hints only. Execution speed
should be maximized where practical, but experiment order and promotion gates
remain fixed.

Source of truth for this plan:

- `D:/BaiduNetdiskDownload/deep-research-report (1).md`

## Fixed Decisions

- Do not pursue energy / congestion / loss target decomposition as a mainline
  path. The local COMED DA sample does not support the assumption that the
  energy component is smoother than total LMP.
- Do not keep iterating on single-model tail reweighting as the main strategy
  for improving both `P50` and `q99`. Recent repo evidence already shows that
  median-focused tweaks can help `q50_mae` while regressing global pinball,
  CRPS, or scenario quality.
- Keep the current canonical runnable unchanged until a planned experiment is
  validated and promoted with the gates below.
- Treat calibration and ensemble as standard post-model steps, not as a
  substitute for weak model or feature design.

## Phase Order

### Phase 1: Benchmark Floor And P50 Mainline

Scheduling note: no fixed timebox; execute as fast as practical.

Goal:

- Establish an indisputable benchmark floor.
- Rebuild a `P50`-friendly mainline before adding more tail pressure.

Required experiment set:

- `E2`: `Naive-168`
- `E3`: `LEAR`
- `E5`: `LightGBM quantile`
- `E6`: `XGBoost quantile`
- `E8`: current `NHITS` full compare
- `E9`: `P50`-friendly `NHITS` / `NBEATSx`

Required feature work:

- Restore `system_load_forecast`
- Restore `price_lag_168`
- Restore `source_lag_168`
- Restore `prior_day_price_max` and `prior_day_price_max_ramp`

Required protocol work:

- Keep `24h` horizon and current rolling split protocol.
- Compare `hour` vs `hour_x_regime` calibration explicitly.
- Use validation-first promotion; do not skip directly to test.

Phase 1 exit gate:

- The best candidate must beat `Naive-168` on validation with significance from
  `DM` or paired bootstrap as appropriate.
- `P50` candidate must improve or at least directionally tighten `q50_mae` /
  `MAE` versus the current mainline without a material regression in pinball or
  CRPS.
- No candidate is promoted if the gain is only median-side while global
  probabilistic quality regresses.

### Phase 2: Tail Expert And Spike-Aware Two-Stage

Scheduling note: no fixed timebox; start immediately after Phase 1 gates are
resolved.

Goal:

- Improve upper-tail and spike behavior without giving back Phase 1 `P50`
  quality.

Required experiment set:

- `E10`: weighted long-grid tail expert
- `E11`: spike-aware two-stage blend
- `E14`: heterogeneous ensemble / QRA on approved inputs

Required model work:

- Build a spike classifier first.
- Keep a separate tail expert rather than forcing one model to optimize every
  regime.
- Blend only upper quantiles when the spike gate is active; do not disturb the
  center of the distribution by default.

Required calibration work:

- Standardize `cqr_asymmetric + hour_x_regime`
- Grid-search `regime_threshold` on validation
- Enforce `min_group_size` safeguards to avoid noisy calibration groups

Phase 2 exit gate:

- Test `q99_excess_mean` and `daily_max_q99_gap_mean` should improve by a
  business-visible amount, with `10%-20%` as the target band from the report.
- `MAE` must not materially worsen relative to the approved Phase 1 mainline.
- `post_crossing_rate` must remain `0`.

### Phase 3: Retrieval And Scenario Layer

Scheduling note: no fixed timebox; defer only until Phases 1 and 2 are stable.

Goal:

- Upgrade from point-only residual memory into quantile-aware retrieval and
  richer joint-distribution modeling.

Allowed only after:

- Phase 1 benchmark floor is complete.
- Phase 2 spike-aware tail path is stable.

Candidate work:

- Quantile-aware retrieval
- Richer market-fundamental pipeline
- Scenario / flow / copula layer improvements

Phase 3 rule:

- Do not start this phase while basic `P50` and upper-tail issues are still
  unresolved in the core backtest.

## Immediate Queue

This is the locked near-term execution order:

1. `E2`
2. `E3`
3. `E5`
4. `E6`
5. `E8`
6. `E9`
7. `hour_x_regime` calibration comparison
8. Promote one Phase 1 mainline candidate or declare no-promotion
9. `E10`
10. `E11`
11. `E14`

## Non-Goals

- No component-target forecasting mainline
- No ad hoc one-off feature detours outside the ordered experiment queue
- No early retrieval-first strategy
- No full scenario-generation push before spike-aware tail work lands
- No canonical promotion from a single metric win

## Working Rules

- Every planned branch must have one config, one experiment note, and one clear
  validation/test decision.
- `artifacts/report/` remains a derived view; promotion decisions must come
  from metrics and diagnostics artifacts.
- Canonical changes require both validation and test evidence.
- If an experiment fails its gate, record the decision and return to the next
  item in this plan instead of expanding the scope.
