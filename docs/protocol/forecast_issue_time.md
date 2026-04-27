# Forecast Issue Time Protocol

This project keeps the v1 time protocol: timestamps are timezone-naive local
market time, and there is no UTC remapping.

## Forecast Target

- Target: COMED day-ahead LMP, represented by canonical column `y`.
- Horizon: one complete forecast day, 24 hourly rows.
- Forecast issue assumption: the forecast is issued before the first hour of the
  forecast day.
- Backtest equivalent: for forecast day `D`, history ends at `D 00:00 - 1h` and
  the future frame is `D 00:00` through `D 23:00`.

## Allowed Inputs

- Future-known load and weather forecast columns configured in
  `features.future_exog`.
- Calendar features derived from each future `ds`.
- Horizon-aligned historical lags where the value is already known at issue
  time, for example `future_price_lag_168(ds) = y(ds - 168h)`.
- Previous complete-day statistics when the source day ends before the forecast
  issue time, for example `prior_day_price_max` and
  `prior_day_price_max_ramp`.

## Boundary Inputs

- `price_lag_24` and other short lags are only valid when the forecast issue
  assumption makes the complete source hours available. Treat them as
  documented experiment choices, not silent defaults.
- Energy, congestion, and loss component history may be used only as historical
  state or diagnostics when those component columns exist locally and are known
  before issue time.

## Forbidden Inputs

- Any horizon-internal realized LMP.
- Any horizon-internal realized energy, congestion, or loss component.
- Any postprocess correction fitted on test targets.
- Any feature that updates hour by hour inside the forecast horizon as if the
  task were a rolling one-hour-ahead forecast.
