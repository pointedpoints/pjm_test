# Feature Availability Matrix

All features promoted into canonical or experiment configs should be classed
before training. The goal is to avoid hidden leakage and to keep P50 and tail
experiments comparable.

| Feature | Class | Available at issue time | Leakage risk | Recommended use | Transform |
|---|---|---:|---:|---|---|
| `zonal_load_forecast` | future-known | yes | low | `futr_exog` | zscore |
| `weather_temp_mean` | future-known | yes | low | `futr_exog` | zscore |
| `weather_temp_spread` | future-known | yes | low | `futr_exog` | zscore |
| `weather_wind_speed_mean` | future-known | yes | low | `futr_exog` | zscore |
| `weather_cloud_cover_mean` | future-known | yes | low | `futr_exog` | zscore |
| `weather_precip_area_fraction` | future-known | yes | low | `futr_exog` | zscore |
| `heating_degree_18` | derived future-known | yes | low | `futr_exog` or `spike_score` | zscore |
| `cooling_degree_22` | derived future-known | yes | low | `futr_exog` or `spike_score` | zscore |
| `load_cooling_pressure` | derived future-known | yes | low | `futr_exog` or `spike_score` | zscore |
| `load_heating_pressure` | derived future-known | yes | low | experiment candidate | zscore |
| `zonal_load_forecast_delta_24` | derived future-known | yes | low | experiment candidate | zscore |
| `future_price_lag_168` | horizon-aligned historical lag | yes | low | experiment candidate `futr_exog` | target-like price transform |
| `future_price_lag_336` | horizon-aligned historical lag | yes | low | experiment candidate `futr_exog` | target-like price transform |
| `price_lag_24` | history-window lag | boundary | medium | documented experiment or hist input | target-like price transform |
| `price_lag_168` | history-window lag | yes | low | hist input | target-like price transform |
| `prior_day_price_max` | previous complete-day state | boundary | medium | experiment candidate or `spike_score` | target-like price transform |
| `prior_day_price_spread` | previous complete-day state | boundary | medium | experiment candidate or `spike_score` | zscore |
| `prior_day_price_max_ramp` | previous complete-day state | boundary | medium | `spike_score` context | zscore |
| `spike_score` | regime context | derived from allowed inputs | low after historical scoring | postprocess context only | rolling historical percentile [0, 1] |
| Horizon realized LMP | ground truth | no | high | forbidden | forbidden |
| Horizon realized energy | ground truth | no | high | forbidden | forbidden |
| Horizon realized congestion/loss | ground truth | no | high | forbidden | forbidden |

## Promotion Rule

A feature can enter a mainline config only when:

- its availability class is documented here,
- it does not depend on horizon-internal realized values,
- validation and test move in the same direction,
- normal-regime P50 does not materially degrade,
- tail improvement is not only width inflation.
