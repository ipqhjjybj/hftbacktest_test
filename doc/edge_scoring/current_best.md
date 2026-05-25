# Current Best Edge-Scored Maker Candidates

更新时间: 2026-05-25

## Old 9-Factor Monthly Best

This is the current monthly benchmark.

```text
period: 20260418-20260518
model: 9 factors, with interactions
model tag: edge_score_wf_train14_20260418_20260518_h250_maker0_with_if_filled
strategy result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_regime_risk_grid_full_p000_a00005.aggregate.csv
```

Parameters:

```text
expected-edge-threshold-bps: 0.10
edge-if-filled-threshold-bps: 0.05
fill-prob-threshold: 0.05
intraday-percentile-gate: on
expected-edge-percentile: 0.95
edge-if-filled-percentile: 0.95
regime-expected-edge-gate: on
regime-min-bid/ask-expected-edge-bps: 0.00
regime-alpha: 0.00005
order_qty: 100
max_position: 300
soft_position: 100
reduce_only_after_soft_position: on
daily_loss_limit_usdt: 0.10
daily_fill_limit: 300
```

Monthly result:

```text
total PnL +0.550429
fills 4177
positive days 19/31
zero-fill days 2
4 月 -0.075700
5 月 +0.626129
last 7 days +0.496950
worst day 20260420 -0.081350
max position max 100
```

## New 22-Factor 7-Day Candidate

This is the current research candidate. It is not yet monthly-validated.

```text
period: 20260512-20260518
model: 22 factors, no interactions
model tag: edge_score_wf_train14_20260512_20260518_h250_new_hf_nointer
strategy result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_e95_i97_m001_a00005_20260512_20260518.aggregate.csv
```

Parameters:

```text
expected-edge-threshold-bps: 0.10
edge-if-filled-threshold-bps: 0.05
fill-prob-threshold: 0.05
intraday-percentile-gate: on
expected-edge-percentile: 0.95
edge-if-filled-percentile: 0.97
regime-expected-edge-gate: on
regime-min-bid/ask-expected-edge-bps: 0.01
regime-alpha: 0.00005
order_qty: 100
max_position: 300
soft_position: 100
reduce_only_after_soft_position: on
daily_loss_limit_usdt: 0.10
daily_fill_limit: 300
```

7-day result:

```text
total PnL +0.559350
fills 1220
PnL/fill +0.00045848
positive days 6/7
worst day 20260512 -0.054200
best day 20260514 +0.216500
avg capture 5.8889 USDT/BTC
max position max 100
```

Same-period old 9f benchmark:

```text
total PnL +0.496950
fills 1490
PnL/fill +0.00033352
positive days 6/7
worst day 20260518 -0.018150
```

## Next Validation

Run a full monthly new22 walk-forward and strategy backtest:

```text
period: 20260418-20260518
model: 22 factors, no interactions first
strategy: e95/i97/regime_min=0.01
```

Decision rule:

```text
If monthly PnL and 4 月 segment improve versus old 9f best:
  promote new22 to current main candidate.

If only last 7 days improve but 4 月 worsens:
  keep old 9f as current main candidate and treat new22 as research branch.
```
