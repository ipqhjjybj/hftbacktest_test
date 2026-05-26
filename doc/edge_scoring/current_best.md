# Current Best Edge-Scored Maker Candidates

更新时间: 2026-05-25

## Current Monthly Best

This is the current main candidate.

```text
period: 20260418-20260518
model: 22 factors, no interactions
model tag: edge_score_wf_train14_20260418_20260518_h250_new_hf_nointer
strategy result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_e95_i97_m001_a00005_20260418_20260518.aggregate.csv
```

Parameters:

```text
expected-edge-threshold-bps: 0.10
edge-if-filled-threshold-bps: 0.05
fill-prob-threshold: 0.05
intraday-percentile-gate: on
expected-edge-percentile: 0.95
edge-if-filled-percentile: 0.97
placement-expected-margin-bps: 0.00
placement-edge-if-filled-margin-bps: 0.00
regime-expected-edge-gate: on
regime-min-bid/ask-expected-edge-bps: 0.01
regime-alpha: 0.00005
order_qty: 100
max_position: 300
soft_position: 100
reduce_only_after_soft_position: on
daily_loss_limit_usdt: 0.10
daily_fill_limit: 300
dynamic_quote: off
```

Monthly result:

```text
total PnL +0.806400
fills 3430
PnL/fill +0.00023510
positive days 18/31
active-day positive days 18/28
4 月 -0.037700
5 月 +0.844100
worst day 20260421 -0.065500
max position max 100
```

## Previous Benchmarks

Old 9-factor static regime/risk best:

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_regime_risk_grid_full_p000_a00005.aggregate.csv

total PnL +0.550429
fills 4177
PnL/fill +0.00013178
positive days 19/31
active-day positive days 19/29
4 月 -0.075700
5 月 +0.626129
worst day 20260420 -0.081350
```

Old 9-factor placement-margin candidate:

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_margin_full_em002_if000_p000_a00005.aggregate.csv

total PnL +0.563879
fills 4059
positive days 23/31
4 月 -0.097750
5 月 +0.661629
worst day 20260423 -0.062100
```

## Decision

```text
Promote new22 e95/i97/regime_min=0.01 to current main candidate.

Reason:
  Higher full-month PnL than both old 9f benchmarks.
  Smaller 4 月 loss than both old 9f benchmarks.
  Stronger 5 月 PnL.
  Better PnL/fill with fewer fills than old 9f static best.

Remaining issue:
  Calendar-day win rate is slightly lower than old 9f static best.
  Bad days remain: 20260421, 20260512, 20260420.
```

## Next Work

Do not continue broad gate sweeps first. Next work should be:

```text
1. Loss attribution for 20260420, 20260421, 20260512.
2. Implement or tune score-aware quote distance on top of new22.
3. Re-test bad days first, then full month.
```

Quote-control status:

```text
Implemented:
  score-aware quote distance
  stale quote control
  side/regime-aware quote widening
  position-age reduce-only control

Result:
  strict and light stacked probes both underperformed the current baseline on bad3+good3.
  decomposed quote-control probes also did not beat baseline.
  Do not promote yet.

Decomposed probe result:
  baseline 6-day +0.299850, bad3 -0.166050, good3 +0.465900.
  quote-distance-only +0.269050, bad3 -0.150700, good3 +0.419750.
  stale-only +0.239450, bad3 -0.160000, good3 +0.399450.
  position-age-only +0.293400, bad3 -0.169250, good3 +0.462650.
  fill-prob-widen-only +0.194350, bad3 -0.170300, good3 +0.364650.

Next:
  only refine quote-distance; it is the only layer that improved bad3,
  but it worsened 20260421 and gave back too much good-day PnL.
```

Initial attribution:

```text
20260421:
  PnL -0.065500, 178 fills.
  fill-side attribution says ask loses most: ask -0.082250, bid +0.016750.
  entry attribution says bid entries lose most: bid -0.046000, ask -0.019500.
  weak entry hours: 12, 13, 15, 17, 18 UTC.

20260512:
  PnL -0.054200, 80 fills.
  fill-side attribution says bid loses most: bid -0.039350, ask -0.014850.
  entry attribution is more balanced but ask entries lose slightly more: ask -0.030200, bid -0.024000.
  weak hours: 13, 14, 16 UTC.

20260420:
  PnL -0.046350, 110 fills.
  fill-side attribution says ask loses more: ask -0.029675, bid -0.016675.
  entry attribution says bid entries lose more: bid -0.029000, ask -0.017350.
  weak entry hours: 14, 15 UTC.
```

Detailed report:

```text
doc/edge_scoring/loss_attribution_new22.md
```
