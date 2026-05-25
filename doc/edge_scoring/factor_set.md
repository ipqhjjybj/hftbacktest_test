# Edge Scoring Factor Set

更新时间: 2026-05-25

## Target Model Outputs

The edge model predicts six targets:

```text
bid_expected_edge_bps
ask_expected_edge_bps
bid_edge_if_filled_bps
ask_edge_if_filled_bps
bid_fill_prob
ask_fill_prob
```

Usage:

```text
expected_edge:
  Main quote quality gate. It already includes non-fill as zero.

edge_if_filled:
  Fill-conditioned adverse-selection / toxicity quality gate.

fill_prob:
  Activity / fillability filter. Do not multiply expected_edge by fill_prob again.
```

## 9-Factor Baseline

Original factors:

```text
spread_bps
queue_imbalance
microprice_bps
trade_flow_imbalance
trade_flow_ewm_imbalance
momentum_100ms_bps
momentum_250ms_bps
momentum_1000ms_bps
vol_1000ms_bps
```

This set is still the monthly benchmark when trained with interactions.

## 22-Factor Set

Added on 2026-05-25.

Additional raw sampled columns:

```text
bid_qty_l2, ask_qty_l2
bid_qty_l3, ask_qty_l3
bid_qty_l4, ask_qty_l4
bid_qty_l5, ask_qty_l5
```

Full factor list:

```text
spread_bps
queue_imbalance
microprice_bps
trade_flow_imbalance
trade_flow_ewm_imbalance
momentum_100ms_bps
momentum_250ms_bps
momentum_1000ms_bps
vol_1000ms_bps
depth_imbalance_3
depth_imbalance_5
weighted_depth_imbalance_5
bid_depth_slope_5
ask_depth_slope_5
top_bid_qty_change
top_ask_qty_change
ofi
ofi_1000ms
trade_qty_1000ms
trade_count_1000ms
momentum_3000ms_bps
vol_250ms_bps
```

## Implementation Notes

Training/research side:

```text
scripts/factor_research/factors.py
scripts/factor_research/bitmex_factor_research.py
scripts/factor_research/edge_scoring.py
```

Strategy realtime side:

```text
scripts/bitmex_edge_scored_maker.py
```

Important consistency rules:

```text
1. FACTOR_NAMES and FACTOR_IDS must stay append-only and order-consistent.
2. edge_scored_maker dynamically uses model mean/std length, so old 9f models still load.
3. 22 factors with interactions produces 275 expanded features, so initial probes used --no-interactions.
```

Verified:

```text
py_compile passed
new 22f train/predict smoke passed
new 22f strategy smoke passed
old 9f model compatibility smoke passed
FACTOR_NAMES and FACTOR_IDS order check passed
```

## Research-Layer 7-Day Comparison

Period:

```text
20260512-20260518
```

Old 9f with interactions:

```text
bid edge_corr avg              0.053483
bid edge_if_filled_corr avg    0.187931
bid top-bottom expected edge   +0.065847 bps
bid top-bottom edge_if_filled  +1.024559 bps

ask edge_corr avg              0.066838
ask edge_if_filled_corr avg    0.217447
ask top-bottom expected edge   +0.078130 bps
ask top-bottom edge_if_filled  +1.159611 bps
```

New 22f no interactions:

```text
bid edge_corr avg              0.055845
bid edge_if_filled_corr avg    0.207945
bid top-bottom expected edge   +0.071253 bps
bid top-bottom edge_if_filled  +1.154374 bps

ask edge_corr avg              0.070940
ask edge_if_filled_corr avg    0.231833
ask top-bottom expected edge   +0.085551 bps
ask top-bottom edge_if_filled  +1.350452 bps
```

Conclusion:

```text
The 22f set adds useful information, especially for edge_if_filled, but it is not a step-change.
Gate parameters must be recalibrated for the new score distribution.
```
