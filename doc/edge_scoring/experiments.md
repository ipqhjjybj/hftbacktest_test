# Edge Scoring Experiment Log

更新时间: 2026-05-25

This file keeps compact experiment summaries. The full historical progress text is archived at:

```text
doc/edge_scoring/archive_2026-05-25_full_progress.md
```

## 2026-05-22/23 Edge Score OOS

Full monthly with_if_filled OOS:

```text
period: 20260418-20260518
train_days: 14
horizon: 250ms
maker_fee: 0
```

Research-layer result:

```text
bid expected-edge top-bottom spread:  +0.1389 bps, 31/31 positive
bid edge_if_filled top-bottom spread: +1.1843 bps, 31/31 positive
ask expected-edge top-bottom spread:  +0.1384 bps, 31/31 positive
ask edge_if_filled top-bottom spread: +1.1060 bps, 31/31 positive
```

Conclusion:

```text
The model ranks quote opportunities. High score usually means better fill quality but lower fill probability.
```

## 2026-05-23 Percentile Gate

Baseline:

```text
edge>=0.10, fill>=0.05, if_filled>=0.05
total PnL -0.938686
fills 6955
positive days 11/31
```

Add intraday pct95 expected + pct95 if_filled:

```text
total PnL -0.320934
fills 5902
positive days 14/31
4 月 -0.487980
5 月 +0.167045
worst day 20260420 -0.204200
```

Conclusion:

```text
Percentile gate reduces tail losses but does not turn the full month positive.
```

## 2026-05-24 Hard Risk

Added:

```text
--reduce-only-after-soft-position
--daily-loss-limit-usdt
--daily-fill-limit
```

Bad-day smoke on 20260504:

```text
pct95 + hard risk:
  PnL +0.011850
  fills 300
  max position 100
```

Full month with same hard risk but no regime improvement:

```text
total PnL -0.344484
fills 6158
positive days 12/31
4 月 -0.580712
5 月 +0.236229
```

Conclusion:

```text
Hard risk cuts some tail days but does not solve persistent bad-regime losses.
```

## 2026-05-24 Regime Gate

Added expected-edge EWM regime gate:

```text
--regime-expected-edge-gate
--regime-expected-edge-warmup-samples
--regime-expected-edge-ewm-alpha
--regime-min-bid-expected-edge-bps
--regime-min-ask-expected-edge-bps
```

First useful full-month combination:

```text
pct95 + regime + hard risk
regime_min -0.02
alpha 0.0001
total PnL +0.271479
fills 5035
positive days 16/31
4 月 -0.280750
5 月 +0.552229
```

Regime sweep found better candidate:

```text
regime_min 0.00
alpha 0.00005
total PnL +0.550429
fills 4177
positive days 19/31
4 月 -0.075700
5 月 +0.626129
```

Conclusion:

```text
Regime gate is the main reason the old 9f strategy became monthly positive.
```

## 2026-05-25 New 22 Factors

Added OFI, multi-level depth, trade intensity, and extra momentum/vol windows.

Research 7-day comparison:

```text
new22 no-interactions improved top-bottom expected-edge spread by about 8-10%.
new22 no-interactions improved top-bottom edge_if_filled spread by about 13-17%.
```

Directly using old 9f gate did not improve 7-day PnL:

```text
old9 best 20260512-20260518:
  PnL +0.496950
  fills 1490

new22 e95/i95/regime_min=0.00:
  PnL +0.465800
  fills 1396
```

Conclusion:

```text
The new factor set changes score distribution; gate needs recalibration.
```

## 2026-05-25 New 22 Gate Sweep

Representative-day grid:

```text
dates: 20260512 20260514 20260518
expected percentiles: 0.95, 0.97
edge_if_filled percentiles: 0.95, 0.97
regime_min: -0.01, 0.00, 0.01
alpha: 0.00005
```

Top three:

```text
1. e95/i95/regime_min=0.01
   total +0.199400, fills 648

2. e95/i97/regime_min=0.01
   total +0.195850, fills 594

3. e97/i95/regime_min=0.01
   total +0.167850, fills 622
```

Full 7-day rerun for top two:

```text
new22 e95/i95/regime_min=0.01:
  total PnL +0.501650
  fills 1320
  PnL/fill +0.00038004

new22 e95/i97/regime_min=0.01:
  total PnL +0.559350
  fills 1220
  PnL/fill +0.00045848
```

Conclusion:

```text
For new22, stricter regime_min=0.01 is clearly better on the 7-day probe.
e95/i97/regime_min=0.01 is the current new22 7-day candidate.
Monthly validation is still required.
```
