# New22 Loss Attribution

更新时间: 2026-05-25

Scope:

```text
strategy result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_e95_i97_m001_a00005_20260418_20260518.aggregate.csv

bad days:
  20260420
  20260421
  20260512
```

## Method

Two attribution views were used:

```text
fill-interval attribution:
  Uses equity_delta_since_prev_fill_usdt from each fill row.
  Useful for locating when equity moved.
  Can assign inventory PnL to the closing side, so it can mislead side diagnosis.

entry round-trip attribution:
  Pairs flat -> position -> flat cycles.
  Attributes realized PnL to the entry quote.
  Better for diagnosing quote quality and adverse selection.
```

## Main Findings

The core issue is adverse selection / stale entry quality, not just low model scores.

```text
Bad-day aggregate round trips:
  days: 20260420, 20260421, 20260512
  round trips: 184
  PnL: -0.166050

If entry spread_capture <= 0:
  PnL -0.126300
  trips 50
  PnL/trip -0.002526

If entry spread_capture < 2 USDT/BTC:
  PnL -0.130800
  trips 56
  PnL/trip -0.002336

If entry spread_capture >= 2 USDT/BTC:
  PnL -0.035250
  trips 128
  PnL/trip -0.000275
```

Full-month cross-check:

```text
Full month:
  round trips 1715
  PnL +0.806400

entry spread_capture < 5:
  PnL -0.648400
  trips 607
  PnL/trip -0.001068

entry spread_capture >= 5:
  PnL +1.454800
  trips 1108
  PnL/trip +0.001313
```

Interpretation:

```text
The strategy is profitable when it captures enough distance from mid at entry.
The worst losses are concentrated in fills where the quote was hit after the mid already moved against it.

entry spread_capture is measured at fill time, so it is not directly usable as a live gate.
It points to the live proxy we should improve:
  quote distance
  quote age / stale quote control
  side/regime-aware widening
```

## Day-Level Attribution

### 20260420

Fill-interval attribution:

```text
total PnL -0.046350
ask fill-side PnL -0.029675
bid fill-side PnL -0.016675

weak hours:
  14 UTC: -0.041750
  15 UTC: -0.014200
```

Entry round-trip attribution:

```text
round trips 55
PnL -0.046350

bid entries:
  PnL -0.029000
  trips 34

ask entries:
  PnL -0.017350
  trips 21

entry spread_capture < 2:
  PnL -0.039450
  trips 18
```

Worst bucket:

```text
20260420 bid entries at 14 UTC:
  PnL -0.031450
  trips 8
  avg entry spread_capture -0.12
  avg quote age 822 ms
  avg entry edge_if_filled 1.132 bps
```

Diagnosis:

```text
Mainly stale/too-close bid entries around 14 UTC.
The model score was not obviously weak, so simply raising score gates is unlikely to solve it.
```

### 20260421

Fill-interval attribution:

```text
total PnL -0.065500
ask fill-side PnL -0.082250
bid fill-side PnL +0.016750

weak hours:
  13 UTC: -0.026350
  15 UTC: -0.018000
  12 UTC: -0.017000
  18 UTC: -0.011400
  17 UTC: -0.010500
```

Entry round-trip attribution:

```text
round trips 89
PnL -0.065500

bid entries:
  PnL -0.046000
  trips 79

ask entries:
  PnL -0.019500
  trips 10

entry spread_capture < 2:
  PnL -0.059950
  trips 26
```

Worst buckets:

```text
20260421 bid entries at 13 UTC:
  PnL -0.026350
  trips 6
  avg entry spread_capture -2.79
  avg quote age 312 ms
  avg entry edge_if_filled 0.685 bps

20260421 bid entries at 12 UTC:
  PnL -0.014300
  trips 3
  avg entry spread_capture -8.75
  avg quote age 337 ms
  avg entry edge_if_filled 0.882 bps
```

Diagnosis:

```text
Although fill-side attribution says ask loses most, entry attribution says the larger opening problem is bid entries.
This is likely long entries being closed by ask fills after price moves down.
```

### 20260512

Fill-interval attribution:

```text
total PnL -0.054200
bid fill-side PnL -0.039350
ask fill-side PnL -0.014850

weak hours:
  13 UTC: -0.026950
  16 UTC: -0.020000
  14 UTC: -0.015550
```

Entry round-trip attribution:

```text
round trips 40
PnL -0.054200

ask entries:
  PnL -0.030200
  trips 20

bid entries:
  PnL -0.024000
  trips 20

entry spread_capture < 2:
  PnL -0.031400
  trips 12

hold time > 30s:
  PnL -0.039500
  trips 11
```

Worst buckets:

```text
20260512 bid entries at 13 UTC:
  PnL -0.015150
  trips 5
  avg entry spread_capture +3.50
  avg quote age 306 ms
  avg entry edge_if_filled 1.115 bps

20260512 ask entries at 13 UTC:
  PnL -0.011800
  trips 2
  avg entry spread_capture +7.00
  avg quote age 1105 ms
  avg entry edge_if_filled 0.941 bps
```

Diagnosis:

```text
This day is less purely a negative-spread-capture problem.
Longer hold time is more important here, suggesting inventory path / delayed exit risk.
```

## Signal Checks

Bad-day aggregate:

```text
edge_if_filled < 1.0:
  PnL -0.111800
  trips 103
  PnL/trip -0.001085

edge_if_filled >= 1.0:
  PnL -0.054250
  trips 81
  PnL/trip -0.000670
```

Full month:

```text
edge_if_filled < 1.0:
  PnL +0.035750
  trips 673
  PnL/trip +0.000053

edge_if_filled >= 1.0:
  PnL +0.770650
  trips 1042
  PnL/trip +0.000740
```

Interpretation:

```text
edge_if_filled is still useful.
But it is not enough as a hard gate because many worst entries still had decent edge_if_filled.
It should drive quote distance / sizing rather than only pass/fail.
```

Regime check:

```text
Full month, entry regime < 0.04:
  PnL +0.106400
  trips 816
  PnL/trip +0.000130

Full month, entry regime >= 0.04:
  PnL +0.700000
  trips 899
  PnL/trip +0.000779
```

Interpretation:

```text
Regime is useful for sizing.
The current hard regime_min=0.01 is too low to distinguish strong from mediocre regimes.
Raising it globally may cut too many fills; better first use it to widen quotes.
```

Hold-time check:

```text
Full month, hold time > 30s:
  PnL -0.270300
  trips 259
  PnL/trip -0.001044

Full month, hold time <= 30s:
  PnL +1.076700
  trips 1456
  PnL/trip +0.000739
```

Interpretation:

```text
Long-held inventory is structurally bad.
For score-aware quote distance, also consider position-age-aware exit or stronger reduce-only behavior.
```

## Recommended Next Experiment

Do not start with another broad gate sweep.

Implement a new22 quote-distance variant:

```text
1. Widen quotes when edge_if_filled is weak or regime EWM is mediocre.
2. Widen quotes when quote age is high, or shorten TTL for weak-score placements.
3. Keep strong-score / strong-regime quotes active so good days do not lose too many fills.
4. Add a position-age exit pressure if inventory remains open beyond 30s.
```

First validation set:

```text
bad days:
  20260420
  20260421
  20260512

good control days:
  20260514
  20260515
  20260517
```

Success criteria:

```text
bad-day PnL improves materially.
good-day PnL does not give back most of the monthly edge.
full-month PnL remains above +0.806400.
worst day improves from -0.065500 toward -0.03~-0.04.
```

## 2026-05-25 Detailed Recompute

Recomputed directly from the current monthly new22 fills:

```text
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_e95_i97_m001_a00005_20260418_20260518_*.fills.csv
```

### Scope Summary

```text
Full month:
  fills 3430
  PnL +0.806400
  PnL/fill +0.00023510
  round trips 1715
  PnL/trip +0.00047020

Bad3:
  days 20260420, 20260421, 20260512
  fills 368
  PnL -0.166050
  PnL/fill -0.00045122
  round trips 184
  PnL/trip -0.00090245

Good3 control:
  days 20260514, 20260515, 20260517
  fills 646
  PnL +0.465900
  PnL/fill +0.00072121
  round trips 323
  PnL/trip +0.00144241
```

### Fill-Side vs Entry-Side

Fill-side attribution on Bad3:

```text
ask fills:
  fills 184
  PnL -0.126775
  PnL/fill -0.00068899

bid fills:
  fills 184
  PnL -0.039275
  PnL/fill -0.00021345
```

Entry round-trip attribution on Bad3:

```text
bid entries:
  trips 133
  PnL -0.099000
  PnL/trip -0.00074436

ask entries:
  trips 51
  PnL -0.067050
  PnL/trip -0.00131471
```

Interpretation:

```text
Fill-side says ask fills lose more.
Entry-side says bid entries lose more in total, while ask entries lose more per trip.

This means side diagnosis must use entry round trips, not only fill side.
Some ask-fill losses are likely closing earlier bad bid entries after price moved down.
```

### Bad Hours

Worst fill-side hours:

```text
20260420 14 UTC:
  PnL -0.041750
  fills 30

20260512 13 UTC:
  PnL -0.026950
  fills 14

20260421 13 UTC:
  PnL -0.026350
  fills 12

20260512 16 UTC:
  PnL -0.020000
  fills 16

20260421 15 UTC:
  PnL -0.018000
  fills 12

20260421 12 UTC:
  PnL -0.017000
  fills 14
```

Worst entry-side buckets:

```text
20260420 14 UTC bid entries:
  trips 8
  PnL -0.031450
  PnL/trip -0.00393125

20260421 13 UTC bid entries:
  trips 6
  PnL -0.026350
  PnL/trip -0.00439167

20260420 15 UTC ask entries:
  trips 4
  PnL -0.015200
  PnL/trip -0.00380000

20260512 13 UTC bid entries:
  trips 5
  PnL -0.015150
  PnL/trip -0.00303000

20260421 15 UTC ask entries:
  trips 1
  PnL -0.014800
```

### Entry Spread Capture

Full month by entry spread_capture quintile:

```text
q1 [-79.50, -2.50]:
  PnL -0.568900
  trips 343
  PnL/trip -0.00165860

q2 [-2.50, +6.50]:
  PnL -0.006350
  trips 343
  PnL/trip -0.00001851

q3 [+6.50, +11.50]:
  PnL +0.180500

q4 [+11.50, +16.50]:
  PnL +0.392600

q5 [+16.50, +60.75]:
  PnL +0.808550
  PnL/trip +0.00235729
```

Bad3 by entry spread_capture quintile:

```text
q1 [-29.50, -5.75]:
  PnL -0.124800
  trips 36
  PnL/trip -0.00346667

q2 [-4.50, +4.50]:
  PnL -0.056150

q3 [+4.75, +10.25]:
  PnL -0.040250

q4 [+10.25, +13.75]:
  PnL +0.005950

q5 [+13.75, +27.50]:
  PnL +0.049200
```

Interpretation:

```text
Entry distance is the clearest separator.
The system is profitable when it captures enough distance at entry.
The worst losses are concentrated in entries that were too close or already adverse at fill.

entry spread_capture is observed at fill time and is not directly tradable.
The live proxy is quote distance / stale quote control.
```

### Hold Time

Full month by round-trip hold time:

```text
q1 [0.06s, 1.30s]:
  PnL +0.541450

q2 [1.30s, 3.17s]:
  PnL +0.376300

q3 [3.18s, 7.83s]:
  PnL +0.216300

q4 [7.83s, 23.45s]:
  PnL -0.010950

q5 [23.47s, 491.37s]:
  PnL -0.316700
```

Bad3 by round-trip hold time:

```text
q1 [0.24s, 1.69s]:
  PnL +0.031950

q2 [1.69s, 3.86s]:
  PnL -0.003400

q3 [3.89s, 9.50s]:
  PnL +0.005850

q4 [9.62s, 25.56s]:
  PnL -0.083350

q5 [26.10s, 172.57s]:
  PnL -0.117100
```

Interpretation:

```text
Long-held inventory is structurally bad.
The effect is much stronger on bad days.
This supports position-age-aware reduce-only pressure or timeout exits.
```

### Score And Regime Checks

Full month by entry edge_if_filled quintile:

```text
q1 [0.000, 0.732]:
  PnL -0.027050

q5 [1.692, 4.242]:
  PnL +0.537800
  PnL/trip +0.00156793
```

Bad3 by entry edge_if_filled quintile:

```text
q1 [0.000, 0.613]:
  PnL -0.036000

q4 [1.089, 1.484]:
  PnL -0.063100

q5 [1.487, 3.096]:
  PnL +0.010850
```

Full month by entry regime quintile:

```text
q1 [0.000, 0.020]:
  PnL +0.013300

q4 [0.052, 0.080]:
  PnL +0.314700

q5 [0.080, 0.246]:
  PnL +0.438800
```

Bad3 by entry regime quintile:

```text
q1 [0.000, 0.016]:
  PnL -0.023150

q3 [0.023, 0.033]:
  PnL -0.045350

q4 [0.034, 0.052]:
  PnL -0.063900

q5 [0.052, 0.115]:
  PnL -0.010550
```

Full month by entry fill probability quintile:

```text
q5 [0.288, 0.648]:
  PnL +0.345100
```

Bad3 by entry fill probability quintile:

```text
q4 [0.233, 0.280]:
  PnL -0.074050

q5 [0.282, 0.442]:
  PnL -0.033900
```

Interpretation:

```text
edge_if_filled and regime are useful across the full month.
They are not sufficient as bad-day hard gates.

High fill probability is good in normal conditions but can become toxic on bad days.
This argues for using fill_prob to adjust quote distance, not only as a pass/fail threshold.
```

### Quote Age And Half Spread

Entry placement age is not a clean monotonic separator:

```text
Full month:
  q1 age [0, 160ms]      PnL +0.208350
  q5 age [670, 4910ms]   PnL +0.138450

Bad3:
  q1 age [0, 150ms]      PnL +0.002900
  q2 age [160, 320ms]    PnL -0.068900
  q5 age [650, 3090ms]   PnL -0.040750
```

Half-spread is almost entirely fixed:

```text
Full month:
  3.0 bps entries:
    trips 1608
    PnL +0.818100

Bad3:
  3.0 bps entries:
    trips 172
    PnL -0.164250
```

Interpretation:

```text
Quote age alone is not enough.
The current fixed 3.0 bps quote distance is the real limitation:
it works in normal regimes but is too close for adverse intervals.
```

### Updated Diagnosis

```text
1. The bad days are not caused by a total lack of model signal.
   edge_if_filled and regime still separate full-month performance.

2. The largest bad-day losses are entry-quality / adverse-selection events:
   too-close entries, negative or weak entry spread_capture, and delayed exits.

3. A global score gate increase is unlikely to be efficient.
   It would remove many good fills and still miss high-score adverse events.

4. The next useful control should change how orders are placed and managed:
   quote distance, stale control, and position-age exit pressure.
```

### Updated Next Experiment

Do a decomposed quote-control probe, not a stacked one:

```text
1. Quote distance only:
   widen weak edge_if_filled / weak regime / high toxic-fill-prob states.

2. Position-age only:
   after 10s / 20s / 30s in inventory, increase reduce-only pressure or tighten exit.

3. Fill-prob-aware widening only:
   keep high fill_prob when regime is strong, widen when regime is mediocre.

4. Stale placement only:
   shorten TTL or widen after quote age passes a threshold.
```

Validation order:

```text
1. Run bad3 + good3 first.
2. Promote only if bad3 improves without giving back most good3 PnL.
3. Then run full 20260418-20260518 month.
```
