# Edge Scoring Experiment Log

更新时间: 2026-05-26

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

## 2026-05-25 Placement Margin Gate

Added placement-time margin gates to the strategy:

```text
--placement-expected-margin-bps
--placement-edge-if-filled-margin-bps
```

Interpretation:

```text
placement expected margin = placement side expected_edge - active expected_edge threshold
placement edge_if_filled margin = placement side edge_if_filled - active edge_if_filled threshold
```

Representative 8-day sweep:

```text
base:          +0.187300 / 1162 fills / worst -0.081350
em002_if000:   +0.202500 / 1100 fills / worst -0.044400
em005_if000:   +0.152750 / 996 fills
em010_if000:   +0.080050 / 916 fills
em000_if010:   +0.199250 / 1112 fills
em000_if025:   +0.190750 / 980 fills
em005_if010:   +0.191900 / 984 fills
em005_if025:   +0.164300 / 880 fills
em010_if025:   +0.077250 / 816 fills
```

Full-month validation for the best 8-day candidate:

```text
old pct95/regime/hard-risk:
  total PnL +0.550429
  fills 4177
  positive days 19/31
  worst day -0.081350

placement em002_if000:
  total PnL +0.563879
  fills 4059
  positive days 23/31
  worst day -0.062100
```

Conclusion:

```text
placement expected margin >= 0.02 bps is the current monthly best.
The gain is small: +0.013450 total PnL with 118 fewer fills.
It improves positive-day count and worst day, but 4 月 worsens from -0.075700 to -0.097750.
```

## 2026-05-25 Label / Target Alignment Check

Goal:

```text
Check whether model scores align with realized fill PnL.
Realized target used here:
  equity_delta_since_prev_fill_usdt / fill_count
```

Samples:

```text
placement_probe_8d_base:
  8 representative days, no placement margin gate
  valid placement rows: 1124 fills
  realized PnL on valid rows: +0.173075

margin_full_em002_if000_month:
  current monthly candidate
  valid placement rows: 3913 fills
  realized PnL on valid rows: +0.493050
```

8-day base alignment:

```text
score                         q1 pnl/fill   q5 pnl/fill   q5-q1
fill_time_expected             -0.000518     +0.000756    +0.001274
placement_edge_if_filled       -0.000445     +0.000797    +0.001243
placement_expected             -0.000456     +0.000749    +0.001205
placement_expected_margin      -0.000303     +0.000722    +0.001026
placement_fill_prob            -0.000190     +0.000302    +0.000492
```

Current full-month candidate alignment:

```text
score                         q1 pnl/fill   q5 pnl/fill   q5-q1
placement_expected_margin      -0.000504     +0.000516    +0.001020
placement_expected             -0.000490     +0.000507    +0.000997
placement_edge_if_filled       -0.000463     +0.000504    +0.000967
fill_time_expected             -0.000454     +0.000509    +0.000963
placement_fill_prob            +0.000235     +0.000254    +0.000019
```

4 月 / 5 月 split on current monthly candidate:

```text
Apr:
  placement_expected q5-q1 pnl/fill      +0.001880
  placement_edge_if_filled q5-q1         +0.001526
  high-score buckets still make money; losses are concentrated in low-score buckets.

May:
  placement_edge_if_filled q5-q1         +0.000662
  placement_expected q5-q1               +0.000487
  overall regime is better, but score sorting is weaker than Apr.
```

Side split on current monthly candidate:

```text
Ask side:
  expected score sorts realized PnL strongly.
  All-month q5-q1 pnl/fill: +0.001746
  Apr ask is the weak area: total valid-row PnL -0.212900.

Bid side:
  edge_if_filled sorts better than expected in May.
  fill_prob is weak to negative as a return target.
```

Conclusion:

```text
The target is broadly aligned with realized fill PnL:
  expected_edge and edge_if_filled both separate bad fills from good fills.

fill_prob should not be treated as a return target:
  it is useful as execution/liquidity information, but weak for PnL ranking.

For improving score quality:
  keep expected_edge and edge_if_filled as primary return targets.
  investigate side/regime-specific calibration, especially Apr ask-side losses.
```

Follow-up: entry-level round-trip attribution.

```text
The first pass used equity_delta_since_prev_fill_usdt on each fill.
That can assign inventory PnL to the closing fill, so it is noisy for judging the opening quote.

Second pass paired fills by date and attributed realized round-trip PnL to the entry fill.
This is closer to the trading target for entry-score quality.
```

Entry-level result on current monthly candidate:

```text
valid-entry closed lots: 1883
entry-attributed realized PnL: +0.538400

score                         q1 pnl/100c   q5 pnl/100c   q5-q1
entry edge_if_filled           -0.000440     +0.000737    +0.001177
entry edge_if_filled margin    -0.000324     +0.000828    +0.001152
entry regime EWM               +0.000223     +0.001053    +0.000830
entry expected                 +0.000269     +0.000598    +0.000329
entry fill_prob                +0.000567     +0.000142    -0.000426
```

Interpretation:

```text
For entry-level realized PnL, edge_if_filled is more aligned than expected_edge.
fill_prob is again not a return target.
However, attribution-only filters can overstate usefulness because they ignore lost opportunity and inventory path.
```

Small strategy probe based on this hypothesis:

```text
8-day probe, same dates as placement-margin sweep:

base:          +0.187300 / 1162 fills / worst -0.081350
em002_if000:   +0.202500 / 1100 fills / worst -0.044400
em002_if050:   +0.144950 / 778 fills  / worst -0.059750
em002_if075:   +0.057100 / 618 fills  / worst -0.046500
em000_if050:   +0.145800 / 790 fills  / worst -0.069850
em000_if075:   +0.102650 / 634 fills  / worst -0.040450
```

Conclusion:

```text
edge_if_filled is useful for score-quality diagnosis, but a hard high if_filled-margin gate is too blunt.
It removes too many good opportunities and underperforms em002_if000 on the 8-day strategy probe.

Next direction should be calibration / quote sizing by edge_if_filled, not simply raising the hard gate.
```

## 2026-05-25 Edge-If-Filled Quote Sizing

Hypothesis:

```text
Use edge_if_filled continuously to size quote aggressiveness.
Do not raise it as a hard gate.
```

Implementation used existing dynamic quote path:

```text
placement expected margin gate: 0.02 bps
placement edge_if_filled margin gate: 0.00 bps
dynamic quote: on
dynamic expected-edge mult: 0.0
dynamic edge_if_filled mult: 0.50
dynamic max tighten: 1.0 bps
dynamic fill_prob widen: 0.0
```

8-day probe:

```text
base:             +0.187300 / 1162 fills / worst -0.081350
em002_if000:      +0.202500 / 1100 fills / worst -0.044400
calib if025/t1:   +0.215850 / 1164 fills / worst -0.079350
calib if050/t1:   +0.257800 / 1218 fills / worst -0.066850
calib if050/t2:   +0.257800 / 1218 fills / worst -0.066850
```

Full-month validation for calib if050/t1:

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_calib_quote_full_if050_t1_p000_a00005.aggregate.csv

static old best:
  total PnL +0.550429
  fills 4177
  positive days 19/31
  4 月 -0.075700
  5 月 +0.626129
  last 7 days +0.496950
  worst day -0.081350

placement em002_if000:
  total PnL +0.563879
  fills 4059
  positive days 23/31
  4 月 -0.097750
  5 月 +0.661629
  last 7 days +0.475100
  worst day -0.062100

calib quote if050/t1:
  total PnL +0.515329
  fills 4499
  positive days 20/31
  4 月 -0.092650
  5 月 +0.607979
  last 7 days +0.509550
  worst day -0.066850
```

Conclusion:

```text
The 8-day probe improved, but the full-month result did not beat the current best.
It increases fills and last-7-day PnL, but reduces full-month PnL versus placement em002_if000.

Keep placement em002_if000 as current monthly best.
Do not promote this quote-sizing version.

If quote sizing is revisited, it needs side/regime conditioning rather than a single global if_filled multiplier.
```

## 2026-05-25 New22 Full-Month Validation

Setup:

```text
period: 20260418-20260518
model: 22 factors, no interactions
model tag: edge_score_wf_train14_20260418_20260518_h250_new_hf_nointer
strategy: expected pct95, edge_if_filled pct97, regime_min 0.01, alpha 0.00005
```

Result:

```text
new22:
  total PnL +0.806400
  fills 3430
  PnL/fill +0.00023510
  positive days 18/31
  active-day positive days 18/28
  4 月 -0.037700
  5 月 +0.844100
  worst day 20260421 -0.065500

old 9f static best:
  total PnL +0.550429
  fills 4177
  PnL/fill +0.00013178
  positive days 19/31
  active-day positive days 19/29
  4 月 -0.075700
  5 月 +0.626129
  worst day 20260420 -0.081350

old 9f placement best:
  total PnL +0.563879
  fills 4059
  positive days 23/31
  4 月 -0.097750
  5 月 +0.661629
  worst day 20260423 -0.062100
```

Conclusion:

```text
Promote new22 e95/i97/regime_min=0.01 to current main candidate.
It improves monthly PnL, 4 月 loss, 5 月 PnL, and PnL/fill.

It is not a finished strategy:
  win rate is slightly lower than old 9f static best.
  bad days remain, especially 20260421, 20260512, 20260420.
```

Initial fill-log attribution:

```text
20260421:
  PnL -0.065500, fills 178.
  ask -0.082250, bid +0.016750.
  weak hours: 12, 13, 15, 17, 18 UTC.

20260512:
  PnL -0.054200, fills 80.
  bid -0.039350, ask -0.014850.
  weak hours: 13, 14, 16 UTC.

20260420:
  PnL -0.046350, fills 110.
  ask -0.029675, bid -0.016675.
  weak hours: 14, 15 UTC.
```

Next:

```text
Do loss attribution before more broad gate sweeps.
Then test score-aware quote distance on bad days first, full month second.
```

## 2026-05-25 New22 Quote-Control Probe

Implemented strategy switches:

```text
score-aware quote distance:
  widen entry quote when edge_if_filled margin or side regime EWM is weak.

stale quote control:
  cancel weak placements earlier than the normal order TTL.

side/regime-aware widening:
  use side-specific expected-edge regime EWM in quote widening.

position-age control:
  after a configurable position age, suppress new entries and quote only reduce side.
```

Validation set:

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

Baseline:

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_e95_i97_m001_a00005_20260418_20260518.aggregate.csv

6-day total +0.299850, fills 1014
bad3 -0.166050, fills 368
good3 +0.465900, fills 646
worst 20260421 -0.065500
```

Strict quote-control probe:

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_quotectrl_probe_bad3_good3.aggregate.csv

params:
  quote_edge_if_filled_widen_target 1.0
  quote_edge_if_filled_widen_mult 0.5
  quote_regime_widen_threshold 0.04
  quote_regime_widen_mult 20.0
  quote_max_score_widen 2.0
  stale_quote_ttl 500ms
  stale min if_filled margin 1.0
  stale min regime 0.04
  position age reduce-only 30s
  reduce tighten 1.5bps

6-day total +0.218500, fills 644
bad3 -0.064650, fills 222
good3 +0.283150, fills 422
worst 20260421 -0.054350
```

Light quote-control probe:

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_quotectrl_probe_light_bad3_good3.aggregate.csv

params:
  quote_edge_if_filled_widen_target 0.75
  quote_edge_if_filled_widen_mult 0.25
  quote_regime_widen_threshold 0.02
  quote_regime_widen_mult 10.0
  quote_max_score_widen 1.0
  stale_quote_ttl 1000ms
  stale min if_filled margin 0.5
  stale min regime 0.02
  position age reduce-only 60s
  reduce tighten 0.5bps

6-day total +0.210900, fills 918
bad3 -0.136600, fills 348
good3 +0.347500, fills 570
worst 20260421 -0.065200
```

Conclusion:

```text
Do not promote either quote-control probe.
Do not run full-month validation for these parameter sets.

Strict version fixes much of 20260420 and 20260512, but gives back too much good-day PnL.
Light version preserves more fills, but barely fixes 20260421 and remains below baseline.
```

Next direction:

```text
Decompose controls instead of stacking all four at once:
  1. stale quote control only
  2. score/regime widening only
  3. position-age reduce-only only

The current stacked controls are too blunt.
```

## 2026-05-25 New22 Decomposed Quote-Control Probes

Purpose:

```text
Split the quote-control stack into separate probes so we can see which layer
actually helps the new22 bad days instead of judging only a combined stack.
```

Validation set:

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

Common baseline params:

```text
model:
  edge_score_wf_train14_20260418_20260518_h250_new_hf_nointer

score gates:
  expected_edge_threshold 0.10
  edge_if_filled_threshold 0.05
  fill_prob_threshold 0.05
  expected_edge_percentile 0.95
  edge_if_filled_percentile 0.97

regime:
  regime_expected_edge_gate on
  regime_min_bid_expected_edge 0.01
  regime_min_ask_expected_edge 0.01
  regime_expected_edge_ewm_alpha 0.00005

risk:
  order_qty 100
  max_position 300
  soft_position 100
  reduce_only_after_soft_position on
  daily_loss_limit 0.10
  daily_fill_limit 300
```

Baseline:

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_e95_i97_m001_a00005_20260418_20260518.aggregate.csv

6-day total +0.299850, fills 1014
bad3 -0.166050
good3 +0.465900
worst 20260421 -0.065500
```

Probe 1: quote distance only

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_decomp_quote_distance_bad3_good3.aggregate.csv

extra params:
  score_aware_quote_distance on
  quote_edge_if_filled_widen_target 0.75
  quote_edge_if_filled_widen_mult 0.25
  quote_regime_widen_threshold 0.02
  quote_regime_widen_mult 10.0
  quote_max_score_widen 1.0

6-day total +0.269050, fills 946
bad3 -0.150700
good3 +0.419750
worst 20260421 -0.073450
```

Probe 2: stale quote control only

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_decomp_stale_only_bad3_good3.aggregate.csv

extra params:
  stale_quote_control on
  stale_quote_ttl 1000ms
  stale_quote_min_edge_if_filled_margin 0.5
  stale_quote_min_regime 0.02

6-day total +0.239450, fills 998
bad3 -0.160000
good3 +0.399450
worst 20260421 -0.067050
```

Probe 3: position-age control only

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_decomp_position_age_bad3_good3.aggregate.csv

extra params:
  position_age_reduce_only 20000ms
  position_age_reduce_tighten 0.5

6-day total +0.293400, fills 1014
bad3 -0.169250
good3 +0.462650
worst 20260421 -0.064800
```

Probe 4: fill-prob widening only

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_decomp_fillprob_widen_bad3_good3.aggregate.csv

extra params:
  dynamic_quote on
  dynamic_quote_expected_edge_mult 0.0
  dynamic_quote_edge_if_filled_mult 0.0
  dynamic_quote_max_tighten 0.0
  dynamic_quote_fill_prob_widen_mult 5.0
  dynamic_quote_fill_prob_baseline 0.20
  dynamic_quote_max_widen 1.0

6-day total +0.194350, fills 904
bad3 -0.170300
good3 +0.364650
worst 20260421 -0.072650
```

Summary table:

```text
variant             total       fills  bad3       good3      worst
baseline            +0.299850   1014   -0.166050  +0.465900  -0.065500
quote_distance      +0.269050    946   -0.150700  +0.419750  -0.073450
stale_only          +0.239450    998   -0.160000  +0.399450  -0.067050
position_age        +0.293400   1014   -0.169250  +0.462650  -0.064800
fillprob_widen      +0.194350    904   -0.170300  +0.364650  -0.072650
```

Daily PnL:

```text
date      baseline   quote_dist  stale_only  pos_age    fillprob
20260420  -0.046350  -0.037450   -0.038050   -0.043300  -0.046800
20260421  -0.065500  -0.073450   -0.067050   -0.064800  -0.072650
20260512  -0.054200  -0.039800   -0.054900   -0.061150  -0.050850
20260514  +0.216500  +0.177250   +0.159450   +0.218400  +0.166500
20260515  +0.130450  +0.124100   +0.128950   +0.128800  +0.097750
20260517  +0.118950  +0.118400   +0.111050   +0.115450  +0.100400
```

Conclusion:

```text
Do not promote any of the 4 decomposed probes as-is.
Do not spend full-month validation time on these exact parameter sets.

quote_distance is the only useful lead:
  bad3 improves from -0.166050 to -0.150700.
  It helps 20260420 and 20260512, but worsens 20260421 and gives back good-day PnL.

position_age is near-neutral but not a bad-day fix:
  total is close to baseline, worst day is slightly better, but bad3 is worse.

stale_only and fillprob_widen are not useful in this form:
  both reduce total PnL materially.
  fillprob_widen is especially bad because it hurts both bad3 and good3.
```

Next direction:

```text
Continue only with quote-distance refinement.

Make it side/day/regime-aware instead of global:
  1. Handle 20260421 separately; current widening makes it worse.
  2. Use entry-side attribution, not fill-side attribution, when deciding which side to widen.
  3. Keep the good-day giveback constraint explicit; a bad-day improvement is not enough
     if it gives back more on 20260514/20260515/20260517.
```

## 2026-05-27 New22 Weak-Regime Score Widening

Purpose:

```text
Refine quote-distance after the decomposed probe showed that global
score-aware widening helped bad3 but worsened 20260421 and gave back too much
good-day PnL.
```

Attribution note:

```text
20260421 baseline vs quote-distance-only showed that the worse result was not
from the main 13 UTC bid-entry bad bucket getting worse.  It came mostly from
removing or changing profitable bid-entry groups around 14/8/20 UTC.

This points to the score-margin widening layer being too global.  Weak regime
widening is steadier, but by itself barely improves bad3.
```

Implementation:

```text
scripts/bitmex_edge_scored_maker.py now supports:
  quote_score_widen_max_regime_bps
  quote_bid_score_widen_mult / quote_ask_score_widen_mult
  quote_bid_regime_widen_mult / quote_ask_regime_widen_mult

Defaults preserve old behavior.
```

Tested variant:

```text
score-aware quote distance on
quote_edge_if_filled_widen_target 0.75
quote_edge_if_filled_widen_mult 0.25
quote_regime_widen_threshold 0.02
quote_regime_widen_mult 10.0
quote_max_score_widen 1.0
quote_score_widen_max_regime 0.02
```

Bad3+good3 result:

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_quotedist_score_weakregime_t002_bad3_good3.aggregate.csv

variant              total       fills  bad3       good3
baseline             +0.299850   1014   -0.166050  +0.465900
quote_distance_old   +0.269050    946   -0.150700  +0.419750
regime_only          +0.297250   1010   -0.164550  +0.461800
weakregime_score     +0.304750   1000   -0.158600  +0.463350

daily:
20260420  -0.043050 vs baseline -0.046350
20260421  -0.064250 vs baseline -0.065500
20260512  -0.051300 vs baseline -0.054200
20260514  +0.207800 vs baseline +0.216500
20260515  +0.134850 vs baseline +0.130450
20260517  +0.120700 vs baseline +0.118950
```

Full-month result:

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_quotedist_score_weakregime_t002_20260418_20260518.aggregate.csv

baseline current best:
  total +0.806400
  fills 3430
  PnL/fill +0.00023510
  positive days 18/31
  active-day positive days 18/28
  4 月 -0.037700
  5 月 +0.844100
  worst 20260421 -0.065500

weakregime_score:
  total +0.802150
  fills 3386
  PnL/fill +0.00023690
  positive days 18/31
  active-day positive days 18/28
  4 月 -0.032350
  5 月 +0.834500
  worst 20260421 -0.064250
```

Conclusion:

```text
Do not promote weakregime_score over the current best.

It passes bad3+good3 and improves 4 月 / worst day slightly, but full-month
PnL is lower by 0.004250 because it gives back enough 5 月 upside
to offset the bad-day improvements.

Main giveback days:
  20260504 -0.014850 vs baseline
  20260514 -0.008700
  20260429 -0.006950

Useful next direction:
  Keep the new side/regime-aware parameters.
  If continuing quote-distance, reduce good-day giveback before another
  full-month run, especially around high-upside days like 20260504/20260514.
```

## 2026-05-27 Good-Day Giveback Constraints

Purpose:

```text
Try to reduce weakregime_score giveback on 20260504/20260514 without losing
the bad3 improvement.
```

Probe 1: tighter regime trigger

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_quotedist_score_regime_t001_bad3_good3.aggregate.csv

params changed from weakregime_score:
  quote_regime_widen_threshold 0.01
  quote_score_widen_max_regime 0.01

6-day result:
  total +0.299850
  bad3 -0.166050
  good3 +0.465900

Conclusion:
  This fully protects good3, but it also removes the bad3 improvement.
  It is effectively back to baseline behavior.
```

Probe 2: bid-focused widening, softer ask widening

```text
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_quotedist_bidfocus_askhalf_bad3_good3.aggregate.csv
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_quotedist_bidfocus_askhalf_20260504.aggregate.csv

params changed from weakregime_score:
  quote_ask_score_widen_mult 0.0
  quote_ask_regime_widen_mult 0.5

6-day result:
  total +0.301250
  bad3 -0.163650
  good3 +0.464900

key days:
  20260504 +0.147200 vs baseline +0.151450 and weakregime +0.136600
  20260514 +0.207700 vs baseline +0.216500 and weakregime +0.207800

Conclusion:
  This recovers part of 20260504 but does not fix 20260514.
  Bad3 improvement is also smaller than weakregime_score.
  Do not run full-month validation for this parameter set.
```

Updated conclusion:

```text
The current quote-distance controls are not selective enough.

Lowering regime trigger protects good days but loses the bad-day fix.
Side-specific ask softening helps 20260504 somewhat but does not solve 20260514.

Do not promote any constrained quote-distance variant.
The current best remains new22 e95/i97/regime_min=0.01/hard-risk.
```

### 20260514 Entry/Fill Pattern Attribution

Purpose:

```text
Explain why weakregime_score gives back 20260514 even though the day has strong
positive PnL and mostly positive side regime.
```

Files compared:

```text
baseline:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_e95_i97_m001_a00005_20260418_20260518_20260514.fills.csv

weakregime_score:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_quotedist_score_weakregime_t002_20260418_20260518_20260514.fills.csv

bidfocus_askhalf:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_quotedist_bidfocus_askhalf_bad3_good3_20260514.fills.csv

regime_t001:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_quotedist_score_regime_t001_bad3_good3_20260514.fills.csv
```

Day-level comparison:

```text
variant          pnl        fills  bid fill pnl  ask fill pnl
baseline         +0.216500  270    +0.113825    +0.102675
weakregime       +0.207800  266    +0.107100    +0.100700
bidfocus_askhalf +0.207700  266    +0.109175    +0.098525
regime_t001      +0.216500  270    +0.113825    +0.102675
```

Main attribution:

```text
The giveback is concentrated in 03 UTC entry sequencing.

Entry round-trip delta vs baseline:
  weakregime 03 UTC ask entries:
    trips 10 -> 8
    PnL +0.004400 -> -0.001350
    delta -0.005750

  weakregime 03 UTC bid entries:
    trips 5 -> 6
    PnL -0.002500 -> -0.005350
    delta -0.002850

Combined 03 UTC entry delta:
  about -0.008600, which explains almost all of the daily
  weakregime giveback vs baseline (-0.008700).
```

Detailed pattern:

```text
Baseline had three consecutive profitable 03 UTC trips:
  bid entry, pnl +0.000350, entry regime +0.0199
  ask entry, pnl +0.003550, entry regime +0.0234
  ask entry, pnl +0.002250, entry regime +0.0293

weakregime changed this local sequence into:
  bid entry, pnl -0.000750, entry regime +0.0196, entry spread 3.124bps
  bid entry, pnl -0.001750, placement record not valid / spread 0 in attribution

After that, the rest of the major profitable 14/15/16/17 UTC buckets are
nearly unchanged.
```

Interpretation:

```text
20260514 is not a broad good-day degradation.
It is a path-dependence issue around 03 UTC caused by quote-distance triggering
in a narrow positive-but-low regime band.

regime_t001 matches baseline exactly, so the harmful trigger band is roughly:
  0.01 < side regime <= 0.02

Disabling or weakening ask-side widening does not fix it because the local path
change starts around early 03 UTC placement timing and then changes the next
entry side sequence.
```

Implication:

```text
Do not keep hand-sweeping global quote-distance knobs.

If quote-distance is revisited, the next test should separate:
  1. score widening only when side regime is negative, not merely below +0.02.
  2. regime widening in low-positive regime vs negative regime.
  3. path-sensitive effects around early-session sequences like 03 UTC.
```

### Negative-Regime-Only Widening

Purpose:

```text
Test whether score/regime widening can keep the bad-day improvement while
avoiding the 20260504/20260514 good-day giveback by only triggering when side
regime is negative.
```

Result:

```text
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_quotedist_negative_regime_bad3_good3_20260504.aggregate.csv
```

Params changed from weakregime_score:

```text
quote_regime_widen_threshold 0.0
quote_score_widen_max_regime 0.0
```

7-day guard set comparison:

```text
variant     total7     bad3       good3      20260504   fills
baseline    +0.451300  -0.166050  +0.465900  +0.151450  1314
weakregime  +0.441350  -0.158600  +0.463350  +0.136600  1300
negreg      +0.451300  -0.166050  +0.465900  +0.151450  1314
```

Day-level check:

```text
date      baseline   negreg
20260420  -0.046350  -0.046350
20260421  -0.065500  -0.065500
20260504  +0.151450  +0.151450
20260512  -0.054200  -0.054200
20260514  +0.216500  +0.216500
20260515  +0.130450  +0.130450
20260517  +0.118950  +0.118950
```

Conclusion:

```text
Negative-regime-only widening fully protects 20260504/20260514, but it also
collapses back to baseline on the bad3 set. The weakregime bad-day improvement
comes from including at least part of the low-positive regime band, while that
same band creates the 20260514 path-dependence giveback.

Do not run full-month validation for this variant.
Do not promote.
```

## 2026-05-26 New22 Previous-Period Validation

Purpose:

```text
Check whether the current new22 main candidate generalizes to the previous
20260317-20260417 period with the same e95/i97/regime_min=0.01/hard-risk params.
```

Data:

```text
Backfilled BitMEX XBTUSDT data from 20260301 to 20260417.
The 14-day walk-forward for test period 20260317-20260417 uses training data
starting at 20260303.
```

Setup:

```text
period: 20260317-20260417
model: 22 factors, no interactions
model tag: edge_score_wf_train14_20260317_20260417_h250_new_hf_nointer
strategy: expected pct95, edge_if_filled pct97, regime_min 0.01, alpha 0.00005
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_e95_i97_m001_a00005_20260317_20260417.aggregate.csv
```

Research-layer walk-forward:

```text
bid avg top-bottom actual edge +0.118239 bps
ask avg top-bottom actual edge +0.115351 bps
bid avg top-bottom fill prob +0.017788
ask avg top-bottom fill prob +0.021413
```

Strategy result:

```text
total PnL +2.175163
gross PnL +2.175850
maker rebate -0.000687
fills 8025
PnL/fill +0.00027105
positive days 27/32
active-day positive days 27/32

3 月段:
  PnL +1.285163
  fills 3953
  PnL/fill +0.00032511
  positive days 13/15
  worst 20260324 -0.048450
  best 20260323 +0.302750

4 月段:
  PnL +0.890000
  fills 4072
  PnL/fill +0.00021857
  positive days 14/17
  worst 20260406 -0.044850
  best 20260412 +0.141950
```

Conclusion:

```text
This is a strong additional validation for the current new22 candidate.
The previous period is materially stronger than 20260418-20260518:
  +2.175163 vs +0.806400 total PnL.
  27/32 positive days vs 18/31.

Do not change the main candidate based on this alone; it reinforces it.
The next bottleneck remains bad-day mechanics and quote-distance refinement,
especially because the later month still contains clear loss days.
```
