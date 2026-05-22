# HFT Factor Research

This folder is a lightweight factor research layer on top of the local BitMEX
npz replay data. It is separate from strategy backtests on purpose: it studies
whether a signal predicts future mid-price movement or maker edge before that
signal is wired into a strategy.

Example:

```bash
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/factor_research/bitmex_factor_research.py \
  --symbol XBTUSDT \
  --dates 20260516 20260517 20260518 \
  --skip-download \
  --sample-interval-ms 100 \
  --horizons-ms 100 250 500 1000
```

Outputs are written to `results/factor_research/`:

- `*.ic.csv`: Pearson IC and rank IC between each factor and label.
- `*.buckets.csv`: quantile bucket statistics for the selected horizon.
- `*.report.md`: a compact markdown report.
- `*.rank_ic_heatmap.png`: factor IC heatmap by forecast horizon.
- `*.bucket_future_ret.png`: future-return bucket curves.
- `*.bucket_maker_edge.png`: approximate maker-edge bucket curves.
- `*.fill_buckets.csv`: bucket statistics for hypothetical BBO fill probability and fill-conditioned edge.
- `*.maker_fill_edge_buckets.csv`: side-aware maker fill edge table with latency, TTL, post-only, and queue assumptions.
- `*.maker_fill_combo_rules.csv`: two-factor AND rule candidates, measured directly on the joint bucket, not inferred from separate one-factor buckets.
- `*.bucket_fill_probability.png`: hypothetical BBO bid/ask fill probability by factor bucket.
- `*.bucket_fill_edge.png`: fill-conditioned maker edge by factor bucket.
- `*.lifecycle_fill_probability.png`: lifecycle maker fill probability by factor bucket.
- `*.lifecycle_edge_if_filled.png`: lifecycle maker edge after modeled fills.
- `*.lifecycle_expected_edge.png`: fill-probability-weighted lifecycle maker edge per quote opportunity.
- optional `*.samples.npz`: sampled factor and label arrays when `--write-samples` is used.

The lifecycle fill layer is still a research approximation, but it adds:

- order entry latency before the quote reaches the book,
- post-only reject checks at the entry timestamp,
- whether the quote is still at BBO or improves BBO,
- displayed queue-ahead assumptions,
- quote TTL,
- fill detection from opposite-side active trade quantity,
- edge measured from the modeled fill timestamp.

The fill analysis uses a conservative simple rule: a hypothetical bid at current
best bid is marked filled only when future sell trade quantity within the
horizon is greater than displayed bid quantity times `--queue-ahead-multiplier`
plus `--hypothetical-order-qty`. Ask-side fill is symmetric with future buy
trade quantity.

## Edge Scoring

The edge scoring layer turns the same factor/label data into continuous scores
instead of fixed bucket rules. It trains separate ridge models for:

- `bid_expected_edge_bps`
- `ask_expected_edge_bps`
- `bid_edge_if_filled_bps`
- `ask_edge_if_filled_bps`
- `bid_fill_prob`
- `ask_fill_prob`

Example train/predict run:

```bash
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/bitmex_edge_score_train_predict.py \
  --skip-download \
  --train-start 20260505 \
  --train-end 20260511 \
  --test-start 20260512 \
  --test-end 20260512 \
  --maker-fee-rate 0.0 \
  --horizon-ms 250 \
  --result-tag edge_score_train7_test_20260512
```

Outputs:

- `*.edge_model.json`: model coefficients, standardization stats, and train diagnostics.
- `*.edge_score_summary.csv`: test-set edge/fill correlation and mean actual edge by side.
- `*.edge_score_buckets.csv`: test-set score buckets; this is the first check that higher score buckets have better realized maker edge.
- optional `*.edge_scores.npz`: timestamped predicted scores when `--write-scores` is set.
