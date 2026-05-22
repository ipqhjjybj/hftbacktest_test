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
- `*.bucket_fill_probability.png`: hypothetical BBO bid/ask fill probability by factor bucket.
- `*.bucket_fill_edge.png`: fill-conditioned maker edge by factor bucket.
- optional `*.samples.npz`: sampled factor and label arrays when `--write-samples` is used.

The first version samples order book state and recent trades. It does not yet
simulate queue position, fill probability, or actual order lifecycle.

The fill analysis uses a conservative simple rule: a hypothetical bid at current
best bid is marked filled only when future sell trade quantity within the
horizon is greater than displayed bid quantity times `--queue-ahead-multiplier`
plus `--hypothetical-order-qty`. Ask-side fill is symmetric with future buy
trade quantity.
