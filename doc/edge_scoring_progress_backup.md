# Edge Scoring / Maker Research Progress Backup

更新时间: 2026-05-25

这份文档只用于快速恢复当前研究上下文。详细实验流水、长命令和逐日结果已拆到 `doc/edge_scoring/`。

## 当前状态

项目已从固定规则做市转向:

```text
edge scoring model + edge_scored_maker strategy
```

当前核心判断:

```text
1. factor_filtered_maker 固定规则方向已弱化，不再作为主线。
2. edge score 研究层 OOS 排序能力成立。
3. 交易层已从负收益推进到月度正收益候选，但收益仍低，仍需稳健性验证。
4. 新 22 高频因子有增量信息；重新校准 gate 后，7 天交易层结果超过旧 9 因子最好组合。
```

## 当前最好候选

旧 9 因子完整月度最好:

```text
period: 20260418-20260518
model: 9 factors, with interactions
strategy: pct95 expected + pct95 if_filled + regime + hard risk
total PnL: +0.550429
fills: 4177
positive days: 19/31
4 月: -0.075700
5 月: +0.626129
worst day: 20260420 -0.081350
```

新 22 因子 7 天最好:

```text
period: 20260512-20260518
model: 22 factors, no interactions
strategy: expected pct95, if_filled pct97, regime_min 0.01, alpha 0.00005
total PnL: +0.559350
fills: 1220
positive days: 6/7
PnL/fill: +0.00045848
worst day: 20260512 -0.054200
```

对比旧 9 因子同 7 天:

```text
old 9f best:
  total PnL +0.496950
  fills 1490
  PnL/fill +0.00033352
  positive days 6/7
```

详细参数和命令见:

```text
doc/edge_scoring/current_best.md
```

## 当前结论

```text
新 22 因子研究层提升:
  expected-edge top-bottom spread 提升约 8-10%
  edge_if_filled top-bottom spread 提升约 13-17%

新 22 因子交易层:
  直接套旧 gate 不提升总 PnL。
  重新校准后，7 天 PnL/fill 明显提高，但交易数下降，且 20260512 仍是主要亏损日。
```

当前不应直接宣布新模型更稳。下一步必须验证:

```text
new22 e95/i97/regime_min=0.01 在完整 20260418-20260518 月度是否仍优于旧 9f best。
```

## 下一步

优先级:

```text
1. 跑完整月度 new22 no-interactions e95/i97/regime_min=0.01。
2. 如果月度优于旧 9f best，再做 4 月亏损日归因。
3. 如果月度不优于旧 9f best，保留 22 因子为研究分支，不替换主候选。
4. 之后再考虑 score-aware quote distance，不要继续只扫 gate。
```

建议月度回测参数:

```text
model-tag:
  需要先补跑 20260418-20260518 的 new22 walk-forward model。

strategy:
  expected-edge-threshold 0.10
  edge-if-filled-threshold 0.05
  fill-prob-threshold 0.05
  expected-edge-percentile 0.95
  edge-if-filled-percentile 0.97
  regime-min 0.01
  regime-alpha 0.00005
  order_qty 100
  max_position 300
  soft_position 100
  reduce_only_after_soft_position
  daily_loss_limit 0.10
  daily_fill_limit 300
```

## Key Files

Code:

```text
scripts/factor_research/factors.py
scripts/factor_research/edge_scoring.py
scripts/bitmex_edge_score_train_predict.py
scripts/bitmex_edge_score_walk_forward.py
scripts/bitmex_edge_scored_maker.py
```

Docs:

```text
doc/edge_scoring/current_best.md
doc/edge_scoring/factor_set.md
doc/edge_scoring/experiments.md
doc/edge_scoring/archive_2026-05-25_full_progress.md
```

Important result files:

```text
results/factor_research/bitmex_xbtusdt_edge_score_wf_train14_20260512_20260518_h250_new_hf_nointer.edge_score_walk_forward_daily.csv
results/factor_research/bitmex_xbtusdt_edge_score_wf_train14_20260512_20260518_h250_new_hf_nointer.edge_score_walk_forward_bucket_summary.csv
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_e95_i97_m001_a00005_20260512_20260518.aggregate.csv
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_regime_risk_grid_full_p000_a00005.aggregate.csv
```

## Maintenance Rule

以后维护规则:

```text
edge_scoring_progress_backup.md:
  只写当前状态、最好结果、下一步、关键路径。

doc/edge_scoring/current_best.md:
  只写当前最好候选和复现命令。

doc/edge_scoring/factor_set.md:
  只写因子定义和训练/策略侧一致性。

doc/edge_scoring/experiments.md:
  只写实验摘要和结果路径，不贴长 JSON。

archive_*.md:
  保存历史全文，平时不更新。
```
