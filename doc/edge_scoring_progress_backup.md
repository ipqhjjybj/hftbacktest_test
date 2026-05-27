# Edge Scoring / Maker Research Checkpoint

更新时间: 2026-05-26

这份文档只作为轻量恢复入口，不记录完整实验流水。

## 用途

```text
1. 快速恢复当前研究主线。
2. 指向当前最好候选、实验日志、因子定义和历史归档。
3. 记录下一步优先级。
```

不在这里维护:

```text
1. 长命令。
2. 逐日结果。
3. 参数 sweep 明细。
4. attribution 表格。
5. 历史实验全文。
```

## 当前主线

```text
strategy:
  edge_scored_maker

main candidate:
  new 22-factor model, no interactions
  + pct95 expected gate
  + pct97 edge_if_filled gate
  + regime_min 0.01
  + hard risk

status:
  new22 已完成完整 20260418-20260518 月度验证。
  月度 PnL +0.806400，优于旧 9f static best +0.550429 和 placement best +0.563879。
  4 月亏损收窄到 -0.037700，5 月收益提升到 +0.844100。
  已补充前一段 20260317-20260417 验证，PnL +2.175163，fills 8025，
  胜率 27/32，3 月段 +1.285163，4 月段 +0.890000。
  两段验证均为正，前一段更强；但 20260418-20260518 胜率没有提升，
  且仍有明显亏损日。

latest probe:
  已完成 new22 quote-control 拆分实验，验证日为 bad3
  20260420/20260421/20260512 加 good3 20260514/20260515/20260517。
  baseline 6-day +0.299850。
  quote-distance-only +0.269050，是唯一改善 bad3 的方向
  （bad3 -0.166050 -> -0.150700），但损失 good3 且 20260421 变差。
  stale-only +0.239450，position-age-only +0.293400，
  fill-prob-widen-only +0.194350，均不应按当前参数推进月度。
```

当前最好候选、参数和复现入口:

```text
doc/edge_scoring/current_best.md
```

实验摘要:

```text
doc/edge_scoring/experiments.md
```

## 下一步

```text
1. 不推广当前 4 个 quote-control 拆分参数，不跑这些参数的完整月度。
2. 继续只 refine quote-distance，因为它是唯一改善 bad3 的层。
3. 下一版 quote-distance 要按 entry-side attribution 做 side/day/regime-aware 控制，
   重点修 20260421，而不是全局一起 widening。
4. 每个新 quote-distance 参数先跑 bad3+good3，只有同时保护 good3 才进入月度。
```

## 文档分工

```text
doc/edge_scoring_progress_backup.md:
  轻量 checkpoint / handoff 入口。
  只写当前主线、当前风险、下一步和文档指针。

doc/edge_scoring/current_best.md:
  当前最好候选、核心参数、核心结果和复现入口。

doc/edge_scoring/experiments.md:
  按时间记录紧凑实验摘要。
  sweep、对比实验、阶段性结论放这里。

doc/edge_scoring/factor_set.md:
  因子定义和训练/策略侧一致性。

doc/edge_scoring/loss_attribution_new22.md:
  new22 主候选的亏损日归因。
  记录 fill-side 和 entry round-trip 两种视角。

doc/edge_scoring/archive_*.md:
  历史全文归档，平时不更新。
```

## 关键文件

```text
scripts/factor_research/factors.py
scripts/factor_research/edge_scoring.py
scripts/bitmex_edge_score_train_predict.py
scripts/bitmex_edge_score_walk_forward.py
scripts/bitmex_edge_scored_maker.py
```
