# Edge Scoring / Maker Research Checkpoint

更新时间: 2026-05-27

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
  已补做 20260504/20260514 good-day giveback 约束 probe。
  weak-regime-gated score widening 原本 bad3+good3 通过：+0.304750，
  但完整月度 +0.802150，低于 current best +0.806400，
  主要回吐来自 20260504/20260514。
  约束 1：把 quote_regime_widen_threshold 和 score_widen_max_regime
  收到 0.01，6 日回到 baseline +0.299850，bad3 改善消失。
  约束 2：bid-focused，ask score widening 关掉且 ask regime widening 减半，
  6 日 +0.301250，bad3 -0.163650，good3 +0.464900；
  20260504 回到 +0.147200，但仍低于 baseline +0.151450，
  20260514 仍 +0.207700，低于 baseline +0.216500。
  结论：两个约束都不跑 full month，不 promote，current best 不变。
  已完成 20260514 entry/fill pattern attribution：
  weakregime 的 20260514 回吐几乎集中在 03 UTC entry sequencing。
  03 UTC ask-entry 从 baseline 的 10 trips / +0.004400 变成
  8 trips / -0.001350，bid-entry 从 5 trips / -0.002500 变成
  6 trips / -0.005350；合计约 -0.008600，解释了当天几乎全部
  giveback。regime_t001 与 baseline 完全一致，说明有害触发带约为
  0.01 < side regime <= 0.02。
  已补做 negative-regime-only widening：
  quote_regime_widen_threshold=0.0 且 quote_score_widen_max_regime=0.0。
  7 日 guard set 结果完全等于 baseline：
  total +0.451300，bad3 -0.166050，good3 +0.465900，20260504 +0.151450。
  结论：它保护 20260504/20260514，但 bad3 改善也消失；
  不跑 full month，不 promote。
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
2. weak-regime-gated score widening 也不推广，虽然 bad3+good3 通过，
   但 full-month 没超过 current best。
3. current best 仍是 new22 e95/i97/regime_min=0.01/hard-risk。
4. 继续 quote-distance 的收益不明确；negative-regime-only 太保守，
   <= +0.02 又误伤 20260514。
5. 后续不要再扫单一全局 regime threshold；如果继续做 quote-distance，
   应改成 pattern-specific guard，例如只在 low-positive regime 同时伴随
   额外 adverse-selection / entry-sequence 风险时才 widening，或转向
   position-age / exit-pressure 控制。
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
