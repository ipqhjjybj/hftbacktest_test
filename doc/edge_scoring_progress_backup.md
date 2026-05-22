# Edge Scoring / Maker Research Progress Backup

更新时间: 2026-05-22

这份文档用于恢复当前研究上下文，避免会话中断后丢失进度。

## 当前阶段

当前已经从固定规则做市切换到 edge scoring 研究阶段。

核心目标不是先看 PnL，而是先验证:

```text
模型给高分的 maker quote opportunity，事后是否真的比低分机会更好。
```

如果这个成立，下一步才把 score 接入真正做市策略。

## 已经放弃/弱化的方向

### factor_filtered_maker 二因子组合规则

已经把原来的单因子规则修成了真正的二因子 AND 组合规则:

```text
factor1 bucket X AND factor2 bucket Y
```

但用户在另一个窗口跑完整 rolling OOS 时，结果基本全亏。

中途状态:

```text
完成 14/31 天
累计 PnL 约 -1.50788
20260501: PnL -0.148950, fills 140
```

结论:

```text
二因子固定规则解决了“不下单”的 bug，但交易层 rolling OOS 失败。
后续不要继续主要优化 fixed rule factor_filtered_maker。
```

## 当前有效方向

### Edge score 模型

新增了 edge scoring 训练/预测层。

当前模型输出 6 个预测目标:

```text
bid_expected_edge_bps
ask_expected_edge_bps
bid_edge_if_filled_bps
ask_edge_if_filled_bps
bid_fill_prob
ask_fill_prob
```

含义:

```text
expected_edge:
  一个 quote opportunity 的整体期望 edge，已经包含不成交为 0 的影响。

edge_if_filled:
  只看成交以后，这笔 maker fill 的成交质量/adverse selection。

fill_prob:
  这个 maker quote 在 TTL 内成交的概率。
```

关系近似为:

```text
expected_edge ~= fill_prob * edge_if_filled
```

注意: 当前策略使用时不能再简单把 `pred_expected_edge * pred_fill_prob` 相乘，否则会重复乘成交概率。更合理的是:

```text
pred_expected_edge 做主 gate
pred_fill_prob 做成交活跃度过滤
pred_edge_if_filled 做成交质量/toxic 过滤
```

## 新增/相关代码文件

### Edge scoring

```text
scripts/factor_research/edge_scoring.py
scripts/bitmex_edge_score_train_predict.py
scripts/bitmex_edge_score_walk_forward.py
```

说明:

```text
edge_scoring.py:
  numpy ridge regression 模型，不依赖 sklearn。
  包括标准化、二阶项、因子交互项、预测、score bucket 评估。

bitmex_edge_score_train_predict.py:
  单次 train/test 入口。
  给定 train dates 和 test dates，输出模型和 OOS 评估。

bitmex_edge_score_walk_forward.py:
  rolling walk-forward driver。
  每天用过去 N 天训练，测试下一天。
  每跑完一天就增量写汇总 CSV。
```

### Factor research / combo rules

相关修改包括:

```text
scripts/factor_research/buckets.py
scripts/factor_research/bitmex_factor_research.py
scripts/bitmex_factor_filtered_maker.py
scripts/bitmex_factor_filtered_walk_forward.py
scripts/factor_research/README.md
```

关键点:

```text
maker_fill_combo_rules.csv:
  训练阶段直接统计二因子组合规则，不再从单因子桶临时拼。
```

## 已验证的小样本

### Edge scoring 小样本验证

命令:

```bash
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/bitmex_edge_score_train_predict.py \
  --skip-download \
  --train-dates 20260512 \
  --test-dates 20260513 \
  --sample-interval-ms 100 \
  --horizon-ms 250 \
  --maker-fee-rate 0.0 \
  --max-samples-per-day 50000 \
  --max-train-samples 30000 \
  --score-buckets 5 \
  --result-tag debug_edge_score_with_if_filled_train_20260512_test_20260513
```

输出:

```text
results/factor_research/bitmex_xbtusdt_debug_edge_score_with_if_filled_train_20260512_test_20260513.edge_model.json
results/factor_research/bitmex_xbtusdt_debug_edge_score_with_if_filled_train_20260512_test_20260513.edge_score_summary.csv
results/factor_research/bitmex_xbtusdt_debug_edge_score_with_if_filled_train_20260512_test_20260513.edge_score_buckets.csv
```

确认字段已包含:

```text
pred_edge_if_filled_mean_bps
actual_edge_if_filled_mean_bps
edge_if_filled_corr
```

## 已完成的 7 天 edge score rolling OOS

测试区间:

```text
20260512 - 20260518
train_days = 14
horizon = 250ms
maker_fee = 0
```

命令:

```bash
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/bitmex_edge_score_walk_forward.py \
  --skip-download \
  --test-start 20260512 \
  --test-end 20260518 \
  --train-days 14 \
  --sample-interval-ms 100 \
  --horizon-ms 250 \
  --maker-fee-rate 0.0 \
  --max-train-samples 2000000 \
  --score-buckets 10 \
  --result-tag edge_score_wf_train14_20260512_20260518_h250_maker0
```

输出:

```text
results/factor_research/bitmex_xbtusdt_edge_score_wf_train14_20260512_20260518_h250_maker0.edge_score_walk_forward_daily.csv
results/factor_research/bitmex_xbtusdt_edge_score_wf_train14_20260512_20260518_h250_maker0.edge_score_walk_forward_bucket_summary.csv
```

结果摘要:

```text
bid edge_corr 每天为正，大约 0.0299 - 0.0838
ask edge_corr 每天为正，大约 0.0457 - 0.1046

bid top bucket - bottom bucket actual edge 平均约 +0.0658 bps
ask top bucket - bottom bucket actual edge 平均约 +0.0781 bps
```

解释:

```text
score 有排序能力，但这还不是策略 PnL。
它只说明高分机会事后确实比低分机会更好。
```

## 用户正在另一个窗口跑的完整 edge score OOS

命令:

```bash
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/bitmex_edge_score_walk_forward.py \
  --skip-download \
  --test-start 20260418 \
  --test-end 20260518 \
  --train-days 14 \
  --sample-interval-ms 100 \
  --horizon-ms 250 \
  --maker-fee-rate 0.0 \
  --max-train-samples 2000000 \
  --score-buckets 10 \
  --result-tag edge_score_wf_train14_20260418_20260518_h250_maker0
```

模型和结果输出目录:

```text
results/factor_research/
```

完整 rolling 汇总文件:

```text
results/factor_research/bitmex_xbtusdt_edge_score_wf_train14_20260418_20260518_h250_maker0.edge_score_walk_forward_daily.csv
results/factor_research/bitmex_xbtusdt_edge_score_wf_train14_20260418_20260518_h250_maker0.edge_score_walk_forward_bucket_summary.csv
```

用户中途观察:

```text
4 天中途汇总:
bid/ask top-bottom actual edge spread 都是 4/4 天为正
平均约 +0.210 / +0.225 bps
但 top bucket fill probability 比 bottom bucket 低
fill spread 平均约 -8.3 / -8.5 个百分点
```

解释:

```text
模型挑出的高分机会成交质量更好，但更难成交。
这是 maker 策略常见的质量/成交率 tradeoff。
后续要看 expected_edge 优势能否抵消 fill_prob 下降。
```

## 当前判断

```text
factor_filtered_maker 固定规则交易层失败。
edge scoring 有研究价值，正在跑完整月度 OOS 验证。
下一步不应该继续优化固定规则，而应该写 edge_scored_maker。
```

## 下一步计划

### 1. 等完整 edge score rolling OOS 跑完

重点看:

```text
edge_corr 是否大多数为正
top bucket actual_expected_edge 是否稳定高于 bottom bucket
top bucket actual_edge_if_filled 是否更好
top bucket fill_prob 是否低到没有交易量
```

### 2. 写真正的 edge_scored_maker 回测策略

建议新增:

```text
scripts/bitmex_edge_scored_maker.py
```

第一版策略逻辑:

```text
每个测试日加载对应 rolling edge_model.json
实时计算当前 factors
实时预测:
  pred_expected_edge
  pred_edge_if_filled
  pred_fill_prob

挂 bid 条件:
  bid_expected_edge > threshold
  bid_fill_prob > min_fill_prob
  bid_edge_if_filled > min_edge_if_filled

ask 同理。
```

第一版先不要做复杂动态报价，只做:

```text
score gate + 现有 maker 报价框架
```

### 3. 做 threshold grid search

建议扫:

```text
expected_edge_threshold: 0.00, 0.01, 0.02, 0.03, 0.05 bps
fill_prob_threshold: 1%, 2%, 3%, 5%, 8%
edge_if_filled_threshold: 0.00, 0.05, 0.10, 0.20 bps
```

观察:

```text
PnL
gross
fills
positive days
max position
avg capture
top bucket usage
```

### 4. 第二版再加动态报价

方向:

```text
expected_edge 高 -> 挂近
expected_edge 低 -> 不挂或挂远
inventory 偏多 -> ask 近、bid 远
inventory 偏空 -> bid 近、ask 远
vol 高 -> 两边挂远
```

## 恢复上下文时优先看的文件

```text
doc/edge_scoring_progress_backup.md
scripts/factor_research/edge_scoring.py
scripts/bitmex_edge_score_train_predict.py
scripts/bitmex_edge_score_walk_forward.py
scripts/factor_research/README.md
```

## 当前 git 状态备注

截至写入本文档前，`git status --short` 显示:

```text
 M .gitignore
 M scripts/factor_research/README.md
?? scripts/bitmex_edge_score_train_predict.py
?? scripts/bitmex_edge_score_walk_forward.py
?? scripts/factor_research/edge_scoring.py
```

`.gitignore` 是已有修改，本次备份不处理它。
