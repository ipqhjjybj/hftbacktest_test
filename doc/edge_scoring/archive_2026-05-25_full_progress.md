# Edge Scoring / Maker Research Progress Backup

更新时间: 2026-05-24

这份文档用于恢复当前研究上下文，避免会话中断后丢失进度。

## 当前阶段

当前已经从固定规则做市切换到 edge scoring + edge_scored_maker 研究阶段。

核心目标不是先看 PnL，而是先验证:

```text
模型给高分的 maker quote opportunity，事后是否真的比低分机会更好。
```

这个已经在月度 OOS 中成立；当前重点是验证 score 接入做市策略后，交易层风控能否把研究 edge 转成稳定 PnL。

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

## 已完成的完整 edge score OOS

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

新增 edge_if_filled 后的完整 rolling 汇总文件:

```text
results/factor_research/bitmex_xbtusdt_edge_score_wf_train14_20260418_20260518_h250_maker0_with_if_filled.edge_score_walk_forward_daily.csv
results/factor_research/bitmex_xbtusdt_edge_score_wf_train14_20260418_20260518_h250_maker0_with_if_filled.edge_score_walk_forward_bucket_summary.csv
```

完整月度 `with_if_filled` 结果摘要:

```text
20260418-20260518, train_days=14, horizon=250ms, maker_fee=0

bid:
  edge_corr 平均/最小/最大 = 0.0766 / 0.0225 / 0.1354
  edge_if_filled_corr 平均/最小/最大 = 0.2260 / 0.1122 / 0.3417
  fill_corr 平均/最小/最大 = 0.1773 / 0.0917 / 0.2726
  top-bottom actual_expected_edge spread 平均 +0.1389 bps，31/31 天为正
  top-bottom actual_edge_if_filled spread 平均 +1.1843 bps，31/31 天为正
  top-bottom fill_prob spread 平均 -4.53 个百分点，22/31 天为负

ask:
  edge_corr 平均/最小/最大 = 0.0776 / 0.0134 / 0.1356
  edge_if_filled_corr 平均/最小/最大 = 0.2169 / 0.0765 / 0.2825
  fill_corr 平均/最小/最大 = 0.1730 / 0.0971 / 0.2601
  top-bottom actual_expected_edge spread 平均 +0.1384 bps，31/31 天为正
  top-bottom actual_edge_if_filled spread 平均 +1.1060 bps，31/31 天为正
  top-bottom fill_prob spread 平均 -4.49 个百分点，22/31 天为负
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
edge scoring 完整月度 OOS 已确认有稳定排序能力:
  expected_edge top-bottom spread 31/31 天为正
  edge_if_filled top-bottom spread 31/31 天为正
但高分机会通常 fill_prob 更低，成交质量和成交率存在 tradeoff。

edge_scored_maker 第一版已经写出并跑了月度回测。
裸 score gate 全月仍亏，intraday percentile gate 明显改善但仍未转正。
下一步重点不是继续优化 fixed rule，而是围绕 edge_scored_maker 做更硬的库存/日内风控和 threshold grid。
```

## 下一步计划

### 1. 完整 edge score rolling OOS 已完成

已确认:

```text
edge_corr 全部为正
top bucket actual_expected_edge 稳定高于 bottom bucket
top bucket actual_edge_if_filled 稳定更好
top bucket fill_prob 多数天低于 bottom bucket，但并非完全没有交易量
```

### 2. edge_scored_maker 回测策略第一版已完成

已新增第一版:

```text
scripts/bitmex_edge_scored_maker.py
```

第一版实现:

```text
加载 rolling edge_model.json
实时计算当前 factors
实时预测 expected_edge / fill_prob / edge_if_filled
用 score gate 决定是否允许 bid/ask 挂单
订单管理、TTL、REST pacing、库存 skew 复用原 maker 框架
```

已做 smoke/grid 初测:

```bash
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/bitmex_edge_scored_maker.py \
  --dates 20260512 20260513 20260514 20260515 20260516 20260517 20260518 \
  --skip-download \
  --model-dir results/factor_research \
  --model-tag edge_score_wf_train14_20260512_20260518_h250_maker0 \
  --expected-edge-threshold-bps 0.10 \
  --fill-prob-threshold 0.05 \
  --edge-if-filled-threshold-bps -999 \
  --maker-fee-rate 0.0 \
  --taker-fee-rate 0.0001 \
  --exchange-model no_partial \
  --result-tag edge_scored_20260512_20260518_e010_f005
```

结果:

```text
20260512-20260518
total PnL +0.266350
gross +0.266350
fills 1800
positive days 5/7
aggregate:
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_20260512_20260518_e010_f005.aggregate.csv
```

注意:

```text
这次历史初测使用的是旧 7 天 rolling 模型，模型里没有 edge_if_filled。
因此 edge-if-filled gate 设置为 -999，实际只启用了 expected_edge + fill_prob gate。
后续已经补跑了包含 edge_if_filled 的 7 天和完整月度版本，见 2026-05-24 同步记录。
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

### 3. 做 threshold / risk grid search

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

注意: 2026-05-24 后，优先级应从单纯抬 score 阈值，转到 score gate + 库存/日内风控一起扫。

### 4. 第二版再加动态报价

方向:

```text
expected_edge 高 -> 挂近
expected_edge 低 -> 不挂或挂远
inventory 偏多 -> ask 近、bid 远
inventory 偏空 -> bid 近、ask 远
vol 高 -> 两边挂远
```

## 2026-05-23 percentile gate 实验

在 `scripts/bitmex_edge_scored_maker.py` 中新增了可选的日内在线 percentile gate。

实现方式:

```text
--intraday-percentile-gate
--expected-edge-percentile
--edge-if-filled-percentile
--fill-prob-percentile
--percentile-warmup-samples
--percentile-update-interval-ms
```

策略用当天已经观察到的预测 score 在线维护 histogram，按分位点生成动态阈值。
预热期不允许新开仓，只允许 reduce side 减仓。
动态阈值和原绝对阈值取更严格者。

已跑完整月度实验:

```bash
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/bitmex_edge_scored_maker.py \
  --dates 20260418 ... 20260518 \
  --skip-download \
  --model-dir results/factor_research \
  --model-tag edge_score_wf_train14_20260418_20260518_h250_maker0_with_if_filled \
  --expected-edge-threshold-bps 0.10 \
  --edge-if-filled-threshold-bps 0.05 \
  --fill-prob-threshold 0.05 \
  --intraday-percentile-gate \
  --expected-edge-percentile 0.95 \
  --edge-if-filled-percentile 0.95 \
  --maker-fee-rate 0.0 \
  --taker-fee-rate 0.0001 \
  --exchange-model no_partial \
  --result-tag edge_scored_e010_if005_f005_pct95_20260418_20260518
```

结果文件:

```text
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_e010_if005_f005_pct95_20260418_20260518.aggregate.csv
```

对比原 `edge>=0.10, fill>=0.05, if_filled>=0.05`:

```text
原参数:
  total PnL -0.938686
  fills 6955
  positive days 11/31
  4 月 -0.531830
  5 月 -0.406856
  last 7 days +0.306900
  max position max 1000

叠加 intraday top 5% expected + top 5% if_filled:
  total PnL -0.320934
  fills 5902
  positive days 14/31
  4 月 -0.487980
  5 月 +0.167045
  last 7 days +0.406290
  max position max 900
```

结论:

```text
percentile gate 有效降低尾部亏损，尤其 20260504 从 -0.4907 改到约 -0.0714。
但全月仍为负，4 月段仍明显亏，不能单独上线。
下一步应把 percentile gate 和更硬的库存/日内风控一起扫:
  max_position 200/300/500
  order_qty 50/100
  daily loss cap
  fill cap
  soft position 后禁止继续开仓
```

## 2026-05-24 本地进度同步

当前本地已有的 edge_scored_maker aggregate:

```text
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_20260418_20260518_e010_f005_if005.aggregate.csv
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_e010_if005_f005_pct95_20260418_20260518.aggregate.csv
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_20260512_20260518_e010_f005_if005.aggregate.csv
```

关键对比:

```text
7 天 with_if_filled gate:
  dates 20260512-20260518
  total PnL +0.306900
  fills 1782
  positive days 4/7
  max position max 800

完整月度裸 gate:
  edge>=0.10, fill>=0.05, if_filled>=0.05
  total PnL -0.938686
  fills 6955
  positive days 11/31
  4 月 -0.531830
  5 月 -0.406856
  last 7 days +0.306900
  worst day 20260504 -0.490700
  max position max 1000

完整月度 intraday percentile gate:
  edge>=0.10, fill>=0.05, if_filled>=0.05
  expected_edge pct95 + edge_if_filled pct95
  total PnL -0.320934
  fills 5902
  positive days 14/31
  4 月 -0.487980
  5 月 +0.167045
  last 7 days +0.406290
  worst day 20260420 -0.204200
  max position max 900
```

额外单日/局部实验:

```text
20260504 pct95 单日复现使用 min_expected_edge=-999, min_edge_if_filled=-999，只保留 fill>=0.05:
  PnL -0.059800
  fills 510
  max position 900

这组与完整月度 pct95 主实验参数不同，不能直接替代主实验结论。
```

当前结论:

```text
edge 模型本身排序能力够稳定，问题主要在交易层:
  1. 高质量 quote 更难成交，fill_prob 下降会吃掉一部分 edge。
  2. 裸 gate 在 4 月尾部日亏损太大。
  3. percentile gate 能降低尾部亏损，但不够。

下一轮应优先扫风控/仓位参数，而不是继续只抬 score 阈值:
  --max-position-contracts 200/300/500
  --soft-position-contracts 100/200/300
  --order-qty-contracts 50/100
  日内 loss cap / fill cap
  soft position 后禁止继续开仓，只允许 reduce side
```

## 2026-05-24 硬风控试跑

在 `scripts/bitmex_edge_scored_maker.py` 加了可选硬风控参数，默认关闭，不改变旧实验:

```text
--reduce-only-after-soft-position
--daily-loss-limit-usdt
--daily-fill-limit
```

实现含义:

```text
reduce-only-after-soft-position:
  abs(position) >= soft_position 后，只允许 reduce side，被动减仓。

daily-loss-limit-usdt:
  当日从初始 equity 到日内最低 equity 的 drawdown 超过阈值后，禁止新开仓，只允许 reduce side。

daily-fill-limit:
  当日 fill 数达到上限后，禁止新开仓，只允许 reduce side。
```

先用坏日 `20260504` smoke:

```bash
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/bitmex_edge_scored_maker.py \
  --dates 20260504 \
  --skip-download \
  --model-dir results/factor_research \
  --model-tag edge_score_wf_train14_20260418_20260518_h250_maker0_with_if_filled \
  --expected-edge-threshold-bps 0.10 \
  --edge-if-filled-threshold-bps 0.05 \
  --fill-prob-threshold 0.05 \
  --intraday-percentile-gate \
  --expected-edge-percentile 0.95 \
  --edge-if-filled-percentile 0.95 \
  --order-qty-contracts 100 \
  --max-position-contracts 300 \
  --soft-position-contracts 100 \
  --reduce-only-after-soft-position \
  --daily-loss-limit-usdt 0.10 \
  --daily-fill-limit 300 \
  --maker-fee-rate 0.0 \
  --taker-fee-rate 0.0001 \
  --exchange-model no_partial \
  --result-tag edge_scored_risktest_20260504_q100_max300_soft100_loss010_fill300
```

结果:

```text
20260504
PnL +0.011850
fills 300
max position 100

对比:
  裸 gate: -0.490700
  pct95:   -0.071400
```

但完整月度同参数不改善:

```text
result:
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_risk_q100_max300_soft100_loss010_fill300_20260418_20260518.aggregate.csv

20260418-20260518:
  total PnL -0.344484
  fills 6158
  positive days 12/31
  4 月 -0.580712
  5 月 +0.236229
  last 7 days +0.431250
  worst day 20260423 -0.105350
  max position max 100

对比 pct95 原实验:
  total PnL -0.320934
  fills 5902
  positive days 14/31
  4 月 -0.487980
  5 月 +0.167045
  last 7 days +0.406290
  worst day 20260420 -0.204200
  max position max 900
```

结论:

```text
硬风控能压掉单日尾部，比如 20260504，但不能解决 4 月连续小亏。
这说明主要问题不是单纯 max_position 太大，而是 regime/日级别 edge 不稳定:
  4 月持续负期望
  5 月尤其最后 7 天明显转好

后续不要只继续扫 max_position/order_qty。
更值得做的是 regime gate:
  用在线统计的日内平均/分位 expected_edge 作为全局开关
  或者先离线验证哪些日级特征能区分 4 月亏损 regime 和 5 月盈利 regime
```

## 2026-05-24 regime gate 开始

按上面的结论，下一步先做最小在线 regime gate，而不是继续扫仓位:

```text
目标:
  用日内正在观察到的 bid/ask expected_edge 运行均值判断当前日/当前边是否值得交易。

第一版:
  --regime-expected-edge-gate
  --regime-expected-edge-warmup-samples
  --regime-expected-edge-ewm-alpha
  --regime-min-bid-expected-edge-bps
  --regime-min-ask-expected-edge-bps

逻辑:
  维护 bid_expected_edge / ask_expected_edge 的日内 EWM。
  预热期和低于阈值时禁止对应 side 新开仓。
  reduce side 仍然允许，用来被动减仓。

先测:
  pct95 原策略 + regime gate，不先叠加硬风控。
  如果能显著砍掉 4 月亏损而保留 5 月收益，再和硬风控组合。
```

## 2026-05-24 regime gate 结果

已在 `scripts/bitmex_edge_scored_maker.py` 中实现:

```text
--regime-expected-edge-gate
--regime-expected-edge-warmup-samples
--regime-expected-edge-ewm-alpha
--regime-min-bid-expected-edge-bps
--regime-min-ask-expected-edge-bps
```

第一组: pct95 + regime gate，不叠加硬风控:

```bash
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/bitmex_edge_scored_maker.py \
  --dates 20260418 ... 20260518 \
  --skip-download \
  --model-dir results/factor_research \
  --model-tag edge_score_wf_train14_20260418_20260518_h250_maker0_with_if_filled \
  --expected-edge-threshold-bps 0.10 \
  --edge-if-filled-threshold-bps 0.05 \
  --fill-prob-threshold 0.05 \
  --intraday-percentile-gate \
  --expected-edge-percentile 0.95 \
  --edge-if-filled-percentile 0.95 \
  --regime-expected-edge-gate \
  --regime-expected-edge-warmup-samples 6000 \
  --regime-expected-edge-ewm-alpha 0.0001 \
  --regime-min-bid-expected-edge-bps -0.02 \
  --regime-min-ask-expected-edge-bps -0.02 \
  --maker-fee-rate 0.0 \
  --taker-fee-rate 0.0001 \
  --exchange-model no_partial \
  --result-tag edge_scored_regime_minneg002_20260418_20260518
```

结果:

```text
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_regime_minneg002_20260418_20260518.aggregate.csv

total PnL -0.058965
fills 4779
positive days 15/31
4 月 -0.491343
5 月 +0.432379
last 7 days +0.437850
max position max 900
worst day 20260426 -0.200286
```

对比 pct95 原策略:

```text
pct95 原策略:
  total PnL -0.320934
  fills 5902
  positive days 14/31
  4 月 -0.487980
  5 月 +0.167045
  last 7 days +0.406290

pct95 + regime:
  total PnL -0.058965
  fills 4779
  positive days 15/31
  4 月 -0.491343
  5 月 +0.432379
  last 7 days +0.437850
```

解释:

```text
regime gate 明显提升 5 月段，且保留最后 7 天收益。
但单独使用时 max position 仍可到 900，20260426 / 20260504 / 20260510 仍有高仓位尾部风险。
```

第二组: pct95 + regime gate + 硬仓位/soft/reduce-only/fill/loss cap:

```bash
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/bitmex_edge_scored_maker.py \
  --dates 20260418 ... 20260518 \
  --skip-download \
  --model-dir results/factor_research \
  --model-tag edge_score_wf_train14_20260418_20260518_h250_maker0_with_if_filled \
  --expected-edge-threshold-bps 0.10 \
  --edge-if-filled-threshold-bps 0.05 \
  --fill-prob-threshold 0.05 \
  --intraday-percentile-gate \
  --expected-edge-percentile 0.95 \
  --edge-if-filled-percentile 0.95 \
  --regime-expected-edge-gate \
  --regime-expected-edge-warmup-samples 6000 \
  --regime-expected-edge-ewm-alpha 0.0001 \
  --regime-min-bid-expected-edge-bps -0.02 \
  --regime-min-ask-expected-edge-bps -0.02 \
  --order-qty-contracts 100 \
  --max-position-contracts 300 \
  --soft-position-contracts 100 \
  --reduce-only-after-soft-position \
  --daily-loss-limit-usdt 0.10 \
  --daily-fill-limit 300 \
  --maker-fee-rate 0.0 \
  --taker-fee-rate 0.0001 \
  --exchange-model no_partial \
  --result-tag edge_scored_regime_risk_minneg002_q100_max300_soft100_loss010_fill300_20260418_20260518
```

结果:

```text
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_regime_risk_minneg002_q100_max300_soft100_loss010_fill300_20260418_20260518.aggregate.csv

total PnL +0.271479
fills 5035
positive days 16/31
4 月 -0.280750
5 月 +0.552229
last 7 days +0.446050
max position max 100
worst day 20260421 -0.097400
```

当前最好组合:

```text
pct95 expected_edge + pct95 edge_if_filled
regime expected_edge EWM gate:
  alpha 0.0001
  warmup 6000 samples
  min bid/ask expected_edge -0.02 bps
hard risk:
  order_qty 100
  max_position 300
  soft_position 100
  reduce-only-after-soft-position
  daily_loss_limit 0.10 USDT
  daily_fill_limit 300

全月从 -0.320934 改到 +0.271479。
这不是最终可上线结果，但说明方向从“证明信号”推进到了“有交易层正收益候选”。
```

下一步:

```text
不要立即加动态报价。
优先做稳健性验证:
  1. 扫 regime_min_expected_edge: -0.03, -0.02, -0.01, 0.00
  2. 扫 alpha: 0.00005, 0.0001, 0.0002
  3. 保持 max_position/soft_position 固定，确认不是单点过拟合
  4. 换 horizon 或后移测试窗口复跑
```

## 2026-05-24 regime gate 参数 sweep

`-0.02 bps` 的来源说明:

```text
它不是训练/优化出来的参数，而是第一轮人工 probe。
当时目标是“不完全要求日内 EWM 为正，但过滤明显负 expected_edge regime”。
因此先取了一个略负的阈值 -0.02 bps，再与 alpha=0.0001 组合试跑。
这个参数必须通过 sweep 验证，不能当作稳定最优值。
```

本轮先做 8 天代表样本 probe，固定:

```text
dates:
  20260420 20260421 20260424 20260426
  20260508 20260513 20260514 20260516

base:
  pct95 expected_edge + pct95 edge_if_filled
  order_qty 100
  max_position 300
  soft_position 100
  reduce-only-after-soft-position
  daily_loss_limit 0.10
  daily_fill_limit 300
  regime_warmup 6000

sweep:
  regime_min_expected_edge: -0.03, -0.02, -0.01, 0.00 bps
  regime_alpha: 0.00005, 0.0001, 0.0002
```

8 天 probe 排序:

```text
threshold alpha    pnl       pos_days fills avg_fills worst
0.00      0.00005  +0.187300 5/8      1162  145.2     -0.081350
-0.01     0.00005  +0.137550 4/8      1258  157.2     -0.100950
-0.01     0.0002   +0.106800 4/8      1386  173.2     -0.086800
0.00      0.0001   +0.101100 4/8      1218  152.2     -0.098850
-0.01     0.0001   +0.097350 3/8      1326  165.8     -0.096100
-0.02     0.0002   +0.085200 3/8      1476  184.5     -0.085250
-0.03     0.00005  +0.076700 3/8      1494  186.8     -0.104950
0.00      0.0002   +0.069600 3/8      1324  165.5     -0.094600
-0.02     0.0001   +0.054000 3/8      1372  171.5     -0.097400
-0.02     0.00005  +0.045800 3/8      1348  168.5     -0.104050
-0.03     0.0002   +0.034100 3/8      1538  192.2     -0.102650
-0.03     0.0001   +0.019850 4/8      1452  181.5     -0.107000
```

probe 结论:

```text
原先 -0.02 / 0.0001 不是局部最优，只是能跑通的启发式点。
更严格的 threshold 明显更好，尤其 0.00 / 0.00005。
alpha 太快不一定更好；0.00 threshold 下 0.00005 明显优于 0.0001 / 0.0002。
```

随后对两个候选做完整 20260418-20260518 回测，并与旧最好组合对比:

```text
old best: threshold=-0.02, alpha=0.0001
  aggregate:
    results/bitmex_xbtusdt_edge_scored_maker_edge_scored_regime_risk_minneg002_q100_max300_soft100_loss010_fill300_20260418_20260518.aggregate.csv
  total PnL +0.271479
  fills 5035, avg/day 162.4
  positive days 16/31
  zero-fill days 1
  4 月 -0.280750
  5 月 +0.552229
  last 7 days +0.446050
  worst day 20260421 -0.097400
  best day 20260514 +0.134900
  max position max 100

new best: threshold=0.00, alpha=0.00005
  aggregate:
    results/bitmex_xbtusdt_edge_scored_maker_edge_scored_regime_risk_grid_full_p000_a00005.aggregate.csv
  total PnL +0.550429
  fills 4177, avg/day 134.7
  positive days 19/31
  zero-fill days 2
  4 月 -0.075700
  5 月 +0.626129
  last 7 days +0.496950
  worst day 20260420 -0.081350
  best day 20260514 +0.126700
  max position max 100

balanced candidate: threshold=-0.01, alpha=0.0002
  aggregate:
    results/bitmex_xbtusdt_edge_scored_maker_edge_scored_regime_risk_grid_full_m001_a0002.aggregate.csv
  total PnL +0.365400
  fills 4920, avg/day 158.7
  positive days 16/31
  zero-fill days 1
  4 月 -0.254100
  5 月 +0.619500
  last 7 days +0.442850
  worst day 20260420 -0.086800
  best day 20260514 +0.148200
  max position max 100
```

`threshold=0.00, alpha=0.00005` 日收益/交易数:

```text
20260418 fills=28  pnl=+0.016050
20260419 fills=262 pnl=-0.005650
20260420 fills=170 pnl=-0.081350
20260421 fills=216 pnl=-0.048450
20260422 fills=208 pnl=-0.035350
20260423 fills=172 pnl=-0.020400
20260424 fills=12  pnl=+0.002500
20260425 fills=0   pnl=+0.000000
20260426 fills=64  pnl=-0.006800
20260427 fills=188 pnl=+0.097800
20260428 fills=2   pnl=+0.001700
20260429 fills=84  pnl=+0.006300
20260430 fills=22  pnl=-0.002050
20260501 fills=46  pnl=+0.001700
20260502 fills=8   pnl=-0.004400
20260503 fills=30  pnl=+0.001100
20260504 fills=300 pnl=+0.040600
20260505 fills=140 pnl=+0.020500
20260506 fills=168 pnl=+0.061850
20260507 fills=102 pnl=+0.019900
20260508 fills=88  pnl=+0.007850
20260509 fills=0   pnl=+0.000000
20260510 fills=215 pnl=-0.033321
20260511 fills=162 pnl=+0.013400
20260512 fills=128 pnl=+0.034150
20260513 fills=180 pnl=+0.095500
20260514 fills=296 pnl=+0.126700
20260515 fills=268 pnl=+0.105150
20260516 fills=136 pnl=+0.091350
20260517 fills=182 pnl=+0.062250
20260518 fills=300 pnl=-0.018150
```

更新后的当前最好组合:

```text
pct95 expected_edge + pct95 edge_if_filled
regime expected_edge EWM gate:
  threshold 0.00 bps
  alpha 0.00005
  warmup 6000 samples
hard risk:
  order_qty 100
  max_position 300
  soft_position 100
  reduce-only-after-soft-position
  daily_loss_limit 0.10 USDT
  daily_fill_limit 300

全月 PnL 从旧最好 +0.271479 提升到 +0.550429。
代价是交易数从 5035 降到 4177，且出现 2 个 zero-fill days。
```

当前判断:

```text
regime gate 的方向成立，但 -0.02 bps 不应继续作为默认。
0.00 / 0.00005 更像当前窗口内的主候选:
  4 月亏损显著收敛
  5 月收益没有被牺牲，反而略提升
  worst day 也更小

但这仍可能是窗口内选择偏差。
下一步必须做:
  1. 后移/扩展测试窗口复跑 0.00/0.00005
  2. 对比 maker fee / taker fee 不同假设
  3. 看 zero-fill days 是否能接受
  4. 再考虑动态报价，不要先加复杂度
```

## 2026-05-25 fill attribution + daily budget/top-N 初查

在 `scripts/bitmex_edge_scored_maker.py` 中新增逐笔 fill attribution 输出，不改变策略行为:

```text
每个 result npz 旁边额外输出:
  *.fills.csv

记录字段包括:
  fill side
  position before/after
  exec_px / mid / spread_capture
  equity_delta_since_prev_fill
  bid/ask predicted expected_edge
  bid/ask predicted edge_if_filled
  bid/ask predicted fill_prob
  side score thresholds / margins
  bid/ask regime EWM
```

注意:

```text
当前 attribution 记录的是“成交检测时”的最新 score，不是下单瞬间绑定到 order id 的 score。
用于第一轮 bucket / top-N 诊断可以接受。
如果后续要做严格因果归因，需要把下单时 score 固定记录到 order id。
```

8 天代表样本:

```text
dates:
  20260420 20260421 20260424 20260426
  20260508 20260513 20260514 20260516

base:
  threshold=0.00
  alpha=0.00005
  pct95 expected_edge + pct95 edge_if_filled
  hard risk 同当前最好组合

aggregate:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_attr_probe_p000_a00005.aggregate.csv

结果:
  total PnL +0.187300
  fills 1162
  positive days 5/8
  worst day -0.081350
```

逐笔 attribution bucket 结果:

```text
按 side_expected_edge_bps 分 5 桶:
  q1 pnl -0.108425, avg/fill -0.0004673
  q2 pnl -0.061150, avg/fill -0.0002636
  q3 pnl +0.086425, avg/fill +0.0003709
  q4 pnl +0.094750, avg/fill +0.0004084
  q5 pnl +0.175700, avg/fill +0.0007541

按 side_edge_if_filled_bps 分 5 桶:
  q1 pnl -0.049175, avg/fill -0.0002120
  q2 pnl -0.083975, avg/fill -0.0003620
  q3 pnl -0.001975, avg/fill -0.0000085
  q4 pnl +0.121650, avg/fill +0.0005244
  q5 pnl +0.200775, avg/fill +0.0008617

按 side_fill_prob 分 5 桶:
  q1 pnl -0.065950, avg/fill -0.0002843
  q2 pnl +0.032300, avg/fill +0.0001392
  q3 pnl +0.091825, avg/fill +0.0003941
  q4 pnl +0.072350, avg/fill +0.0003119
  q5 pnl +0.056775, avg/fill +0.0002437

按 side_regime_ewm_bps 分 5 桶:
  q1 pnl +0.002650, avg/fill +0.0000114
  q2 pnl +0.032200, avg/fill +0.0001388
  q3 pnl -0.011125, avg/fill -0.0000477
  q4 pnl -0.003600, avg/fill -0.0000155
  q5 pnl +0.167175, avg/fill +0.0007175
```

初步解释:

```text
expected_edge / edge_if_filled 的排序在真实 fills 上仍然有用:
  低分桶明显亏
  高分桶贡献绝大多数收益

fill_prob 不是越高越好:
  最低 fill_prob 桶亏
  中高桶较好
  最高桶反而不如 q3/q4

regime EWM 只有最高桶明显好，中间桶没有单调性。
这说明 regime gate 更适合做粗过滤，不适合直接当连续排序 score。
```

离线 daily budget/top-N attribution:

```text
说明:
  这只是从已发生 fills 中按预测 score 选择 top-N 后重算 equity_delta。
  它不是严格回放，因为被过滤的 fill 会改变后续库存路径。

chronological daily budget:
  N=40  pnl +0.096000, fills 292, pos_days 6/8, worst -0.003450
  N=80  pnl +0.054300, fills 556, pos_days 4/8, worst -0.012600
  N=120 pnl +0.085600, fills 764, pos_days 5/8, worst -0.049300
  N=300 pnl +0.187300, fills 1162, pos_days 5/8, worst -0.081350

top-N by side_expected_margin:
  N=40  pnl +0.131625, fills 292, pos_days 5/8, worst -0.012550
  N=80  pnl +0.219125, fills 556, pos_days 4/8, worst -0.033400
  N=120 pnl +0.290625, fills 764, pos_days 6/8, worst -0.068650

top-N by side_edge_if_filled_margin:
  N=40  pnl +0.141425, fills 292, pos_days 6/8, worst -0.015200
  N=80  pnl +0.157450, fills 556, pos_days 4/8, worst -0.033625
  N=120 pnl +0.232775, fills 764, pos_days 5/8, worst -0.059925

top-N by composite_margin = expected_margin + 0.1 * edge_if_filled_margin:
  N=40  pnl +0.126925, fills 292, pos_days 5/8, worst -0.011900
  N=80  pnl +0.227250, fills 556, pos_days 6/8, worst -0.033775
  N=120 pnl +0.272700, fills 764, pos_days 5/8, worst -0.066450
```

离线结论:

```text
从已发生 fills 看，top score 子集确实更赚钱。
但这还不能直接当策略结果，必须用真实回测验证。
```

真实 daily budget / top-N 近似回测:

```text
base current best, same 8 days:
  pct95/pct95, fill cap 300
  PnL +0.187300
  fills 1162
  positive days 5/8
  worst -0.081350

expected pct97 / if_filled pct95 / fill cap 120:
  PnL +0.093700
  fills 766
  positive days 5/8
  worst -0.055100

expected pct95 / if_filled pct97 / fill cap 160:
  PnL +0.087500
  fills 922
  positive days 5/8
  worst -0.076350

expected pct97 / if_filled pct97 / fill cap 160:
  PnL +0.094200
  fills 904
  positive days 5/8
  worst -0.086250

expected pct98 / if_filled pct98 / fill cap 160:
  PnL +0.143700
  fills 834
  positive days 6/8
  worst -0.044900
```

真实回测结论:

```text
更严格 percentile + fill budget 可以降低最差日和交易量，
但目前没有提高 8 天总收益。
离线 top-N 的收益提升没有在真实回放中兑现，说明库存路径/成交路径影响很大。

其中 pct98/pct98/fill160 是防守型候选:
  总收益低于 base
  但 positive days 更多，worst day 更小
```

side-specific 初查:

```text
逐笔 attribution 按成交 side:
  bid fills: n=581, pnl +0.180250, avg/fill +0.0003102
  ask fills: n=581, pnl +0.007050, avg/fill +0.0000121

但直接做 side-entry gate 的真实回测显示，逐笔 side attribution 会误导。
原因是 bid/ask fill 本身和库存路径绑定，不能简单把一侧 fill 当独立贡献。
```

8 天真实 side-entry probe:

```text
two-sided base:
  PnL +0.187300
  fills 1162
  positive days 5/8
  worst -0.081350

ask-entry only:
  用 regime_min_bid_expected_edge=999 关闭 bid 新开仓，bid 只做 reduce side。
  PnL +0.189100
  fills 698
  positive days 4/8
  worst -0.020900

bid-entry only:
  用 regime_min_ask_expected_edge=999 关闭 ask 新开仓，ask 只做 reduce side。
  PnL +0.068300
  fills 786
  positive days 4/8
  worst -0.052550
```

因为 ask-entry only 在 8 天样本上风险更小，做了完整月度确认:

```text
aggregate:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_side_full_ask_entries_p000_a00005.aggregate.csv

ask-entry only full 20260418-20260518:
  total PnL +0.447179
  fills 2657, avg/day 85.7
  positive days 17/31
  zero-fill days 5
  4 月 -0.120350
  5 月 +0.567529
  last 7 days +0.390000
  worst day 20260422 -0.105050
  best day 20260517 +0.107100

two-sided current best:
  total PnL +0.550429
  fills 4177, avg/day 134.7
  positive days 19/31
  zero-fill days 2
  4 月 -0.075700
  5 月 +0.626129
  last 7 days +0.496950
  worst day 20260420 -0.081350
  best day 20260514 +0.126700
```

当前结论:

```text
1. fill attribution 确认高 expected_edge / high edge_if_filled fills 更赚钱。
2. 简单 daily fill budget 和更严格 percentile 目前主要降低风险，不提升收益。
3. side-entry 单边化不是新最好；ask-entry full 比双边少交易但收益/4 月/worst 都更差。
4. 下一步更应该做“下单时 score 绑定 order id”的严格 attribution，
   然后基于 placement score 做真实门控，而不是继续只提高 percentile。
```

## 2026-05-25 dynamic quote 初试

在 `scripts/bitmex_edge_scored_maker.py` 中新增可选动态报价，默认关闭:

```text
--dynamic-quote
--dynamic-quote-expected-edge-mult
--dynamic-quote-edge-if-filled-mult
--dynamic-quote-max-tighten-bps
--dynamic-quote-fill-prob-widen-mult
--dynamic-quote-fill-prob-baseline
--dynamic-quote-max-widen-bps
```

实现逻辑:

```text
原报价:
  half_spread = base_half_spread + inventory_spread

动态报价:
  expected_margin = side_expected_edge - side_expected_threshold
  if_margin = side_edge_if_filled - side_if_threshold
  tighten = expected_mult * max(expected_margin, 0)
          + if_mult * max(if_margin, 0)
  tighten capped by max_tighten_bps

  fill_prob_excess = max(side_fill_prob - fill_prob_baseline, 0)
  widen = fill_prob_widen_mult * fill_prob_excess
  widen capped by max_widen_bps

  half_spread = base_half_spread + inventory_spread - tighten + widen
  half_spread 不低于 1 tick
```

解释:

```text
高 expected_edge / high edge_if_filled 时挂近一点，提高成交率。
高 fill_prob 可能代表更容易成交但更 toxic，所以保留可选 widen 项。
reduce side 不套动态报价，继续让库存 skew 决定。
```

8 天代表样本 probe，对比当前静态 best:

```text
static best:
  PnL +0.187300
  fills 1162
  positive days 5/8
  worst -0.081350

dynamic e1_if025_t2:
  expected_mult=1.0, if_mult=0.25, max_tighten=2.0, fill_widen=0
  PnL +0.224750
  fills 1276
  positive days 5/8
  worst -0.098450

dynamic e2_if05_t2:
  expected_mult=2.0, if_mult=0.5, max_tighten=2.0, fill_widen=0
  PnL +0.197500
  fills 1402
  positive days 4/8
  worst -0.090700

dynamic e3_if075_t25:
  expected_mult=3.0, if_mult=0.75, max_tighten=2.5, fill_widen=0
  PnL +0.072350
  fills 1474
  positive days 3/8
  worst -0.099550

dynamic e2_if05_t2_fw4:
  expected_mult=2.0, if_mult=0.5, max_tighten=2.0
  fill_prob_widen_mult=4.0, fill_prob_baseline=0.20
  PnL +0.213500
  fills 1322
  positive days 5/8
  worst -0.082600
```

8 天结论:

```text
动态报价能增加交易数和样本收益，但贴近过头会恶化 worst day。
带 fill_prob widen 的版本更均衡，worst 接近静态 best。
```

完整月度确认:

```text
static current best:
  aggregate:
    results/bitmex_xbtusdt_edge_scored_maker_edge_scored_regime_risk_grid_full_p000_a00005.aggregate.csv
  total PnL +0.550429
  fills 4177, avg/day 134.7
  positive days 19/31
  zero-fill days 2
  4 月 -0.075700
  5 月 +0.626129
  last 7 days +0.496950
  worst day 20260420 -0.081350
  best day 20260514 +0.126700

dynamic e1_if025_t2:
  aggregate:
    results/bitmex_xbtusdt_edge_scored_maker_edge_scored_dynamic_full_e1_if025_t2_p000_a00005.aggregate.csv
  total PnL +0.516479
  fills 4609, avg/day 148.7
  positive days 22/31
  zero-fill days 1
  4 月 -0.113700
  5 月 +0.630179
  last 7 days +0.528300
  worst day 20260420 -0.098450
  best day 20260514 +0.157650

dynamic e2_if05_t2_fw4:
  aggregate:
    results/bitmex_xbtusdt_edge_scored_maker_edge_scored_dynamic_full_e2_if05_t2_fw4_p000_a00005.aggregate.csv
  total PnL +0.544279
  fills 4831, avg/day 155.8
  positive days 20/31
  zero-fill days 1
  4 月 -0.123250
  5 月 +0.667529
  last 7 days +0.474600
  worst day 20260420 -0.082600
  best day 20260514 +0.140150
```

dynamic e2_if05_t2_fw4 日级:

```text
20260418 fills=36  pnl=+0.021400
20260419 fills=300 pnl=+0.008200
20260420 fills=206 pnl=-0.082600
20260421 fills=230 pnl=-0.036950
20260422 fills=238 pnl=-0.064900
20260423 fills=218 pnl=-0.043950
20260424 fills=14  pnl=+0.002600
20260425 fills=0   pnl=+0.000000
20260426 fills=84  pnl=-0.018750
20260427 fills=214 pnl=+0.100250
20260428 fills=4   pnl=-0.002400
20260429 fills=90  pnl=+0.000900
20260430 fills=28  pnl=-0.007050
20260501 fills=58  pnl=+0.001950
20260502 fills=8   pnl=-0.004900
20260503 fills=36  pnl=+0.001500
20260504 fills=300 pnl=+0.049300
20260505 fills=172 pnl=+0.026800
20260506 fills=222 pnl=+0.064000
20260507 fills=128 pnl=-0.003400
20260508 fills=112 pnl=+0.017150
20260509 fills=2   pnl=+0.001950
20260510 fills=299 pnl=+0.033429
20260511 fills=188 pnl=+0.005150
20260512 fills=172 pnl=-0.020900
20260513 fills=216 pnl=+0.130600
20260514 fills=300 pnl=+0.140150
20260515 fills=292 pnl=+0.093350
20260516 fills=160 pnl=+0.061300
20260517 fills=204 pnl=+0.038500
20260518 fills=300 pnl=+0.031600
```

当前结论:

```text
动态报价没有超过静态 current best。

它的收益结构是:
  交易数增加
  5 月收益提高或接近
  但 4 月亏损扩大
  spread capture 平均下降

最接近可用的是 dynamic e2_if05_t2_fw4:
  PnL 只比静态少约 0.00615
  fills 多 654
  worst day 接近静态
  但 4 月明显更差，说明动态贴近会放大坏 regime 暴露

因此当前默认仍应保持静态报价。
动态报价如果继续做，应和更强的 regime / placement-score gate 绑定，
不要单独用 score margin 贴近。
```

## 2026-05-25 placement-score attribution

进一步修正 fill attribution:

```text
之前 *.fills.csv 记录的是“成交检测时”的最新 score。
这对第一轮诊断有用，但不等价于下单时的决策信息。

现在在 scripts/bitmex_edge_scored_maker.py 中给 bid/ask 两个常驻 order id
分别绑定当前有效订单的 placement record。
每次 submit / modify 时记录:
  placement timestamp
  action: submit / modify
  target_px
  target half_spread
  position
  bid/ask predicted expected_edge
  bid/ask predicted edge_if_filled
  bid/ask predicted fill_prob
  side score thresholds / margins
  bid/ask regime EWM

每次 fill 时按成交 side 取对应 order id 的 placement record 写入 *.fills.csv。
cancel / ttl cancel / filter cancel 时清空对应 placement record。
```

新增 `.fills.csv` 字段:

```text
placement_valid
placement_action
placement_timestamp_ns
placement_age_ns
placement_target_px
placement_half_spread_bps
placement_position_contracts
placement_bid_expected_edge_bps
placement_ask_expected_edge_bps
placement_bid_edge_if_filled_bps
placement_ask_edge_if_filled_bps
placement_bid_fill_prob
placement_ask_fill_prob
placement_side_expected_edge_bps
placement_side_edge_if_filled_bps
placement_side_fill_prob
placement_side_expected_threshold_bps
placement_side_edge_if_filled_threshold_bps
placement_side_fill_prob_threshold
placement_side_expected_margin_bps
placement_side_edge_if_filled_margin_bps
placement_side_fill_prob_margin
placement_bid_regime_expected_edge_ewm_bps
placement_ask_regime_expected_edge_ewm_bps
```

Smoke:

```text
date 20260513
result:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_placement_smoke_p000_a00005_20260513.fills.csv

PnL / fills 与原静态 best 单日一致:
  +0.095500 / 180 fills

placement valid:
  178 / 180 fills
  valid rate 98.89%

placement_age:
  min 10.0ms
  avg 578.2ms
  p50 约 420.0ms
  p95 约 1830.0ms
  max 2900.0ms
```

8 天代表样本:

```text
aggregate:
  results/bitmex_xbtusdt_edge_scored_maker_edge_scored_placement_probe_p000_a00005.aggregate.csv

PnL / fills 与原 8 天静态 best 一致:
  +0.187300 / 1162 fills

placement valid:
  1124 / 1162 fills
  valid rate 96.73%
  valid fills PnL +0.173075

placement_age:
  min 0.0ms
  avg 567.5ms
  p50 420.0ms
  p95 1830.0ms
  max 6000.0ms

placement_action:
  submit 484
  modify 640
```

下单时 score bucket:

```text
按 placement_side_expected_edge_bps 分 5 桶:
  q1 pnl -0.095475, avg/fill -0.0004262
  q2 pnl -0.023850, avg/fill -0.0001060
  q3 pnl +0.066525, avg/fill +0.0002957
  q4 pnl +0.057375, avg/fill +0.0002550
  q5 pnl +0.168500, avg/fill +0.0007489

按 placement_side_edge_if_filled_bps 分 5 桶:
  q1 pnl -0.100350, avg/fill -0.0004480
  q2 pnl +0.036400, avg/fill +0.0001618
  q3 pnl -0.000550, avg/fill -0.0000024
  q4 pnl +0.058175, avg/fill +0.0002586
  q5 pnl +0.179400, avg/fill +0.0007973

按 placement_side_expected_margin_bps 分 5 桶:
  q1 pnl -0.066825, avg/fill -0.0002983
  q2 pnl -0.022325, avg/fill -0.0000992
  q3 pnl +0.039850, avg/fill +0.0001771
  q4 pnl +0.059825, avg/fill +0.0002659
  q5 pnl +0.162550, avg/fill +0.0007224

按 placement_side_edge_if_filled_margin_bps 分 5 桶:
  q1 pnl -0.058450, avg/fill -0.0002609
  q2 pnl +0.027450, avg/fill +0.0001220
  q3 pnl +0.023550, avg/fill +0.0001047
  q4 pnl +0.057700, avg/fill +0.0002564
  q5 pnl +0.122825, avg/fill +0.0005459
```

placement half-spread bucket:

```text
q1/q2 主要是 3.0bps 附近，PnL 明显为正。
q3/q4 主要混入 4.5bps 库存加宽，PnL 偏弱。
这说明 soft inventory / reduce-only 路径对 fill attribution 影响很大，
不能只看 score 做独立判断。
```

离线 placement top-N attribution:

```text
说明:
  仍然不是严格策略回放。
  它只是用 placement score 对已发生 fills 做重排/截取。

top-N by placement_side_expected_margin:
  N=40  pnl +0.133400, fills 291, pos_days 7/8, worst -0.007725
  N=80  pnl +0.172225, fills 554, pos_days 6/8, worst -0.006725
  N=120 pnl +0.220900, fills 757, pos_days 6/8, worst -0.026450
  N=160 pnl +0.166450, fills 931, pos_days 5/8, worst -0.085700

top-N by placement_side_edge_if_filled_margin:
  N=40  pnl +0.120825, fills 291, pos_days 7/8, worst -0.010100
  N=80  pnl +0.158725, fills 554, pos_days 5/8, worst -0.019975
  N=120 pnl +0.172950, fills 757, pos_days 5/8, worst -0.040900
  N=160 pnl +0.188825, fills 931, pos_days 5/8, worst -0.076275

top-N by placement_side_fill_prob:
  N=40  pnl +0.068450, fills 291, pos_days 7/8, worst -0.007825
  N=80  pnl +0.093775, fills 554, pos_days 6/8, worst -0.024500
  N=120 pnl +0.193325, fills 757, pos_days 5/8, worst -0.032550
  N=160 pnl +0.188200, fills 931, pos_days 5/8, worst -0.074825
```

placement attribution 结论:

```text
1. 下单时 expected_edge / edge_if_filled 仍然有排序能力。
   低 placement score 桶亏，高 placement score 桶赚钱。

2. placement-score attribution 比 fill-time attribution 更保守。
   这说明成交前 score 会漂移，不能只用 fill-time score 做门控。

3. placement top-N 离线结果仍显示 N=80/120 有潜力降低 worst day，
   但上一次真实 percentile / budget 回测没有兑现。
   原因大概率是库存路径和成交路径被过滤后改变。

4. 下一步如果继续优化，应直接做 placement-score gate 的真实回放:
   例如在下单/改单时要求 placement expected_margin / if_margin 达到更高阈值，
   而不是事后从 fills 里 top-N。

5. 动态报价也应使用 placement score + regime 条件，不应直接按 fill-time score 解释。
```

## 2026-05-25 高频因子扩展

背景:

```text
当前最好组合已经月度为正，但收益偏低。
判断之一是原始 9 个高频因子信息量偏少，尤其缺少多档盘口、盘口挂撤变化、OFI、成交强度。
```

已扩展训练/研究侧:

```text
scripts/factor_research/factors.py
scripts/factor_research/bitmex_factor_research.py
```

原始采样列从 9 列扩展到 17 列:

```text
ts, bid, ask, bid_qty, ask_qty,
buy_qty, sell_qty, buy_count, sell_count,
bid_qty_l2, ask_qty_l2,
bid_qty_l3, ask_qty_l3,
bid_qty_l4, ask_qty_l4,
bid_qty_l5, ask_qty_l5
```

因子从 9 个扩展到 22 个:

```text
原有:
  spread_bps
  queue_imbalance
  microprice_bps
  trade_flow_imbalance
  trade_flow_ewm_imbalance
  momentum_100ms_bps
  momentum_250ms_bps
  momentum_1000ms_bps
  vol_1000ms_bps

新增:
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

已扩展策略侧实时预测:

```text
scripts/bitmex_edge_scored_maker.py
```

关键实现:

```text
1. FACTOR_IDS 与训练侧 FACTOR_NAMES 保持 append-only 一致。
2. 实时维护 bid/ask qty、trade qty/count、OFI 的 ring buffer。
3. predict_scores 不再硬编码 9 个因子，而是按 model["mean"] 长度动态计算。
4. load_edge_model 按因子数动态推导 expanded feature length。
5. 因此旧 9 因子模型仍可加载，新 22 因子模型也可加载。
```

验证:

```bash
.venv/bin/python -m py_compile \
  scripts/factor_research/factors.py \
  scripts/factor_research/bitmex_factor_research.py \
  scripts/bitmex_edge_score_train_predict.py \
  scripts/bitmex_edge_scored_maker.py
```

通过。

新因子训练 smoke:

```bash
PYTHONPATH=py-hftbacktest:scripts .venv/bin/python scripts/bitmex_edge_score_train_predict.py \
  --skip-download \
  --train-dates 20260517 \
  --test-dates 20260518 \
  --sample-interval-ms 100 \
  --horizon-ms 250 \
  --maker-fee-rate 0.0 \
  --max-samples-per-day 20000 \
  --max-train-samples 10000 \
  --score-buckets 5 \
  --no-interactions \
  --result-tag smoke_new_hf_factors
```

输出:

```text
results/factor_research/bitmex_xbtusdt_smoke_new_hf_factors.edge_model.json
results/factor_research/bitmex_xbtusdt_smoke_new_hf_factors.edge_score_summary.csv
results/factor_research/bitmex_xbtusdt_smoke_new_hf_factors.edge_score_buckets.csv
```

新 22 因子模型策略 smoke:

```text
20260518:
  PnL +0.033750
  fills 98
  max position 200
```

旧 9 因子模型兼容性 smoke:

```text
20260518:
  脚本可正常加载旧 rolling edge_model.json 并完成回测。
```

注意:

```text
22 个因子如果继续启用二阶项 + 两两交互，会产生 275 个 expanded features。
全量 2M 样本训练内存会明显高于原 9 因子版本。
下一轮正式 walk-forward 建议先用:
  --no-interactions
或降低:
  --max-train-samples
确认新因子是否提升 edge_corr / top-bottom spread 后，再决定是否恢复交互项。
```

### 新因子 7 天 walk-forward probe

命令:

```bash
PYTHONPATH=py-hftbacktest:scripts .venv/bin/python scripts/bitmex_edge_score_walk_forward.py \
  --skip-download \
  --test-start 20260512 \
  --test-end 20260518 \
  --train-days 14 \
  --sample-interval-ms 100 \
  --horizon-ms 250 \
  --maker-fee-rate 0.0 \
  --max-train-samples 500000 \
  --score-buckets 10 \
  --no-interactions \
  --result-tag edge_score_wf_train14_20260512_20260518_h250_new_hf_nointer
```

输出:

```text
results/factor_research/bitmex_xbtusdt_edge_score_wf_train14_20260512_20260518_h250_new_hf_nointer.edge_score_walk_forward_daily.csv
results/factor_research/bitmex_xbtusdt_edge_score_wf_train14_20260512_20260518_h250_new_hf_nointer.edge_score_walk_forward_bucket_summary.csv
```

对比旧 9 因子 with_if_filled 7 天 rolling OOS:

```text
old 9 factors, with interactions:
  bid:
    edge_corr avg              0.053483
    edge_if_filled_corr avg    0.187931
    fill_corr avg              0.153761
    top-bottom expected edge   +0.065847 bps, 7/7 days positive
    top-bottom edge_if_filled  +1.024559 bps, 7/7 days positive
    top-bottom fill_prob       +0.008099

  ask:
    edge_corr avg              0.066838
    edge_if_filled_corr avg    0.217447
    fill_corr avg              0.144892
    top-bottom expected edge   +0.078130 bps, 7/7 days positive
    top-bottom edge_if_filled  +1.159611 bps, 7/7 days positive
    top-bottom fill_prob       +0.020600

new 22 factors, no interactions, max_train_samples=500k:
  bid:
    edge_corr avg              0.055845
    edge_if_filled_corr avg    0.207945
    fill_corr avg              0.172059
    top-bottom expected edge   +0.071253 bps, 7/7 days positive
    top-bottom edge_if_filled  +1.154374 bps, 7/7 days positive
    top-bottom fill_prob       +0.013891

  ask:
    edge_corr avg              0.070940
    edge_if_filled_corr avg    0.231833
    fill_corr avg              0.162775
    top-bottom expected edge   +0.085551 bps, 7/7 days positive
    top-bottom edge_if_filled  +1.350452 bps, 7/7 days positive
    top-bottom fill_prob       +0.021931
```

结论:

```text
新 22 因子 no-interactions 相比旧 9 因子 with-interactions 仍有小幅提升:
  bid expected-edge top-bottom spread:  +8.2%
  ask expected-edge top-bottom spread:  +9.5%
  bid edge_if_filled top-bottom spread: +12.7%
  ask edge_if_filled top-bottom spread: +16.5%

edge_corr 提升不大，但 edge_if_filled_corr、fill_corr、top-bottom spread 都一致改善。
这说明新增 OFI / 多档盘口 / 成交强度因子有增量信息，但不是数量级提升。

下一步应把新 22 因子模型接入当前最好策略组合做 7 天交易层对比:
  pct95 expected + pct95 if_filled
  regime threshold 0.00, alpha 0.00005
  hard risk q100/max300/soft100/loss0.10/fill300

若 7 天 PnL/fill 或 daily PnL 有改善，再考虑完整 20260418-20260518 月度复跑。
```

### 新 22 因子交易层 7 天回测

命令:

```bash
PYTHONPATH=py-hftbacktest:scripts .venv/bin/python scripts/bitmex_edge_scored_maker.py \
  --dates 20260512 20260513 20260514 20260515 20260516 20260517 20260518 \
  --skip-download \
  --model-dir results/factor_research \
  --model-tag edge_score_wf_train14_20260512_20260518_h250_new_hf_nointer \
  --expected-edge-threshold-bps 0.10 \
  --edge-if-filled-threshold-bps 0.05 \
  --fill-prob-threshold 0.05 \
  --intraday-percentile-gate \
  --expected-edge-percentile 0.95 \
  --edge-if-filled-percentile 0.95 \
  --regime-expected-edge-gate \
  --regime-expected-edge-warmup-samples 6000 \
  --regime-expected-edge-ewm-alpha 0.00005 \
  --regime-min-bid-expected-edge-bps 0.00 \
  --regime-min-ask-expected-edge-bps 0.00 \
  --order-qty-contracts 100 \
  --max-position-contracts 300 \
  --soft-position-contracts 100 \
  --reduce-only-after-soft-position \
  --daily-loss-limit-usdt 0.10 \
  --daily-fill-limit 300 \
  --maker-fee-rate 0.0 \
  --taker-fee-rate 0.0001 \
  --exchange-model no_partial \
  --result-tag edge_scored_new22_nointer_bestparams_20260512_20260518
```

输出:

```text
results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_nointer_bestparams_20260512_20260518.aggregate.csv
```

同区间对比旧最好组合:

```text
old best 9f p000/a00005, 20260512-20260518:
  total PnL +0.496950
  fills 1490
  PnL/fill +0.00033352
  positive days 6/7
  worst day 20260518 -0.018150
  best day 20260514 +0.126700
  max position max 100
  avg capture 5.3111 USDT/BTC

new 22f no-interactions, same trading params:
  total PnL +0.465800
  fills 1396
  PnL/fill +0.00033367
  positive days 6/7
  worst day 20260512 -0.082200
  best day 20260514 +0.180950
  max position max 100
  avg capture 4.9964 USDT/BTC
```

逐日:

```text
old:
  20260512 +0.034150 fills 128
  20260513 +0.095500 fills 180
  20260514 +0.126700 fills 296
  20260515 +0.105150 fills 268
  20260516 +0.091350 fills 136
  20260517 +0.062250 fills 182
  20260518 -0.018150 fills 300

new:
  20260512 -0.082200 fills 98
  20260513 +0.054200 fills 160
  20260514 +0.180950 fills 300
  20260515 +0.119700 fills 256
  20260516 +0.059150 fills 134
  20260517 +0.098550 fills 166
  20260518 +0.035450 fills 282
```

交易层结论:

```text
新 22 因子提升了研究层排序能力，但直接接入当前最好策略参数后，
7 天交易层没有提高总 PnL:
  +0.496950 -> +0.465800

不过 PnL/fill 基本相同:
  old +0.00033352
  new +0.00033367

说明新因子没有降低单笔质量，但当前 pct95/regime 参数不是为新模型重新校准的。
主要问题是 20260512 单日从盈利变成 -0.0822，吃掉了 20260514/18 的改善。

下一步如果继续新因子方向，应先对新模型单独重扫:
  expected_edge_percentile: 0.93, 0.95, 0.97
  edge_if_filled_percentile: 0.93, 0.95, 0.97
  regime_min: -0.01, 0.00, 0.01
  regime_alpha: 0.00005, 0.0001

不要直接拿旧 9 因子的最优 gate 当新 22 因子的最优 gate。
```

### 新 22 因子 gate 小网格

先用代表日做小网格:

```text
dates:
  20260512 20260514 20260518

固定:
  new 22 factors no-interactions
  regime_alpha 0.00005
  order_qty 100
  max_position 300
  soft_position 100
  reduce-only-after-soft-position
  daily_loss_limit 0.10
  daily_fill_limit 300

sweep:
  expected_edge_percentile: 0.95, 0.97
  edge_if_filled_percentile: 0.95, 0.97
  regime_min_expected_edge: -0.01, 0.00, 0.01
```

3 日 probe 排序:

```text
01 e095_i095_m001_a00005 total=+0.199400 fills=648 ppf=0.00030772 d12=-0.060650 d14=+0.224200 d18=+0.035850
02 e095_i097_m001_a00005 total=+0.195850 fills=594 ppf=0.00032971 d12=-0.054200 d14=+0.216500 d18=+0.033550
03 e097_i095_m001_a00005 total=+0.167850 fills=622 ppf=0.00026986 d12=-0.054100 d14=+0.210700 d18=+0.011250
04 e095_i097_mneg001_a00005 total=+0.157400 fills=654 ppf=0.00024067 d12=-0.078950 d14=+0.205550 d18=+0.030800
05 e095_i097_m000_a00005 total=+0.148100 fills=644 ppf=0.00022997 d12=-0.081550 d14=+0.202500 d18=+0.027150
06 e097_i097_m001_a00005 total=+0.140950 fills=574 ppf=0.00024556 d12=-0.055050 d14=+0.164750 d18=+0.031250
07 e095_i095_mneg001_a00005 total=+0.136250 fills=696 ppf=0.00019576 d12=-0.080650 d14=+0.182950 d18=+0.033950
08 e095_i095_m000_a00005 total=+0.134200 fills=680 ppf=0.00019735 d12=-0.082200 d14=+0.180950 d18=+0.035450
09 e097_i095_mneg001_a00005 total=+0.126400 fills=684 ppf=0.00018480 d12=-0.077350 d14=+0.197550 d18=+0.006200
10 e097_i095_m000_a00005 total=+0.123450 fills=672 ppf=0.00018371 d12=-0.079950 d14=+0.195500 d18=+0.007900
11 e097_i097_mneg001_a00005 total=+0.112100 fills=632 ppf=0.00017737 d12=-0.074650 d14=+0.159550 d18=+0.027200
12 e097_i097_m000_a00005 total=+0.107900 fills=626 ppf=0.00017236 d12=-0.073200 d14=+0.157550 d18=+0.023550
```

probe 结论:

```text
1. regime_min=0.01 明显优于 -0.01 / 0.00。
   说明新 22 因子模型需要更严格的日内 regime gate。

2. expected_edge_percentile=0.97 不好。
   它减少交易，但没有提高总收益。

3. edge_if_filled_percentile=0.97 在代表日上提高 PnL/fill，
   但总 PnL 与 0.95 很接近。
```

随后完整 7 天复跑前两名:

```text
old 9f best, 20260512-20260518:
  total PnL +0.496950
  fills 1490
  PnL/fill +0.00033352
  positive days 6/7
  worst 20260518 -0.018150
  best 20260514 +0.126700
  avg capture 5.3111 USDT/BTC

new 22f, e95/i95/regime_min=0.01:
  aggregate:
    results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_e95_i95_m001_a00005_20260512_20260518.aggregate.csv
  total PnL +0.501650
  fills 1320
  PnL/fill +0.00038004
  positive days 6/7
  worst 20260512 -0.060650
  best 20260514 +0.224200
  avg capture 5.4131 USDT/BTC

new 22f, e95/i97/regime_min=0.01:
  aggregate:
    results/bitmex_xbtusdt_edge_scored_maker_edge_scored_new22_e95_i97_m001_a00005_20260512_20260518.aggregate.csv
  total PnL +0.559350
  fills 1220
  PnL/fill +0.00045848
  positive days 6/7
  worst 20260512 -0.054200
  best 20260514 +0.216500
  avg capture 5.8889 USDT/BTC
```

7 天结论:

```text
新 22 因子如果重新校准 gate，可以超过旧 9 因子 7 天交易层结果。

当前新 22 因子 7 天最佳候选:
  expected_edge_percentile 0.95
  edge_if_filled_percentile 0.97
  regime_min_expected_edge 0.01
  regime_alpha 0.00005

它相对旧 9f best:
  total PnL:  +0.496950 -> +0.559350
  PnL/fill:   +0.00033352 -> +0.00045848
  fills:      1490 -> 1220
  worst day:  -0.018150 -> -0.054200

也就是说:
  每笔质量明显更好，但交易数更少，且 20260512 仍有更大的单日亏损。

下一步不应直接宣布新模型更优，而应跑完整 20260418-20260518 月度:
  new22 e95/i97/regime_min=0.01
并重点看 4 月段是否重新变差。
```

## 恢复上下文时优先看的文件

```text
doc/edge_scoring_progress_backup.md
scripts/factor_research/edge_scoring.py
scripts/bitmex_edge_score_train_predict.py
scripts/bitmex_edge_score_walk_forward.py
scripts/bitmex_edge_scored_maker.py
scripts/factor_research/README.md
```

## 当前 git 状态备注

截至 2026-05-24 同步本文档前，`git status --short` 显示:

```text
 M doc/edge_scoring_progress_backup.md
?? scripts/bitmex_edge_scored_maker.py
```

`scripts/bitmex_edge_scored_maker.py` 当前还是未跟踪文件，提交前需要确认是否纳入版本控制。
