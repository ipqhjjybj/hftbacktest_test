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
