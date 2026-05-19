# 单市场做市策略说明

本文档说明当前 `hftbacktest` 中 4 个 BitMEX `XBTUSDT` 单市场做市策略的原理、实现细节、核心参数和回测表现。

## 实现位置

- `advanced`: `scripts/bitmex_single_market_mm_backtest.py`
- `fixed_spread`: `scripts/bitmex_single_market_fixed_spread_mm_backtest.py`
- `avellaneda_stoikov`: `scripts/bitmex_single_market_as_mm_backtest.py`
- `ladder_grid`: `scripts/bitmex_single_market_ladder_mm_backtest.py`
- 三个 variant 共用主体: `scripts/bitmex_single_market_mm_variants.py`

## 共同回测假设

当前 4 个策略都是单市场 maker 策略，目标是在 BitMEX `XBTUSDT` 上被动挂限价单成交。

共同假设：

- 合约线性计价: `1 contract = 0.000001 BTC`
- 默认单笔数量: `100 contracts`
- tick size: `0.5`
- lot size: `100`
- queue model: `risk_adverse_queue_model`
- exchange model: `no_partial_fill_exchange`
- entry latency: `80 ms`
- response latency: `40 ms`
- REST 最小间隔: `350 ms`
- quote 更新周期: `10 ms`
- 日终会强制平仓，确保最终仓位归零

手续费口径：

- 无手续费版本: `maker_fee_rate = 0.0`, `taker_fee_rate = 0.0`
- rebate 版本: `maker_fee_rate = -0.0002`, `taker_fee_rate = 0.0001`
- 这里 `-0.0002` 是费率，即 `-2 bps` maker rebate
- `total_fee_usdt` 为负数时表示 rebate 收入

## 共同风控指标

回测 summary 中常看的字段：

- `total_pnl_usdt`: 最终净 PnL
- `total_fee_usdt`: 手续费或 rebate，负数代表返佣收入
- `maker_fills`: maker 成交次数
- `toxic_fill_events`: 成交后短时间内价格向不利方向移动的次数
- `avg_spread_capture_usdt_per_btc`: 平均每 BTC 成交捕获的价差
- `max_position_contracts_seen`: 回测中出现过的最大绝对持仓
- `force_close_pnl_usdt`: 日终强平造成的 PnL

`toxic_fill_events` 不是交易所字段，而是回测里用来衡量 adverse selection 的诊断指标。它表示订单成交后，mid price 很快向不利方向移动，说明这笔 maker fill 很可能是被有信息流或冲击流打中。

## 1. fixed_spread

### 策略思想

`fixed_spread` 是最基础的做市 baseline。它不做复杂预测，只围绕 fair price 双边挂一档：

- bid 挂在 fair price 下方
- ask 挂在 fair price 上方
- spread 基本固定
- 库存接近上限时停止继续加仓方向

它的作用是作为对照组，判断当前市场在简单固定价差下是否有 maker edge。

### 价格计算

代码中 fair price 默认使用盘口 microprice：

```text
mid = (best_bid + best_ask) / 2
fair = (best_ask * best_bid_qty + best_bid * best_ask_qty) / (best_bid_qty + best_ask_qty)
```

基础 half spread：

```text
half_spread_bps = max(BASE_HALF_SPREAD_BPS, MIN_HALF_SPREAD_TICKS * tick_size / mid * 10000)
```

默认：

```text
BASE_HALF_SPREAD_BPS = 3.0
MIN_HALF_SPREAD_TICKS = 1.0
```

最终挂单价格：

```text
bid = floor_to_tick(min(fair * (1 - half_spread_bps / 10000), best_bid))
ask = ceil_to_tick(max(fair * (1 + half_spread_bps / 10000), best_ask))
```

### 挂撤单逻辑

每边只挂 1 个订单：

- bid order id: `10001`
- ask order id: `20001`

触发以下情况会撤单或改价：

- quote TTL 到期
- 当前目标价格变化
- 当前目标数量变化
- 持仓接近硬上限，不允许继续加仓
- REST pacing 不允许时跳过本次操作

### 库存控制

`fixed_spread` 没有强库存 skew，只做硬限制：

```text
如果 pos + order_qty > MAX_POSITION_CONTRACTS，不再挂 bid
如果 pos - order_qty < -MAX_POSITION_CONTRACTS，不再挂 ask
```

默认：

```text
SOFT_POSITION_CONTRACTS = 500
MAX_POSITION_CONTRACTS = 1000
```

### 优点

- 逻辑简单，容易作为基准
- 回测解释性强
- 手续费敏感度容易看出来

### 风险

- 遇到趋势行情容易被连续打中
- 没有主动避开 toxic flow
- 经常可能打到最大持仓
- 如果没有 maker rebate，长期不一定有正收益

## 2. Avellaneda-Stoikov

### 策略思想

`avellaneda_stoikov` 是基于 Avellaneda-Stoikov 思路的库存风险做市版本。它不是简单围绕 mid 固定挂单，而是根据库存、风险厌恶、波动率和时间 horizon 调整 reservation price 和 spread。

核心目标：

- 库存多时，降低 reservation price，鼓励卖出、减少买入
- 库存空时，提高 reservation price，鼓励买回、减少卖出
- 波动越大，spread 越宽
- 库存越接近 soft limit，spread 越宽

### 价格计算

库存比例：

```text
inv = clamp(position / SOFT_POSITION_CONTRACTS, -1, 1)
```

波动率使用最近窗口内 mid price 变动的绝对值：

```text
vol_bps = abs(current_mid / past_mid - 1) * 10000
```

reservation price 近似：

```text
variance_bps2 = vol_bps * vol_bps
horizon_seconds = AS_HORIZON_NS / 1e9
inventory_shift_bps = inv * AS_RISK_AVERSION * variance_bps2 * horizon_seconds / 100
anchor = fair * (1 - inventory_shift_bps / 10000)
```

half spread：

```text
inventory_spread_bps = abs(inv) * AS_INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT
half_spread_bps = max(
    BASE_HALF_SPREAD_BPS
    + AS_LIQUIDITY_SPREAD_BPS
    + VOL_SPREAD_MULTIPLIER * vol_bps
    + inventory_spread_bps,
    min_half_spread_bps
)
```

默认参数：

```text
AS_RISK_AVERSION = 0.20
AS_HORIZON_NS = 2_000_000_000
AS_LIQUIDITY_SPREAD_BPS = 1.0
AS_INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT = 1.5
VOL_SPREAD_MULTIPLIER = 0.5
```

最终价格：

```text
bid = floor_to_tick(min(anchor * (1 - half_spread_bps / 10000), best_bid))
ask = ceil_to_tick(max(anchor * (1 + half_spread_bps / 10000), best_ask))
```

### 挂撤单逻辑

每边只挂 1 个订单，和 `fixed_spread` 相同：

- TTL 到期撤单
- 目标价格或数量变化时改单
- 加仓方向超过 `MAX_POSITION_CONTRACTS` 时停止挂单
- REST pacing 不允许时跳过

### 优点

- 有理论上的库存风险调整
- 成交量比 fixed / advanced / ladder 少很多
- 手续费暴露更小
- 比 fixed spread 更适合库存敏感场景

### 风险

- 当前实现仍是近似版本，不是完整闭式最优解
- 参数非常敏感，尤其 `risk_aversion` 和 horizon
- 成交少，单日 PnL 波动仍可能很大
- 回测中也会触及 `1000 contracts` 最大持仓

## 3. advanced

### 策略思想

`advanced` 是原始单市场做市策略。它比 `fixed_spread` 多了几个实用过滤器：

- microprice 作为 fair price
- 库存 skew
- 短周期 momentum toxic cancel
- microprice toxic cancel
- 波动率加宽 spread
- TTL 撤单

它的目标不是只吃 rebate，而是尽量避开明显不利的盘口状态。

### fair price 与库存 skew

先计算 microprice：

```text
fair = (best_ask * best_bid_qty + best_bid * best_ask_qty) / (best_bid_qty + best_ask_qty)
```

库存比例：

```text
inv_ratio = clamp(position / SOFT_POSITION_CONTRACTS, -1, 1)
```

reservation price：

```text
reservation = fair * (1 - inv_ratio * INVENTORY_SKEW_BPS_AT_SOFT_LIMIT / 10000)
```

默认：

```text
INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = 4.0
```

含义：

- 当前多头库存越大，reservation 越低，bid/ask 都整体下移，鼓励卖出、不鼓励继续买入
- 当前空头库存越大，reservation 越高，bid/ask 都整体上移，鼓励买回、不鼓励继续卖出

### spread 计算

基础 half spread：

```text
half_spread_bps = BASE_HALF_SPREAD_BPS + volatility_penalty_bps
```

波动率惩罚：

```text
volatility_penalty_bps = abs(recent_mid_move_bps) * VOL_SPREAD_MULTIPLIER
```

默认：

```text
BASE_HALF_SPREAD_BPS = 3.0
VOL_WINDOW_NS = 1_000_000_000
VOL_SPREAD_MULTIPLIER = 0.5
```

最终价格：

```text
bid = floor_to_tick(min(reservation * (1 - half_spread_bps / 10000), best_bid))
ask = ceil_to_tick(max(reservation * (1 + half_spread_bps / 10000), best_ask))
```

### toxic cancel 逻辑

`advanced` 会在挂单前检查当前一侧是否 toxic。

短周期 momentum：

```text
SHORT_MOMENTUM_WINDOW_NS = 100_000_000
MOMENTUM_CANCEL_BPS = 0.8
```

如果最近 `100 ms` mid price 快速下跌，则 bid 侧更危险；如果快速上涨，则 ask 侧更危险。

microprice 偏离：

```text
MICROPRICE_CANCEL_BPS = 0.3
```

如果 microprice 明显偏向下方，说明卖压更强，bid 侧容易被打；如果 microprice 明显偏向上方，ask 侧容易被打。

触发 toxic signal 时：

- 如果已有该侧订单，尝试撤单
- 如果没有订单，则 suppress 该侧新挂单
- 如果 REST pacing 不允许，则本轮跳过，可能导致订单继续暴露

### 库存控制

`advanced` 同时使用 skew 和硬限制：

```text
如果 pos + order_qty > MAX_POSITION_CONTRACTS，不再挂 bid
如果 pos - order_qty < -MAX_POSITION_CONTRACTS，不再挂 ask
```

它不会因为达到 soft limit 就完全停止加仓，而是先通过 skew 调整价格。

### 优点

- 比 fixed spread 有更多不利行情过滤
- 加入 maker rebate 后月度表现稳定
- 回测中最大持仓低于 fixed / AS / ladder
- 适合作为后续实盘保守版本的基础

### 风险

- toxic fill 统计仍然很高，说明很多成交后仍会短期不利移动
- 撤单依赖 REST pacing，实盘可能撤不掉
- 频繁判断和撤单会带来订单管理压力
- 如果没有 maker rebate，当前参数月度仍为负

## 4. ladder_grid

### 策略思想

`ladder_grid` 是多层网格做市。它不是每边只挂一档，而是在 bid/ask 两侧各挂多层订单：

```text
bid levels: b1, b2, b3, ...
ask levels: s1, s2, s3, ...
```

越靠近 fair price 的订单越容易成交，越远的订单承担补库存和吃 rebate 的角色。

当前默认：

```text
LADDER_LEVELS = 3
LADDER_SPACING_BPS = 2.0
LADDER_MIN_SPACING_TICKS = 1.0
LADDER_INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = 4.0
```

### anchor 价格

先计算 fair price，再按库存做整体偏移：

```text
anchor = fair * (1 - inv * LADDER_INVENTORY_SKEW_BPS_AT_SOFT_LIMIT / 10000)
```

含义和 advanced 类似：

- 多头库存越大，整体报价下移
- 空头库存越大，整体报价上移

### 网格间距

每层间距：

```text
spacing_bps = max(
    LADDER_SPACING_BPS + VOL_SPREAD_MULTIPLIER * vol_bps,
    LADDER_MIN_SPACING_TICKS * tick_size / mid * 10000
)
```

第 `n` 层 half spread：

```text
half_spread_bps = spacing_bps * (level_idx + 1)
```

所以默认 3 层大致是：

```text
第 1 层: anchor +/- 1 * spacing
第 2 层: anchor +/- 2 * spacing
第 3 层: anchor +/- 3 * spacing
```

最终价格：

```text
bid_n = floor_to_tick(min(anchor * (1 - half_spread_bps / 10000), best_bid))
ask_n = ceil_to_tick(max(anchor * (1 + half_spread_bps / 10000), best_ask))
```

### 层数和库存控制

`ladder_grid` 会根据库存动态减少加仓方向的层数：

```text
如果当前多头库存较大，减少 bid 层数
如果当前空头库存较大，减少 ask 层数
```

但当前实现里，加仓方向最少仍会保留 1 层，除非触发硬上限：

```text
如果 pos + order_qty * projected_levels > MAX_POSITION_CONTRACTS，不挂该 bid level
如果 pos - order_qty * projected_levels < -MAX_POSITION_CONTRACTS，不挂该 ask level
```

默认 `LADDER_LEVELS = 3` 时，正常情况下每边最多 3 个活跃订单。两边合计最多 6 个订单。若实盘同一侧看到超过 3 个活跃订单，一般不是策略意图，而应检查订单状态同步、撤单回报和本地订单去重。

### 挂撤单逻辑

每一层都有固定 order id：

```text
bid: 10001, 10002, 10003, ...
ask: 20001, 20002, 20003, ...
```

每层独立管理：

- TTL 到期撤单
- 目标价格变化则 modify
- 库存限制触发则撤单或停止挂单
- REST pacing 不允许时跳过

### 为什么它回测收益高

加入 `maker_fee_rate = -0.0002` 后，`ladder_grid` 月度收益最高，核心原因是成交量最大：

```text
成交 base: 4.8632 BTC
maker fills: 48650
net rebate: -76.20611 USDT
月度 PnL: +62.14096 USDT
```

它本质上更像一个高换手 rebate capture 策略。价差交易本身并不一定强，主要收益来自大量 maker 成交带来的 rebate。

### 优点

- 高频成交，能最大化 maker rebate 暴露
- 多层订单适合震荡行情
- 库存通过多层成交自然来回换手
- 在当前 fee 口径下月度所有天数为正

### 风险

- 极度依赖 maker rebate
- 成交量很大，实盘订单限制和撤改单限制压力最大
- 如果真实成交率低于回测，收益会显著下降
- 趋势行情可能积累库存并被打穿
- 如果某些成交被判为 taker 或失去 maker rebate，收益会快速恶化
- 网页上看到同侧超过配置层数时，需要优先排查订单状态同步问题

## 月度回测对比

回测区间：

```text
20260418 - 20260518
XBTUSDT
QUOTE_TTL_MS = 2000
```

### 无手续费 / 无 rebate

| 策略 | 总 PnL | 胜/负天 | fills | toxic | 成交 BTC | max pos |
|---|---:|---:|---:|---:|---:|---:|
| fixed_spread | -2.38940 | 16/15 | 12244 | 1088 | 1.2236 | 1000 |
| avellaneda_stoikov | -2.28505 | 16/15 | 1765 | 225 | 0.1765 | 1000 |
| advanced | -3.71385 | 8/23 | 13436 | 8657 | 1.3436 | 900 |
| ladder_grid | -14.06515 | 4/27 | 48650 | 4139 | 4.8632 | 1000 |

结论：无 rebate 时，4 个策略当前参数都不能月度赚钱。

### maker rebate -2bps / taker fee 1bp

| 策略 | 总 PnL | net fee/rebate | 胜/负天 | fills | toxic | 成交 BTC | max pos |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_spread | +16.64363 | -19.03303 | 24/7 | 12244 | 1088 | 1.2236 | 1000 |
| avellaneda_stoikov | +0.32322 | -2.60827 | 16/15 | 1765 | 225 | 0.1765 | 1000 |
| advanced | +17.33177 | -21.04562 | 30/1 | 13436 | 8657 | 1.3436 | 900 |
| ladder_grid | +62.14096 | -76.20611 | 31/0 | 48650 | 4139 | 4.8632 | 1000 |

结论：加入 `-2bps` maker rebate 后，结果显著转正。收益排序为：

```text
ladder_grid > advanced > fixed_spread > avellaneda_stoikov
```

但这也说明策略收益高度依赖 maker rebate，而不是纯 spread capture。

## 实盘选择建议

### 如果目标是保守实盘观察

优先考虑：

```text
advanced
fixed_spread
```

理由：

- 月度收益为正
- 成交量中等
- `advanced` 最大持仓相对低一些
- fixed spread 更容易解释和排查

### 如果目标是最大化 rebate

可以考虑：

```text
ladder_grid
```

但需要严格监控：

- 同侧活跃订单数量
- REST 限速和撤改单失败
- maker/taker 标记是否真实符合预期
- 单日成交量是否明显偏离回测
- 持仓是否频繁接近 `MAX_POSITION_CONTRACTS`

### 不建议直接依赖 AS 当前版本

`avellaneda_stoikov` 成交少，月度只是小幅转正，且单日波动较大。它适合继续作为库存模型研究方向，但当前参数不是最好的实盘候选。

## 后续优化方向

优先级建议：

1. 给 `advanced` 和 `ladder_grid` 加更强的 one-sided quote 逻辑。
2. 降低 `MAX_POSITION_CONTRACTS`，例如从 `1000` 扫到 `500` 或 `700`。
3. 对 `QUOTE_TTL_MS` 做 walk-forward 参数扫描，不只看单月最优。
4. 实盘记录每笔成交是否真的是 maker，并和回测 maker fill 数量对齐。
5. 对 `ladder_grid` 增加最大活跃订单数断言，防止交易所状态同步异常导致重复挂单。
6. 单独分析扣除 rebate 后的 price PnL，避免误把 rebate 策略理解成纯 alpha 策略。

## 一句话总结

这 4 个策略里，`fixed_spread` 是基准，`avellaneda_stoikov` 是库存理论模型，`advanced` 是带 toxic 过滤的实用单档做市，`ladder_grid` 是高换手多层 rebate capture。当前回测显示：没有 maker rebate 时它们都不赚钱；有 `-2bps` maker rebate 时，`advanced` 和 `ladder_grid` 表现最好，其中 `ladder_grid` 收益最高但实盘约束和订单管理风险也最大。
