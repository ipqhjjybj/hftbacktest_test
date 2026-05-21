# Ladder Grid 做市策略说明

本文说明当前 `ladder_grid` 单市场做市策略的设计逻辑、参数含义、挂单价格、下单量、库存控制、波动率影响、撤单/改单规则，以及回测和实盘代码之间的关键差异。

对应代码：

- 回测：`/Users/zhuoheng.shen/git/hftbacktest/scripts/bitmex_single_market_mm_variants.py`
- 回测入口：`/Users/zhuoheng.shen/git/hftbacktest/scripts/bitmex_single_market_ladder_mm_backtest.py`
- 实盘策略：`/Users/zhuoheng.shen/git/market_maker_single_market/src/strategy/ladder_grid.rs`
- 实盘执行层：`/Users/zhuoheng.shen/git/market_maker_single_market/src/main.rs`

## 1. 策略目标

`ladder_grid` 是一个多层被动 maker 策略。

它不是只在最优买卖价附近各挂一单，而是在买卖两侧同时挂多层订单：

```text
b1 / s1
b2 / s2
b3 / s3
...
```

每一层离参考价格更远。越内层越容易成交，越外层价格越保守。

策略主要想赚三类收益：

```text
1. maker rebate
   被动挂单成交后获得 maker 返佣。

2. spread capture
   买在 bid 附近，卖在 ask 附近，价格没有明显单边移动时赚价差。

3. 短周期 mean reversion
   价格短暂打到某一侧后回归，仓位通过反方向挂单出掉。
```

主要风险是：

```text
1. adverse selection
   买单成交后价格继续跌，卖单成交后价格继续涨。

2. inventory risk
   单边行情连续打穿多层订单，库存越来越偏。

3. partial fill / residual inventory
   实盘中订单可能只成交一部分，剩余撤单后留下非 ORDER_QTY 整数倍的仓位。

4. execution mismatch
   回测 no_partial 模型不会模拟所有实盘执行细节。
```

## 2. 核心参数

常用参数如下：

```text
ORDER_QTY
    每一层基础下单数量，单位是 contracts。

GRID_LEVELS
    ladder 层数。GRID_LEVELS=3 时，两侧最多 b1/s1/b2/s2/b3/s3。

GRID_SPACING_BPS
    每层之间的基础间距，单位 bps。

GRID_MIN_SPACING_TICKS
    每层最小间距，单位 tick。

BASE_HALF_SPREAD_BPS
    对 ladder_grid 回测版本来说，目前主要保留在公共参数中；
    ladder_grid 的真实层间距离主要由 GRID_SPACING_BPS 和波动率惩罚决定。

VOL_SPREAD_MULTIPLIER
    波动率对 grid spacing 的放大倍数。

GRID_INVENTORY_SKEW_BPS
    库存达到 soft limit 时，报价中心偏移多少 bps。

SOFT_POSITION_CONTRACTS
    软库存阈值。仓位接近这个值时，会明显减少加仓方向层数，并移动报价中心。

MAX_POSITION_CONTRACTS
    硬最大仓位。继续成交会超过该仓位时，对应方向不再挂单。

QUOTE_TTL_MS
    订单最大存活时间。超过后撤掉重新评估。

REST_MIN_INTERVAL_MS
    REST 下单、改单、撤单之间的最小间隔。

MIN_AMEND_TICKS
    实盘中目标价格变化小于该 tick 数时，不改单。
```

示例参数：

```text
ORDER_QTY=100
GRID_LEVELS=3
GRID_SPACING_BPS=2.0
GRID_MIN_SPACING_TICKS=1
GRID_INVENTORY_SKEW_BPS=4.0
SOFT_POSITION_CONTRACTS=500
MAX_POSITION_CONTRACTS=1000
QUOTE_TTL_MS=5000
REST_MIN_INTERVAL_MS=700
```

## 3. 合约数量口径

### XBTUSDT

当前 XBTUSDT 回测按线性合约处理：

```text
CONTRACT_SIZE_BTC = 0.000001
ORDER_QTY = 100 contracts

每层基础下单 BTC 数量：
100 * 0.000001 = 0.0001 BTC
```

如果 BTC 价格是 80,000 USDT，则一层名义价值大约：

```text
0.0001 * 80,000 = 8 USDT
```

### XBTUSD

XBTUSD 是 inverse 合约：

```text
1 contract = 1 USD 面值
BTC 数量 = contracts / price
```

如果 BTC 价格是 80,000 USDT：

```text
100 contracts = 100 / 80,000 = 0.00125 BTC
```

所以同样是 `ORDER_QTY=100`，XBTUSD 的 BTC 敞口会明显大于 XBTUSDT。

这也是为什么 XBTUSD 回测里的 `base_btc` 明显更大。

## 4. 参考价格 fair / anchor

策略不是直接用简单 mid 作为报价中心，而是优先使用 microprice。

mid：

```text
mid = (best_bid + best_ask) / 2
```

microprice：

```text
microprice = (best_ask * best_bid_qty + best_bid * best_ask_qty)
             / (best_bid_qty + best_ask_qty)
```

microprice 的含义是：

```text
bid size 越大，说明买盘更强，microprice 越靠近 ask。
ask size 越大，说明卖盘更强，microprice 越靠近 bid。
```

实盘 ladder_grid 使用：

```text
fair = microprice
```

然后根据库存偏移得到 anchor：

```text
inv_ratio = clamp(position / SOFT_POSITION_CONTRACTS, -1, 1)

anchor = microprice * (1 - inv_ratio * GRID_INVENTORY_SKEW_BPS / 10000)
```

如果当前多仓：

```text
position > 0
inv_ratio > 0
anchor 下移
buy 报价更远
sell 报价更近
```

如果当前空仓：

```text
position < 0
inv_ratio < 0
anchor 上移
buy 报价更近
sell 报价更远
```

这就是库存对挂单价格的第一层影响。

## 5. 挂单间距设计

ladder_grid 的核心是每一层距离 anchor 的间距。

当前逻辑：

```text
min_spacing_bps = GRID_MIN_SPACING_TICKS * tick_size / mid * 10000

spacing_bps = max(
    GRID_SPACING_BPS + volatility_penalty_bps,
    min_spacing_bps
)

level 1 half_bps = spacing_bps * 1
level 2 half_bps = spacing_bps * 2
level 3 half_bps = spacing_bps * 3
...
```

然后：

```text
bid_level_n = anchor * (1 - half_bps / 10000)
ask_level_n = anchor * (1 + half_bps / 10000)
```

最后做 tick rounding，并且保证不穿过当前盘口：

```text
bid = floor_to_tick(min(raw_bid, best_bid))
ask = ceil_to_tick(max(raw_ask, best_ask))
```

也就是说：

```text
buy 单不会高于 best_bid
sell 单不会低于 best_ask
```

示例：

```text
mid = 100000
anchor = 100000
GRID_SPACING_BPS = 2
volatility_penalty_bps = 0
GRID_LEVELS = 3
```

理论挂单大概是：

```text
b1 = 99980   距 anchor 2 bps
s1 = 100020  距 anchor 2 bps

b2 = 99960   距 anchor 4 bps
s2 = 100040  距 anchor 4 bps

b3 = 99940   距 anchor 6 bps
s3 = 100060  距 anchor 6 bps
```

如果波动率惩罚为 1 bps：

```text
spacing_bps = 2 + 1 = 3

b1/s1 距 3 bps
b2/s2 距 6 bps
b3/s3 距 9 bps
```

所以波动越大，所有层都会被整体拉开。

## 6. 波动率如何影响间距

实盘中 `volatility_penalty_bps` 来自最近一段时间的 mid price 变化。

代码逻辑接近：

```text
past_mid = 约 1 秒前的 mid
move_bps = abs((mid_now / past_mid - 1) * 10000)
volatility_penalty_bps = move_bps * VOL_SPREAD_MULTIPLIER
```

如果：

```text
VOL_SPREAD_MULTIPLIER=0.5
最近 1 秒 mid 变动 = 4 bps
```

则：

```text
volatility_penalty_bps = 4 * 0.5 = 2 bps
```

如果原始 `GRID_SPACING_BPS=2`：

```text
spacing_bps = 2 + 2 = 4 bps
```

三层距离变成：

```text
level 1: 4 bps
level 2: 8 bps
level 3: 12 bps
```

这能降低波动行情中被连续打穿的概率，但也会降低成交频率。

## 7. 库存如何影响挂单价格

库存首先通过 `anchor` 偏移影响价格。

公式：

```text
inv_ratio = clamp(position / SOFT_POSITION_CONTRACTS, -1, 1)
anchor = microprice * (1 - inv_ratio * GRID_INVENTORY_SKEW_BPS / 10000)
```

假设：

```text
SOFT_POSITION_CONTRACTS = 500
GRID_INVENTORY_SKEW_BPS = 4
position = +250
```

则：

```text
inv_ratio = 250 / 500 = 0.5
anchor 下移 = 0.5 * 4 = 2 bps
```

结果：

```text
buy 报价整体更低，不容易继续加多仓
sell 报价整体更低，更容易卖出减仓
```

如果：

```text
position = -250
```

则：

```text
anchor 上移 2 bps
buy 更容易成交，用来减空仓
sell 更远，不容易继续加空仓
```

这个设计的核心是：

```text
库存偏多时，让策略更愿意卖出、更不愿意继续买入。
库存偏空时，让策略更愿意买入、更不愿意继续卖出。
```

## 8. 库存如何影响挂单层数

库存还会减少“加仓方向”的 ladder 层数。

当前逻辑：

```text
if side is Buy and position > 0:
    reduce = ceil(inv_ratio * (levels - 1))
    active_buy_levels = max(1, levels - reduce)

if side is Sell and position < 0:
    reduce = ceil(abs(inv_ratio) * (levels - 1))
    active_sell_levels = max(1, levels - reduce)
```

注意：

```text
soft inventory 逻辑至少保留 1 层。
真正完全不挂加仓方向，要靠 MAX_POSITION 硬限制或实盘 residual qty 逻辑。
```

例子：

```text
GRID_LEVELS = 3
SOFT_POSITION_CONTRACTS = 500
position = +250
```

则：

```text
inv_ratio = 0.5
reduce = ceil(0.5 * (3 - 1)) = 1

buy active levels = 3 - 1 = 2
sell active levels = 3
```

实际变成：

```text
buy:  b1, b2
sell: s1, s2, s3
```

如果：

```text
position = +500
inv_ratio = 1
```

则：

```text
reduce = ceil(1 * 2) = 2
buy active levels = 1
```

实际变成：

```text
buy:  b1
sell: s1, s2, s3
```

这意味着库存越偏，策略越少在同方向加仓。

## 9. MAX_POSITION 硬限制

硬限制用来防止继续加仓超过最大仓位。

回测逻辑：

```text
projected_levels = level_idx + 1

Buy:
    if position + ORDER_QTY * projected_levels > MAX_POSITION:
        不挂该层

Sell:
    if position - ORDER_QTY * projected_levels < -MAX_POSITION:
        不挂该层
```

例子：

```text
ORDER_QTY = 100
MAX_POSITION = 1000
position = +900
```

则：

```text
b1 成交后 position = +1000，可以挂
b2 成交后 position = +1100，不挂
b3 成交后 position = +1200，不挂
```

如果：

```text
position = +1000
```

则：

```text
所有 buy 层都不挂
sell 层仍然允许挂，用来减仓
```

## 10. 下单量设计

### 回测版

当前回测版 `ladder_grid` 每个可挂层使用固定数量：

```text
qty = ORDER_QTY
```

例如：

```text
ORDER_QTY=100
GRID_LEVELS=3
```

满状态下理论挂单为：

```text
b1 100
s1 100
b2 100
s2 100
b3 100
s3 100
```

单侧最大挂单量：

```text
100 * 3 = 300 contracts
```

双侧最大挂单量：

```text
100 * 3 * 2 = 600 contracts
```

但实际会受库存层数、MAX_POSITION、REST pacing、TTL、已有订单状态影响。

### 实盘版

实盘版最近加入了 residual inventory 减仓数量逻辑。

如果某一侧是加仓方向：

```text
qty = ORDER_QTY
```

如果某一侧是减仓方向：

```text
qty_level_n = min(ORDER_QTY, abs(position) - ORDER_QTY * level_idx)
```

如果计算结果小于等于 0，则该减仓层不挂。

例子：

```text
ORDER_QTY = 100
position = +70
```

sell 是减仓方向：

```text
s1 qty = 70
s2 qty = 0，不挂
s3 qty = 0，不挂
```

这样如果 s1 成交：

```text
position: +70 -> 0
```

不会因为挂 `sell 100` 导致：

```text
position: +70 -> -30
```

再看一个例子：

```text
ORDER_QTY = 100
position = +250
```

sell 减仓方向：

```text
s1 qty = 100
s2 qty = 100
s3 qty = 50
```

如果三层同时全部成交：

```text
position: +250 -> 0
```

不会反向开空。

这个逻辑主要是为了解决实盘 partial fill 后留下非整百仓位的问题。

## 11. Partial Fill 和残余仓位

实盘中一个 100 contracts 的订单可能只成交一部分：

```text
buy 100
成交 70
剩余 30 后续被撤掉
```

这时真实仓位是：

```text
position = +70
```

如果继续按固定 `sell 100` 减仓，成交后会变成：

```text
position = -30
```

所以实盘策略现在做了 residual reduce sizing：

```text
position = +70 时，只挂 sell 70
position = -30 时，只挂 buy 30
```

这让残余仓位更容易回到 0。

当前回测 `no_partial` 模型不会模拟 partial fill：

```text
订单要么 100 全成
要么 0 不成
```

因此：

```text
回测中 residual inventory 问题不明显。
实盘中 residual inventory 很重要。
```

如果后续要让回测更贴近实盘，应补充：

```text
1. partial fill
2. leavesQty 生命周期
3. amend 后不自动补满
4. residual reduce sizing
```

## 12. 撤单和改单逻辑

### TTL 撤单

订单超过 `QUOTE_TTL_MS` 后会撤掉。

目的：

```text
1. 防止旧价格长时间留在市场
2. 重新计算 anchor、spacing、库存状态
3. 避免行情状态变化后仍保留旧单
```

副作用：

```text
1. 频繁撤单会丢队列位置
2. 订单在市场上的有效时间下降
3. REST 请求增多
```

### 改单

回测版：

```text
existing.price != target_px 或 existing.qty != ORDER_QTY 时改单
```

实盘版：

```text
abs(existing.price - target_price) >= MIN_AMEND_TICKS * tick_size 时才改单
```

实盘这么做是为了少动：

```text
1. 减少 REST 请求
2. 保留队列位置
3. 降低被 rate limit 的概率
```

实盘还会检查剩余数量：

```text
如果现有 leaves_qty > target_leaves_qty，则缩小订单。
如果现有 leaves_qty < target_leaves_qty，则不主动补满。
```

这个设计是为了避免 partial fill 后无意中扩大暴露。

### REST pacing

策略不是每次 target 变化都立刻发 REST。

它受：

```text
REST_MIN_INTERVAL_MS
REST_RATE_LIMIT_WINDOW_MS
REST_RATE_LIMIT_MAX_REQUESTS
ORDER_INFLIGHT_MS
ORDER_ERROR_COOLDOWN_MS
```

这些限制影响。

如果 REST pacing 没通过：

```text
本次不下单/改单/撤单
等待后续 tick
```

这也是为什么回测和实盘日志里会看到大量 pacing skip。

## 13. 挂单顺序

实盘 `ladder_grid` 的 configured slots 顺序是：

```text
b1, s1, b2, s2, b3, s3, ...
```

并且执行层会做轮转调度，避免 REST 资源长期偏向某一侧。

目的：

```text
1. 买卖两侧更公平
2. 避免只更新 buy 或只更新 sell
3. 在 REST pacing 紧张时降低单侧偏置
```

## 14. 手续费和 PnL

策略非常依赖 maker fee。

当前常用设置：

```text
maker_fee = -0.0002
taker_fee = 0.0001
```

负 maker fee 表示返佣。

PnL 可以拆成：

```text
net_pnl = gross_pnl + maker_rebate - taker_fee_or_force_close_cost
```

其中：

```text
gross_pnl
    不考虑手续费/返佣的交易损益。

maker_rebate
    被动成交获得的返佣。

net_pnl
    最终总 PnL。
```

如果：

```text
gross_pnl < 0
net_pnl > 0
```

说明策略主要靠 rebate 覆盖 adverse selection。

如果：

```text
gross_pnl > 0
net_pnl > 0
```

说明即使不完全依赖 rebate，成交质量也相对更好。

## 15. 回测结果参考

### XBTUSDT 20260518

参数：

```text
symbol = XBTUSDT
asset_type = linear
date = 20260518
exchange_model = no_partial
levels = 3
ORDER_QTY = 100
QUOTE_TTL_MS = 5000
REST_MIN_INTERVAL_MS = 700
maker_fee = -0.0002
taker_fee = 0.0001
```

结果：

```text
Net PnL:        +3.1179 USDT
Gross PnL:      -1.2642 USDT
Maker rebate:   +4.3821 USDT
maker fills:     2852
buy fills:       1426
sell fills:      1426
成交量:          0.2852 BTC
最大仓位:        600 contracts / 0.0006 BTC
最终仓位:        0
toxic fills:     326
```

解读：

```text
XBTUSDT 这天是赚钱的，但 gross PnL 为负，主要靠 rebate 转正。
```

### XBTUSD 20260418-20260518

参数：

```text
symbol = XBTUSD
asset_type = inverse
exchange_model = no_partial
levels = 3
ORDER_QTY = 100
BASE_HALF_SPREAD_BPS = 3.0
GRID_SPACING_BPS = 2.0
QUOTE_TTL_MS = 5000
REST_MIN_INTERVAL_MS = 700
SOFT_POSITION = 500
MAX_POSITION = 1000
maker_fee = -0.0002
taker_fee = 0.0001
```

结果：

```text
天数: 31
赚钱天数: 31 / 31
总 PnL: +1931.84 USDT
gross PnL: +315.92 USDT
maker rebate: +1615.92 USDT
总 fills: 80810
日均 fills: 2606.77
总成交量: 103.3639 BTC
最大仓位: 500 contracts
最好一天: 20260504 +129.42 USDT
最差一天: 20260509 +10.44 USDT
```

解读：

```text
XBTUSD 这个月回测明显强于 XBTUSDT。
更关键的是，XBTUSD 月度 gross PnL 也是正的，不只是纯靠 rebate。
```

但需要注意：

```text
1. 这是 no_partial 模型。
2. no_partial 对真实排队和 partial fill 仍然偏乐观。
3. 实盘仍需重点对比 fills/day、avg fill qty、partial ratio、markout。
```

## 16. base_btc 的含义

`base_btc` 是累计成交的 BTC 数量，不是盈利，也不是最大持仓。

对 XBTUSD：

```text
base_btc = sum(abs(contracts / price))
```

对 XBTUSDT：

```text
base_btc = sum(abs(contracts * CONTRACT_SIZE_BTC))
```

例如：

```text
XBTUSD price = 80,000
100 contracts = 100 / 80,000 = 0.00125 BTC
```

如果一天里买卖累计很多笔，`base_btc` 会累计增加。

它表达的是交易活跃度：

```text
base_btc 越大，成交量越大。
max_position_base 表示最大库存。
final_position_base 表示最终库存。
```

## 17. 策略适合的行情

更适合：

```text
1. 盘口深度好
2. 有足够 maker 成交
3. 短期价格有来回波动
4. maker rebate 较高
5. adverse selection 不严重
```

不适合：

```text
1. 单边快速行情
2. 买单成交后持续下跌
3. 卖单成交后持续上涨
4. 盘口突然变薄
5. REST 延迟或撤单延迟过高
```

## 18. 关键监控指标

实盘运行时不要只看 PnL，需要同时看：

```text
fills/day
avg fill qty
partial fill ratio
order fill ratio
gross PnL
maker rebate
net PnL
max inventory
inventory holding time
markout 1s / 3s / 10s
post-only reject
REST pacing skip
cancel/amend frequency
```

尤其要关注：

```text
1. gross PnL 是否持续为正
2. 如果 gross PnL 为负，rebate 是否稳定覆盖
3. fill 后 1s/3s/10s markout 是否恶化
4. 实盘 fills 是否和 no_partial 回测同量级
5. 实盘 partial fill 是否导致大量残余仓位
```

## 19. 当前策略的主要不足

当前版本仍有一些明显不足：

```text
1. 回测 no_partial 不模拟 partial fill。
2. 回测版没有实盘 residual reduce sizing。
3. toxic signal 对 ladder_grid 还不够细。
4. 没有 inventory age 超时减仓逻辑。
5. 没有按时段/波动状态动态调整 levels。
6. 没有跨市场 hedge。
7. 回测和实盘的 REST、ack、cancel latency 仍需继续校准。
```

## 20. 后续优化方向

优先级建议：

```text
1. 回测补 partial fill / leavesQty / residual reduce sizing。
2. 给 ladder_grid 增加 inventory age。
3. toxic 时跳过第一层，而不是只靠撤单。
4. 动态 levels：波动高时减少层数，波动低时恢复层数。
5. 动态 spacing：根据 markout、volatility、spread 状态调整间距。
6. 统计真实 execComm，确认 maker rebate 口径。
7. 如果单市场 maker leg 稳定有 edge，再考虑多市场 hedge。
```

一个更保守的 ladder 状态机可以是：

```text
abs(position) < 0.2 * SOFT:
    正常双边挂单

0.2 * SOFT <= abs(position) < 0.5 * SOFT:
    加仓方向减少一层
    减仓方向保持全部层

0.5 * SOFT <= abs(position) < 0.8 * SOFT:
    加仓方向只保留外层或只保留一层
    减仓方向靠近 anchor

abs(position) >= 0.8 * SOFT:
    只挂减仓方向

inventory_age 超过阈值:
    减仓单更靠近盘口
```

这会比单纯扩大或缩小 `GRID_SPACING_BPS` 更稳定。

