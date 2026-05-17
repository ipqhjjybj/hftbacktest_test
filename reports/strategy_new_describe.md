# BitMEX/Gate 库存型做市策略说明

## 策略目标

这个策略不是逐笔无风险套利。它的目标是用 BitMEX maker 成交赚取相对更好的被动成交价，同时允许短时间库存偏离，尽量用后续 BitMEX 反向 maker 成交把库存拉回；只有库存超时或超限时，才用 Gate IOC 作为兜底对冲。

核心取舍是：

```text
减少每笔 Gate taker hedge 成本
换取有限、可控的短时间库存风险
```

## 交易结构

- BitMEX 是 maker venue。
- Gate 是 panic hedge / risk-off venue。
- BitMEX 同时维护 bid 和 ask，但会根据库存和当日买卖成交数量动态停掉某一侧。
- Gate 不再每笔 BitMEX 成交后立刻 hedge。
- Gate 只在净库存超过阈值并持有过久，或超过硬阈值时对冲。

## 主要参数

当前脚本参数在 `scripts/bitmex_gate_inventory_backtest.py`：

```python
BASE_BID_SPREAD_RATIO = 0.0008
BASE_ASK_SPREAD_RATIO = 0.0008
INVENTORY_SKEW_BPS = 12.0
MAX_REDUCE_QUOTE_CROSS_BPS = 2.0

SOFT_INVENTORY_LIMIT_BASE = 0.0010
HARD_INVENTORY_LIMIT_BASE = 0.0025
MAX_INVENTORY_HOLD_NS = 3_000_000_000

MAX_POSITION_BASE = 0.0030
MAX_GROSS_POSITION_BASE = 0.02

TARGET_BITMEX_BUY_FILLS = 10
TARGET_BITMEX_SELL_FILLS = 10
MAX_FILL_COUNT_IMBALANCE = 1
```

含义：

- `BASE_BID_SPREAD_RATIO` / `BASE_ASK_SPREAD_RATIO`: 正常情况下 BitMEX bid/ask 相对 Gate anchor 的基础报价距离。
- `INVENTORY_SKEW_BPS`: 库存偏离后，减仓方向报价变 aggressive、加仓方向报价变 conservative 的力度。
- `MAX_REDUCE_QUOTE_CROSS_BPS`: 减仓方向最多允许相对 Gate anchor 穿多少 bps，提高反向 maker 成交概率。
- `SOFT_INVENTORY_LIMIT_BASE`: 净库存超过该值后开始计时，并强烈偏向减仓。
- `HARD_INVENTORY_LIMIT_BASE`: 净库存超过该值后立刻 Gate hedge。
- `MAX_INVENTORY_HOLD_NS`: 净库存超过 soft limit 后最多持有多久，超过后 Gate hedge。
- `MAX_POSITION_BASE`: 单次加仓后的净风险上限。
- `MAX_GROSS_POSITION_BASE`: `abs(bitmex_base) + abs(gate_base)` 的总仓位占用上限。
- `TARGET_BITMEX_BUY_FILLS` / `TARGET_BITMEX_SELL_FILLS`: 希望当天 BitMEX maker 买卖成交数量尽量接近的目标。
- `MAX_FILL_COUNT_IMBALANCE`: 买卖成交数量最大允许领先差距，超过后停止领先方向。

## 报价逻辑

### BitMEX bid

BitMEX bid 参考 Gate best bid 和 BitMEX best bid：

```text
bid_price = min(gate_bid * (1 - bid_spread), bitmex_best_bid)
```

其中：

```text
bid_spread = BASE_BID_SPREAD_RATIO + inventory_skew
```

如果净库存偏多，bid 会变远，降低继续买入的概率。  
如果净库存偏空，bid 会变近，甚至允许轻微穿 Gate anchor，用来减空仓。

### BitMEX ask

BitMEX ask 参考 Gate best ask 和 BitMEX best ask：

```text
ask_price = max(gate_ask * (1 + ask_spread), bitmex_best_ask)
```

其中：

```text
ask_spread = BASE_ASK_SPREAD_RATIO - inventory_skew
```

如果净库存偏多，ask 会变近，鼓励卖出减仓。  
如果净库存偏空，ask 会变远，降低继续卖出的概率。

## 库存控制

策略持续计算：

```text
net_base = bitmex_position_base + gate_position_base
gross_base = abs(bitmex_position_base) + abs(gate_position_base)
```

控制规则：

- `abs(net_base) <= soft limit`: 正常做市。
- `abs(net_base) > soft limit`: 进入库存偏离状态，记录持有时间。
- `net_base > soft limit`: 净多 BTC，停止或压低 bid，保留/加强 ask。
- `net_base < -soft limit`: 净空 BTC，停止或压低 ask，保留/加强 bid。
- `abs(net_base) >= hard limit`: 立即 Gate IOC hedge。
- `abs(net_base) > soft limit` 且持有超过 `MAX_INVENTORY_HOLD_NS`: Gate IOC hedge。
- `gross_base` 接近 `MAX_GROSS_POSITION_BASE`: 停止继续增加总仓位。

## 买卖成交数量配平

为了避免下跌日只成交 bid、上涨日只成交 ask，策略加入成交数量约束：

```text
如果 buy_fills - sell_fills >= MAX_FILL_COUNT_IMBALANCE:
    停止继续挂 bid

如果 sell_fills - buy_fills >= MAX_FILL_COUNT_IMBALANCE:
    停止继续挂 ask
```

例外：

- 如果当前净空，bid 是减仓方向，即使 buy 数量领先限制存在，也允许 bid。
- 如果当前净多，ask 是减仓方向，即使 sell 数量领先限制存在，也允许 ask。

这个规则的目标是让当天 BitMEX maker 成交尽量接近：

```text
10 买 / 10 卖
```

但这是约束，不是保证。maker 单是否成交取决于市场是否打到报价。强行保证 10/10 通常需要 taker 或更激进报价，会改变策略风险性质。

## Gate 对冲逻辑

Gate hedge 是兜底风险控制，不是主要收益来源。

Gate hedge 触发条件：

```text
abs(net_base) >= HARD_INVENTORY_LIMIT_BASE
```

或者：

```text
abs(net_base) > SOFT_INVENTORY_LIMIT_BASE
and inventory_hold_time >= MAX_INVENTORY_HOLD_NS
```

触发后，策略用 Gate IOC limit 把净库存向 0 对冲。

为什么仍然需要 Gate：

- BitMEX 反向 maker 不一定成交。
- 单边行情会让库存持续累积。
- Gate hedge 可以限制库存风险和日终强平风险。

## 与逐笔 hedge 策略的区别

逐笔 hedge：

```text
BitMEX maker fill -> 立即 Gate taker hedge
```

优点：

- 净风险时间短。
- 仓位暴露小。

缺点：

- 每笔都吃 Gate taker 成本、滑点和延迟。
- 小价差很容易被 hedge 成本吃掉。

库存策略：

```text
BitMEX maker fill -> 暂时持库存 -> 等 BitMEX 反向 maker fill
                  -> 超时/超限才 Gate hedge
```

优点：

- 减少逐笔 Gate taker hedge。
- 有机会用 BitMEX maker 买卖回补库存。

缺点：

- 有裸库存风险。
- 单边行情里仍然可能频繁 Gate hedge。
- 成交数量会下降，特别是在买卖数量配平约束打开后。

## 当前回测结论

当前回测仍使用 `0` 手续费模型，因此只能看策略形态和风险，不可直接视为实盘收益。

加入买卖配平和 gross limit 后，结果更保守：

- `20260512`: BitMEX `buy=1`, `sell=0`, PnL 约 `+0.0361 USDT`
- `20260513`: BitMEX `buy=2`, `sell=1`, PnL 约 `+0.0762 USDT`
- `20260514`: BitMEX `buy=4`, `sell=3`, PnL 约 `+0.2874 USDT`

可以看到，买卖数量更均衡，但成交数明显下降。这符合预期：策略宁愿少成交，也避免在单边行情里持续接同一方向的 maker fill。

## 后续改进方向

1. 加真实手续费模型。
2. 对 `TARGET_BITMEX_BUY_FILLS` / `TARGET_BITMEX_SELL_FILLS` 做参数扫描。
3. 对 `BASE_*_SPREAD_RATIO`、`INVENTORY_SKEW_BPS`、`MAX_REDUCE_QUOTE_CROSS_BPS` 做网格搜索。
4. 增加趋势过滤：下跌行情减少 bid，上涨行情减少 ask。
5. 增加分时段成交目标，而不是全天固定 10/10。
6. 增加 Gate hedge 滑点和成交失败统计。
7. 把回测中的订单延迟参数校准到实盘日志分布。
