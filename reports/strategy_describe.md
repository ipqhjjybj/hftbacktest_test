# BitMEX XBTUSD vs Gate BTCUSDT 做市套利策略说明

本文描述 `scripts/bitmex_gate_mm_arb_backtest.py` 当前实现的回测策略。

## 标的和数据

- 回测日期: `2026-05-12`
- Maker 交易所: BitMEX `XBTUSD`
- Hedge 交易所: Gate futures `BTC_USDT`
- 数据来源: Tardis
- BitMEX 使用 `trades` 和 `incremental_book_L2`
- Gate 使用 `trades` 和 `book_ticker`

脚本会先从 `.env` 读取 `TARDIS_API_KEY`，下载 CSV gzip 数据，再转换成 hftbacktest 使用的 `.npz` 数据。

## 当前策略参数

```text
OPEN_LONG_SPREAD_RATIO = 0.00035
CLOSE_SPREAD_RATIO = 0.00016
MAX_POSITION_BASE = 10000.0
ORDER_UPDATE_INTERVAL_NS = 10_000_000
```

交易和合约参数:

```text
BitMEX tick size = 0.1
BitMEX lot size = 100 USD
BitMEX 每次挂单量 = 100 USD

Gate tick size = 0.1
Gate lot size = 1 contract
Gate contract size = 0.0001 BTC
```

回测费用和延迟:

```text
BitMEX fee = 0
Gate fee = 0
Order latency = 0
改单间隔 = 10ms
```

## 挂单逻辑

策略每隔 `10ms` 读取一次 Gate best bid / best ask，并更新 BitMEX maker 挂单。

BitMEX 买单价格:

```text
bitmex_bid = gate_bid * (1 - OPEN_LONG_SPREAD_RATIO)
```

BitMEX 卖单价格:

```text
bitmex_ask = gate_ask * (1 - CLOSE_SPREAD_RATIO)
```

价格会按 BitMEX tick size 做取整:

- 买单价格向下取整到 tick
- 卖单价格向上取整到 tick

BitMEX 挂单使用 post-only limit order。

## 仓位限制

仓位限制按 BitMEX 的 BTC base 仓位计算:

```text
bitmex_base = bitmex_contract_position / bitmex_mid_price
```

买单是否允许挂出:

- 当前 BitMEX base 仓位为负时，允许挂买单，用于减空仓
- 当前 BitMEX base 仓位为正或零时，只有买完后不超过 `MAX_POSITION_BASE` 才允许挂买单

卖单是否允许挂出:

- 当前 BitMEX base 仓位为正时，允许挂卖单，用于减多仓
- 当前 BitMEX base 仓位为负或零时，只有卖完后不超过负向 `MAX_POSITION_BASE` 才允许挂卖单

如果 `MAX_POSITION_BASE` 小于 BitMEX 最小下单量对应的 BTC 数量，脚本会使用一笔 BitMEX 最小单作为有效仓位上限，保证策略至少能挂出一笔 `100 USD` 的最小订单。

## 成交后 Gate 对冲

当 BitMEX maker 单发生成交后，策略立即计算两边合计 base 敞口:

```text
net_base = bitmex_base + gate_base
```

其中:

```text
gate_base = gate_contract_position * 0.0001 BTC
```

对冲规则:

- 如果 `net_base > 0`，说明整体多 BTC，在 Gate 用 taker 单卖出对冲
- 如果 `net_base < 0`，说明整体空 BTC，在 Gate 用 taker 单买入对冲
- 对冲数量按 Gate contract size 转换，并向下取整到 Gate lot size

Gate 对冲使用 IOC market order。

## PnL 和风险统计

脚本会记录以下核心指标:

- 总 PnL
- 总成交 base
- BitMEX maker 成交次数
- BitMEX 买成交次数
- BitMEX 卖成交次数
- Gate hedge 成交次数
- 实际配对边际 PnL
- 平均实际配对边际
- BitMEX 买 -> Gate 卖平均边际
- BitMEX 卖 -> Gate 买平均边际
- BitMEX 最大仓位
- Gate 最大仓位
- 期间最大持仓量 gross
- 最大净敞口
- 平均 hedge 延迟
- 平均 hedge 滑点
- 日终强制平仓 PnL
- 最终 BitMEX 仓位
- 最终 Gate 仓位
- 最终净仓位

总 PnL 由 BitMEX BTC equity 按 BitMEX mid price 折算成 USDT 后，加上 Gate USDT equity 得到。

## 日终处理

回测结束时，策略会:

1. 撤掉 BitMEX 和 Gate 的所有未成交订单
2. 用 taker 单平掉 BitMEX 剩余仓位
3. 用 taker 单平掉 Gate 剩余仓位
4. 记录日终强制平仓 PnL
5. 输出最终仓位是否归零

## 输出文件

每次运行脚本后，结果会写入:

```text
results/bitmex_xbtusd_gate_btc_usdt_mm_arb_20260512.npz
results/bitmex_xbtusd_gate_btc_usdt_mm_arb_20260512.summary.json
results/bitmex_xbtusd_gate_btc_usdt_mm_arb_20260512.report.md
```
