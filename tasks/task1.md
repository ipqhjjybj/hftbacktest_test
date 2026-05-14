# 任务 1：BitMEX XBTUSD vs Gate BTCUSDT 做市套利回测

## 目标

使用 Tardis 数据回测一个 maker/taker 套利策略：

- Maker 交易所：BitMEX `XBTUSD`
- 对冲交易所：`gate-io-futures` `BTCUSDT`
- 回测日期：`2026-05-12`
- Tardis API key 已配置在 `.env`

## 准备工作

1. 下载 `2026-05-12` 的 BitMEX 和 Gate futures 数据。
2. 配置 `hftbacktest`，让两个交易所的数据在同一时间轴上回放。
3. 运行策略回测，并把结果保存到 `results/` 目录。

## 基础参数

```text
OPEN_LONG_SPREAD_RATIO=0.00035
CLOSE_SPREAD_RATIO=0.00016
MAX_POSITION_BASE=10000
```

## 策略逻辑

读取 Gate 最优买卖价：

```text
gate_bid = Gate BTCUSDT best bid
gate_ask = Gate BTCUSDT best ask
```

在 BitMEX 上挂单：

```text
bitmex_bid = gate_bid * (1 - bid_spread)
bitmex_ask = gate_ask * (1 - ask_spread)
```

当 BitMEX maker 单成交后，立即在 Gate 上用 taker 单对冲对应的 base 数量。

当 BitMEX 某个方向的 base 仓位达到 `MAX_POSITION_BASE` 后，停止继续挂该方向的订单。

回测日结束时，撤掉所有未成交挂单，并用 taker 单平掉 BitMEX 和 Gate 的全部剩余仓位，最终仓位必须归零。

## 动态价差

把固定价差改成带库存偏斜的动态价差：

```text
inventory_ratio = bitmex_position_base / max_position_base
inventory_skew = inventory_ratio * 0.00020

bid_spread = clamp(
    OPEN_LONG_SPREAD_RATIO + inventory_skew + vol_widen - basis_skew,
    0.00010,
    0.00070
)

ask_spread = clamp(
    CLOSE_SPREAD_RATIO + inventory_skew - vol_widen - basis_skew,
    -0.00020,
    0.00030
)


bid_spread = OPEN_LONG_SPREAD_RATIO

ask_spread = CLOSE_SPREAD_RATIO
```

第一版可以先设：

```text
vol_widen = 0
basis_skew = 0
```

## 订单管理

- Gate best bid 变化时，更新 BitMEX 买单。
- Gate best ask 变化时，更新 BitMEX 卖单。
- BitMEX 挂单必须保持 post-only。
- BitMEX 成交后，在 Gate 上用 taker 单对冲。
- 尽量让 BitMEX 和 Gate 的 base 仓位保持一致。
- 回测结束时撤掉所有挂单，并强制平掉所有剩余仓位。

## 输出结果

生成一份简洁报告，至少包含：

- 总成交 base 数量
- 已实现 PnL
- BitMEX 最大仓位
- Gate 最大仓位
- 最大净敞口
- BitMEX maker 成交次数
- Gate hedge 成交次数
- 平均 hedge 延迟
- 平均 hedge 滑点
- 日终强制平仓 PnL
- 最终 BitMEX 仓位
- 最终 Gate 仓位
