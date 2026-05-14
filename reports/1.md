挂单价差是 
bid_spread = OPEN_LONG_SPREAD_RATIO
ask_spread = CLOSE_SPREAD_RATIO

bitmex_bid = gate_bid * (1 - bid_spread)
bitmex_ask = gate_ask * (1 - ask_spread)


OPEN_LONG_SPREAD_RATIO 开仓价差
CLOSE_SPREAD_RATIO 平仓价差
MAX_POSITION_BASE 最大持仓量


参数:
OPEN_LONG_SPREAD_RATIO=0.00035
CLOSE_SPREAD_RATIO=0.00016
MAX_POSITION_BASE=10000

========== 回测结果1 ==========
结论: 赚钱 +86.0622 USDT
总成交 base: 6.83118839 BTC

BitMEX maker 成交次数: 5508
买成交: 3439
卖成交: 2069
Gate hedge 成交次数: 5504

实际配对边际 PnL: +83.2042 USDT
平均配对边际: +12.1819 USDT/BTC
BitMEX买 -> Gate卖: +27.6937 USDT/BTC
BitMEX卖 -> Gate买: -13.6111 USDT/BTC

BitMEX 最大仓位: 2.16518147 BTC
Gate 最大仓位: 2.16500000 BTC
期间最大持仓量 gross: 4.33018147 BTC
最大净敞口: 0.00326755 BTC
日终强平 PnL: -0.1703 USDT
手续费: 0
最终仓位归零: 是



参数:
OPEN_LONG_SPREAD_RATIO=0.00035
CLOSE_SPREAD_RATIO=0.00016
MAX_POSITION_BASE=0.01

========== 回测结果2 ==========

结论: 赚钱 +5.4697 USDT
总成交 base: 0.96813905 BTC

BitMEX maker 成交次数: 782
买成交: 387
卖成交: 395
Gate hedge 成交次数: 782

实际配对边际 PnL: +5.5123 USDT
平均配对边际: +5.9006 USDT/BTC
BitMEX买 -> Gate卖: +26.7408 USDT/BTC
BitMEX卖 -> Gate买: -14.5175 USDT/BTC

BitMEX 最大仓位: 0.01002962 BTC
Gate 最大仓位: 0.00990000 BTC
期间最大持仓量 gross: 0.01992962 BTC
最大净敞口: 0.00013007 BTC
日终强平 PnL: -0.0010 USDT
手续费: 0
最终仓位归零: 是



参数:
OPEN_LONG_SPREAD_RATIO=0.00035
CLOSE_SPREAD_RATIO=0.00016
MAX_POSITION_BASE=0.0001
========== 回测结果3 ==========

结论: 赚钱 +3.4734 USDT
总成交 base: 0.59803053 BTC
BitMEX maker 成交次数: 483
买成交: 241
卖成交: 242
Gate hedge 成交次数: 483
实际配对边际 PnL: +3.4627 USDT
平均配对边际: +5.9743 USDT/BTC
BitMEX买 -> Gate卖: +26.7498 USDT/BTC
BitMEX卖 -> Gate买: -14.7153 USDT/BTC
最大净敞口: 0.00005370 BTC
期间最大持仓量 gross: 0.00245370 BTC
最终仓位归零: 是
手续费: 0


