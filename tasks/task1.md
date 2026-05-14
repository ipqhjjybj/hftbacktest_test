

.env 是 TARDIS_API_KEY 的配置， 现在我要你用 BITMEX 的 XBTUSD 2026-05-12 的数据，以及 gate-io-futures 的BTCUSDT 2026-05-12 的数据，做一个做市套利策略的回测

当然，你得先下好数据，配置好hftbacktest， 然后再进行回测

OPEN_LONG_SPREAD_RATIO=0.00040
CLOSE_SPREAD_RATIO=-0.001
MAX_POSITION_BASE=0.079

策略描述:
    读取 gate价格 买一bid1, 卖一ask1 
    在 BITMEX 上 挂买单 bid1 * (1 - OPEN_LONG_SPREAD_RATIO)
    在 gate-io-futures 上 挂卖单 ask1 * (1 - CLOSE_SPREAD_RATIO)

    1.如果 gate价格发生变化，则对应的修改 BITMEX 上的买单和卖单
    2.如果 bitmex 这边发生成交， 则对应的在 gate 对冲对应的量，做到两边仓位一致
    3.程序有最大仓位 MAX_POSITION_BASE， 买单达到最大仓位，就不在买单侧挂单了，卖单同理
