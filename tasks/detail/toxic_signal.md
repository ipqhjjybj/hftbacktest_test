

比较实用的 toxic signal 组成

我建议先用 4 类，不要一上来搞太复杂。

1. Microprice 偏离

microprice 可以理解为盘口压力价格：

microprice = (ask * bid_qty + bid * ask_qty) / (bid_qty + ask_qty)

如果 microprice 明显高于 mid，说明买盘压力更强；明显低于 mid，说明卖盘压力更强。

microprice_bps = (microprice - mid) / mid * 10000

用法：

microprice_bps > +0.8  买方强，少挂/撤 sell
microprice_bps < -0.8  卖方强，少挂/撤 buy

这个适合捕捉盘口一档压力。

2. 短期 momentum

看最近几百毫秒到几秒的 mid price 变化。

例如：

momentum_1s_bps = (mid_now - mid_1s_ago) / mid_1s_ago * 10000

用法：

momentum_1s_bps > +1.0  价格上冲，sell 更危险
momentum_1s_bps < -1.0  价格下砸，buy 更危险

这个比 microprice 更偏价格行为。

3. Spread / volatility 放大

如果短时间内 mid 抖动很大，或者 spread 变宽，说明市场状态变差。

可以算：

vol_bps = rolling_std(mid_return, 2s 或 5s)

用法：

vol_bps 高于阈值：
    加宽 spread
    减少 ladder 层数
    只挂减仓方向

这个不是判断方向，而是判断“现在是否适合做 maker”。

4. Trade pressure / aggressive flow

如果 npz 里有 trade 数据，最好用主动成交方向。

例如最近 1 秒：

buy_aggr_qty = 主动买成交量
sell_aggr_qty = 主动卖成交量

trade_imbalance = (buy_aggr_qty - sell_aggr_qty) / (buy_aggr_qty + sell_aggr_qty)

用法：

trade_imbalance > +0.6  主动买很强，sell 风险高
trade_imbalance < -0.6  主动卖很强，buy 风险高

这个通常比单纯盘口更有效，因为它看到的是已经发生的主动吃单。

———

  我建议 toxic signal 最终不要输出一个单值，而是输出两个方向风险：

  toxic_buy_risk   # bid 被打后继续下跌的风险
  toxic_sell_risk  # ask 被打后继续上涨的风险

  例如：

  toxic_buy_risk =
      weight1 * max(0, -microprice_bps)
    + weight2 * max(0, -momentum_bps)
    + weight3 * max(0, -trade_imbalance)
    + weight4 * vol_bps

  toxic_sell_risk =
      weight1 * max(0, microprice_bps)
    + weight2 * max(0, momentum_bps)
    + weight3 * max(0, trade_imbalance)
    + weight4 * vol_bps

  然后策略动作分层处理。

  轻度 toxic：加宽

  buy_risk 高：
      bid half_spread += risk_penalty_bps

  sell_risk 高：
      ask half_spread += risk_penalty_bps

  中度 toxic：跳过第一层

  对 ladder/grid 很有用：

  buy_risk 高：
      不挂 b1，只挂 b2/b3...

  sell_risk 高：
      不挂 s1，只挂 s2/s3...

  这样不会完全停止交易，但避开最容易被打的内层单。

  重度 toxic：单边撤单

  buy_risk 很高：
      取消所有 buy quotes

  sell_risk 很高：
      取消所有 sell quotes

  极端 toxic：只允许减仓

  如果你已经有仓位，强 toxic 下应该优先控制库存。

  持多仓 + 下跌 toxic：
      不再挂 buy
      只挂 sell 减仓

  持空仓 + 上涨 toxic：
      不再挂 sell
      只挂 buy 减仓

  ———

  对于你现在的 ladder/grid，我觉得最值得做的是这个版本：

  1. 正常状态：
     b1/s1/b2/s2/b3/s3 都挂

  2. 轻微 toxic：
     对危险方向加宽 1-3 bps

  3. 中等 toxic：
     危险方向跳过 level 1，只挂 level 2+

  4. 严重 toxic：
     危险方向全部不挂

  5. 如果库存已经偏大：
     toxic 方向禁止加仓，只允许减仓

  这样比“有信号就全撤”更平滑，也比“只加 spread”更有效。

  一个比较实用的初版阈值可以是：

  microprice_threshold_bps = 0.8
  momentum_threshold_bps = 1.0
  vol_threshold_bps = 2.0
  trade_imbalance_threshold = 0.6

  最终 toxic score：

  score_buy =
      1.0 * max(0, -microprice_bps - 0.8)
    + 1.0 * max(0, -momentum_bps - 1.0)
    + 2.0 * max(0, -trade_imbalance - 0.6)
    + 0.5 * max(0, vol_bps - 2.0)

  score_sell =
      1.0 * max(0, microprice_bps - 0.8)
    + 1.0 * max(0, momentum_bps - 1.0)
    + 2.0 * max(0, trade_imbalance - 0.6)
    + 0.5 * max(0, vol_bps - 2.0)

  动作：

  score < 0.5      正常挂单
  0.5 - 1.5        加宽
  1.5 - 3.0        跳过第一层
  > 3.0            危险方向不挂

  我个人建议你不要先追求 toxic signal 预测很准，而是先看它能不能改善这几个指标：

  gross_pnl 是否变好
  fill 后 1s/3s/10s markout 是否变好
  成交数是否下降太多
  库存峰值是否下降
  亏损日是否改善

  尤其是 markout 很关键。
  如果加了 toxic signal 后，maker rebate 可能少一些，但 fill 后 1s/3s/10s 的 adverse selection 明显下降，那这个信号就是有价值的。