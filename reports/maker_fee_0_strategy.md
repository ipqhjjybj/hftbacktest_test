
当 maker fee 为 0时， 依旧能盈利的策略

| 策略 | 标的 | 区间 | gross | 说明 |
    |---|---|---:|---:|---|
    | queue_aware_ladder | XBTUSDT | 20260516-20260518 | +0.2053 | 低频、保守，成交少 |
    | toxic_flow_filtered_maker | XBTUSDT | 20260516-20260518 | +0.0513 | 有一点 gross alpha，但很小 |
    | cooldown_inventory_maker | XBTUSDT | 20260516-20260518 | +0.0881 | 三天 gross 为正 |
    | ladder_grid | XBTUSD | 20260418-20260518 | +315.9187 | XBTUSD 反向合约，不是 XBTUSDT |

所以如果只看 XBTUSDT，当前比较值得继续研究的是：

  1. queue_aware_ladder
  2. toxic_flow_filtered_maker
  3. cooldown_inventory_maker

  