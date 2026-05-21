# Other HFT Strategy Backtests

这些脚本是学习用的单市场高频策略，不复用前面的 fixed spread、Avellaneda-Stoikov、ladder/grid 或 advanced 做市逻辑。

统一入口：

```bash
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/other_hft/bitmex_other_hft_strategies.py \
  --skip-download \
  --strategy order_flow_momentum \
  --dates 20260518
```

也可以用分类包装脚本：

```bash
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/other_hft/bitmex_order_flow_momentum.py --skip-download --dates 20260518
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/other_hft/bitmex_queue_imbalance_breakout.py --skip-download --dates 20260518
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/other_hft/bitmex_liquidity_fade.py --skip-download --dates 20260518
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/other_hft/bitmex_mean_reversion_scalper.py --skip-download --dates 20260518
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/other_hft/bitmex_passive_entry_active_exit.py --skip-download --dates 20260518
PYTHONPATH=py-hftbacktest .venv/bin/python scripts/other_hft/bitmex_maker_first_exit.py --skip-download --dates 20260518
```

## 策略分类

- `order_flow_momentum`: 主动方向策略。用最近市场成交的主动买卖量差做信号，顺订单流方向用 IOC limit 入场。
- `queue_imbalance_breakout`: 主动方向策略。用 best bid / best ask 队列失衡和 microprice 判断短线突破。
- `liquidity_fade`: 主动方向策略。观察买一或卖一流动性突然撤退，顺流动性变薄方向入场。
- `mean_reversion_scalper`: 主动反转策略。短时间价格冲击过大但订单流没有继续确认时反向入场。
- `passive_entry_active_exit`: 半被动策略。用订单流+盘口 alpha 决定方向，被动挂 maker 单入场，仓位不利或信号反转时主动 IOC 出场。
- `maker_first_exit`: 半被动策略。被动 maker 入场后，先挂 maker 平仓单；只有信号反转、止损或持仓超时才用 IOC emergency exit。

## 默认执行模型

- 默认 `--exchange-model no_partial`，符合你之前说的“下次不要用 live_l2，用最开始乐观估计模型”。
- 主动策略默认会产生 taker 成交，默认 `--taker-fee-rate 0.0001`。
- 半被动策略入场用 GTX maker，出场用 IOC limit。
- `maker_first_exit` 入场和平常出场都用 GTX maker，emergency exit 才用 IOC limit。

这些策略都是研究模板，参数没有调优。建议先固定同一批日期横向比较，再扫阈值、持仓时间、止盈止损和手续费。
