from __future__ import annotations

from pathlib import Path


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def relative_link(target: Path, base_file: Path | None) -> str:
    if base_file is None:
        return str(target)
    try:
        return target.relative_to(base_file.parent).as_posix()
    except ValueError:
        return str(target)


def top_ic_rows(ic_rows: list[dict[str, object]], bucket_label: str, limit: int = 5) -> list[dict[str, object]]:
    rows = [row for row in ic_rows if row["label"] == bucket_label]
    rows.sort(key=lambda row: abs(float(row["rank_ic"])), reverse=True)
    return rows[:limit]


def render_report(
    symbol: str,
    dates: list[str],
    sample_interval_ms: int,
    horizons_ms: tuple[int, ...],
    bucket_label: str,
    ic_rows: list[dict[str, object]],
    bucket_rows: list[dict[str, object]],
    maker_fill_bucket_rows: list[dict[str, object]] | None,
    output_files: dict[str, Path],
    chart_files: dict[str, Path] | None = None,
    report_path: Path | None = None,
) -> str:
    chart_files = chart_files or {}
    lines: list[str] = []
    lines.append(f"# 高频因子研究报告: {symbol}")
    lines.append("")
    lines.append(f"- 数据日期: {', '.join(dates)}")
    lines.append(f"- 采样间隔: {sample_interval_ms}ms")
    lines.append(f"- 预测周期: {', '.join(str(x) + 'ms' for x in horizons_ms)}")
    lines.append(f"- 当前分桶观察标签: `{bucket_label}`")
    lines.append("")
    lines.append("## 怎么读")
    lines.append("")
    lines.append("- IC 衡量因子和未来 mid-price 变化的方向关系。")
    lines.append("- IC > 0：因子越大，未来 mid 越倾向上涨。")
    lines.append("- IC < 0：因子越大，未来 mid 越倾向下跌。")
    lines.append("- 分桶图比单个 IC 更直观：横轴从低因子值到高因子值，重点看曲线是否有稳定斜率。")
    lines.append("- maker edge 图只是粗略估计；lifecycle maker edge 图进一步加入入场延迟、TTL、post-only 和队列假设。")
    lines.append("")
    if chart_files:
        lines.append("## 图表")
        lines.append("")
        if "rank_ic_heatmap" in chart_files:
            lines.append("### 1. Rank IC 热力图")
            lines.append("")
            lines.append("颜色越深代表方向关系越强；红色偏上涨预测，蓝色偏下跌预测。")
            lines.append("")
            lines.append(f"![Rank IC Heatmap]({relative_link(chart_files['rank_ic_heatmap'], report_path)})")
            lines.append("")
        if "bucket_future_ret" in chart_files:
            lines.append("### 2. 分桶后的未来收益")
            lines.append("")
            lines.append("每张小图对应一个因子。横轴越往右，代表因子值越高；纵轴是之后 mid-price 的平均变化。")
            lines.append("")
            lines.append(f"![Future Return By Bucket]({relative_link(chart_files['bucket_future_ret'], report_path)})")
            lines.append("")
        if "bucket_maker_edge" in chart_files:
            lines.append("### 3. 分桶后的近似 maker edge")
            lines.append("")
            lines.append("这张图用来判断某个因子区间更适合挂 bid 还是挂 ask。它是研究入口，不等于最终策略 PnL。")
            lines.append("")
            lines.append(f"![Maker Edge By Bucket]({relative_link(chart_files['bucket_maker_edge'], report_path)})")
            lines.append("")
        if "bucket_fill_probability" in chart_files:
            lines.append("### 4. 假设挂在 BBO 的成交概率")
            lines.append("")
            lines.append("这里假设在当前 best bid/ask 排队，未来相反方向主动成交量吃掉当前队列和自己的订单后，才算可能成交。")
            lines.append("")
            lines.append(f"![Fill Probability By Bucket]({relative_link(chart_files['bucket_fill_probability'], report_path)})")
            lines.append("")
        if "bucket_fill_edge" in chart_files:
            lines.append("### 5. 假设成交后的 maker edge")
            lines.append("")
            lines.append("这张图只统计上面假设模型里会成交的样本，用来看不同因子区间成交后是否更容易被 toxic flow 打。")
            lines.append("")
            lines.append(f"![Fill Edge By Bucket]({relative_link(chart_files['bucket_fill_edge'], report_path)})")
            lines.append("")
        if "lifecycle_fill_probability" in chart_files:
            lines.append("### 6. 加入订单生命周期后的成交概率")
            lines.append("")
            lines.append("这里先等 quote 经过入场延迟进入订单簿，再检查 post-only 是否有效、是否仍在 BBO/改善 BBO，并在 TTL 内用相反方向主动成交量估算是否吃掉队列。")
            lines.append("")
            lines.append(f"![Lifecycle Fill Probability]({relative_link(chart_files['lifecycle_fill_probability'], report_path)})")
            lines.append("")
        if "lifecycle_edge_if_filled" in chart_files:
            lines.append("### 7. 加入订单生命周期后的成交条件 edge")
            lines.append("")
            lines.append("这张图只看模型里会成交的样本。它更接近 maker 策略需要回答的问题：被成交以后，这笔单是否有正 edge。")
            lines.append("")
            lines.append(f"![Lifecycle Edge If Filled]({relative_link(chart_files['lifecycle_edge_if_filled'], report_path)})")
            lines.append("")
        if "lifecycle_expected_edge" in chart_files:
            lines.append("### 8. 每次报价机会的期望 edge")
            lines.append("")
            lines.append("这是成交概率乘以成交后 edge 的粗略结果；数值会比成交条件 edge 小很多，但更适合用来筛报价条件。")
            lines.append("")
            lines.append(f"![Lifecycle Expected Edge]({relative_link(chart_files['lifecycle_expected_edge'], report_path)})")
            lines.append("")
    lines.append("## 这次样本的直接结论")
    lines.append("")
    lines.append(f"按 `{bucket_label}` 的绝对 rank IC 排序，最明显的几个因子是：")
    lines.append("")
    lines.append("| 因子 | pearson_ic | rank_ic | 直观含义 |")
    lines.append("|---|---:|---:|---|")
    for row in top_ic_rows(ic_rows, bucket_label):
        rank_ic = float(row["rank_ic"])
        direction = "因子越高，未来 mid 越偏上涨" if rank_ic > 0 else "因子越高，未来 mid 越偏下跌"
        lines.append(
            "| {factor} | {pearson_ic} | {rank_ic} | {direction} |".format(
                factor=row["factor"],
                pearson_ic=fmt(row["pearson_ic"], 6),
                rank_ic=fmt(rank_ic, 6),
                direction=direction,
            )
        )
    lines.append("")
    lines.append("如果一个因子的热力图颜色深、分桶曲线也比较平滑，它才更值得放进策略。")
    lines.append("如果只有 IC 好看，但分桶曲线跳来跳去，通常说明它不够稳定。")
    lines.append("")
    if maker_fill_bucket_rows:
        rows = list(maker_fill_bucket_rows)
        rows.sort(
            key=lambda row: (
                float(row["expected_edge_bps"]) if str(row["expected_edge_bps"]) != "nan" else -1e9
            ),
            reverse=True,
        )
        lines.append("## Maker Fill Edge 正桶候选")
        lines.append("")
        lines.append("下面按 `expected_edge_bps` 从高到低列出前 20 个单因子桶。它不是最终策略，只是告诉我们哪些状态值得转成挂单规则。")
        lines.append("")
        lines.append("| factor | bucket | side | samples | fill_prob | edge_if_filled_bps | expected_edge_bps | positive_fill_frac | post_only_reject_frac |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for row in rows[:20]:
            lines.append(
                "| {factor} | {bucket} | {side} | {samples} | {fill_prob} | {edge} | {expected} | {positive} | {reject} |".format(
                    factor=row["factor"],
                    bucket=row["bucket"],
                    side=row["side"],
                    samples=row["samples"],
                    fill_prob=fmt(float(row["fill_prob"]), 6),
                    edge=fmt(float(row["edge_if_filled_bps"]), 6),
                    expected=fmt(float(row["expected_edge_bps"]), 6),
                    positive=fmt(float(row["positive_fill_frac"]), 4),
                    reject=fmt(float(row["post_only_reject_frac"]), 4),
                )
            )
        lines.append("")
    lines.append("## IC 明细表")
    lines.append("")
    lines.append("| factor | label | samples | pearson_ic | rank_ic |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in ic_rows:
        lines.append(
            "| {factor} | {label} | {samples} | {pearson_ic} | {rank_ic} |".format(
                factor=row["factor"],
                label=row["label"],
                samples=row["samples"],
                pearson_ic=fmt(row["pearson_ic"], 6),
                rank_ic=fmt(row["rank_ic"], 6),
            )
        )
    lines.append("")
    lines.append("## 分桶表预览")
    lines.append("")
    lines.append("下面只是前几行预览，完整结果看 CSV。")
    lines.append("")
    lines.append("| factor | bucket | samples | factor_mean | label_mean_bps | positive_frac |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in bucket_rows[:80]:
        lines.append(
            "| {factor} | {bucket} | {samples} | {factor_mean} | {label_mean} | {positive} |".format(
                factor=row["factor"],
                bucket=row["bucket"],
                samples=row["samples"],
                factor_mean=fmt(row["factor_mean"], 6),
                label_mean=fmt(row[bucket_label], 6),
                positive=fmt(row["positive_label_frac"], 4),
            )
        )
    lines.append("")
    lines.append("## 文件")
    lines.append("")
    for name, path in output_files.items():
        lines.append(f"- {name}: `{path}`")
    for name, path in chart_files.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")
    lines.append("## 注意")
    lines.append("")
    lines.append("- lifecycle fill 仍然是研究模型，不等于交易所真实撮合；它使用采样窗口里的相反方向主动成交量近似吃队列过程。")
    lines.append("- 如果采样间隔太粗，time-to-fill 和队列消耗会被离散化；可以降低 `--sample-interval-ms` 做更细研究。")
    lines.append("- future return 为正，表示采样之后 mid-price 上涨。")
    lines.append("- maker edge label 已包含配置的 maker fee/rebate；本报告如果传 `--maker-fee-rate 0`，edge 就是不含 maker 返佣的 gross edge。")
    return "\n".join(lines) + "\n"
