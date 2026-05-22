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
    lines.append("- maker edge 图只是粗略估计，已经包含 maker 返佣，但还没有包含排队位置、撤单延迟和真实订单生命周期。")
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
    lines.append("- 第一版只采样市场状态，不模拟真实挂单、排队和撤单。")
    lines.append("- BBO 成交概率是假设模型，不等于真实实盘 fill；它用于判断哪个因子 bucket 更容易成交和更容易 toxic。")
    lines.append("- future return 为正，表示采样之后 mid-price 上涨。")
    lines.append("- maker edge label 已包含配置的 maker rebate，但只是用未来 mid 做粗略评估。")
    return "\n".join(lines) + "\n"
