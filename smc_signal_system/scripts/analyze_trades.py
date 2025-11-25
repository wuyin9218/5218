"""Analyze trades from a backtest run and generate behavior report."""

import argparse
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd


def format_pct(count: int, total: int) -> str:
    if total == 0:
        return "0 (0.0%)"
    pct = (count / total) * 100
    return f"{count} ({pct:.1f}%)"


def analyze_trades(run_dir: Path) -> str:
    trades_path = run_dir / "trades.csv"
    if not trades_path.exists():
        raise FileNotFoundError(f"trades.csv not found in {run_dir}")

    df = pd.read_csv(trades_path)
    if df.empty:
        raise ValueError("trades.csv is empty – nothing to analyze.")

    # Parse timestamps
    for col in ["entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    if "exit_reason" not in df.columns:
        df["exit_reason"] = "unknown"

    # Holding duration in minutes
    df["holding_minutes"] = (
        (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60.0
    )

    lines = []
    md_lines = [
        "# Trade Behaviour Report",
        f"- Run directory: `{run_dir}`",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    total_trades = len(df)
    lines.append(f"Total trades: {total_trades}")
    md_lines.append(f"Total trades: **{total_trades}**")
    md_lines.append("")

    # Per-symbol statistics
    lines.append("\nPer-symbol breakdown:")
    md_lines.append("## Per-Symbol Summary")
    for symbol, grp in df.groupby("symbol"):
        symbol_total = len(grp)
        long_count = (grp["side"].str.lower() == "buy").sum()
        short_count = symbol_total - long_count
        avg_hold = grp["holding_minutes"].mean()
        max_hold = grp["holding_minutes"].max()

        exit_counts = (
            grp["exit_reason"]
            .fillna("unknown")
            .str.lower()
            .value_counts()
            .to_dict()
        )

        lines.append(
            f"  {symbol}: total={symbol_total}, "
            f"long={format_pct(long_count, symbol_total)}, "
            f"short={format_pct(short_count, symbol_total)}, "
            f"avg_hold={avg_hold:.1f}m, max_hold={max_hold:.1f}m"
        )

        md_lines.append(f"### {symbol}")
        md_lines.append(f"- Total trades: **{symbol_total}**")
        md_lines.append(
            f"- Long trades: {format_pct(long_count, symbol_total)}, "
            f"Short trades: {format_pct(short_count, symbol_total)}"
        )
        md_lines.append(
            f"- Avg holding time: {avg_hold:.1f} min, Max holding time: {max_hold:.1f} min"
        )
        md_lines.append("- Exit reasons:")
        for reason, count in exit_counts.items():
            md_lines.append(f"  - {reason}: {format_pct(count, symbol_total)}")
        md_lines.append("")

    # Daily trade counts
    df["entry_date"] = df["entry_time"].dt.date
    daily_counts = df.groupby("entry_date").size()
    min_daily = int(daily_counts.min())
    max_daily = int(daily_counts.max())
    avg_daily = daily_counts.mean()

    lines.append("\nDaily trade counts:")
    md_lines.append("## Daily Trade Counts")
    md_lines.append(
        f"Min: {min_daily}, Max: {max_daily}, Average: {avg_daily:.2f} trades/day"
    )

    dense_threshold = 5
    dense_days = daily_counts[daily_counts >= dense_threshold]

    for date, count in daily_counts.items():
        lines.append(f"  {date}: {count}")
        md_lines.append(f"- {date}: {count}")

    md_lines.append("")
    if not dense_days.empty:
        md_lines.append(
            f"### Days with >= {dense_threshold} trades (possible high-activity sessions)"
        )
        for date, count in dense_days.items():
            md_lines.append(f"- {date}: {count} trades")
    else:
        md_lines.append(
            f"### No days exceeded {dense_threshold} trades – activity evenly distributed."
        )

    md_lines.append("")
    md_lines.append("_End of report_")

    report_text = "\n".join(lines)
    markdown_report = "\n".join(md_lines)

    report_path = run_dir / "behaviour_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trades.csv and generate behavior report."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="backtests/skeleton",
        help="Path to backtest run directory (default: backtests/skeleton)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    try:
        report_text = analyze_trades(run_dir)
        print(report_text)
        print(f"\nReport saved to {run_dir / 'behaviour_report.md'}")
    except Exception as exc:
        print(f"Error analyzing trades: {exc}")


if __name__ == "__main__":
    main()

