"""Generate preview push messages from backtest trades."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.loader import TelegramConfig, load_global_config
from io_layer.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    # Older Python versions or environments that don't support reconfigure
    pass


def load_configs(global_config_path: Path) -> dict:
    """Load global and risk configuration needed for formatting."""
    with open(global_config_path, "r", encoding="utf-8") as f:
        global_cfg = yaml.safe_load(f)

    risk_path = global_config_path.parent / "risk.yaml"
    with open(risk_path, "r", encoding="utf-8") as f:
        risk_cfg = yaml.safe_load(f)["risk"]

    data_cfg = global_cfg.get("data", {})
    intervals = data_cfg.get("intervals") or []
    primary_interval = intervals[0] if intervals else "5m"
    strategy_cfg = (global_cfg.get("strategies") or {}).get("model_a", {}) or {}

    strategy_rr_target = (
        strategy_cfg.get("rr_target")
        or strategy_cfg.get("target_rr")
        or strategy_cfg.get("min_rr")
    )
    try:
        strategy_rr_target = float(strategy_rr_target)
    except (TypeError, ValueError):
        strategy_rr_target = 2.0

    return {
        "timeframe": primary_interval,
        "initial_balance": float(global_cfg.get("backtest", {}).get("initial_balance", 10000)),
        "risk_per_trade_pct": float(risk_cfg.get("risk_per_trade_pct", 1.0)),
        "strategy_rr_target": strategy_rr_target,
    }


def load_trades(run_dir: Path) -> pd.DataFrame:
    trades_path = run_dir / "trades.csv"
    if not trades_path.exists():
        raise FileNotFoundError(f"trades.csv not found in {run_dir}")

    df = pd.read_csv(trades_path)
    if df.empty:
        raise ValueError("trades.csv is empty.")

    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    if "exit_reason" not in df.columns:
        df["exit_reason"] = "unknown"

    df["exit_reason"] = df["exit_reason"].fillna("unknown").str.lower()
    df = df.sort_values("entry_time")
    # Keep only trades with exit reason
    df = df[df["exit_reason"] != ""]
    return df


def format_message(row: pd.Series, cfg: dict) -> str:
    symbol = row["symbol"]
    base_symbol = symbol.replace("USDT", "")
    side = str(row["side"]).lower()
    side_text = "多头" if side == "buy" else "空头"
    timeframe = cfg["timeframe"]

    entry_time = row["entry_time"]
    entry_time_str = entry_time.strftime("%Y-%m-%d %H:%M") if not pd.isna(entry_time) else "未知时间"

    entry_price = float(row.get("entry_price", 0))
    pnl = float(row.get("pnl", 0))
    hold_minutes_td = (
        row["exit_time"] - row["entry_time"]
        if pd.notna(row["exit_time"]) and pd.notna(row["entry_time"])
        else pd.Timedelta(0)
    )
    hold_minutes = int(round(hold_minutes_td.total_seconds() / 60)) if hold_minutes_td else 0

    # Planned RR (from trades or config)
    planned_rr = None
    for col in ("planned_rr", "target_rr", "rr_plan", "rr_target"):
        if col in row:
            try:
                planned_rr = float(row[col])
                break
            except (TypeError, ValueError):
                continue
    if planned_rr is None:
        planned_rr = cfg.get("strategy_rr_target", 2.0)

    # Actual RR from trade record or fallback to pnl/risk
    actual_r = None
    for col in ("actual_rr", "realized_rr", "rr", "result_rr"):
        if col in row:
            try:
                candidate = float(row.get(col))
            except (TypeError, ValueError):
                continue
            else:
                actual_r = candidate
                break
    if actual_r is None or pd.isna(actual_r) or (abs(actual_r) < 1e-6 and abs(pnl) > 1e-6):
        risk_amount = cfg["initial_balance"] * (cfg["risk_per_trade_pct"] / 100.0)
        actual_r = (pnl / risk_amount) if risk_amount > 0 else 0.0

    epsilon = 1e-6
    if actual_r > epsilon:
        result_label = "止盈"
    elif actual_r < -epsilon:
        result_label = "止损"
    else:
        result_label = "平手"

    planned_rr_str = f"{planned_rr:.2f}"
    actual_r_str = f"{actual_r:+.2f}"
    pnl_str = f"{pnl:+.2f}"

    stop_loss_val = (
        row.get("stop_loss_price")
        if "stop_loss_price" in row
        else row.get("stop_loss")
    )
    take_profit_val = (
        row.get("take_profit_price")
        if "take_profit_price" in row
        else row.get("take_profit")
    )

    parts = [
        f"[{base_symbol} {timeframe} {side_text}] {entry_time_str} 开仓 @ {entry_price:.2f}",
    ]

    def _has_value(value) -> bool:
        try:
            return value is not None and not pd.isna(value)
        except Exception:
            return False

    if _has_value(stop_loss_val):
        parts.append(f"止损 {float(stop_loss_val):.2f}")
    if _has_value(take_profit_val):
        parts.append(f"目标 {float(take_profit_val):.2f}")

    parts.append(f"RR计划={planned_rr_str}")
    parts.append(f"持仓 {hold_minutes} 分钟")
    parts.append(f"结果: {result_label} {actual_r_str}R ({pnl_str} USDT)")

    message = "，".join(parts)
    return message


def generate_messages(run_dir: Path, limit: int, global_config_path: Path) -> List[dict]:
    cfg = load_configs(global_config_path)
    df = load_trades(run_dir)
    if df.empty:
        return []
    if limit:
        df = df.head(limit)

    messages = []
    for _, row in df.iterrows():
        message = format_message(row, cfg)
        trade_id = f"{row['symbol']}_{row['entry_time']}"
        messages.append({"trade_id": trade_id, "message": message})
    return messages


def main():
    parser = argparse.ArgumentParser(description="Preview push messages from trades.csv")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="backtests/skeleton",
        help="Path to backtest run directory (default: backtests/skeleton)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of trades to preview (default: 50)",
    )
    parser.add_argument(
        "--global-config",
        type=str,
        default="config/global.yaml",
        help="Path to global config for formatting/notifications (default: config/global.yaml)",
    )
    parser.add_argument(
        "--send-telegram",
        action="store_true",
        help="Send each preview message to Telegram using notifications config.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    global_config_path = Path(args.global_config)
    try:
        messages = generate_messages(run_dir, args.limit, global_config_path)
    except Exception as exc:
        print(f"Error generating messages: {exc}")
        return

    if not messages:
        print("No trades available for preview.")
        return

    notifier: Optional[TelegramNotifier] = None
    if args.send_telegram:
        if not global_config_path.exists():
            logger.warning("无法找到全局配置 %s，跳过 Telegram 发送。", global_config_path)
        else:
            global_cfg_obj = load_global_config(str(global_config_path))
            notifications = getattr(global_cfg_obj, "notifications", None)
            telegram_cfg: Optional[TelegramConfig] = (
                getattr(notifications, "telegram", None) if notifications else None
            )
            if telegram_cfg is None:
                logger.warning(
                    "[WARN] 配置中未找到 notifications.telegram，跳过 Telegram 发送。"
                )
            else:
                notifier = TelegramNotifier(telegram_cfg)

    # Print to console
    for msg in messages:
        print(msg["message"])
        if notifier is not None:
            sent = notifier.send_message(msg["message"])
            if not sent:
                print("[WARN] Telegram 发送失败，详见日志。")

    # Save to CSV
    output_path = run_dir / "push_messages.csv"
    pd.DataFrame(messages).to_csv(output_path, index=False)
    print(f"\nPreview messages saved to {output_path}")


if __name__ == "__main__":
    main()

