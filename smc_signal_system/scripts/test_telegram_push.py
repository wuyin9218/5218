"""Send a test Telegram message using the project configuration."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.loader import TelegramConfig, load_global_config  # noqa: E402
from io_layer.telegram_notifier import TelegramNotifier  # noqa: E402


def load_telegram_config(config_path: Path) -> TelegramConfig | None:
    """Return the Telegram configuration from the global config file."""
    global_config = load_global_config(str(config_path))
    notifications = getattr(global_config, "notifications", None)
    telegram_cfg = getattr(notifications, "telegram", None) if notifications else None
    return telegram_cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a test Telegram message using project configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/global.yaml",
        help="Path to the global configuration file (default: config/global.yaml).",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="SMC 信号系统测试消息",
        help="Message text to send (default: 'SMC 信号系统测试消息').",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        raise SystemExit(1)

    telegram_cfg = load_telegram_config(config_path)
    if not telegram_cfg:
        print("[SKIP] 未在配置中找到 notifications.telegram，已跳过发送。")
        return

    if not telegram_cfg.enabled:
        print("[SKIP] Telegram 通知未启用（notifications.telegram.enabled=false），已跳过发送。")
        return

    notifier = TelegramNotifier(telegram_cfg)
    success = notifier.send_message(args.message)

    if success:
        print("[OK] Telegram message sent")
    else:
        error_msg = notifier.last_error or "unknown error"
        print(f"[ERROR] Telegram message failed: {error_msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

