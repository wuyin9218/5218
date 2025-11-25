"""Simple Telegram notification sender."""

from __future__ import annotations

from typing import Optional

import logging

import requests

from config.loader import TelegramConfig

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send text messages to a Telegram chat using a bot token."""

    API_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, config: Optional[TelegramConfig] = None) -> None:
        self.config = config
        self.last_error: Optional[str] = None

    def send_message(self, text: str) -> bool:
        """Send a Telegram message. Returns True on success (or when disabled)."""
        self.last_error = None
        if not self.config or not self.config.enabled:
            logger.info(
                "[SKIP] Telegram 通知未启用（notifications.telegram.enabled=false），已跳过发送。"
            )
            return True

        token = (self.config.bot_token or "").strip()
        chat_id = (self.config.chat_id or "").strip()
        if not token or not chat_id:
            self.last_error = "Missing bot_token or chat_id."
            logger.warning("Telegram notifier missing bot token or chat_id.")
            return False

        try:
            resp = requests.post(
                self.API_TEMPLATE.format(token=token),
                timeout=10,
                data={"chat_id": chat_id, "text": text},
            )
            if resp.status_code != 200:
                self.last_error = f"HTTP {resp.status_code}: {resp.text}"
                logger.error("Telegram API error %s: %s", resp.status_code, resp.text)
                return False
            data = resp.json()
            if not data.get("ok"):
                self.last_error = str(data)
                logger.error("Telegram API responded with error: %s", data)
                return False
            return True
        except Exception as exc:  # pragma: no cover
            self.last_error = str(exc)
            logger.error("Telegram send_message failed: %s", exc)
            return False

