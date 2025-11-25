"""Configuration loader using dataclasses and YAML."""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import yaml


@dataclass
class ProjectConfig:
    """Project-level configuration."""
    name: str
    timezone: str
    data_dir: str
    backtest_dir: str


@dataclass
class BinanceDataConfig:
    """Binance-specific data options."""
    offline_fallback: bool = False

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "BinanceDataConfig":
        data = data or {}
        return cls(offline_fallback=bool(data.get("offline_fallback", False)))


@dataclass
class DataConfig:
    """Data fetching configuration."""
    exchange: str
    start_date: str
    end_date: str
    intervals: List[str]
    limit_per_call: int
    binance: BinanceDataConfig = field(default_factory=BinanceDataConfig)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_balance: float
    fee_bps: float
    slippage_bps: float
    max_trades_per_day: int
    seed: int


@dataclass
class TelegramConfig:
    """Telegram notifier configuration."""
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TelegramConfig":
        data = data or {}
        token = os.getenv("TELEGRAM_BOT_TOKEN", data.get("bot_token", ""))
        chat_id = os.getenv("TELEGRAM_CHAT_ID", data.get("chat_id", ""))
        enabled = bool(data.get("enabled", False))
        return cls(enabled=enabled, bot_token=token, chat_id=chat_id)


@dataclass
class NotificationsConfig:
    """Notifications configuration wrapper."""
    telegram: Optional[TelegramConfig] = None


@dataclass
class GlobalConfig:
    """Global configuration container."""
    project: ProjectConfig
    data: DataConfig
    backtest: BacktestConfig
    strategies: Optional[dict] = None
    notifications: Optional[NotificationsConfig] = None

    @classmethod
    def from_yaml(cls, path: str) -> "GlobalConfig":
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        strategies = data.get('strategies', None)
        notifications_raw = data.get('notifications') or {}
        telegram_cfg_data = notifications_raw.get('telegram')
        telegram = (
            TelegramConfig.from_dict(telegram_cfg_data) if telegram_cfg_data else None
        )
        notifications = NotificationsConfig(telegram=telegram) if telegram else None

        data_section = data['data']
        binance_cfg = BinanceDataConfig.from_dict(data_section.get('binance'))

        data_config = DataConfig(
            exchange=data_section['exchange'],
            start_date=data_section['start_date'],
            end_date=data_section['end_date'],
            intervals=data_section['intervals'],
            limit_per_call=data_section.get('limit_per_call', 1500),
            binance=binance_cfg,
        )

        return cls(
            project=ProjectConfig(**data['project']),
            data=data_config,
            backtest=BacktestConfig(**data['backtest']),
            strategies=strategies,
            notifications=notifications,
        )


def load_global_config(path: str) -> GlobalConfig:
    """Convenience helper mirroring other modules."""
    return GlobalConfig.from_yaml(path)


@dataclass
class SymbolsConfig:
    """Symbols configuration."""
    symbols: List[str]
    market: str

    @classmethod
    def from_yaml(cls, path: str) -> "SymbolsConfig":
        """Load symbols configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    risk_per_trade_pct: float
    daily_loss_limit_pct: float
    consecutive_loss_limit: int
    cooldown_minutes: int
    min_signal_interval_minutes: int
    min_rr: float
    daily_loss_limit_mode: str = "day_equity"
    min_stop_distance_pct: float = 0.0005
    min_stop_distance_abs: float = 0.0

    @classmethod
    def from_yaml(cls, path: str) -> "RiskConfig":
        """Load risk configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(**data['risk'])


@dataclass
class ManualBlackout:
    """Manual blackout period configuration."""
    name: str
    start_utc: str
    end_utc: str


@dataclass
class NewsFilterConfig:
    """News filter configuration."""
    enabled: bool
    blackout_minutes_before: int
    blackout_minutes_after: int
    manual_blackouts: List[ManualBlackout]

    @classmethod
    def from_yaml(cls, path: str) -> "NewsFilterConfig":
        """Load news filter configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        blackouts = []
        if data.get('news_filter', {}).get('manual_blackouts'):
            for b in data['news_filter']['manual_blackouts']:
                blackouts.append(ManualBlackout(**b))
        
        return cls(
            enabled=data['news_filter']['enabled'],
            blackout_minutes_before=data['news_filter']['blackout_minutes_before'],
            blackout_minutes_after=data['news_filter']['blackout_minutes_after'],
            manual_blackouts=blackouts
        )



