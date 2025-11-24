"""Configuration loader using dataclasses and YAML."""

from dataclasses import dataclass
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
class DataConfig:
    """Data fetching configuration."""
    exchange: str
    start_date: str
    end_date: str
    intervals: List[str]
    limit_per_call: int


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_balance: float
    fee_bps: float
    slippage_bps: float
    max_trades_per_day: int
    seed: int


@dataclass
class GlobalConfig:
    """Global configuration container."""
    project: ProjectConfig
    data: DataConfig
    backtest: BacktestConfig
    strategies: Optional[dict] = None

    @classmethod
    def from_yaml(cls, path: str) -> "GlobalConfig":
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        strategies = data.get('strategies', None)
        
        return cls(
            project=ProjectConfig(**data['project']),
            data=DataConfig(**data['data']),
            backtest=BacktestConfig(**data['backtest']),
            strategies=strategies
        )


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



