"""Base classes for trading strategies."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
from abc import ABC, abstractmethod


@dataclass
class Signal:
    """
    Normalized trading signal produced by a strategy.
    """
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    model: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        """
        Convert to the plain dict format expected by BacktestRunner.
        """
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": float(self.entry_price),
            "stop_loss": float(self.stop_loss),
            "take_profit": float(self.take_profit) if self.take_profit is not None else None,
        }


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, current_time: datetime) -> Optional[Signal]:
        """
        Generate a trading signal at current_time based on historical data.

        Args:
            data: DataFrame of OHLCV up to and including current_time.
            current_time: Current timestamp
        
        Returns:
            Signal object or None if no signal
        """
        raise NotImplementedError

