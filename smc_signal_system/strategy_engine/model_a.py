"""Model A: Trend-following Order Block pullback with FVG confirmation."""

from datetime import datetime
from typing import Optional, List
import pandas as pd
from strategy_engine.base import BaseStrategy, Signal
from smc_engine.indicators import SMCIndicators
from smc_engine.structure import SMCStructure, OrderBlock, BOSCHOC, FVG


class TrendOBFVGStrategy(BaseStrategy):
    """
    Model A: Trend-following Order Block pullback with FVG confirmation.

    This is a first, simplified version. It:
    - infers trend from the latest BOS in bos_choch;
    - finds the latest order block in the trend direction;
    - requires price to close inside that OB;
    - optionally requires at least one FVG in the trend direction.
    """

    def __init__(
        self,
        symbol: str,
        rr_target: float = 2.0,
        structure_lookback: int = 200,
    ) -> None:
        super().__init__(symbol)
        self.rr_target = rr_target
        self.structure_lookback = structure_lookback
        try:
            self.indicators = SMCIndicators()
        except ImportError:
            # If smartmoneyconcepts library is not installed, set to None
            # generate_signal will handle this gracefully
            self.indicators = None

    def generate_signal(self, data: pd.DataFrame, current_time: datetime) -> Optional[Signal]:
        """
        Generate trading signal based on SMC structure.
        
        Args:
            data: Historical OHLCV data
            current_time: Current timestamp
        
        Returns:
            Signal object or None
        """
        if data.empty:
            return None

        # Ensure current_time exists in the index; if not, use last row.
        if current_time in data.index:
            row = data.loc[current_time]
        else:
            row = data.iloc[-1]
            current_time = row.name if isinstance(row.name, datetime) else pd.to_datetime(row.name)

        # Restrict to recent history for structure calculation
        window = data.loc[:current_time].tail(self.structure_lookback)
        if window.empty:
            return None

        if self.indicators is None:
            # SMC library not installed, cannot generate signals
            return None
        
        try:
            structure = self.indicators.calculate_all(window)
        except Exception:
            # If SMC indicators fail, return None
            return None

        trend = self._detect_trend(structure)
        if trend is None:
            return None

        ob = self._find_latest_ob(structure, trend)
        if ob is None:
            return None

        if not self._price_in_ob(row, ob, trend):
            return None

        if not self._has_fvg_confirmation(structure, trend):
            return None

        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if trend == "bullish":
            side = "buy"
            entry_price = close
            stop_loss = min(ob.low, low)
            risk = entry_price - stop_loss
            if risk <= 0:
                return None
            take_profit = entry_price + self.rr_target * risk
        else:
            side = "sell"
            entry_price = close
            stop_loss = max(ob.high, high)
            risk = stop_loss - entry_price
            if risk <= 0:
                return None
            take_profit = entry_price - self.rr_target * risk

        return Signal(
            symbol=self.symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            model="SMC_MODEL_A_OB_FVG",
            metadata={
                "trend": trend,
                "ob_time": ob.time.isoformat() if hasattr(ob.time, 'isoformat') else str(ob.time),
            },
        )

    # -------- helpers --------

    def _detect_trend(self, structure: SMCStructure) -> Optional[str]:
        """
        Infer trend from the latest BOS in bos_choch.
        Returns 'bullish', 'bearish', or None.
        """
        bos_list: List[BOSCHOC] = [
            b for b in structure.bos_choch
            if b.bos_type.upper() == "BOS"
        ]
        if not bos_list:
            return None

        bos_list = sorted(bos_list, key=lambda b: b.time)
        last_bos = bos_list[-1]
        direction = last_bos.direction.lower()

        if "bull" in direction:
            return "bullish"
        if "bear" in direction:
            return "bearish"
        return None

    def _find_latest_ob(self, structure: SMCStructure, trend: str) -> Optional[OrderBlock]:
        """
        Find the latest order block in the trend direction.
        """
        obs: List[OrderBlock] = [
            o for o in structure.order_blocks
            if o.direction.lower() == trend
        ]
        if not obs:
            return None
        obs = sorted(obs, key=lambda o: o.time)
        return obs[-1]

    def _price_in_ob(self, row: pd.Series, ob: OrderBlock, trend: str) -> bool:
        """
        Check if current price is inside the order block.
        """
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if trend == "bullish":
            return ob.low <= close <= ob.high
        else:
            return ob.low <= close <= ob.high

    def _has_fvg_confirmation(self, structure: SMCStructure, trend: str) -> bool:
        """
        Check if there is at least one FVG in the trend direction.
        """
        fvgs: List[FVG] = [
            f for f in structure.fvgs
            if f.direction.lower() == trend
        ]
        return len(fvgs) > 0

