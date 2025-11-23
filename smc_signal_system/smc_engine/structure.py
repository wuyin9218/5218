"""Unified structure output dataclasses for SMC signals."""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
import pandas as pd


@dataclass
class FVG:
    """Fair Value Gap structure."""
    start_time: datetime
    end_time: datetime
    high: float
    low: float
    direction: str  # 'bullish' or 'bearish'
    filled: bool = False


@dataclass
class SwingPoint:
    """Swing high or swing low point."""
    time: datetime
    price: float
    swing_type: str  # 'high' or 'low'


@dataclass
class BOSCHOC:
    """Break of Structure / Change of Character."""
    time: datetime
    price: float
    bos_type: str  # 'BOS' or 'CHOC'
    direction: str  # 'bullish' or 'bearish'


@dataclass
class OrderBlock:
    """Order block structure."""
    time: datetime
    high: float
    low: float
    direction: str  # 'bullish' or 'bearish'
    touched: bool = False


@dataclass
class LiquidityZone:
    """Liquidity zone (support/resistance)."""
    time: datetime
    price: float
    zone_type: str  # 'support' or 'resistance'
    strength: float  # 0.0 to 1.0


@dataclass
class PreviousHighLow:
    """Previous high or low level."""
    time: datetime
    price: float
    level_type: str  # 'high' or 'low'


@dataclass
class Session:
    """Trading session information."""
    start_time: datetime
    end_time: datetime
    session_type: str  # 'asian', 'london', 'new_york', etc.
    high: float
    low: float
    close: float


@dataclass
class Retracement:
    """Retracement level."""
    time: datetime
    price: float
    level: float  # 0.236, 0.382, 0.5, 0.618, 0.786
    direction: str  # 'bullish' or 'bearish'


@dataclass
class SMCStructure:
    """Unified SMC structure output containing all indicators."""
    timestamp: datetime
    fvgs: List[FVG]
    swing_points: List[SwingPoint]
    bos_choch: List[BOSCHOC]
    order_blocks: List[OrderBlock]
    liquidity_zones: List[LiquidityZone]
    previous_highs_lows: List[PreviousHighLow]
    sessions: List[Session]
    retracements: List[Retracement]
    
    def to_dict(self) -> dict:
        """Convert structure to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "fvgs": [self._fvg_to_dict(f) for f in self.fvgs],
            "swing_points": [self._swing_to_dict(s) for s in self.swing_points],
            "bos_choch": [self._bos_to_dict(b) for b in self.bos_choch],
            "order_blocks": [self._ob_to_dict(o) for o in self.order_blocks],
            "liquidity_zones": [self._liq_to_dict(l) for l in self.liquidity_zones],
            "previous_highs_lows": [self._phl_to_dict(p) for p in self.previous_highs_lows],
            "sessions": [self._session_to_dict(s) for s in self.sessions],
            "retracements": [self._ret_to_dict(r) for r in self.retracements]
        }
    
    @staticmethod
    def _fvg_to_dict(fvg: FVG) -> dict:
        return {
            "start_time": fvg.start_time.isoformat(),
            "end_time": fvg.end_time.isoformat(),
            "high": fvg.high,
            "low": fvg.low,
            "direction": fvg.direction,
            "filled": fvg.filled
        }
    
    @staticmethod
    def _swing_to_dict(swing: SwingPoint) -> dict:
        return {
            "time": swing.time.isoformat(),
            "price": swing.price,
            "swing_type": swing.swing_type
        }
    
    @staticmethod
    def _bos_to_dict(bos: BOSCHOC) -> dict:
        return {
            "time": bos.time.isoformat(),
            "price": bos.price,
            "bos_type": bos.bos_type,
            "direction": bos.direction
        }
    
    @staticmethod
    def _ob_to_dict(ob: OrderBlock) -> dict:
        return {
            "time": ob.time.isoformat(),
            "high": ob.high,
            "low": ob.low,
            "direction": ob.direction,
            "touched": ob.touched
        }
    
    @staticmethod
    def _liq_to_dict(liq: LiquidityZone) -> dict:
        return {
            "time": liq.time.isoformat(),
            "price": liq.price,
            "zone_type": liq.zone_type,
            "strength": liq.strength
        }
    
    @staticmethod
    def _phl_to_dict(phl: PreviousHighLow) -> dict:
        return {
            "time": phl.time.isoformat(),
            "price": phl.price,
            "level_type": phl.level_type
        }
    
    @staticmethod
    def _session_to_dict(session: Session) -> dict:
        return {
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat(),
            "session_type": session.session_type,
            "high": session.high,
            "low": session.low,
            "close": session.close
        }
    
    @staticmethod
    def _ret_to_dict(ret: Retracement) -> dict:
        return {
            "time": ret.time.isoformat(),
            "price": ret.price,
            "level": ret.level,
            "direction": ret.direction
        }

