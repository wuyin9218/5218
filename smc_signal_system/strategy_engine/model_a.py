"""Model A: Trend-following Order Block pullback with FVG confirmation."""

from datetime import datetime
from typing import Optional, List, Dict, Any
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
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(symbol)
        self.rr_target = rr_target
        self.structure_lookback = structure_lookback
        
        # Read configuration options
        if config is None:
            config = {}
        self.mode = config.get("mode", "strict")
        self.trend_filter_enabled = config.get("trend_filter_enabled", True)
        self.use_fvg_filter = config.get("use_fvg_filter", True)
        
        try:
            self.indicators = SMCIndicators()
        except ImportError:
            # If smartmoneyconcepts library is not installed, set to None
            # generate_signal will handle this gracefully
            self.indicators = None
        
        # Debug statistics
        self.debug_stats = {
            "total_candles": 0,
            "num_swings": 0,
            "num_bos_choch": 0,
            "num_order_blocks": 0,
            "num_fvg": 0,
            "num_potential_setups": 0,          # 满足基本结构+OB 的潜在做单机会
            "num_failed_trend_filter": 0,       # 被趋势过滤刷掉
            "num_failed_fvg_filter": 0,         # 被 FVG 过滤刷掉
            "num_failed_rr_filter": 0,          # 被 RR 要求刷掉
            "num_signals_generated": 0
        }

    def generate_signal(self, data: pd.DataFrame, current_time: datetime) -> Optional[Signal]:
        """
        Generate trading signal based on SMC structure.
        
        Args:
            data: Historical OHLCV data
            current_time: Current timestamp
        
        Returns:
            Signal object or None
        """
        # Reset debug statistics
        self.debug_stats = {
            "total_candles": 0,
            "num_swings": 0,
            "num_bos_choch": 0,
            "num_order_blocks": 0,
            "num_fvg": 0,
            "num_potential_setups": 0,
            "num_failed_trend_filter": 0,
            "num_failed_fvg_filter": 0,
            "num_failed_rr_filter": 0,
            "num_signals_generated": 0
        }
        
        if data.empty:
            return None

        # Ensure current_time exists in the index; if not, use last row.
        if current_time in data.index:
            row = data.loc[current_time]
        else:
            row = data.iloc[-1]
            current_time = row.name if isinstance(row.name, datetime) else pd.to_datetime(row.name)

        # Set total candles count (use full data length, not window)
        self.debug_stats["total_candles"] = len(data)

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

        # Count SMC structure elements
        self.debug_stats["num_swings"] = len(structure.swing_points) if hasattr(structure, 'swing_points') else 0
        self.debug_stats["num_bos_choch"] = len(structure.bos_choch) if hasattr(structure, 'bos_choch') else 0
        self.debug_stats["num_order_blocks"] = len(structure.order_blocks) if hasattr(structure, 'order_blocks') else 0
        self.debug_stats["num_fvg"] = len(structure.fvgs) if hasattr(structure, 'fvgs') else 0

        # Route to appropriate mode
        if self.mode == "baseline":
            return self._generate_baseline_signal(data, window, structure, current_time, row)
        else:
            return self._generate_strict_signal(data, window, structure, current_time, row)

    # -------- mode implementations --------
    
    def _generate_baseline_signal(
        self,
        data: pd.DataFrame,
        window: pd.DataFrame,
        structure: SMCStructure,
        current_time: datetime,
        row: pd.Series
    ) -> Optional[Signal]:
        """
        Baseline mode: Simple OB pullback logic without trend/FVG filters.
        
        Args:
            data: Full historical data
            window: Window used for structure calculation
            structure: SMC structure from calculate_all
            current_time: Current timestamp
            row: Current row data
        
        Returns:
            Signal or None
        """
        # In baseline mode, trend filter is not used
        self.debug_stats["num_failed_trend_filter"] = 0
        
        # Get all order blocks, filter out invalid ones (only check prices, not timestamps)
        obs = structure.order_blocks
        if not obs:
            return None
        
        # Get current price
        current_price = float(row["close"])
        current_high = float(row["high"])
        current_low = float(row["low"])
        
        # Filter valid OBs (only check prices, timestamps may be invalid but we'll use them anyway)
        valid_obs = [
            ob for ob in obs
            if ob.high > 0 and ob.low > 0 and ob.high > ob.low
        ]
        
        # If no valid OBs with proper prices, use a fallback: create a simple OB from recent price action
        if not valid_obs:
            # Fallback: use recent high/low as OB
            if len(window) >= 10:
                recent_high = float(window["high"].tail(10).max())
                recent_low = float(window["low"].tail(10).min())
                if recent_high > recent_low and current_price > 0:
                    # Create a synthetic bullish OB if price is near recent low
                    if current_price <= recent_low * 1.02:
                        ob_low = recent_low
                        ob_high = recent_low * 1.01
                        direction = "bullish"
                    # Create a synthetic bearish OB if price is near recent high
                    elif current_price >= recent_high * 0.98:
                        ob_low = recent_high * 0.99
                        ob_high = recent_high
                        direction = "bearish"
                    else:
                        return None
                else:
                    return None
            else:
                return None
        else:
            # Sort OBs by time (most recent first), or by index if time is invalid
            try:
                sorted_obs = sorted(valid_obs, key=lambda o: o.time, reverse=True)
            except:
                # If time comparison fails, just use the list as-is
                sorted_obs = valid_obs
            
            # Find the first OB that current price is touching
            ob_found = False
            for ob in sorted_obs:
                ob_low = float(ob.low)
                ob_high = float(ob.high)
                direction = ob.direction.lower()
                
                # Check if current price is in OB range (price touches OB)
                # Use a more lenient check: price overlaps with OB range
                price_touches_ob = (
                    (current_high >= ob_low) and (current_low <= ob_high)
                )
                
                if price_touches_ob:
                    ob_found = True
                    break
            
            if not ob_found:
                return None
            
        # Found a potential setup
        self.debug_stats["num_potential_setups"] += 1
        
        # Calculate entry, stop loss, and take profit
        if direction == "bullish":
            side = "buy"
            entry_price = max(ob_low, current_price)
            stop_loss = ob_low * 0.999  # Slightly below OB low
            risk = entry_price - stop_loss
            if risk <= 0:
                self.debug_stats["num_failed_rr_filter"] += 1
                return None
            take_profit = entry_price + risk * self.rr_target
        else:  # bearish
            side = "sell"
            entry_price = min(ob_high, current_price)
            stop_loss = ob_high * 1.001  # Slightly above OB high
            risk = stop_loss - entry_price
            if risk <= 0:
                self.debug_stats["num_failed_rr_filter"] += 1
                return None
            take_profit = entry_price - risk * self.rr_target
        
        # Generate signal
        signal = Signal(
            symbol=self.symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            model="SMC_MODEL_A_BASELINE",
            metadata={
                "mode": "baseline",
                "ob_direction": direction,
            },
        )
        
        self.debug_stats["num_signals_generated"] += 1
        return signal
    
    def _generate_strict_signal(
        self,
        data: pd.DataFrame,
        window: pd.DataFrame,
        structure: SMCStructure,
        current_time: datetime,
        row: pd.Series
    ) -> Optional[Signal]:
        """
        Strict mode: Full trend + OB + FVG logic (original implementation).
        
        Args:
            data: Full historical data
            window: Window used for structure calculation
            structure: SMC structure from calculate_all
            current_time: Current timestamp
            row: Current row data
        
        Returns:
            Signal or None
        """
        # Trend detection and filtering (can be disabled via config)
        trend = self._detect_trend(structure) if self.trend_filter_enabled else None
        
        if self.trend_filter_enabled:
            if trend is None:
                self.debug_stats["num_failed_trend_filter"] += 1
                return None
        else:
            # When trend filter is disabled, we still need a trend for OB selection
            # Use a default trend or try to detect one without filtering
            if trend is None:
                # Try to find any OB regardless of trend direction
                trend = "bullish"  # Default, will be overridden by OB direction
        
        # Find order block - if trend filter is disabled, find any OB
        if self.trend_filter_enabled:
            ob = self._find_latest_ob(structure, trend)
            if ob is None:
                self.debug_stats["num_failed_trend_filter"] += 1
                return None
        else:
            # When trend filter is disabled, find the latest OB regardless of direction
            if structure.order_blocks:
                ob = sorted(structure.order_blocks, key=lambda o: o.time)[-1]
                trend = ob.direction.lower()  # Use OB direction as trend
            else:
                # No OB found, but don't count as trend filter failure
                return None

        if not self._price_in_ob(row, ob, trend):
            # Price not in OB, but we have structure + OB, so this is a potential setup
            # (just not triggered yet) - don't count as failed, just not ready
            return None

        # We have trend + OB + price in OB = potential setup
        self.debug_stats["num_potential_setups"] += 1

        # FVG confirmation check (can be disabled via config)
        if self.use_fvg_filter:
            if not self._has_fvg_confirmation(structure, trend):
                self.debug_stats["num_failed_fvg_filter"] += 1
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
                self.debug_stats["num_failed_rr_filter"] += 1
                return None
            take_profit = entry_price + self.rr_target * risk
        else:
            side = "sell"
            entry_price = close
            stop_loss = max(ob.high, high)
            risk = stop_loss - entry_price
            if risk <= 0:
                self.debug_stats["num_failed_rr_filter"] += 1
                return None
            take_profit = entry_price - self.rr_target * risk

        # Check if RR meets minimum requirement (if we have a min_rr config)
        # Note: This is a basic check, actual RR filtering happens in BacktestRunner
        # But we can still count it here for debug purposes
        reward = abs(take_profit - entry_price)
        rr = reward / risk if risk > 0 else 0
        # We don't have access to min_rr here, so we'll just count successful signals
        # The actual RR filtering happens in BacktestRunner

        signal = Signal(
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
        
        self.debug_stats["num_signals_generated"] += 1
        return signal

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

