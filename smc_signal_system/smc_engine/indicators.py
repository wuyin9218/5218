"""SMC indicators wrapper for smartmoneyconcepts library."""

import pandas as pd
from typing import List, Optional
from datetime import datetime
from smc_engine.structure import (
    FVG, SwingPoint, BOSCHOC, OrderBlock, LiquidityZone,
    PreviousHighLow, Session, Retracement, SMCStructure
)

try:
    from smartmoneyconcepts import smc
except ImportError:
    # Fallback if library not installed
    smc = None


class SMCIndicators:
    """Wrapper for smartmoneyconcepts.smc functions."""
    
    def __init__(self, swing_left: int = 10, swing_right: int = 10):
        """
        Initialize SMC indicators.
        
        Args:
            swing_left: Left swing lookback period
            swing_right: Right swing lookback period
        """
        if smc is None:
            raise ImportError(
                "smartmoneyconcepts library not installed. "
                "Install with: pip install smart-money-concepts==0.0.26"
            )
        self.swing_left = swing_left
        self.swing_right = swing_right
    
    def _parse_result_to_fvgs(self, result: pd.DataFrame) -> List[FVG]:
        """Parse FVG result DataFrame to list of FVG objects."""
        fvgs = []
        if isinstance(result, pd.DataFrame) and not result.empty:
            for idx, row in result.iterrows():
                fvgs.append(FVG(
                    start_time=idx if isinstance(idx, datetime) else pd.to_datetime(idx),
                    end_time=idx if isinstance(idx, datetime) else pd.to_datetime(idx),
                    high=row.get('high', row.get('top', 0)),
                    low=row.get('low', row.get('bottom', 0)),
                    direction=row.get('direction', 'bullish'),
                    filled=row.get('filled', False)
                ))
        return fvgs
    
    def _parse_result_to_swings(self, result: pd.DataFrame) -> List[SwingPoint]:
        """Parse swing highs/lows result DataFrame to list of SwingPoint objects."""
        swings = []
        if isinstance(result, pd.DataFrame) and not result.empty:
            for idx, row in result.iterrows():
                swing_type = 'high' if 'high' in str(row.get('type', '')).lower() else 'low'
                swings.append(SwingPoint(
                    time=idx if isinstance(idx, datetime) else pd.to_datetime(idx),
                    price=row.get('price', row.get('high', row.get('low', 0))),
                    swing_type=swing_type
                ))
        return swings
    
    def _parse_result_to_bos_choch(self, result: pd.DataFrame) -> List[BOSCHOC]:
        """Parse BOS/CHOC result DataFrame to list of BOSCHOC objects."""
        bos_list = []
        if isinstance(result, pd.DataFrame) and not result.empty:
            for idx, row in result.iterrows():
                bos_type = 'BOS' if 'bos' in str(row.get('type', '')).lower() else 'CHOC'
                direction = 'bullish' if 'bull' in str(row.get('direction', '')).lower() else 'bearish'
                bos_list.append(BOSCHOC(
                    time=idx if isinstance(idx, datetime) else pd.to_datetime(idx),
                    price=row.get('price', 0),
                    bos_type=bos_type,
                    direction=direction
                ))
        return bos_list
    
    def _parse_result_to_order_blocks(self, result: pd.DataFrame) -> List[OrderBlock]:
        """Parse Order Block result DataFrame to list of OrderBlock objects."""
        obs = []
        if isinstance(result, pd.DataFrame) and not result.empty:
            for idx, row in result.iterrows():
                obs.append(OrderBlock(
                    time=idx if isinstance(idx, datetime) else pd.to_datetime(idx),
                    high=row.get('high', 0),
                    low=row.get('low', 0),
                    direction=row.get('direction', 'bullish'),
                    touched=row.get('touched', False)
                ))
        return obs
    
    def _parse_result_to_liquidity_zones(self, result: pd.DataFrame) -> List[LiquidityZone]:
        """Parse liquidity result DataFrame to list of LiquidityZone objects."""
        zones = []
        if isinstance(result, pd.DataFrame) and not result.empty:
            for idx, row in result.iterrows():
                zone_type = 'support' if 'support' in str(row.get('type', '')).lower() else 'resistance'
                zones.append(LiquidityZone(
                    time=idx if isinstance(idx, datetime) else pd.to_datetime(idx),
                    price=row.get('price', 0),
                    zone_type=zone_type,
                    strength=row.get('strength', 0.5)
                ))
        return zones
    
    def _parse_result_to_previous_highs_lows(self, result: pd.DataFrame) -> List[PreviousHighLow]:
        """Parse previous high/low result DataFrame to list of PreviousHighLow objects."""
        phls = []
        if isinstance(result, pd.DataFrame) and not result.empty:
            for idx, row in result.iterrows():
                level_type = 'high' if 'high' in str(row.get('type', '')).lower() else 'low'
                phls.append(PreviousHighLow(
                    time=idx if isinstance(idx, datetime) else pd.to_datetime(idx),
                    price=row.get('price', 0),
                    level_type=level_type
                ))
        return phls
    
    def _parse_result_to_sessions(self, result: pd.DataFrame) -> List[Session]:
        """Parse sessions result DataFrame to list of Session objects."""
        sessions = []
        if isinstance(result, pd.DataFrame) and not result.empty:
            for idx, row in result.iterrows():
                sessions.append(Session(
                    start_time=row.get('start_time', idx),
                    end_time=row.get('end_time', idx),
                    session_type=row.get('session_type', 'unknown'),
                    high=row.get('high', 0),
                    low=row.get('low', 0),
                    close=row.get('close', 0)
                ))
        return sessions
    
    def _parse_result_to_retracements(self, result: pd.DataFrame) -> List[Retracement]:
        """Parse retracements result DataFrame to list of Retracement objects."""
        rets = []
        if isinstance(result, pd.DataFrame) and not result.empty:
            for idx, row in result.iterrows():
                rets.append(Retracement(
                    time=idx if isinstance(idx, datetime) else pd.to_datetime(idx),
                    price=row.get('price', 0),
                    level=row.get('level', 0.5),
                    direction=row.get('direction', 'bullish')
                ))
        return rets
    
    def fvg(
        self,
        df: pd.DataFrame,
        join_consecutive: bool = False
    ) -> List[FVG]:
        """
        Calculate Fair Value Gaps.
        
        Args:
            df: DataFrame with OHLC data
            join_consecutive: Whether to join consecutive FVGs
        
        Returns:
            List of FVG structures
        """
        try:
            result = smc.fvg(df, join_consecutive=join_consecutive)
            return self._parse_result_to_fvgs(result)
        except Exception as e:
            print(f"Warning: FVG calculation failed: {e}")
            return []
    
    def swing_highs_lows(
        self,
        df: pd.DataFrame,
        swing_length: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate swing highs and lows.
        
        Args:
            df: DataFrame with OHLC data
            swing_length: Swing length (defaults to max of swing_left and swing_right)
        
        Returns:
            DataFrame with swing highs/lows
        """
        try:
            if swing_length is None:
                swing_length = max(int(self.swing_left), int(self.swing_right))
            result = smc.swing_highs_lows(df, swing_length=swing_length)
            return result if isinstance(result, pd.DataFrame) else pd.DataFrame()
        except Exception as e:
            print(f"Warning: Swing highs/lows calculation failed: {e}")
            return pd.DataFrame()
    
    def bos_choch(
        self,
        df: pd.DataFrame,
        swing_highs_lows: Optional[pd.DataFrame] = None
    ) -> List[BOSCHOC]:
        """
        Calculate Break of Structure and Change of Character.
        
        Args:
            df: DataFrame with OHLC data
            swing_highs_lows: Pre-calculated swing highs/lows DataFrame
        
        Returns:
            List of BOS/CHOC structures
        """
        try:
            if swing_highs_lows is None:
                swing_highs_lows = self.swing_highs_lows(df)
            result = smc.bos_choch(df, swing_highs_lows=swing_highs_lows, close_break=True)
            return self._parse_result_to_bos_choch(result)
        except Exception as e:
            print(f"Warning: BOS/CHOC calculation failed: {e}")
            return []
    
    def ob(
        self,
        df: pd.DataFrame,
        swing_highs_lows: Optional[pd.DataFrame] = None,
        close_mitigation: bool = False
    ) -> List[OrderBlock]:
        """
        Calculate Order Blocks.
        
        Args:
            df: DataFrame with OHLC data
            swing_highs_lows: Pre-calculated swing highs/lows DataFrame
            close_mitigation: Whether to use close mitigation
        
        Returns:
            List of order blocks
        """
        try:
            if swing_highs_lows is None:
                swing_highs_lows = self.swing_highs_lows(df)
            result = smc.ob(df, swing_highs_lows=swing_highs_lows, close_mitigation=close_mitigation)
            return self._parse_result_to_order_blocks(result)
        except Exception as e:
            print(f"Warning: OB calculation failed: {e}")
            return []
    
    def liquidity(
        self,
        df: pd.DataFrame,
        swing_highs_lows: Optional[pd.DataFrame] = None,
        range_percent: float = 0.01
    ) -> List[LiquidityZone]:
        """
        Calculate liquidity zones.
        
        Args:
            df: DataFrame with OHLC data
            swing_highs_lows: Pre-calculated swing highs/lows DataFrame
            range_percent: Range percent for liquidity calculation
        
        Returns:
            List of liquidity zones
        """
        try:
            if swing_highs_lows is None:
                swing_highs_lows = self.swing_highs_lows(df)
            result = smc.liquidity(df, swing_highs_lows=swing_highs_lows, range_percent=range_percent)
            return self._parse_result_to_liquidity_zones(result)
        except Exception as e:
            print(f"Warning: Liquidity calculation failed: {e}")
            return []
    
    def previous_high_low(
        self,
        df: pd.DataFrame,
        time_frame: str = "1D"
    ) -> List[PreviousHighLow]:
        """
        Calculate previous highs and lows.
        
        Args:
            df: DataFrame with OHLC data
            time_frame: Time frame for previous high/low (default: "1D")
        
        Returns:
            List of previous high/low levels
        """
        try:
            result = smc.previous_high_low(df, time_frame=time_frame)
            return self._parse_result_to_previous_highs_lows(result)
        except Exception as e:
            print(f"Warning: Previous high/low calculation failed: {e}")
            return []
    
    def sessions(
        self,
        df: pd.DataFrame,
        session: str = "New York",
        time_zone: str = "UTC"
    ) -> List[Session]:
        """
        Calculate trading sessions.
        
        Args:
            df: DataFrame with OHLC data
            session: Session name (e.g., "New York", "London", "Asian")
            time_zone: Time zone (default: "UTC")
        
        Returns:
            List of session structures
        """
        try:
            result = smc.sessions(df, session=session, time_zone=time_zone)
            return self._parse_result_to_sessions(result)
        except Exception as e:
            print(f"Warning: Sessions calculation failed: {e}")
            return []
    
    def retracements(
        self,
        df: pd.DataFrame,
        swing_highs_lows: Optional[pd.DataFrame] = None
    ) -> List[Retracement]:
        """
        Calculate retracement levels.
        
        Args:
            df: DataFrame with OHLC data
            swing_highs_lows: Pre-calculated swing highs/lows DataFrame
        
        Returns:
            List of retracement levels
        """
        try:
            if swing_highs_lows is None:
                swing_highs_lows = self.swing_highs_lows(df)
            result = smc.retracements(df, swing_highs_lows=swing_highs_lows)
            return self._parse_result_to_retracements(result)
        except Exception as e:
            print(f"Warning: Retracements calculation failed: {e}")
            return []
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        timestamp: Optional[datetime] = None
    ) -> SMCStructure:
        """
        Calculate all SMC indicators and return unified structure.
        
        This method calculates swing_highs_lows once and reuses it for other indicators.
        
        Args:
            df: DataFrame with OHLC data
            timestamp: Current timestamp (defaults to last row index)
        
        Returns:
            SMCStructure with all indicators
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if timestamp is None:
            timestamp = df.index[-1] if isinstance(df.index[-1], datetime) else pd.to_datetime(df.index[-1])
        
        # Calculate swing_highs_lows once and reuse
        swing_length = max(int(self.swing_left), int(self.swing_right))
        swing_df = None
        try:
            swing_df = self.swing_highs_lows(df, swing_length=swing_length)
        except Exception as e:
            print(f"Warning: Swing calculation failed in calculate_all: {e}")
            swing_df = pd.DataFrame()
        
        # Calculate all indicators, reusing swing_df where needed
        fvgs = []
        swing_points = []
        bos_choch_list = []
        order_blocks = []
        liquidity_zones = []
        previous_highs_lows = []
        sessions_list = []
        retracements_list = []
        
        # FVG (doesn't need swing_highs_lows)
        try:
            fvgs = self.fvg(df, join_consecutive=False)
        except Exception as e:
            print(f"Warning: FVG calculation failed in calculate_all: {e}")
        
        # Swing points (parse from swing_df)
        try:
            swing_points = self._parse_result_to_swings(swing_df) if not swing_df.empty else []
        except Exception as e:
            print(f"Warning: Swing points parsing failed in calculate_all: {e}")
        
        # BOS/CHOC (needs swing_highs_lows)
        try:
            bos_choch_list = self.bos_choch(df, swing_highs_lows=swing_df) if not swing_df.empty else []
        except Exception as e:
            print(f"Warning: BOS/CHOC calculation failed in calculate_all: {e}")
        
        # Order Blocks (needs swing_highs_lows)
        try:
            order_blocks = self.ob(df, swing_highs_lows=swing_df, close_mitigation=False) if not swing_df.empty else []
        except Exception as e:
            print(f"Warning: OB calculation failed in calculate_all: {e}")
        
        # Liquidity (needs swing_highs_lows)
        try:
            liquidity_zones = self.liquidity(df, swing_highs_lows=swing_df, range_percent=0.01) if not swing_df.empty else []
        except Exception as e:
            print(f"Warning: Liquidity calculation failed in calculate_all: {e}")
        
        # Previous high/low (doesn't need swing_highs_lows)
        try:
            previous_highs_lows = self.previous_high_low(df, time_frame="1D")
        except Exception as e:
            print(f"Warning: Previous high/low calculation failed in calculate_all: {e}")
        
        # Sessions (doesn't need swing_highs_lows)
        try:
            sessions_list = self.sessions(df, session="New York", time_zone="UTC")
        except Exception as e:
            print(f"Warning: Sessions calculation failed in calculate_all: {e}")
        
        # Retracements (needs swing_highs_lows)
        try:
            retracements_list = self.retracements(df, swing_highs_lows=swing_df) if not swing_df.empty else []
        except Exception as e:
            print(f"Warning: Retracements calculation failed in calculate_all: {e}")
        
        return SMCStructure(
            timestamp=timestamp,
            fvgs=fvgs,
            swing_points=swing_points,
            bos_choch=bos_choch_list,
            order_blocks=order_blocks,
            liquidity_zones=liquidity_zones,
            previous_highs_lows=previous_highs_lows,
            sessions=sessions_list,
            retracements=retracements_list
        )
