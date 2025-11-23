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
    
    def __init__(self):
        """Initialize SMC indicators."""
        if smc is None:
            raise ImportError(
                "smartmoneyconcepts library not installed. "
                "Install with: pip install smart-money-concepts==0.0.26"
            )
    
    def fvg(
        self,
        df: pd.DataFrame,
        lookback: int = 3
    ) -> List[FVG]:
        """
        Calculate Fair Value Gaps.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of candles to look back
        
        Returns:
            List of FVG structures
        """
        try:
            result = smc.fvg(df, lookback=lookback)
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
            
        except Exception as e:
            print(f"Warning: FVG calculation failed: {e}")
            return []
    
    def swing_highs_lows(
        self,
        df: pd.DataFrame,
        left: int = 5,
        right: int = 5
    ) -> List[SwingPoint]:
        """
        Calculate swing highs and lows.
        
        Args:
            df: DataFrame with OHLC data
            left: Left lookback period
            right: Right lookback period
        
        Returns:
            List of swing points
        """
        try:
            result = smc.swing_highs_lows(df, left=left, right=right)
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
            
        except Exception as e:
            print(f"Warning: Swing highs/lows calculation failed: {e}")
            return []
    
    def bos_choch(
        self,
        df: pd.DataFrame,
        swing_left: int = 5,
        swing_right: int = 5
    ) -> List[BOSCHOC]:
        """
        Calculate Break of Structure and Change of Character.
        
        Args:
            df: DataFrame with OHLC data
            swing_left: Left swing lookback
            swing_right: Right swing lookback
        
        Returns:
            List of BOS/CHOC structures
        """
        try:
            result = smc.bos_choch(df, swing_left=swing_left, swing_right=swing_right)
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
            
        except Exception as e:
            print(f"Warning: BOS/CHOC calculation failed: {e}")
            return []
    
    def ob(
        self,
        df: pd.DataFrame,
        swing_left: int = 5,
        swing_right: int = 5
    ) -> List[OrderBlock]:
        """
        Calculate Order Blocks.
        
        Args:
            df: DataFrame with OHLC data
            swing_left: Left swing lookback
            swing_right: Right swing lookback
        
        Returns:
            List of order blocks
        """
        try:
            result = smc.ob(df, swing_left=swing_left, swing_right=swing_right)
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
            
        except Exception as e:
            print(f"Warning: Order block calculation failed: {e}")
            return []
    
    def liquidity(
        self,
        df: pd.DataFrame,
        left: int = 5,
        right: int = 5
    ) -> List[LiquidityZone]:
        """
        Calculate liquidity zones.
        
        Args:
            df: DataFrame with OHLC data
            left: Left lookback period
            right: Right lookback period
        
        Returns:
            List of liquidity zones
        """
        try:
            result = smc.liquidity(df, left=left, right=right)
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
            
        except Exception as e:
            print(f"Warning: Liquidity calculation failed: {e}")
            return []
    
    def previous_high_low(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> List[PreviousHighLow]:
        """
        Calculate previous highs and lows.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Lookback period
        
        Returns:
            List of previous high/low levels
        """
        try:
            result = smc.previous_high_low(df, lookback=lookback)
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
            
        except Exception as e:
            print(f"Warning: Previous high/low calculation failed: {e}")
            return []
    
    def sessions(
        self,
        df: pd.DataFrame,
        session_type: str = "all"
    ) -> List[Session]:
        """
        Calculate trading sessions.
        
        Args:
            df: DataFrame with OHLC data
            session_type: Type of session ('asian', 'london', 'new_york', 'all')
        
        Returns:
            List of session structures
        """
        try:
            result = smc.sessions(df, session_type=session_type)
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
            
        except Exception as e:
            print(f"Warning: Sessions calculation failed: {e}")
            return []
    
    def retracements(
        self,
        df: pd.DataFrame,
        swing_left: int = 5,
        swing_right: int = 5
    ) -> List[Retracement]:
        """
        Calculate retracement levels.
        
        Args:
            df: DataFrame with OHLC data
            swing_left: Left swing lookback
            swing_right: Right swing lookback
        
        Returns:
            List of retracement levels
        """
        try:
            result = smc.retracements(df, swing_left=swing_left, swing_right=swing_right)
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
        
        return SMCStructure(
            timestamp=timestamp,
            fvgs=self.fvg(df),
            swing_points=self.swing_highs_lows(df),
            bos_choch=self.bos_choch(df),
            order_blocks=self.ob(df),
            liquidity_zones=self.liquidity(df),
            previous_highs_lows=self.previous_high_low(df),
            sessions=self.sessions(df),
            retracements=self.retracements(df)
        )

