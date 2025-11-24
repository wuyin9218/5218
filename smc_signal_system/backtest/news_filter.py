"""News filter for blackout periods."""

import pandas as pd
from datetime import datetime
from typing import List
from config.loader import NewsFilterConfig


class NewsFilter:
    """Filter for news blackout periods."""
    
    def __init__(self, config: NewsFilterConfig):
        """
        Initialize news filter.
        
        Args:
            config: News filter configuration
        """
        self.config = config
        self.blackout_periods: List[tuple] = []
        
        # Parse manual blackouts
        if config.manual_blackouts:
            for blackout in config.manual_blackouts:
                start_utc = pd.to_datetime(blackout.start_utc, utc=True)
                end_utc = pd.to_datetime(blackout.end_utc, utc=True)
                
                # Apply before/after buffer
                start_with_buffer = start_utc - pd.Timedelta(minutes=config.blackout_minutes_before)
                end_with_buffer = end_utc + pd.Timedelta(minutes=config.blackout_minutes_after)
                
                self.blackout_periods.append((start_with_buffer, end_with_buffer))
    
    def is_blackout(self, ts: datetime) -> bool:
        """
        Check if timestamp is in a blackout period.
        
        Args:
            ts: Timestamp to check
        
        Returns:
            True if timestamp is in blackout period
        """
        # If filter is disabled, always return False
        if not self.config.enabled:
            return False
        
        # Convert timestamp to pandas Timestamp (assume UTC if naive)
        if isinstance(ts, pd.Timestamp):
            ts_pd = ts
        else:
            ts_pd = pd.to_datetime(ts)
        
        # If naive, assume UTC
        if ts_pd.tz is None:
            ts_pd = ts_pd.tz_localize('UTC')
        else:
            ts_pd = ts_pd.tz_convert('UTC')
        
        # Check if timestamp falls within any blackout period
        for start, end in self.blackout_periods:
            if start <= ts_pd <= end:
                return True
        
        return False



