"""Local cache for klines data to prevent redundant API calls."""

import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa


class DataCache:
    """Local cache for market data using Parquet format."""
    
    def __init__(self, cache_dir: str = "./data"):
        """
        Initialize data cache.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, symbol: str, interval: str) -> Path:
        """Get cache file path for symbol and interval."""
        return self.cache_dir / f"{symbol}_{interval}.parquet"
    
    def get(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get cached data for symbol and interval.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start datetime filter
            end_time: End datetime filter
        
        Returns:
            Cached DataFrame or None if not found
        """
        cache_path = self._get_cache_path(symbol, interval)
        
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            
            # Filter by time range if specified
            if start_time or end_time:
                if start_time:
                    df = df[df.index >= start_time]
                if end_time:
                    df = df[df.index <= end_time]
            
            return df if not df.empty else None
            
        except Exception as e:
            print(f"Warning: Failed to read cache {cache_path}: {e}")
            return None
    
    def save(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """
        Save DataFrame to cache.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            df: DataFrame to cache
        """
        if df.empty:
            return
        
        cache_path = self._get_cache_path(symbol, interval)
        
        try:
            # If cache exists, merge with existing data
            if cache_path.exists():
                existing_df = pd.read_parquet(cache_path)
                # Combine and remove duplicates
                combined = pd.concat([existing_df, df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                combined.to_parquet(cache_path, index=True)
            else:
                df.to_parquet(cache_path, index=True)
                
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_path}: {e}")
    
    def clear(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> None:
        """
        Clear cache for symbol and/or interval.
        
        Args:
            symbol: Trading pair symbol (optional, clears all if None)
            interval: Time interval (optional, clears all if None)
        """
        if symbol and interval:
            cache_path = self._get_cache_path(symbol, interval)
            if cache_path.exists():
                cache_path.unlink()
        else:
            # Clear all matching files
            pattern = f"{symbol or '*'}_{interval or '*'}.parquet"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()



