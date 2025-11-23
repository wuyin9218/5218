"""Binance USD-M Futures REST API client for fetching klines."""

import requests
import pandas as pd
from typing import List, Optional
from datetime import datetime
import time


class BinanceRestClient:
    """Client for fetching klines from Binance USD-M Futures API."""
    
    BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
    
    # Interval mapping
    INTERVAL_MAP = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M"
    }
    
    def __init__(self, limit_per_call: int = 1500):
        """
        Initialize Binance REST client.
        
        Args:
            limit_per_call: Maximum number of klines per API call
        """
        self.limit_per_call = limit_per_call
    
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch klines from Binance USD-M Futures.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (5m, 15m, 1h, 4h, 1d)
            start_time: Start datetime (optional)
            end_time: End datetime (optional)
            limit: Maximum number of klines to fetch (optional)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, etc.
        """
        if interval not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval}")
        
        params = {
            "symbol": symbol,
            "interval": self.INTERVAL_MAP[interval],
            "limit": limit or self.limit_per_call
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        all_klines = []
        while True:
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                klines = response.json()
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # If we got fewer than requested, we're done
                if len(klines) < params["limit"]:
                    break
                
                # If limit is specified and we have enough, break
                if limit and len(all_klines) >= limit:
                    all_klines = all_klines[:limit]
                    break
                
                # Update start_time for next batch
                last_timestamp = klines[-1][0]
                params["startTime"] = last_timestamp + 1
                
                # Rate limiting
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Failed to fetch klines: {e}")
        
        if not all_klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        
        df = df.set_index("timestamp")
        df = df.sort_index()
        
        return df[["open", "high", "low", "close", "volume"]]

