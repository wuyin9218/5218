"""Binance USD-M Futures REST API client for fetching klines."""

import requests
import pandas as pd
from typing import List, Optional
from datetime import datetime
import time


class BinanceRestClient:
    """Client for fetching klines from Binance USD-M Futures API."""
    
    BASE_URL = "https://fapi.binance.com"
    
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
    
    def __init__(
        self,
        timeout: float = 10.0,
        offline_fallback: bool = False,
        limit_per_call: int = 1500,
    ):
        """
        Initialize Binance REST client.
        
        Args:
            timeout: Request timeout in seconds
            offline_fallback: If True, fall back to dummy data when API is unreachable
        """
        self.session = requests.Session()
        self.timeout = timeout
        self.offline_fallback = offline_fallback
        # Binance Futures API limit is 1500 per request; enforce upper bound
        self.limit_per_call = min(limit_per_call, 1500)
    
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
        
        request_limit = min(limit or self.limit_per_call, self.limit_per_call)
        ms_interval = self._interval_to_milliseconds(interval)
        start_ts = int(start_time.timestamp() * 1000) if start_time else None
        end_ts = int(end_time.timestamp() * 1000) if end_time else None
        next_start = start_ts
        all_klines: List[List] = []
        paginate = any(ts is not None for ts in (start_ts, end_ts)) or bool(limit)

        try:
            while True:
                params = {
                    "symbol": symbol,
                    "interval": self.INTERVAL_MAP[interval],
                    "limit": request_limit,
                }
                if next_start is not None:
                    params["startTime"] = next_start
                if end_ts is not None:
                    params["endTime"] = end_ts

                resp = self.session.get(
                    f"{self.BASE_URL}/fapi/v1/klines",
                    params=params,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                rows = resp.json()

                if not rows:
                    break

                all_klines.extend(rows)

                if limit and len(all_klines) >= limit:
                    all_klines = all_klines[:limit]
                    break

                # If we are not paginating (no explicit start/end and no limit),
                # stop after the first batch.
                if not paginate:
                    break

                if len(rows) < request_limit:
                    break

                last_open_time = rows[-1][0]
                next_start = last_open_time + (ms_interval or 1)

                if end_ts is not None and next_start >= end_ts:
                    break

                # Respect API rate limits
                time.sleep(0.1)

        except requests.RequestException:
            if self.offline_fallback:
                print(f"[WARN] Binance request failed for {symbol} {interval}, using offline dummy data instead.")
                return self._generate_dummy_klines(
                    symbol, interval, start_time, end_time, limit or self.limit_per_call
                )
            raise
        
        if not all_klines:
            # Return empty DataFrame with correct columns when no data
            return pd.DataFrame(columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
                "ignore",
            ])
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
            "ignore"
        ])
        
        # Convert types
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        for col in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
            df[col] = df[col].astype(float)
        
        df = df.set_index("open_time").sort_index()
        df.index.name = "timestamp"

        if start_time is not None:
            df = df[df.index >= start_time]
        if end_time is not None:
            df = df[df.index < end_time]
        
        return df[["open", "high", "low", "close", "volume"]]
    
    def _generate_dummy_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int,
    ) -> pd.DataFrame:
        """
        Generate deterministic dummy OHLCV data for offline backtesting.
        
        This is ONLY for development/testing when Binance API is not reachable.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_time: Start datetime (optional)
            end_time: End datetime (optional)
            limit: Maximum number of klines to generate
        
        Returns:
            DataFrame with OHLCV data
        """
        import numpy as np
        
        # If no start/end, generate a fixed-length sequence
        if start_time is None or end_time is None:
            periods = min(500, limit)
            end_time = datetime.utcnow()
            freq = "1h"
            date_index = pd.date_range(end=end_time, periods=periods, freq=freq)
        else:
            # Convert interval to pandas frequency string
            freq_map = {
                "1m": "1min",
                "3m": "3min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "1h": "1h",
                "2h": "2h",
                "4h": "4h",
                "1d": "1D",
            }
            freq = freq_map.get(interval, "1h")
            date_index = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        n = len(date_index)
        if n == 0:
            return pd.DataFrame()
        
        # Limit the number of rows
        if n > limit:
            date_index = date_index[:limit]
            n = limit
        
        # For reproducibility, use symbol+interval to construct a stable seed
        seed = (hash(symbol + interval) % (2**32))
        rng = np.random.default_rng(seed)
        
        # Generate simple random walk prices
        price = 100 + rng.standard_normal(n).cumsum()
        high = price + rng.random(n) * 2
        low = price - rng.random(n) * 2
        open_ = price + rng.standard_normal(n) * 0.5
        close = price + rng.standard_normal(n) * 0.5
        volume = rng.random(n) * 100
        
        df = pd.DataFrame(
            {
                "open_time": date_index,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                # These fields are placeholders for compatibility with existing code
                "close_time": date_index,
                "quote_asset_volume": volume,
                "number_of_trades": rng.integers(1, 100, size=n),
                "taker_buy_base_asset_volume": volume * 0.5,
                "taker_buy_quote_asset_volume": volume * 0.5,
                "ignore": 0,
            }
        )
        
        df = df.set_index("open_time")
        df.index.name = "timestamp"
        
        return df[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def _interval_to_milliseconds(interval: str) -> Optional[int]:
        """Return interval length in milliseconds."""
        mapping = {
            "1m": 60_000,
            "3m": 3 * 60_000,
            "5m": 5 * 60_000,
            "15m": 15 * 60_000,
            "30m": 30 * 60_000,
            "1h": 60 * 60_000,
            "2h": 2 * 60 * 60_000,
            "4h": 4 * 60 * 60_000,
            "6h": 6 * 60 * 60_000,
            "8h": 8 * 60 * 60_000,
            "12h": 12 * 60 * 60_000,
            "1d": 24 * 60 * 60_000,
            "3d": 3 * 24 * 60 * 60_000,
            "1w": 7 * 24 * 60 * 60_000,
            "1M": 30 * 24 * 60 * 60_000,
        }
        return mapping.get(interval)

