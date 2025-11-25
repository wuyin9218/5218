"""Utilities for aligning time ranges to completed candles."""

from datetime import datetime, timedelta
from typing import Union

import pandas as pd


def interval_to_timedelta(interval: str) -> timedelta:
    """
    Convert an interval string like '5m', '1h', or '1d' into a timedelta.

    Args:
        interval: Interval string using Binance-style suffixes.

    Returns:
        timedelta representing the interval length.

    Raises:
        ValueError: If the interval cannot be parsed.
    """
    try:
        delta = pd.to_timedelta(interval)
    except ValueError as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Unsupported interval: {interval}") from exc

    if delta <= pd.Timedelta(0):
        raise ValueError(f"Interval must be positive, got {interval}")

    return timedelta(seconds=delta.total_seconds())


def align_end_to_closed_bar(end_dt: Union[datetime, pd.Timestamp], interval: str) -> datetime:
    """
    Align a desired end timestamp to the open time of the last fully closed candle.

    Rules:
        - If `end_dt` lands exactly on a candle boundary, treat it as the *close* of that candle
          and return the previous candle's open time.
        - If `end_dt` falls inside an in-progress candle, floor it to the candle's open time
          and return the previous candle's open time.

    Args:
        end_dt: Desired inclusive end timestamp.
        interval: Candle interval string compatible with `interval_to_timedelta`.

    Returns:
        datetime representing the open time of the last closed candle.
    """
    delta = interval_to_timedelta(interval)
    end_ts = pd.Timestamp(end_dt)

    floored = end_ts.floor(freq=interval)
    aligned_open = floored.to_pydatetime() - delta
    return aligned_open


