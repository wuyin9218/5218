"""Quick utility script to verify Binance kline fetching over long ranges."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.loader import load_global_config  # noqa: E402
from data_layer.binance_rest import BinanceRestClient  # noqa: E402


def _parse_date(value: str) -> datetime:
    """Parse YYYY-MM-DD into a UTC datetime."""
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Binance klines over a date range to validate pagination."
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="5m")
    parser.add_argument(
        "--start",
        type=str,
        help="Start date YYYY-MM-DD (defaults to config.data.start_date)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date YYYY-MM-DD (defaults to config.data.end_date)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/global.yaml",
        help="Path to global config (default: config/global.yaml)",
    )
    args = parser.parse_args()

    global_cfg = load_global_config(args.config)
    binance_cfg = getattr(global_cfg.data, "binance", None)

    start_date = _parse_date(args.start) if args.start else pd.to_datetime(global_cfg.data.start_date).tz_localize("UTC")
    end_date = _parse_date(args.end) if args.end else pd.to_datetime(global_cfg.data.end_date).tz_localize("UTC")

    client = BinanceRestClient(
        offline_fallback=binance_cfg.offline_fallback if binance_cfg else False,
        limit_per_call=global_cfg.data.limit_per_call,
    )

    df = client.fetch_klines(
        symbol=args.symbol.upper(),
        interval=args.interval,
        start_time=start_date.to_pydatetime(),
        end_time=end_date.to_pydatetime(),
    )

    if df.empty:
        print("No data returned.")
        return

    print(f"Fetched {len(df)} candles for {args.symbol} {args.interval}")
    print(f"First candle: {df.index.min()}")
    print(f"Last candle:  {df.index.max()}")


if __name__ == "__main__":
    main()





