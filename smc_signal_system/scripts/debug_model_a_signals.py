"""Debug script to check Model A (TrendOBFVGStrategy) raw signal generation.

This script generates raw signals from Model A strategy without backtest risk filters,
to verify that the strategy itself is producing signals.

Usage:
    python scripts/debug_model_a_signals.py --symbol ETHUSDT
    python scripts/debug_model_a_signals.py --symbol BTCUSDT
    python scripts/debug_model_a_signals.py --symbol SOLUSDT
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import (
    GlobalConfig, SymbolsConfig
)
from data_layer.binance_rest import BinanceRestClient
from data_layer.cache import DataCache
from strategy_engine.model_a import TrendOBFVGStrategy
from data_layer.time_utils import align_end_to_closed_bar


def main():
    """Main entry point for debug script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Debug Model A raw signal generation")
    parser.add_argument(
        "--symbol",
        type=str,
        default="ETHUSDT",
        help="Trading symbol to debug (default: ETHUSDT)"
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help="Maximum number of candles to process for debugging (default: all)",
    )
    args = parser.parse_args()
    
    symbol = args.symbol.upper()  # Ensure uppercase
    
    print("=" * 60)
    print("Debugging Model A raw signals...")
    print("=" * 60)
    
    # Load configurations
    config_dir = Path(__file__).parent.parent / "config"
    global_config = GlobalConfig.from_yaml(str(config_dir / "global.yaml"))
    
    print(f"Symbol: {symbol}")
    print(f"Date range: {global_config.data.start_date} to {global_config.data.end_date}")
    print(f"Aligned end (last closed bar open): {effective_end}")
    print()
    
    # Initialize data components (same as run_backtest)
    binance_cfg = getattr(global_config.data, "binance", None)
    client = BinanceRestClient(
        offline_fallback=binance_cfg.offline_fallback if binance_cfg else False,
        limit_per_call=global_config.data.limit_per_call,
    )
    cache = DataCache(cache_dir=global_config.project.data_dir)
    
    # Get primary interval from config
    primary_interval = (
        global_config.data.intervals[0]
        if getattr(global_config.data, "intervals", None) and len(global_config.data.intervals) > 0
        else "1h"
    )
    print(f"Using interval: {primary_interval}")
    
    # Load data (same as run_backtest)
    start_date = pd.to_datetime(global_config.data.start_date)
    end_date = pd.to_datetime(global_config.data.end_date)
    effective_end = align_end_to_closed_bar(end_date, primary_interval)
    if effective_end < start_date:
        effective_end = start_date
    
    df = cache.get(symbol, primary_interval, start_date, end_date)
    
    # Check if cached data covers the required range
    need_refetch = False
    if df is None or df.empty:
        need_refetch = True
        print(f"  No cache found, fetching {symbol} {primary_interval} data...")
    else:
        # Check if cached data covers the required date range
        cached_start = df.index.min() if not df.empty else None
        cached_end = df.index.max() if not df.empty else None
        
        # Check if we have enough data in the required range
        filtered_df = df[(df.index >= start_date) & (df.index <= effective_end)]
        if filtered_df.empty or len(filtered_df) < 100:  # Require at least 100 candles
            need_refetch = True
            print(f"  Cached data insufficient (got {len(filtered_df)} candles, need ~1500), fetching {symbol} {primary_interval} data...")
        elif cached_start > start_date or (cached_end and cached_end < effective_end):
            need_refetch = True
            print(f"  Cached data range ({cached_start} to {cached_end}) doesn't cover required range ({start_date} to {effective_end}), fetching...")
    
    if need_refetch:
        df = client.fetch_klines(
            symbol=symbol,
            interval=primary_interval,
            start_time=start_date,
            end_time=end_date
        )
        
        if not df.empty:
            cache.save(symbol, primary_interval, df)
    
    if df.empty:
        print(f"  Error: No data for {symbol}")
        return
    
    # Filter by date range (aligned to last closed candle)
    df = df[(df.index >= start_date) & (df.index <= effective_end)]
    
    if df.empty:
        print(f"  Error: No data in date range for {symbol}")
        return
    
    print(f"Loaded {len(df)} candles")
    print()
    
    # Load strategy config from global config
    strategy_config = {}
    if global_config.strategies and 'model_a' in global_config.strategies:
        strategy_config = global_config.strategies['model_a']
    
    # Instantiate Model A strategy (same parameters as run_backtest)
    strategy = TrendOBFVGStrategy(symbol=symbol, config=strategy_config)
    print(f"Strategy initialized: {strategy.__class__.__name__}")
    print(f"  Mode: {strategy.mode}")
    print(f"  RR target: {strategy.rr_target}")
    print(f"  Structure lookback: {strategy.structure_lookback}")
    print(f"  Trend filter enabled: {strategy.trend_filter_enabled}")
    print(f"  FVG filter enabled: {strategy.use_fvg_filter}")
    print()
    
    # Generate raw signals for all time points
    print("Generating raw signals...")
    raw_signals = []
    
    # Accumulate debug statistics across all calls
    accumulated_stats = {
        "total_candles": 0,
        "num_swings": 0,
        "num_bos_choch": 0,
        "num_order_blocks": 0,
        "num_fvg": 0,
        "num_potential_setups": 0,
        "num_failed_trend_filter": 0,
        "num_failed_fvg_filter": 0,
        "num_failed_rr_filter": 0,
        "num_signals_generated": 0,
        "skipped_ob_duplicate": 0,
    }
    
    total_bars = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        if args.max_bars is not None and i >= args.max_bars:
            print(f"[DEBUG] Reached max-bars limit ({args.max_bars}), stopping early.")
            break

        if i % 1000 == 0:
            print(f"[DEBUG] processed {i} / {total_bars} bars...")

        current_time = idx if isinstance(idx, datetime) else pd.to_datetime(idx)
        
        # Use data up to current time (same as backtest)
        current_data = df.loc[:current_time]
        
        # Generate signal
        signal = strategy.generate_signal(current_data, current_time)
        
        # Accumulate debug statistics from this call
        if hasattr(strategy, 'debug_stats'):
            for key in accumulated_stats:
                if key in strategy.debug_stats:
                    if key == "total_candles":
                        # For total_candles, use the max value (since it's the same for all calls)
                        accumulated_stats[key] = max(accumulated_stats[key], strategy.debug_stats[key])
                    elif key in ["num_swings", "num_bos_choch", "num_order_blocks", "num_fvg"]:
                        # For structure counts, use the max value (they represent current window state)
                        accumulated_stats[key] = max(accumulated_stats[key], strategy.debug_stats[key])
                    else:
                        # For filter counts and signals, accumulate
                        accumulated_stats[key] += strategy.debug_stats[key]
        
        if signal is not None:
            # Calculate RR
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price) if signal.take_profit else 0
            rr = (reward / risk) if risk > 0 else None
            
            raw_signals.append({
                'time': current_time,
                'symbol': signal.symbol,
                'side': signal.side,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'rr': rr,
                'model': signal.model,
                'trend': signal.metadata.get('trend', signal.metadata.get('ob_direction', 'unknown')),
                'ob_time': signal.metadata.get('ob_time', 'unknown'),
            })
    
    # Print debug statistics
    print()
    print("=" * 60)
    print("Debug stats:")
    print("=" * 60)
    for key, value in accumulated_stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Print summary
    print("=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Total raw signals: {len(raw_signals)}")
    
    if raw_signals:
        long_signals = [s for s in raw_signals if s['side'] == 'buy']
        short_signals = [s for s in raw_signals if s['side'] == 'sell']
        
        print(f"  Long signals:  {len(long_signals)}")
        print(f"  Short signals: {len(short_signals)}")
        print()
        
        # Print first 5 signals as samples
        print("Sample signals (first 5):")
        print("-" * 60)
        for i, sig in enumerate(raw_signals[:5], 1):
            print(f"{i}. Time: {sig['time']}")
            print(f"   Side: {sig['side']}, Entry: {sig['entry_price']:.2f}")
            print(f"   Stop Loss: {sig['stop_loss']:.2f}, Take Profit: {sig['take_profit']:.2f}")
            rr_str = f"{sig['rr']:.2f}" if sig.get('rr') is not None else "N/A"
            trend_info = sig.get('trend', 'unknown')
            print(f"   RR: {rr_str}, Direction: {trend_info}")
            print()
    else:
        print("  No signals generated.")
        print()
        print("Possible reasons:")
        print("  - SMC indicators library not installed")
        print("  - No BOS/CHOC detected in data")
        print("  - No order blocks found")
        print("  - Price not in order block range")
        print("  - No FVG confirmation")
        print()
    
    # Save to CSV if signals exist
    if raw_signals:
        output_dir = Path(global_config.project.backtest_dir) / "debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "model_a_raw_signals.csv"
        
        signals_df = pd.DataFrame(raw_signals)
        signals_df.to_csv(output_path, index=False)
        
        print(f"Raw signals saved to: {output_path}")
        print(f"  Total signals: {len(raw_signals)}")
    
    print("=" * 60)
    print("Debug complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

