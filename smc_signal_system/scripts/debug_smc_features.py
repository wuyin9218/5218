"""Debug script to check SMC features calculation directly.

This script calculates SMC features (swings, BOS/CHOCH, OB, FVG) directly
using smc_engine/indicators.py to verify the indicator layer is working correctly.

Usage:
    python scripts/debug_smc_features.py
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import GlobalConfig
from data_layer.binance_rest import BinanceRestClient
from data_layer.cache import DataCache
from smc_engine.indicators import SMCIndicators
from smc_engine.structure import BOSCHOC, OrderBlock, FVG, SwingPoint


def main():
    """Main entry point for debug script."""
    print("=" * 70)
    print("Debugging SMC features calculation...")
    print("=" * 70)
    
    # Load configurations
    config_dir = Path(__file__).parent.parent / "config"
    global_config = GlobalConfig.from_yaml(str(config_dir / "global.yaml"))
    
    # Use ETHUSDT 5m as specified
    symbol = "ETHUSDT"
    interval = "5m"
    start_date = pd.to_datetime(global_config.data.start_date)
    end_date = pd.to_datetime(global_config.data.end_date)
    
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval}")
    print(f"Date range: {start_date.date()} ~ {end_date.date()}")
    print()
    
    # Initialize data components (same as run_backtest)
    binance_cfg = getattr(global_config.data, "binance", None)
    client = BinanceRestClient(
        offline_fallback=binance_cfg.offline_fallback if binance_cfg else False,
        limit_per_call=global_config.data.limit_per_call,
    )
    cache = DataCache(cache_dir=global_config.project.data_dir)
    
    # Load data
    print("Loading data...")
    df = cache.get(symbol, interval, start_date, end_date)
    
    if df is None or df.empty:
        print(f"  Fetching {symbol} {interval} data...")
        df = client.fetch_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_date,
            end_time=end_date
        )
        
        if not df.empty:
            cache.save(symbol, interval, df)
    
    if df.empty:
        print(f"  Error: No data for {symbol}")
        return
    
    # Filter by date range
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    if df.empty:
        print(f"  Error: No data in date range for {symbol}")
        return
    
    print(f"Loaded {len(df)} candles")
    print()
    
    # Initialize SMC indicators (same as TrendOBFVGStrategy)
    try:
        indicators = SMCIndicators()
    except ImportError as e:
        print(f"Error: SMC indicators not available: {e}")
        return
    
    # Use same structure_lookback as TrendOBFVGStrategy
    structure_lookback = 200
    
    print("Calculating SMC features...")
    print(f"Using structure_lookback: {structure_lookback} (same as TrendOBFVGStrategy)")
    print()
    
    # Simulate strategy behavior: calculate features for the last window (same as strategy does)
    # The strategy uses the last `structure_lookback` candles for structure calculation
    window = df.tail(structure_lookback)
    
    if window.empty:
        print("Error: Window is empty")
        return
    
    # Use calculate_all to get all features (same as TrendOBFVGStrategy)
    # This ensures we use the same parsing logic
    try:
        structure = indicators.calculate_all(window)
    except Exception as e:
        print(f"Error calculating SMC features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Extract features from structure (same as TrendOBFVGStrategy does)
    swing_points = structure.swing_points
    bos_choch_list = structure.bos_choch
    ob_list = structure.order_blocks
    fvg_list = structure.fvgs
    
    # Count features exactly as TrendOBFVGStrategy does in debug_stats
    num_swing_highs = len([s for s in swing_points if s.swing_type == 'high'])
    num_swing_lows = len([s for s in swing_points if s.swing_type == 'low'])
    num_swings_total = len(swing_points)
    num_bos_choch = len(bos_choch_list)
    num_order_blocks = len(ob_list)
    num_fvg = len(fvg_list)
    
    # Print summary (same format as TrendOBFVGStrategy debug_stats)
    print("=" * 70)
    print(f"Debugging SMC features for {symbol} {interval} ({start_date.date()} ~ {end_date.date()})")
    print("=" * 70)
    print(f"Total candles in dataset: {len(df)}")
    print(f"Window size (structure_lookback): {len(window)}")
    print()
    print("Feature counts (should match TrendOBFVGStrategy.debug_stats):")
    print(f"  Swings:          {num_swings_total} total ({num_swing_highs} highs / {num_swing_lows} lows)")
    print(f"  BOS/CHOCH:       {num_bos_choch} events")
    print(f"  Order Blocks:    {num_order_blocks}")
    print(f"  FVG:             {num_fvg}")
    print()
    print("Note: These counts are calculated using the same logic as TrendOBFVGStrategy")
    print("      (using indicators.calculate_all() with structure_lookback window)")
    print()
    
    # Print first 5 BOS/CHOCH (only valid ones with proper timestamps and prices)
    valid_bos_choch = [b for b in bos_choch_list if b.time.year > 2000 and b.price > 0]
    if valid_bos_choch:
        print("First 5 BOS/CHOCH:")
        for i, bos in enumerate(sorted(valid_bos_choch, key=lambda x: x.time)[:5], 1):
            print(f"  {i}. {bos.time} | {bos.bos_type} | {bos.direction} | Price: {bos.price:.2f}")
    else:
        print("First 5 BOS/CHOCH: (none - all entries have invalid timestamps/prices)")
    print()
    
    # Print first 5 OB (only valid ones)
    valid_ob = [ob for ob in ob_list if ob.time.year > 2000 and (ob.high > 0 or ob.low > 0)]
    if valid_ob:
        print("First 5 OB:")
        for i, ob in enumerate(sorted(valid_ob, key=lambda x: x.time)[:5], 1):
            print(f"  {i}. {ob.time} | {ob.direction} | Range: [{ob.low:.2f}, {ob.high:.2f}]")
    else:
        print("First 5 OB: (none - all entries have invalid timestamps/prices)")
    print()
    
    # Print first 5 FVG (only valid ones)
    valid_fvg = [fvg for fvg in fvg_list if fvg.start_time.year > 2000 and (fvg.high > 0 or fvg.low > 0)]
    if valid_fvg:
        print("First 5 FVG:")
        for i, fvg in enumerate(sorted(valid_fvg, key=lambda x: x.start_time)[:5], 1):
            print(f"  {i}. {fvg.start_time} | {fvg.direction} | Range: [{fvg.low:.2f}, {fvg.high:.2f}]")
    else:
        print("First 5 FVG: (none - all entries have invalid timestamps/prices)")
    print()
    
    print("=" * 70)
    print("Debug complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

