"""Command-line script to run backtest and output results."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import (
    GlobalConfig,
    SymbolsConfig,
    RiskConfig,
    NewsFilterConfig,
)
from data_layer.binance_rest import BinanceRestClient
from data_layer.cache import DataCache
from backtest.runner import BacktestRunner
from backtest.slippage_fee import SlippageFeeModel
from backtest.metrics import Trade
from backtest.news_filter import NewsFilter
from strategy_engine.model_a import TrendOBFVGStrategy
from io_layer.telegram_notifier import TelegramNotifier
from data_layer.time_utils import align_end_to_closed_bar


def dummy_signal_generator(df: pd.DataFrame, current_time: datetime) -> dict:
    """
    Dummy signal generator for skeleton backtest.
    
    This is a placeholder that generates no signals.
    In a real implementation, this would use SMC indicators to generate signals.
    
    Args:
        df: Historical data up to current_time
        current_time: Current timestamp
    
    Returns:
        Signal dictionary or None
    """
    # Placeholder: return None (no signals)
    # In real implementation, this would analyze SMC structure and generate signals
    return None


def run_backtest(
    global_config: GlobalConfig,
    symbols_config: SymbolsConfig,
    risk_config: RiskConfig,
    news_filter_config: NewsFilterConfig,
    telegram_notifier: Optional[TelegramNotifier] = None,
) -> Tuple[list, dict, dict]:
    """
    Run backtest for all symbols.
    
    Args:
        global_config: Global configuration
        symbols_config: Symbols configuration
        risk_config: Risk configuration
        news_filter_config: News filter configuration
    
    Returns:
        Tuple of (all_trades, combined_metrics, signal_stats_dict)
    """
    # Initialize signal statistics dictionary
    signal_stats_dict = {}
    
    # Initialize components
    binance_cfg = getattr(global_config.data, "binance", None)
    client = BinanceRestClient(
        offline_fallback=binance_cfg.offline_fallback if binance_cfg else False,
        limit_per_call=global_config.data.limit_per_call,
    )
    cache = DataCache(cache_dir=global_config.project.data_dir)
    fee_model = SlippageFeeModel(
        fee_bps=global_config.backtest.fee_bps,
        slippage_bps=global_config.backtest.slippage_bps
    )
    news_filter = NewsFilter(news_filter_config)
    
    all_trades = []
    all_metrics = []
    
    # Process each symbol
    for symbol in symbols_config.symbols:
        print(f"Processing {symbol}...")
        
        # Fetch data for each interval (use primary interval for backtest)
        # Get primary interval from config
        primary_interval = (
            global_config.data.intervals[0]
            if getattr(global_config.data, "intervals", None) and len(global_config.data.intervals) > 0
            else "1h"
        )
        
        # Try to get from cache first
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
            print(f"  Fetching {symbol} {primary_interval} data...")
        else:
            # Check if cached data covers the required date range
            cached_start = df.index.min() if not df.empty else None
            cached_end = df.index.max() if not df.empty else None
            
            # Check if we have enough data in the required range
            filtered_df = df[(df.index >= start_date) & (df.index <= effective_end)]
            if filtered_df.empty or len(filtered_df) < 100:  # Require at least 100 candles
                need_refetch = True
                print(f"  Cached data insufficient (got {len(filtered_df)} candles), fetching {symbol} {primary_interval} data...")
            elif cached_start > start_date or (cached_end and cached_end < effective_end):
                need_refetch = True
                print(f"  Cached data range doesn't cover required range, fetching {symbol} {primary_interval} data...")
        
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
            print(f"  Warning: No data for {symbol}")
            continue
        
        # Filter by date range (aligned to last closed candle)
        df = df[(df.index >= start_date) & (df.index <= effective_end)]
        
        if df.empty:
            print(f"  Warning: No data in date range for {symbol}")
            continue
        
        # Run backtest
        runner = BacktestRunner(
            global_config,
            risk_config,
            fee_model,
            news_filter=news_filter,
            telegram_notifier=telegram_notifier,
        )
        
        # Load strategy config from global config
        strategy_config = {}
        if global_config.strategies and 'model_a' in global_config.strategies:
            strategy_config = global_config.strategies['model_a']
        
        # Instantiate strategy for this symbol
        strategy = TrendOBFVGStrategy(symbol=symbol, config=strategy_config)
        
        def signal_gen(data, time):
            sig = strategy.generate_signal(data, time)
            if sig is None:
                return None
            return sig.to_dict()
        
        trades, metrics, signal_stats = runner.run(df, signal_gen)
        
        all_trades.extend(trades)
        all_metrics.append({
            'symbol': symbol,
            **metrics
        })
        
        # Store signal statistics per symbol
        signal_stats_dict[symbol] = signal_stats
        
        print(f"  Completed {symbol}: {metrics.get('total_trades', 0)} trades")
    
    # Combine metrics
    if all_metrics:
        combined_metrics = {
            'total_trades': sum(m.get('total_trades', 0) for m in all_metrics),
            'winning_trades': sum(m.get('winning_trades', 0) for m in all_metrics),
            'losing_trades': sum(m.get('losing_trades', 0) for m in all_metrics),
            'total_pnl': sum(m.get('total_pnl', 0) for m in all_metrics),
            'total_fees': sum(m.get('total_fees', 0) for m in all_metrics),
            'net_pnl': sum(m.get('net_pnl', 0) for m in all_metrics),
        }
        
        if combined_metrics['total_trades'] > 0:
            combined_metrics['win_rate'] = (
                combined_metrics['winning_trades'] / combined_metrics['total_trades'] * 100
            )
        else:
            combined_metrics['win_rate'] = 0.0
        
        combined_metrics['symbols'] = all_metrics
    else:
        combined_metrics = {}
    
    return all_trades, combined_metrics, signal_stats_dict


def save_results(
    trades: list,
    metrics: dict,
    output_dir: Path,
    phase: str = "skeleton",
    signal_stats: Optional[dict] = None
):
    """
    Save backtest results to files.
    
    Args:
        trades: List of trade records
        metrics: Metrics dictionary
        output_dir: Output directory
        phase: Phase name (e.g., "skeleton")
        signal_stats: Signal pipeline statistics dictionary
    """
    output_dir = output_dir / phase
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary.json
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Save trades.csv
    if trades:
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'symbol': t.symbol,
            'side': t.side,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'fees': t.fees,
            'rr': t.rr,
            'exit_reason': getattr(t, 'exit_reason', 'unknown')
        } for t in trades])
        
        trades_path = output_dir / "trades.csv"
        trades_df.to_csv(trades_path, index=False)
    else:
        # Create empty CSV with headers
        trades_path = output_dir / "trades.csv"
        empty_df = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'symbol', 'side',
            'entry_price', 'exit_price', 'quantity',
            'pnl', 'pnl_pct', 'fees', 'rr', 'exit_reason'
        ])
        empty_df.to_csv(trades_path, index=False)
    
    # Save signal_stats.json
    if signal_stats:
        stats_path = output_dir / "signal_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(signal_stats, f, indent=2, ensure_ascii=False)
        print(f"  Signal stats: {stats_path}")
    
    print(f"\nResults saved to:")
    print(f"  Summary: {summary_path}")
    print(f"  Trades: {trades_path}")


def main():
    """Main entry point for backtest script."""
    parser = argparse.ArgumentParser(description="Run SMC signal system backtest")
    parser.add_argument(
        "--config",
        type=str,
        default="config/global.yaml",
        help="Path to global config YAML file"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="config/symbols.yaml",
        help="Path to symbols config YAML file"
    )
    parser.add_argument(
        "--risk",
        type=str,
        default="config/risk.yaml",
        help="Path to risk config YAML file"
    )
    parser.add_argument(
        "--news-filter",
        type=str,
        default="config/news_filter.yaml",
        help="Path to news filter config YAML file"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="skeleton",
        help="Phase name for output directory"
    )
    
    args = parser.parse_args()
    
    # Load configurations
    try:
        global_config = GlobalConfig.from_yaml(args.config)
        symbols_config = SymbolsConfig.from_yaml(args.symbols)
        risk_config = RiskConfig.from_yaml(args.risk)
        news_filter_config = NewsFilterConfig.from_yaml(args.news_filter)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Run backtest
    print("Starting backtest...")
    print(f"Date range: {global_config.data.start_date} to {global_config.data.end_date}")
    print(f"Symbols: {', '.join(symbols_config.symbols)}")
    print()
    
    # Telegram notifier reads from the same config passed via --config to keep alerts aligned with the chosen backtest/global config
    telegram_notifier: Optional[TelegramNotifier] = None
    notifications = getattr(global_config, "notifications", None)
    telegram_cfg = getattr(notifications, "telegram", None) if notifications else None
    if telegram_cfg and telegram_cfg.enabled:
        telegram_notifier = TelegramNotifier(telegram_cfg)

    try:
        trades, metrics, signal_stats = run_backtest(
            global_config,
            symbols_config,
            risk_config,
            news_filter_config,
            telegram_notifier=telegram_notifier,
        )
        
        # Print signal pipeline stats
        if signal_stats:
            print("\n=== Signal pipeline stats ===")
            for symbol, stats in signal_stats.items():
                print(
                    f"{symbol}: raw={stats.get('raw_signals', 0)}, "
                    f"after_news={stats.get('after_news_filter', 0)}, "
                    f"after_min_rr={stats.get('after_min_rr', 0)}, "
                    f"executed={stats.get('executed_trades', 0)}, "
                    f"skipped_open={stats.get('skipped_open_position', 0)}, "
                    f"skipped_max_trades={stats.get('skipped_max_trades_per_day', 0)}, "
                    f"skipped_min_interval={stats.get('skipped_min_interval', 0)}, "
                    f"skipped_other={stats.get('skipped_other', 0)}"
                )
            print("=" * 30)
        
        # Save results
        output_dir = Path(global_config.project.backtest_dir)
        save_results(trades, metrics, output_dir, args.phase, signal_stats)
        
        # Print summary
        print("\n=== Backtest Summary ===")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
        print(f"Net PnL: {metrics.get('net_pnl', 0):.2f}")
        print(f"Total Fees: {metrics.get('total_fees', 0):.2f}")
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
