"""
Regression test to ensure backtest results are consistent.

Note: This test does NOT depend on real Binance API access.
If Binance requests fail, the system will automatically fall back to offline dummy data.
The focus is on verifying:
1. Same configuration produces identical results across multiple runs
2. Output file schemas are correct
"""

import pytest
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import (
    GlobalConfig, SymbolsConfig, RiskConfig, NewsFilterConfig
)
from backtest.runner import BacktestRunner
from backtest.slippage_fee import SlippageFeeModel
from scripts.run_backtest import run_backtest, save_results


def test_backtest_consistency():
    """
    Test that running the same backtest 3 times produces identical results.
    
    This ensures no randomness or state leakage between runs.
    
    Note: Uses offline dummy data if Binance API is unreachable.
    """
    # Load configurations
    config_dir = Path(__file__).parent.parent / "config"
    
    global_config = GlobalConfig.from_yaml(str(config_dir / "global.yaml"))
    symbols_config = SymbolsConfig.from_yaml(str(config_dir / "symbols.yaml"))
    risk_config = RiskConfig.from_yaml(str(config_dir / "risk.yaml"))
    news_filter_config = NewsFilterConfig.from_yaml(str(config_dir / "news_filter.yaml"))
    
    # Run backtest 3 times
    results = []
    for i in range(3):
        print(f"\nRunning backtest iteration {i+1}/3...")
        
        # Create a fresh runner for each iteration
        trades, metrics = run_backtest(
            global_config,
            symbols_config,
            risk_config,
            news_filter_config
        )
        
        # Store results
        results.append({
            'trades': trades,
            'metrics': metrics
        })
    
    # Compare results
    assert len(results) == 3, "Should have 3 backtest runs"
    
    # Check metrics consistency
    metrics_0 = results[0]['metrics']
    metrics_1 = results[1]['metrics']
    metrics_2 = results[2]['metrics']
    
    # Compare key metrics
    assert metrics_0.get('total_trades') == metrics_1.get('total_trades'), \
        "Total trades should be consistent"
    assert metrics_1.get('total_trades') == metrics_2.get('total_trades'), \
        "Total trades should be consistent"
    
    assert metrics_0.get('winning_trades') == metrics_1.get('winning_trades'), \
        "Winning trades should be consistent"
    assert metrics_1.get('winning_trades') == metrics_2.get('winning_trades'), \
        "Winning trades should be consistent"
    
    # Compare net PnL (allowing for floating point precision)
    net_pnl_0 = metrics_0.get('net_pnl', 0)
    net_pnl_1 = metrics_1.get('net_pnl', 0)
    net_pnl_2 = metrics_2.get('net_pnl', 0)
    
    assert abs(net_pnl_0 - net_pnl_1) < 0.01, \
        f"Net PnL should be consistent: {net_pnl_0} vs {net_pnl_1}"
    assert abs(net_pnl_1 - net_pnl_2) < 0.01, \
        f"Net PnL should be consistent: {net_pnl_1} vs {net_pnl_2}"
    
    # Compare trades count
    trades_0 = results[0]['trades']
    trades_1 = results[1]['trades']
    trades_2 = results[2]['trades']
    
    assert len(trades_0) == len(trades_1), \
        f"Trade count should be consistent: {len(trades_0)} vs {len(trades_1)}"
    assert len(trades_1) == len(trades_2), \
        f"Trade count should be consistent: {len(trades_1)} vs {len(trades_2)}"
    
    # If there are trades, compare them
    if trades_0:
        for i, (t0, t1, t2) in enumerate(zip(trades_0, trades_1, trades_2)):
            assert t0.symbol == t1.symbol == t2.symbol, \
                f"Trade {i} symbol should be consistent"
            assert abs(t0.pnl - t1.pnl) < 0.01, \
                f"Trade {i} PnL should be consistent: {t0.pnl} vs {t1.pnl}"
            assert abs(t1.pnl - t2.pnl) < 0.01, \
                f"Trade {i} PnL should be consistent: {t1.pnl} vs {t2.pnl}"
    
    print("\n✓ All 3 backtest runs produced identical results")


def test_backtest_output_schema():
    """
    Test that backtest output files have correct schema.
    
    Note: Uses offline dummy data if Binance API is unreachable.
    """
    config_dir = Path(__file__).parent.parent / "config"
    output_dir = Path(__file__).parent.parent / "backtests" / "test_schema"
    
    global_config = GlobalConfig.from_yaml(str(config_dir / "global.yaml"))
    symbols_config = SymbolsConfig.from_yaml(str(config_dir / "symbols.yaml"))
    risk_config = RiskConfig.from_yaml(str(config_dir / "risk.yaml"))
    news_filter_config = NewsFilterConfig.from_yaml(str(config_dir / "news_filter.yaml"))
    
    # Run backtest
    trades, metrics = run_backtest(
        global_config,
        symbols_config,
        risk_config,
        news_filter_config
    )
    
    # Save results
    save_results(trades, metrics, output_dir.parent, "test_schema")
    
    # Check summary.json schema
    summary_path = output_dir / "summary.json"
    assert summary_path.exists(), "summary.json should exist"
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    # Check required fields
    required_fields = ['total_trades', 'winning_trades', 'losing_trades', 'win_rate']
    for field in required_fields:
        assert field in summary, f"summary.json should have '{field}' field"
    
    # Check trades.csv schema
    trades_path = output_dir / "trades.csv"
    assert trades_path.exists(), "trades.csv should exist"
    
    trades_df = pd.read_csv(trades_path)
    
    # Check required columns
    required_columns = [
        'entry_time', 'exit_time', 'symbol', 'side',
        'entry_price', 'exit_price', 'quantity',
        'pnl', 'pnl_pct', 'fees', 'rr'
    ]
    for col in required_columns:
        assert col in trades_df.columns, f"trades.csv should have '{col}' column"
    
    # Cleanup
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print("\n✓ Output schema is correct")

