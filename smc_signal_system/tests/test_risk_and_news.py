"""Tests for risk management and news filtering."""

from datetime import datetime
from dataclasses import dataclass
from typing import List
from backtest.runner import BacktestRunner
from backtest.slippage_fee import SlippageFeeModel
from config.loader import GlobalConfig, BacktestConfig, DataConfig, ProjectConfig, RiskConfig


@dataclass
class TestGlobalConfig:
    """Test global config."""
    backtest: BacktestConfig


def test_min_rr_filter():
    """
    Test that minimum RR filter prevents trades with insufficient risk-reward ratio.
    """
    # Create test configurations
    backtest_config = BacktestConfig(
        initial_balance=10000,
        fee_bps=1.0,  # Small fee for testing
        slippage_bps=0.5,  # Small slippage for testing
        max_trades_per_day=10,
        seed=42
    )
    
    risk_config = RiskConfig(
        risk_per_trade_pct=1.0,
        daily_loss_limit_pct=10.0,
        consecutive_loss_limit=3,
        cooldown_minutes=240,
        min_signal_interval_minutes=180,
        min_rr=2.0  # Minimum RR of 2.0
    )
    
    # Create a minimal global config for BacktestRunner
    project_config = ProjectConfig(
        name="test",
        timezone="UTC",
        data_dir="./data",
        backtest_dir="./backtests"
    )
    
    data_config = DataConfig(
        exchange="test",
        start_date="2023-01-01",
        end_date="2024-01-01",
        intervals=["1h"],
        limit_per_call=1000
    )
    
    global_config = GlobalConfig(
        project=project_config,
        data=data_config,
        backtest=backtest_config
    )
    
    # Create fee model
    fee_model = SlippageFeeModel(fee_bps=1.0, slippage_bps=0.5)
    
    # Initialize runner
    runner = BacktestRunner(global_config, risk_config, fee_model)
    
    # Test case 1: RR ≈ 1.4 < 2.0, should be rejected
    entry_price = 100.0
    stop_loss = 95.0
    take_profit = 107.0  # RR = (107-100)/(100-95) = 7/5 = 1.4
    
    result1 = runner.enter_position(
        symbol="TEST",
        side="buy",
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        entry_time=datetime(2023, 1, 1, 12, 0, 0)
    )
    
    # Should return False and not open position
    assert result1 is False, "Trade with RR < 2.0 should be rejected"
    assert "TEST" not in runner.open_positions, "Position should not be opened"
    
    # Test case 2: RR = 2.0, should be accepted
    entry_price = 100.0
    stop_loss = 95.0
    take_profit = 110.0  # RR = (110-100)/(100-95) = 10/5 = 2.0
    
    result2 = runner.enter_position(
        symbol="TEST",
        side="buy",
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        entry_time=datetime(2023, 1, 1, 13, 0, 0)
    )
    
    # Should return True and open position
    assert result2 is True, "Trade with RR = 2.0 should be accepted"
    assert "TEST" in runner.open_positions, "Position should be opened"
    
    # Verify position details
    pos = runner.open_positions["TEST"]
    assert pos['entry_price'] == entry_price
    assert pos['stop_loss'] == stop_loss
    assert pos['take_profit'] == take_profit
    
    print("Minimum RR filter test passed")

