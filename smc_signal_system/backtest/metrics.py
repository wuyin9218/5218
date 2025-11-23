"""Backtest metrics calculation."""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class Trade:
    """Single trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    rr: Optional[float] = None  # Risk-reward ratio


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    total_fees: float
    net_pnl: float
    max_drawdown: float
    max_drawdown_pct: float
    expectancy: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    sharpe_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None


class MetricsCalculator:
    """Calculate backtest performance metrics."""
    
    @staticmethod
    def calculate(trades: List[Trade], initial_balance: float) -> BacktestMetrics:
        """
        Calculate comprehensive backtest metrics.
        
        Args:
            trades: List of trade records
            initial_balance: Initial account balance
        
        Returns:
            BacktestMetrics object
        """
        if not trades:
            return MetricsCalculator._empty_metrics()
        
        df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'pnl': t.pnl,
            'fees': t.fees,
            'rr': t.rr
        } for t in trades])
        
        # Basic counts
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        # PnL metrics
        total_pnl = sum(t.pnl for t in trades)
        total_fees = sum(t.fees for t in trades)
        net_pnl = total_pnl - total_fees
        
        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        # Average win/loss
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        # Largest win/loss
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0
        
        # Expectancy
        expectancy = (win_rate / 100.0 * avg_win) + ((100 - win_rate) / 100.0 * avg_loss)
        
        # Drawdown calculation
        cumulative_pnl = df['pnl'].cumsum() - df['fees'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0.0
        max_drawdown_pct = (max_drawdown / initial_balance * 100) if initial_balance > 0 else 0.0
        
        # Sharpe ratio (simplified, using returns)
        if len(trades) > 1:
            returns = df['pnl'].values / initial_balance
            sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = None
        
        # Calmar ratio
        calmar_ratio = (net_pnl / initial_balance / max_drawdown_pct * 100) if max_drawdown_pct > 0 else None
        
        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            total_fees=total_fees,
            net_pnl=net_pnl,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            expectancy=expectancy,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio
        )
    
    @staticmethod
    def _empty_metrics() -> BacktestMetrics:
        """Return empty metrics for no trades."""
        return BacktestMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            total_pnl=0.0,
            total_fees=0.0,
            net_pnl=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            expectancy=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            sharpe_ratio=None,
            calmar_ratio=None
        )
    
    @staticmethod
    def to_dict(metrics: BacktestMetrics) -> dict:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate": round(metrics.win_rate, 2),
            "profit_factor": round(metrics.profit_factor, 2),
            "total_pnl": round(metrics.total_pnl, 2),
            "total_fees": round(metrics.total_fees, 2),
            "net_pnl": round(metrics.net_pnl, 2),
            "max_drawdown": round(metrics.max_drawdown, 2),
            "max_drawdown_pct": round(metrics.max_drawdown_pct, 2),
            "expectancy": round(metrics.expectancy, 2),
            "avg_win": round(metrics.avg_win, 2),
            "avg_loss": round(metrics.avg_loss, 2),
            "largest_win": round(metrics.largest_win, 2),
            "largest_loss": round(metrics.largest_loss, 2),
            "sharpe_ratio": round(metrics.sharpe_ratio, 4) if metrics.sharpe_ratio is not None else None,
            "calmar_ratio": round(metrics.calmar_ratio, 4) if metrics.calmar_ratio is not None else None
        }

