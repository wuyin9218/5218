"""Event-driven backtest runner for SMC signals."""

import logging
from datetime import datetime, date
from typing import List, Optional, Callable, Tuple

import numpy as np
import pandas as pd

from backtest.slippage_fee import SlippageFeeModel
from backtest.metrics import Trade, MetricsCalculator
from config.loader import GlobalConfig, RiskConfig
from io_layer.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Event-driven backtest engine."""
    
    def __init__(
        self,
        global_config: GlobalConfig,
        risk_config: RiskConfig,
        slippage_fee_model: SlippageFeeModel,
        news_filter=None,
        telegram_notifier: Optional[TelegramNotifier] = None,
    ):
        """
        Initialize backtest runner.
        
        Args:
            global_config: Global configuration
            risk_config: Risk management configuration
            slippage_fee_model: Slippage and fee model
            news_filter: News filter instance (optional)
        """
        self.config = global_config
        self.risk_config = risk_config
        self.fee_model = slippage_fee_model
        self.news_filter = news_filter
        self.telegram_notifier = telegram_notifier
        
        # State
        self.initial_balance = global_config.backtest.initial_balance
        self.balance = global_config.backtest.initial_balance
        self.current_equity = self.balance
        self.trades: List[Trade] = []
        self.open_positions: dict = {}  # symbol -> position info
        self.daily_trades: dict = {}  # date -> count
        self.daily_pnl: dict = {}  # date -> 当日 net_pnl 累计
        self.equity_curve: list = []  # (timestamp, equity)
        self.consecutive_losses = 0
        self.last_trade_time: Optional[datetime] = None
        self.cooldown_until: Optional[datetime] = None
        self.current_date: Optional[date] = None
        self.day_start_equity: float = self.current_equity
        self.day_realized_pnl: float = 0.0
        self.day_trading_locked: bool = False
        
        # Signal pipeline statistics
        self._init_signal_stats()
        
        # Initialize equity curve with starting point
        self.equity_curve.append((None, self.initial_balance))
        
        # Set random seed for reproducibility
        np.random.seed(global_config.backtest.seed)
    
    def can_trade(self, current_time: datetime) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is allowed based on risk rules.
        
        Args:
            current_time: Current timestamp
        
        Returns:
            Tuple of (allowed flag, reason string if not allowed)
        """
        # Ensure day state is aligned with current time
        self._ensure_day_state(current_time)

        # Reset cooldown if it has expired
        self._reset_cooldown_if_needed(current_time)

        # Check cooldown window
        if self.cooldown_until and current_time < self.cooldown_until:
            return False, 'cooldown'
        
        # Check daily loss lock
        if self.day_trading_locked:
            return False, 'daily_loss'

        # Check daily trade limit
        date_key = current_time.date()
        if date_key in self.daily_trades:
            if self.daily_trades[date_key] >= self.config.backtest.max_trades_per_day:
                return False, 'max_trades'
        
        # Check minimum signal interval
        if self.last_trade_time:
            minutes_since = (current_time - self.last_trade_time).total_seconds() / 60
            if minutes_since < self.risk_config.min_signal_interval_minutes:
                return False, 'min_interval'
        
        return True, None
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate position size based on risk per trade.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
        
        Returns:
            Position size in base currency
        """
        risk_amount = self.balance * (self.risk_config.risk_per_trade_pct / 100.0)
        price_risk = abs(entry_price - stop_loss_price)

        min_pct = getattr(self.risk_config, "min_stop_distance_pct", 0.0)
        min_abs = getattr(self.risk_config, "min_stop_distance_abs", 0.0)

        min_dist_pct = entry_price * min_pct if min_pct and entry_price > 0 else 0.0
        min_dist_abs = min_abs if min_abs and min_abs > 0 else 0.0
        min_distance = max(min_dist_pct, min_dist_abs)

        adjusted_price_risk = max(price_risk, min_distance)

        if adjusted_price_risk <= 0:
            return 0.0

        if adjusted_price_risk > price_risk and self.signal_stats.get('adjusted_min_stop_distance') is not None:
            self.signal_stats['adjusted_min_stop_distance'] += 1
        
        quantity = risk_amount / adjusted_price_risk
        return quantity
    
    def enter_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: Optional[float],
        entry_time: datetime,
        check_rr: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Enter a new position.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price (optional)
            entry_time: Entry timestamp
            check_rr: Whether to check RR filter (for statistics)
        Returns:
            Tuple of (success flag, failure reason)
        """
        can_trade_allowed, blocked_reason = self.can_trade(entry_time)
        if not can_trade_allowed:
            return False, blocked_reason or 'other'
        
        if symbol in self.open_positions:
            return False, 'open_position'  # Already have position
        
        # Check minimum RR filter (最小 RR 过滤)
        if check_rr and take_profit_price is not None and stop_loss_price != entry_price:
            risk = abs(entry_price - stop_loss_price)
            reward = abs(take_profit_price - entry_price)
            if risk > 0:
                rr = reward / risk
                if rr < self.risk_config.min_rr:
                    return False, 'other'  # RR 不满足要求，不开仓
        
        # Calculate position size
        quantity = self.calculate_position_size(entry_price, stop_loss_price)
        if quantity <= 0:
            return False, 'other'
        
        # Calculate entry cost with slippage and fees
        exec_price = self.fee_model.get_execution_price(entry_price, side)
        entry_fee = self.fee_model.calculate_fee(exec_price, quantity)
        entry_cost = exec_price * quantity + entry_fee
        
        # Check if we have enough balance
        if entry_cost > self.balance:
            return False, 'other'
        
        # Enter position
        self.open_positions[symbol] = {
            'side': side,
            'entry_price': entry_price,
            'exec_entry_price': exec_price,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'quantity': quantity,
            'entry_time': entry_time,
            'entry_fee': entry_fee,
            'entry_cost': entry_cost
        }
        
        self.balance -= entry_cost
        self.last_trade_time = entry_time
        
        # Update daily trade count
        date_key = entry_time.date()
        self.daily_trades[date_key] = self.daily_trades.get(date_key, 0) + 1

        self._notify_telegram_on_execution(
            symbol=symbol,
            side=side,
            entry_time=entry_time,
            entry_price=entry_price,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
        )

        return True, None
    
    def check_exit_conditions(
        self,
        symbol: str,
        current_price: float,
        current_time: datetime
    ) -> Optional[str]:
        """
        Check if position should be exited.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            current_time: Current timestamp
        
        Returns:
            Exit reason ('stop_loss', 'take_profit', 'time') or None
        """
        if symbol not in self.open_positions:
            return None
        
        pos = self.open_positions[symbol]
        
        # Check stop loss
        if pos['side'] == 'buy':
            if current_price <= pos['stop_loss']:
                return 'stop_loss'
        else:  # sell
            if current_price >= pos['stop_loss']:
                return 'stop_loss'
        
        # Check take profit
        if pos['take_profit']:
            if pos['side'] == 'buy':
                if current_price >= pos['take_profit']:
                    return 'take_profit'
            else:  # sell
                if current_price <= pos['take_profit']:
                    return 'take_profit'
        
        return None
    
    def exit_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ) -> bool:
        """
        Exit an open position.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: Reason for exit
        
        Returns:
            True if position exited successfully
        """
        if symbol not in self.open_positions:
            return False
        
        pos = self.open_positions[symbol]
        
        # Calculate exit cost with slippage and fees
        exit_side = 'sell' if pos['side'] == 'buy' else 'buy'
        exec_exit_price = self.fee_model.get_execution_price(exit_price, exit_side)
        exit_fee = self.fee_model.calculate_fee(exec_exit_price, pos['quantity'])
        
        # Calculate PnL
        if pos['side'] == 'buy':
            pnl = (exec_exit_price - pos['exec_entry_price']) * pos['quantity']
        else:  # sell
            pnl = (pos['exec_entry_price'] - exec_exit_price) * pos['quantity']
        
        net_pnl = pnl - pos['entry_fee'] - exit_fee
        pnl_pct = (net_pnl / pos['entry_cost']) * 100 if pos['entry_cost'] > 0 else 0.0
        
        # Calculate risk-reward ratio
        risk = abs(pos['entry_price'] - pos['stop_loss']) * pos['quantity']
        reward = abs(exec_exit_price - pos['entry_price']) * pos['quantity'] if net_pnl > 0 else 0
        rr = (reward / risk) if risk > 0 else None
        
        # Update balance
        exit_proceeds = exec_exit_price * pos['quantity'] - exit_fee
        self.balance += exit_proceeds
        
        # Update daily PnL (legacy stats)
        date_key = exit_time.date()
        self.daily_pnl[date_key] = self.daily_pnl.get(date_key, 0.0) + net_pnl
        self._update_day_pnl(net_pnl, exit_time)
        
        # Update equity curve
        self.equity_curve.append((exit_time, self.balance))
        
        # Record trade
        trade = Trade(
            entry_time=pos['entry_time'],
            exit_time=exit_time,
            symbol=symbol,
            side=pos['side'],
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            quantity=pos['quantity'],
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            fees=pos['entry_fee'] + exit_fee,
            rr=rr
        )
        self.trades.append(trade)
        
        # Update state
        if net_pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.risk_config.consecutive_loss_limit:
                self.cooldown_until = exit_time + pd.Timedelta(
                    minutes=self.risk_config.cooldown_minutes
                )
                logger.warning(
                    "Hit max consecutive losses (%d) at %s for %s. Cooling down until %s",
                    self.consecutive_losses,
                    exit_time,
                    symbol,
                    self.cooldown_until,
                )
        else:
            self.consecutive_losses = 0
        
        # Remove position
        del self.open_positions[symbol]
        
        return True
    
    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame, datetime], Optional[dict]]
    ) -> Tuple[List[Trade], dict, dict]:
        """
        Run backtest on data.
        
        Args:
            data: DataFrame with OHLC data, indexed by timestamp
            signal_generator: Function that takes (df, current_time) and returns signal dict or None
        
        Returns:
            Tuple of (trades list, metrics dict, signal_stats dict)
        """
        if data.empty:
            return [], {}, self.signal_stats
        
        # Reset signal statistics
        self._init_signal_stats()
        
        # Process each candle (only use close price to avoid lookahead)
        for idx, row in data.iterrows():
            current_time = idx if isinstance(idx, datetime) else pd.to_datetime(idx)
            current_price = row['close']

            # Roll day state when date changes
            self._ensure_day_state(current_time)
            
            # Check exit conditions for open positions
            for symbol in list(self.open_positions.keys()):
                exit_reason = self.check_exit_conditions(symbol, current_price, current_time)
                if exit_reason:
                    self.exit_position(symbol, current_price, current_time, exit_reason)
            
            # Generate signal (only use data up to current candle)
            current_data = data.loc[:current_time]
            signal = signal_generator(current_data, current_time)
            
            # Count raw signals
            if signal:
                self.signal_stats['raw_signals'] += 1
                
                # Check news blackout (新闻黑窗过滤)
                if self.news_filter is not None and self.news_filter.is_blackout(current_time):
                    self._increment_skip('news_blackout')
                    continue  # Skip signal generation during blackout
                
                self.signal_stats['after_news_filter'] += 1
                
                symbol = signal.get('symbol', 'UNKNOWN')
                side = signal.get('side', 'buy')
                entry_price = signal.get('entry_price', current_price)
                stop_loss = signal.get('stop_loss')
                take_profit = signal.get('take_profit')
                
                if stop_loss is not None:
                    # Check RR before attempting to enter (for statistics)
                    passed_rr = True
                    if take_profit is not None and stop_loss != entry_price:
                        risk = abs(entry_price - stop_loss)
                        reward = abs(take_profit - entry_price)
                        if risk > 0:
                            rr = reward / risk
                            if rr < self.risk_config.min_rr:
                                passed_rr = False
                    
                    if passed_rr:
                        self.signal_stats['after_min_rr'] += 1
                        
                        # Try to enter position
                        success, fail_reason = self.enter_position(
                            symbol=symbol,
                            side=side,
                            entry_price=entry_price,
                            stop_loss_price=stop_loss,
                            take_profit_price=take_profit,
                            entry_time=current_time,
                            check_rr=False  # Already checked above
                        )
                        
                        if not success:
                            self._increment_skip(fail_reason)
                    else:
                        self._increment_skip('min_rr')
                else:
                    self._increment_skip('missing_stop_loss')
        
        # Close all remaining positions at end
        final_time = data.index[-1]
        final_price = data.iloc[-1]['close']
        for symbol in list(self.open_positions.keys()):
            self.exit_position(symbol, final_price, final_time, 'end_of_data')
        
        # Update executed_trades count (final count after all positions closed)
        self.signal_stats['executed_trades'] = len(self.trades)
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate(self.trades, self.config.backtest.initial_balance)
        metrics_dict = MetricsCalculator.to_dict(metrics)
        
        return self.trades, metrics_dict, self.signal_stats

    def _init_signal_stats(self) -> None:
        """Initialize signal pipeline statistics."""
        self.signal_stats = {
            'raw_signals': 0,
            'after_news_filter': 0,
            'after_min_rr': 0,
            'executed_trades': 0,
            'skipped_open_position': 0,
            'skipped_max_trades_per_day': 0,
            'skipped_min_interval': 0,
            'skipped_other': 0,
            'adjusted_min_stop_distance': 0,
        }
    
    def _increment_skip(self, reason: Optional[str]) -> None:
        """Increment skipped signal counters based on reason."""
        mapping = {
            'open_position': 'skipped_open_position',
            'max_trades': 'skipped_max_trades_per_day',
            'min_interval': 'skipped_min_interval',
            'daily_loss': 'skipped_other',
        }
        key = mapping.get(reason, 'skipped_other')
        if key not in self.signal_stats:
            # Ensure key exists even if signal_stats wasn't reset properly
            self.signal_stats[key] = 0
        self.signal_stats[key] += 1

    def _reset_cooldown_if_needed(self, current_time: datetime) -> None:
        """Clear cooldown and consecutive loss counter when cooldown expires."""
        if self.cooldown_until and current_time >= self.cooldown_until:
            self.cooldown_until = None
            if self.consecutive_losses != 0:
                logger.info(
                    "Cooldown expired at %s, resetting consecutive losses.",
                    current_time,
                )
            self.consecutive_losses = 0

    def _ensure_day_state(self, current_time: datetime) -> None:
        """Ensure daily tracking state matches the provided timestamp."""
        bar_date = current_time.date()
        if self.current_date is None or bar_date != self.current_date:
            self.current_date = bar_date
            self._reset_day_state()

    def _reset_day_state(self) -> None:
        """Reset per-day tracking fields at the start of a new trading day."""
        self.day_start_equity = self.current_equity
        self.day_realized_pnl = 0.0
        self.day_trading_locked = False

    def _update_day_pnl(self, net_pnl: float, timestamp: datetime) -> None:
        """Update daily realized PnL after a trade is closed."""
        self.current_equity = self.balance
        self._ensure_day_state(timestamp)
        self.day_realized_pnl += net_pnl
        self._check_daily_loss_limit(timestamp)

    def _check_daily_loss_limit(self, now: datetime) -> None:
        """Lock trading for the rest of the day if daily loss threshold is exceeded."""
        pct = self.risk_config.daily_loss_limit_pct
        if pct <= 0:
            return

        mode = getattr(self.risk_config, "daily_loss_limit_mode", "day_equity")
        base = self.initial_balance if mode == "initial_equity" else self.day_start_equity
        if base <= 0:
            return

        max_loss = base * (pct / 100.0)

        if -self.day_realized_pnl >= max_loss:
            if not self.day_trading_locked:
                self.day_trading_locked = True
                logger.warning(
                    "Hit daily loss limit (%.2f%% of %.2f) on %s, locking trading for the rest of the day.",
                    pct,
                    base,
                    self.current_date,
                )

    def _notify_telegram_on_execution(
        self,
        *,
        symbol: str,
        side: str,
        entry_time: datetime,
        entry_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> None:
        """Send Telegram notification when a trade is opened."""
        if self.telegram_notifier is None:
            return

        try:
            timeframe = (
                self.config.data.intervals[0]
                if getattr(self.config.data, "intervals", None)
                else "N/A"
            )
        except Exception:  # pragma: no cover
            timeframe = "N/A"

        direction = "多头" if side.lower() == "buy" else "空头"
        entry_time_str = (
            entry_time.strftime("%Y-%m-%d %H:%M") if isinstance(entry_time, datetime) else str(entry_time)
        )

        def _fmt_price(value: Optional[float]) -> str:
            return f"{value:.2f}" if value is not None else "N/A"

        planned_rr = None
        if stop_loss is not None and take_profit is not None and stop_loss != entry_price:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            if risk > 0:
                planned_rr = reward / risk

        rr_str = f"{planned_rr:.2f}" if planned_rr is not None else "N/A"

        message = (
            f"[{symbol} {timeframe} {direction}] {entry_time_str} 开仓 @ {entry_price:.2f}, "
            f"SL={_fmt_price(stop_loss)}, TP={_fmt_price(take_profit)}, 计划RR={rr_str}"
        )

        sent = self.telegram_notifier.send_message(message)
        if not sent:
            logger.warning("Telegram trade notification failed for %s", symbol)

