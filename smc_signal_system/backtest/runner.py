"""Event-driven backtest runner for SMC signals."""

import pandas as pd
from typing import List, Optional, Callable, Tuple
from datetime import datetime
import numpy as np
from backtest.slippage_fee import SlippageFeeModel
from backtest.metrics import Trade, MetricsCalculator
from config.loader import GlobalConfig, RiskConfig


class BacktestRunner:
    """Event-driven backtest engine."""
    
    def __init__(
        self,
        global_config: GlobalConfig,
        risk_config: RiskConfig,
        slippage_fee_model: SlippageFeeModel
    ):
        """
        Initialize backtest runner.
        
        Args:
            global_config: Global configuration
            risk_config: Risk management configuration
            slippage_fee_model: Slippage and fee model
        """
        self.config = global_config
        self.risk_config = risk_config
        self.fee_model = slippage_fee_model
        
        # State
        self.balance = global_config.backtest.initial_balance
        self.trades: List[Trade] = []
        self.open_positions: dict = {}  # symbol -> position info
        self.daily_trades: dict = {}  # date -> count
        self.consecutive_losses = 0
        self.last_trade_time: Optional[datetime] = None
        self.cooldown_until: Optional[datetime] = None
        
        # Set random seed for reproducibility
        np.random.seed(global_config.backtest.seed)
    
    def can_trade(self, current_time: datetime) -> bool:
        """
        Check if trading is allowed based on risk rules.
        
        Args:
            current_time: Current timestamp
        
        Returns:
            True if trading is allowed
        """
        # Check cooldown
        if self.cooldown_until and current_time < self.cooldown_until:
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.risk_config.consecutive_loss_limit:
            return False
        
        # Check daily trade limit
        date_key = current_time.date()
        if date_key in self.daily_trades:
            if self.daily_trades[date_key] >= self.config.backtest.max_trades_per_day:
                return False
        
        # Check minimum signal interval
        if self.last_trade_time:
            minutes_since = (current_time - self.last_trade_time).total_seconds() / 60
            if minutes_since < self.risk_config.min_signal_interval_minutes:
                return False
        
        return True
    
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
        
        if price_risk == 0:
            return 0.0
        
        quantity = risk_amount / price_risk
        return quantity
    
    def enter_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: Optional[float],
        entry_time: datetime
    ) -> bool:
        """
        Enter a new position.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price (optional)
            entry_time: Entry timestamp
        
        Returns:
            True if position entered successfully
        """
        if not self.can_trade(entry_time):
            return False
        
        if symbol in self.open_positions:
            return False  # Already have position
        
        # Calculate position size
        quantity = self.calculate_position_size(entry_price, stop_loss_price)
        if quantity <= 0:
            return False
        
        # Calculate entry cost with slippage and fees
        exec_price = self.fee_model.get_execution_price(entry_price, side)
        entry_fee = self.fee_model.calculate_fee(exec_price, quantity)
        entry_cost = exec_price * quantity + entry_fee
        
        # Check if we have enough balance
        if entry_cost > self.balance:
            return False
        
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
        
        return True
    
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
            # Check if we need cooldown
            if self.consecutive_losses >= self.risk_config.consecutive_loss_limit:
                self.cooldown_until = exit_time + pd.Timedelta(minutes=self.risk_config.cooldown_minutes)
        else:
            self.consecutive_losses = 0
        
        # Remove position
        del self.open_positions[symbol]
        
        return True
    
    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame, datetime], Optional[dict]]
    ) -> Tuple[List[Trade], dict]:
        """
        Run backtest on data.
        
        Args:
            data: DataFrame with OHLC data, indexed by timestamp
            signal_generator: Function that takes (df, current_time) and returns signal dict or None
        
        Returns:
            Tuple of (trades list, metrics dict)
        """
        if data.empty:
            return [], {}
        
        # Process each candle (only use close price to avoid lookahead)
        for idx, row in data.iterrows():
            current_time = idx if isinstance(idx, datetime) else pd.to_datetime(idx)
            current_price = row['close']
            
            # Check exit conditions for open positions
            for symbol in list(self.open_positions.keys()):
                exit_reason = self.check_exit_conditions(symbol, current_price, current_time)
                if exit_reason:
                    self.exit_position(symbol, current_price, current_time, exit_reason)
            
            # Generate signal (only use data up to current candle)
            current_data = data.loc[:current_time]
            signal = signal_generator(current_data, current_time)
            
            if signal:
                symbol = signal.get('symbol', 'UNKNOWN')
                side = signal.get('side', 'buy')
                entry_price = signal.get('entry_price', current_price)
                stop_loss = signal.get('stop_loss')
                take_profit = signal.get('take_profit')
                
                if stop_loss:
                    self.enter_position(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        stop_loss_price=stop_loss,
                        take_profit_price=take_profit,
                        entry_time=current_time
                    )
        
        # Close all remaining positions at end
        final_time = data.index[-1]
        final_price = data.iloc[-1]['close']
        for symbol in list(self.open_positions.keys()):
            self.exit_position(symbol, final_price, final_time, 'end_of_data')
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate(self.trades, self.config.backtest.initial_balance)
        metrics_dict = MetricsCalculator.to_dict(metrics)
        
        return self.trades, metrics_dict

