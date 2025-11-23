"""Slippage and fee models for backtesting."""

from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass
class SlippageFeeModel:
    """Model for calculating slippage and fees."""
    
    fee_bps: float  # Fee in basis points (e.g., 4.0 = 0.04%)
    slippage_bps: float  # Slippage in basis points (e.g., 2.0 = 0.02%)
    
    def calculate_fee(self, price: float, quantity: float) -> float:
        """
        Calculate trading fee.
        
        Args:
            price: Entry/exit price
            quantity: Trade quantity
        
        Returns:
            Fee amount in quote currency
        """
        notional = price * quantity
        fee = notional * (self.fee_bps / 10000.0)
        return fee
    
    def apply_slippage(
        self,
        price: float,
        side: Literal["buy", "sell"]
    ) -> float:
        """
        Apply slippage to price.
        
        Args:
            price: Original price
            side: 'buy' or 'sell'
        
        Returns:
            Price with slippage applied
        """
        slippage_pct = self.slippage_bps / 10000.0
        
        if side == "buy":
            # Buy orders pay more (slippage increases price)
            return price * (1 + slippage_pct)
        else:
            # Sell orders receive less (slippage decreases price)
            return price * (1 - slippage_pct)
    
    def get_execution_price(
        self,
        price: float,
        side: Literal["buy", "sell"]
    ) -> float:
        """
        Get execution price including slippage.
        
        Args:
            price: Original price
            side: 'buy' or 'sell'
        
        Returns:
            Execution price with slippage
        """
        return self.apply_slippage(price, side)
    
    def calculate_total_cost(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: Literal["buy", "sell"]
    ) -> Tuple[float, float, float]:
        """
        Calculate total cost including fees and slippage.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            side: 'buy' or 'sell'
        
        Returns:
            Tuple of (entry_cost, exit_cost, total_fees)
        """
        # Apply slippage
        exec_entry = self.get_execution_price(entry_price, side)
        exec_exit = self.get_execution_price(exit_price, "sell" if side == "buy" else "buy")
        
        # Calculate costs
        entry_cost = exec_entry * quantity
        exit_cost = exec_exit * quantity
        
        # Calculate fees
        entry_fee = self.calculate_fee(exec_entry, quantity)
        exit_fee = self.calculate_fee(exec_exit, quantity)
        total_fees = entry_fee + exit_fee
        
        return entry_cost, exit_cost, total_fees

