"""
Base Strategy Class

Abstract base class for all trading strategies.
Provides standardized interface for strategy implementation.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class PositionType(Enum):
    """Position type enumeration"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """
    Represents a trading position
    """
    symbol: str
    position_type: PositionType
    size: float  # Position size (positive for both long and short)
    entry_price: float
    entry_time: pd.Timestamp
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    
    @property
    def is_open(self) -> bool:
        """Check if position is still open"""
        return self.exit_price is None
    
    @property
    def pnl(self) -> float:
        """Calculate position P&L (unrealized if position is open)"""
        if self.exit_price is None:
            return 0.0  # Cannot calculate unrealized P&L without current price
        
        if self.position_type == PositionType.LONG:
            return self.size * (self.exit_price - self.entry_price)
        elif self.position_type == PositionType.SHORT:
            return self.size * (self.entry_price - self.exit_price)
        else:
            return 0.0
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L given current price"""
        if self.position_type == PositionType.LONG:
            return self.size * (current_price - self.entry_price)
        elif self.position_type == PositionType.SHORT:
            return self.size * (self.entry_price - current_price)
        else:
            return 0.0


@dataclass 
class Trade:
    """
    Represents a completed trade (closed position)
    """
    symbol: str
    position_type: PositionType
    size: float
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    commission: float = 0.0
    
    @property
    def pnl(self) -> float:
        """Calculate trade P&L including commission"""
        if self.position_type == PositionType.LONG:
            gross_pnl = self.size * (self.exit_price - self.entry_price)
        elif self.position_type == PositionType.SHORT:
            gross_pnl = self.size * (self.entry_price - self.exit_price)
        else:
            gross_pnl = 0.0
            
        return gross_pnl - self.commission
    
    @property
    def return_pct(self) -> float:
        """Calculate trade return percentage"""
        notional = self.size * self.entry_price
        return (self.pnl / notional) * 100 if notional != 0 else 0.0
    
    @property
    def holding_period(self) -> pd.Timedelta:
        """Calculate holding period"""
        return self.exit_time - self.entry_time


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Provides standardized interface for strategy configuration,
    execution, and performance tracking.
    """
    
    def __init__(self, name: str, initial_capital: float = 1000000.0, 
                 commission_rate: float = 0.001):
        """
        Initialize strategy with basic parameters.
        
        Args:
            name: Strategy name for identification
            initial_capital: Starting capital amount
            commission_rate: Commission rate (e.g., 0.001 = 0.1%)
        """
        self.name = name
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.completed_trades: List[Trade] = []
        self.cash = initial_capital
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """
        Generate trading signals for the current timestamp.
        
        Args:
            data: Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
            timestamp: Current timestamp for signal generation
            
        Returns:
            Dict containing trading signals and metadata
        """
        pass
    
    @abstractmethod
    def execute_rebalance(self, data: pd.DataFrame, timestamp: pd.Timestamp, 
                         signals: Dict[str, Any]) -> None:
        """
        Execute portfolio rebalancing based on generated signals.
        
        Args:
            data: Multi-symbol OHLCV data
            timestamp: Current timestamp
            signals: Trading signals from generate_signals()
        """
        pass
    
    def close_position(self, symbol: str, exit_price: float, exit_time: pd.Timestamp) -> Trade:
        """
        Close an existing position and create a completed trade.
        
        Args:
            symbol: Symbol to close position for
            exit_price: Exit price
            exit_time: Exit timestamp
            
        Returns:
            Completed Trade object
        """
        if symbol not in self.positions:
            raise ValueError(f"No open position found for symbol {symbol}")
        
        position = self.positions[symbol]
        if not position.is_open:
            raise ValueError(f"Position for {symbol} is already closed")
        
        # Calculate commission for closing
        notional = position.size * exit_price
        exit_commission = notional * self.commission_rate
        
        # Total commission includes both entry and exit commissions
        entry_notional = position.size * position.entry_price
        entry_commission = entry_notional * self.commission_rate
        total_commission = entry_commission + exit_commission
        
        # Create completed trade
        trade = Trade(
            symbol=position.symbol,
            position_type=position.position_type,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=exit_time,
            commission=total_commission  # Total commission for the complete trade
        )
        
        # Update cash based on position type (only exit transaction)
        if position.position_type == PositionType.LONG:
            # Long position: sell and get cash back (minus exit commission)
            self.cash += notional - exit_commission
            
        elif position.position_type == PositionType.SHORT:
            # Short position: buy back and pay cash (plus exit commission)
            self.cash -= notional + exit_commission
        
        # Remove position and add to completed trades
        del self.positions[symbol]
        self.completed_trades.append(trade)
        
        return trade
    
    def open_position(self, symbol: str, position_type: PositionType, size: float,
                     entry_price: float, entry_time: pd.Timestamp) -> Position:
        """
        Open a new position.
        
        Args:
            symbol: Symbol to open position for
            position_type: Long or short position
            size: Position size (always positive)
            entry_price: Entry price
            entry_time: Entry timestamp
            
        Returns:
            New Position object
        """
        if symbol in self.positions:
            raise ValueError(f"Position already exists for symbol {symbol}")
        
        # Calculate commission
        notional = size * entry_price
        commission = notional * self.commission_rate
        
        if position_type == PositionType.LONG:
            # Long positions require cash to buy
            required_cash = notional + commission
            if required_cash > self.cash:
                raise ValueError(f"Insufficient cash for LONG position. Required: {required_cash}, Available: {self.cash}")
            
            # Deduct cash for long position
            self.cash -= required_cash
            
        elif position_type == PositionType.SHORT:
            # Short positions: we receive cash from selling, but pay commission
            # In a real trading system, we'd need margin requirements
            # For backtesting, we just deduct commission and add the proceeds
            if commission > self.cash:
                raise ValueError(f"Insufficient cash for SHORT commission. Required: {commission}, Available: {self.cash}")
            
            # Add cash from short sale, deduct commission
            self.cash += notional - commission
            
        else:
            raise ValueError(f"Invalid position type: {position_type}")
        
        # Create and store position
        position = Position(
            symbol=symbol,
            position_type=position_type,
            size=size,
            entry_price=entry_price,
            entry_time=entry_time
        )
        
        self.positions[symbol] = position
        return position
    
    def close_all_positions(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> List[Trade]:
        """
        Close all open positions at current market prices.
        
        Args:
            data: Multi-symbol OHLCV data
            timestamp: Current timestamp
            
        Returns:
            List of completed trades
        """
        trades = []
        
        # Get current prices for all symbols with open positions
        current_data = data.loc[data.index.get_level_values('open_time') == timestamp]
        
        for symbol, position in list(self.positions.items()):
            try:
                # Get current price (use open price for actual execution)
                symbol_data = current_data.loc[current_data.index.get_level_values('symbol') == symbol]
                if len(symbol_data) == 0:
                    raise ValueError(f"No current data available for {symbol}")

                # Use open price to simulate realistic execution at the start of the bar
                current_price = symbol_data['open'].iloc[0]
                trade = self.close_position(symbol, current_price, timestamp)
                trades.append(trade)
                
            except Exception as e:
                print(f"Warning: Could not close position for {symbol}: {e}")
                continue
        
        return trades
    
    def calculate_portfolio_value(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> float:
        """
        Calculate total portfolio value including unrealized P&L.
        
        Args:
            data: Multi-symbol OHLCV data
            timestamp: Current timestamp
            
        Returns:
            Total portfolio value
        """
        portfolio_value = self.cash
        
        # Add unrealized P&L from open positions
        current_data = data.loc[data.index.get_level_values('open_time') == timestamp]
        
        for symbol, position in self.positions.items():
            try:
                symbol_data = current_data.loc[current_data.index.get_level_values('symbol') == symbol]
                if len(symbol_data) > 0:
                    current_price = symbol_data['close'].iloc[0]
                    unrealized_pnl = position.calculate_unrealized_pnl(current_price)
                    
                    # For portfolio value calculation, we only add the unrealized P&L
                    # The notional value is already accounted for in cash flow
                    portfolio_value += unrealized_pnl
                    
                # Note: No need to add entry notional as it's already reflected in cash
            except Exception as e:
                print(f"Warning: Could not get current price for {symbol}: {e}")
                # Skip this position if we can't get current price
                continue
        
        return portfolio_value

    def __str__(self) -> str:
        return f"{self.name}(capital={self.initial_capital:,.0f}, commission={self.commission_rate:.3f})"
    
    def __repr__(self) -> str:
        return self.__str__()