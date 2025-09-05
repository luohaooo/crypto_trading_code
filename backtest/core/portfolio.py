from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: Optional[float] = None
    
    def get_market_value(self, current_price: float) -> float:
        """Calculate current market value of position."""
        if self.side == 'long':
            return self.quantity * current_price
        else:  # short position
            return self.quantity * (2 * self.entry_price - current_price)
    
    def get_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L of position."""
        if self.side == 'long':
            return self.quantity * (current_price - self.entry_price)
        else:  # short position
            return self.quantity * (self.entry_price - current_price)


@dataclass
class Trade:
    """Represents a trade execution."""
    symbol: str
    side: str  # 'long' or 'short' 
    action: str  # 'open', 'close', 'rebalance'
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    reason: str = 'rebalance'
    
    @property
    def value(self) -> float:
        """Trade value (quantity * price)."""
        return abs(self.quantity * self.price)


class Portfolio:
    """
    Portfolio management for market-neutral backtesting.
    
    Tracks long/short positions, calculates P&L, and manages
    portfolio rebalancing for market-neutral strategies.
    """
    
    def __init__(self, initial_capital: float):
        """
        Initialize portfolio with starting capital.
        
        Args:
            initial_capital: Starting capital amount
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        
    def execute_trade(self, trade: Trade) -> bool:
        """
        Execute a trade and update portfolio state.
        
        Args:
            trade: Trade to execute
            
        Returns:
            True if trade executed successfully
        """
        try:
            # Calculate trade costs
            trade_value = trade.value
            total_cost = trade_value + trade.commission
            
            if trade.action == 'open':
                # Open new position
                if trade.side == 'long' and total_cost > self.cash:
                    return False  # Insufficient cash
                
                # Create new position
                position = Position(
                    symbol=trade.symbol,
                    side=trade.side,
                    quantity=trade.quantity,
                    entry_price=trade.price,
                    entry_time=trade.timestamp
                )
                
                self.positions[trade.symbol] = position
                
                # Update cash (for long positions, reduce cash; for short, increase cash)
                if trade.side == 'long':
                    self.cash -= total_cost
                else:  # short position
                    self.cash += trade_value - trade.commission
                    
            elif trade.action == 'close':
                # Close existing position
                if trade.symbol not in self.positions:
                    return False  # No position to close
                    
                position = self.positions[trade.symbol]
                
                # Update cash based on position close
                if position.side == 'long':
                    self.cash += trade_value - trade.commission
                else:  # short position
                    # For short positions, we need to buy back at current price
                    realized_pnl = position.quantity * (position.entry_price - trade.price)
                    self.cash += realized_pnl - trade.commission
                
                # Remove position
                del self.positions[trade.symbol]
                
            elif trade.action == 'rebalance':
                # Rebalancing combines close + open in one step
                if trade.symbol in self.positions:
                    # Close existing position first
                    old_position = self.positions[trade.symbol]
                    close_trade = Trade(
                        symbol=trade.symbol,
                        side=old_position.side,
                        action='close',
                        quantity=old_position.quantity,
                        price=trade.price,
                        timestamp=trade.timestamp,
                        commission=trade.commission / 2  # Split commission
                    )
                    self.execute_trade(close_trade)
                
                # Open new position
                open_trade = Trade(
                    symbol=trade.symbol,
                    side=trade.side,
                    action='open',
                    quantity=trade.quantity,
                    price=trade.price,
                    timestamp=trade.timestamp,
                    commission=trade.commission / 2  # Split commission
                )
                self.execute_trade(open_trade)
            
            # Record trade
            self.trade_history.append(trade)
            return True
            
        except Exception as e:
            print(f"Error executing trade: {e}")
            return False
    
    def get_current_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_position_value(self, symbol_prices: Dict[str, float]) -> float:
        """Calculate total value of all positions."""
        total_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in symbol_prices:
                current_price = symbol_prices[symbol]
                market_value = position.get_market_value(current_price)
                total_value += market_value
        
        return total_value
    
    def calculate_total_value(self, symbol_data: Dict[str, pd.DataFrame], date: datetime) -> float:
        """Calculate total portfolio value including cash and positions."""
        # Get current prices for all symbols
        symbol_prices = {}
        for symbol, df in symbol_data.items():
            if symbol in self.positions:
                # Find closest price to the date
                price_data = df[df['open_time'] <= date]
                if len(price_data) > 0:
                    symbol_prices[symbol] = price_data['close'].iloc[-1]
        
        position_value = self.get_position_value(symbol_prices)
        return self.cash + position_value
    
    def get_long_exposure(self, symbol_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total long exposure."""
        long_exposure = 0.0
        
        for symbol, position in self.positions.items():
            if position.side == 'long' and symbol in symbol_data:
                df = symbol_data[symbol]
                if len(df) > 0:
                    current_price = df['close'].iloc[-1]
                    long_exposure += position.quantity * current_price
        
        return long_exposure
    
    def get_short_exposure(self, symbol_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total short exposure."""
        short_exposure = 0.0
        
        for symbol, position in self.positions.items():
            if position.side == 'short' and symbol in symbol_data:
                df = symbol_data[symbol]
                if len(df) > 0:
                    current_price = df['close'].iloc[-1] 
                    short_exposure += position.quantity * current_price
        
        return short_exposure
    
    def get_net_exposure(self, symbol_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate net exposure (long - short)."""
        return self.get_long_exposure(symbol_data) - self.get_short_exposure(symbol_data)
    
    def get_portfolio_metrics(self, symbol_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get comprehensive portfolio metrics."""
        total_value = self.calculate_total_value(symbol_data, datetime.now())
        long_exposure = self.get_long_exposure(symbol_data)
        short_exposure = self.get_short_exposure(symbol_data)
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'position_count': len(self.positions),
            'long_positions': len([p for p in self.positions.values() if p.side == 'long']),
            'short_positions': len([p for p in self.positions.values() if p.side == 'short']),
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': long_exposure - short_exposure,
            'gross_exposure': long_exposure + short_exposure,
            'cash_pct': (self.cash / total_value) * 100 if total_value > 0 else 0
        }
    
    def get_unrealized_pnl(self, symbol_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate unrealized P&L for all positions."""
        unrealized_pnl = {}
        
        for symbol, position in self.positions.items():
            if symbol in symbol_data:
                df = symbol_data[symbol]
                if len(df) > 0:
                    current_price = df['close'].iloc[-1]
                    pnl = position.get_pnl(current_price)
                    unrealized_pnl[symbol] = pnl
        
        return unrealized_pnl
    
    def close_all_positions(self, symbol_data: Dict[str, pd.DataFrame], timestamp: datetime) -> List[Trade]:
        """Close all open positions."""
        close_trades = []
        
        for symbol, position in list(self.positions.items()):
            if symbol in symbol_data:
                df = symbol_data[symbol]
                if len(df) > 0:
                    current_price = df['close'].iloc[-1]
                    
                    close_trade = Trade(
                        symbol=symbol,
                        side=position.side,
                        action='close',
                        quantity=position.quantity,
                        price=current_price,
                        timestamp=timestamp,
                        reason='close_all'
                    )
                    
                    if self.execute_trade(close_trade):
                        close_trades.append(close_trade)
        
        return close_trades
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trade_history.clear()
        self.portfolio_history.clear()