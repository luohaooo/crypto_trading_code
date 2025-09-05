from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
from .portfolio import Portfolio, Trade, Position


class Rebalancer:
    """
    Market-neutral portfolio rebalancing engine.
    
    Handles the logic for creating long/short positions based on factor scores
    and managing portfolio rebalancing with transaction cost considerations.
    """
    
    def __init__(self, transaction_cost: float = 0.001):
        """
        Initialize rebalancer.
        
        Args:
            transaction_cost: Transaction cost as decimal (0.001 = 0.1%)
        """
        self.transaction_cost = transaction_cost
    
    def generate_rebalancing_trades(self, 
                                   current_portfolio: Portfolio,
                                   factor_scores: Dict[str, float],
                                   target_long_count: int,
                                   target_short_count: int,
                                   rebalance_date: datetime,
                                   symbol_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[Trade]:
        """
        Generate trades for portfolio rebalancing based on factor scores.
        
        Args:
            current_portfolio: Current portfolio state
            factor_scores: Dictionary mapping symbol to factor score
            target_long_count: Number of long positions to hold
            target_short_count: Number of short positions to hold
            rebalance_date: Date of rebalancing
            symbol_data: Optional symbol price data for current prices
            
        Returns:
            List of trades to execute
        """
        trades = []
        
        if not factor_scores:
            return trades
        
        # Sort symbols by factor scores
        sorted_symbols = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top symbols for long positions
        long_symbols = [symbol for symbol, score in sorted_symbols[:target_long_count]]
        
        # Select bottom symbols for short positions 
        short_symbols = [symbol for symbol, score in sorted_symbols[-target_short_count:]]
        
        # Calculate target allocations
        total_portfolio_value = current_portfolio.cash
        if symbol_data:
            total_portfolio_value = current_portfolio.calculate_total_value(symbol_data, rebalance_date)
        
        # Allocate 50% to long positions, 50% to short positions
        long_allocation = total_portfolio_value * 0.5
        short_allocation = total_portfolio_value * 0.5
        
        # Calculate position sizes
        long_position_size = long_allocation / target_long_count if target_long_count > 0 else 0
        short_position_size = short_allocation / target_short_count if target_short_count > 0 else 0
        
        # Generate trades for long positions
        for symbol in long_symbols:
            trades.extend(self._generate_symbol_trades(
                symbol=symbol,
                target_side='long',
                target_value=long_position_size,
                current_portfolio=current_portfolio,
                rebalance_date=rebalance_date,
                symbol_data=symbol_data
            ))
        
        # Generate trades for short positions
        for symbol in short_symbols:
            trades.extend(self._generate_symbol_trades(
                symbol=symbol,
                target_side='short',
                target_value=short_position_size,
                current_portfolio=current_portfolio,
                rebalance_date=rebalance_date,
                symbol_data=symbol_data
            ))
        
        # Close positions that are no longer in target universe
        current_symbols = set(current_portfolio.positions.keys())
        target_symbols = set(long_symbols + short_symbols)
        symbols_to_close = current_symbols - target_symbols
        
        for symbol in symbols_to_close:
            close_trade = self._generate_close_trade(
                symbol=symbol,
                current_portfolio=current_portfolio,
                rebalance_date=rebalance_date,
                symbol_data=symbol_data
            )
            if close_trade:
                trades.append(close_trade)
        
        return trades
    
    def _generate_symbol_trades(self, 
                               symbol: str,
                               target_side: str,
                               target_value: float,
                               current_portfolio: Portfolio,
                               rebalance_date: datetime,
                               symbol_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[Trade]:
        """
        Generate trades for a specific symbol to reach target allocation.
        
        Args:
            symbol: Symbol to trade
            target_side: 'long' or 'short'
            target_value: Target position value
            current_portfolio: Current portfolio state
            rebalance_date: Rebalancing date
            symbol_data: Optional price data
            
        Returns:
            List of trades for this symbol
        """
        trades = []
        
        # Get current price
        current_price = self._get_current_price(symbol, rebalance_date, symbol_data)
        if current_price <= 0:
            return trades
        
        # Calculate target quantity
        target_quantity = target_value / current_price
        
        # Check current position
        current_position = current_portfolio.positions.get(symbol)
        
        if current_position is None:
            # No current position - create new one
            trade = Trade(
                symbol=symbol,
                side=target_side,
                action='open',
                quantity=target_quantity,
                price=current_price,
                timestamp=rebalance_date,
                commission=target_value * self.transaction_cost,
                reason='rebalance_open'
            )
            trades.append(trade)
            
        else:
            # Have current position - check if rebalancing needed
            current_value = abs(current_position.quantity * current_price)
            value_difference = abs(current_value - target_value)
            
            # Only rebalance if difference is significant (>5% of target)
            if (value_difference / target_value) > 0.05 or current_position.side != target_side:
                # Close current position and open new one
                close_trade = Trade(
                    symbol=symbol,
                    side=current_position.side,
                    action='close',
                    quantity=current_position.quantity,
                    price=current_price,
                    timestamp=rebalance_date,
                    commission=current_value * self.transaction_cost / 2,
                    reason='rebalance_close'
                )
                trades.append(close_trade)
                
                # Open new position
                open_trade = Trade(
                    symbol=symbol,
                    side=target_side,
                    action='open',
                    quantity=target_quantity,
                    price=current_price,
                    timestamp=rebalance_date,
                    commission=target_value * self.transaction_cost / 2,
                    reason='rebalance_open'
                )
                trades.append(open_trade)
        
        return trades
    
    def _generate_close_trade(self, 
                             symbol: str,
                             current_portfolio: Portfolio,
                             rebalance_date: datetime,
                             symbol_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[Trade]:
        """
        Generate trade to close position for symbol no longer in target universe.
        
        Args:
            symbol: Symbol to close
            current_portfolio: Current portfolio state
            rebalance_date: Rebalancing date
            symbol_data: Optional price data
            
        Returns:
            Close trade or None if no position exists
        """
        current_position = current_portfolio.positions.get(symbol)
        if not current_position:
            return None
        
        current_price = self._get_current_price(symbol, rebalance_date, symbol_data)
        if current_price <= 0:
            return None
        
        position_value = abs(current_position.quantity * current_price)
        
        return Trade(
            symbol=symbol,
            side=current_position.side,
            action='close',
            quantity=current_position.quantity,
            price=current_price,
            timestamp=rebalance_date,
            commission=position_value * self.transaction_cost,
            reason='remove_from_universe'
        )
    
    def _get_current_price(self, 
                          symbol: str, 
                          date: datetime,
                          symbol_data: Optional[Dict[str, pd.DataFrame]] = None) -> float:
        """
        Get current price for a symbol at the given date.
        
        Args:
            symbol: Symbol to get price for
            date: Date to get price at
            symbol_data: Optional price data
            
        Returns:
            Current price or 0.0 if not available
        """
        if not symbol_data or symbol not in symbol_data:
            return 0.0
        
        df = symbol_data[symbol]
        if df is None or len(df) == 0:
            return 0.0
        
        # Find closest price data to the date
        price_data = df[df['open_time'] <= date]
        if len(price_data) == 0:
            # If no data before date, use first available
            price_data = df
        
        if len(price_data) > 0:
            return float(price_data['close'].iloc[-1])
        
        return 0.0
    
    def calculate_rebalancing_cost(self, trades: List[Trade]) -> float:
        """
        Calculate total cost of rebalancing trades.
        
        Args:
            trades: List of trades to execute
            
        Returns:
            Total commission cost
        """
        return sum(trade.commission for trade in trades)
    
    def optimize_trades(self, trades: List[Trade], min_trade_value: float = 100.0) -> List[Trade]:
        """
        Optimize trades by filtering out small trades to reduce transaction costs.
        
        Args:
            trades: Original list of trades
            min_trade_value: Minimum trade value to execute
            
        Returns:
            Filtered list of trades
        """
        optimized_trades = []
        
        for trade in trades:
            if trade.value >= min_trade_value:
                optimized_trades.append(trade)
        
        return optimized_trades
    
    def get_rebalancing_summary(self, trades: List[Trade]) -> Dict[str, Any]:
        """
        Get summary statistics for a rebalancing operation.
        
        Args:
            trades: List of trades executed
            
        Returns:
            Dictionary with rebalancing statistics
        """
        if not trades:
            return {}
        
        total_value = sum(trade.value for trade in trades)
        total_commission = sum(trade.commission for trade in trades)
        
        long_trades = [t for t in trades if t.side == 'long']
        short_trades = [t for t in trades if t.side == 'short']
        open_trades = [t for t in trades if t.action == 'open']
        close_trades = [t for t in trades if t.action == 'close']
        
        return {
            'total_trades': len(trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'open_trades': len(open_trades),
            'close_trades': len(close_trades),
            'total_value': total_value,
            'total_commission': total_commission,
            'commission_pct': (total_commission / total_value * 100) if total_value > 0 else 0,
            'symbols_traded': len(set(trade.symbol for trade in trades))
        }