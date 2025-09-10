"""
Neutral Strategy Implementation

Market-neutral strategy that goes long top-ranked symbols and short bottom-ranked symbols
based on factor scores. Rebalances at specified intervals.
"""

import sys
import os

# IMPORTANT: Set paths before any imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

neural_strategy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, neural_strategy_root)

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from .base_strategy import BaseStrategy, PositionType, Position, Trade
from .factors.base_factor import BaseFactor
from .factors.returns_factor import ReturnsFactor, MomentumFactor, VolatilityAdjustedReturnsFactor

# Import from main project utils using importlib
import importlib.util
spec = importlib.util.spec_from_file_location("data_loader", os.path.join(project_root, "utils", "data_loader.py"))
data_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader_module)
get_multiple_symbols_data = data_loader_module.get_multiple_symbols_data


class NeutralStrategy(BaseStrategy):
    """
    Market-neutral strategy implementation.
    
    Strategy mechanics:
    1. Calculate factor scores for all symbols
    2. Rank symbols by factor scores
    3. Go long top N symbols with 50% of capital
    4. Go short bottom N symbols with 50% of capital
    5. Rebalance at specified intervals
    """
    
    def __init__(self, 
                 factor: BaseFactor,
                 top_n: int = 10,
                 bottom_n: int = 10,
                 initial_capital: float = 1000000.0,
                 commission_rate: float = 0.001,
                 name: Optional[str] = None):
        """
        Initialize neutral strategy.
        
        Args:
            factor: Factor to use for symbol ranking
            top_n: Number of long positions
            bottom_n: Number of short positions  
            initial_capital: Starting capital
            commission_rate: Trading commission rate
            name: Strategy name (auto-generated if None)
        """
        if name is None:
            name = f"NeutralStrategy_{factor.name}_{top_n}L_{bottom_n}S"
        
        super().__init__(name, initial_capital, commission_rate)
        
        self.factor = factor
        self.top_n = top_n
        self.bottom_n = bottom_n
        
        # Strategy state
        self.long_allocation = 0.5  # 50% for long positions
        self.short_allocation = 0.5  # 50% for short positions
        self.rebalance_count = 0
        
        # Track target positions for monitoring
        self.target_long_symbols: List[str] = []
        self.target_short_symbols: List[str] = []
        
    def generate_signals(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """
        Generate trading signals based on factor rankings.
        
        Args:
            data: Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
            timestamp: Current timestamp
            
        Returns:
            Dictionary containing target positions and metadata
        """
        # Validate that we have sufficient data for factor calculation
        if not self.factor.validate_data(data, timestamp):
            return {
                'action': 'no_signal',
                'reason': 'insufficient_data',
                'timestamp': timestamp
            }
        
        # Get factor rankings
        rankings = self.factor.rank_symbols(data, timestamp, self.top_n, self.bottom_n)
        
        long_symbols = rankings['long']
        short_symbols = rankings['short']
        
        # Calculate position sizes based on available capital
        current_portfolio_value = self.calculate_portfolio_value(data, timestamp)
        
        long_capital = current_portfolio_value * self.long_allocation
        short_capital = current_portfolio_value * self.short_allocation
        
        # Calculate individual position sizes
        long_position_size = long_capital / len(long_symbols) if long_symbols else 0
        short_position_size = short_capital / len(short_symbols) if short_symbols else 0
        
        # Get current prices for position sizing
        current_data = data.loc[data.index.get_level_values('open_time') == timestamp]
        
        target_positions = {}
        
        # Long positions
        for symbol in long_symbols:
            try:
                symbol_data = current_data.loc[
                    current_data.index.get_level_values('symbol') == symbol
                ]
                if len(symbol_data) > 0:
                    current_price = symbol_data['close'].iloc[0]
                    shares = long_position_size / current_price
                    target_positions[symbol] = {
                        'position_type': PositionType.LONG,
                        'size': shares,
                        'price': current_price
                    }
            except Exception as e:
                print(f"Warning: Could not calculate position size for {symbol}: {e}")
        
        # Short positions
        for symbol in short_symbols:
            try:
                symbol_data = current_data.loc[
                    current_data.index.get_level_values('symbol') == symbol
                ]
                if len(symbol_data) > 0:
                    current_price = symbol_data['close'].iloc[0]
                    shares = short_position_size / current_price
                    target_positions[symbol] = {
                        'position_type': PositionType.SHORT,
                        'size': shares,
                        'price': current_price
                    }
            except Exception as e:
                print(f"Warning: Could not calculate position size for {symbol}: {e}")
        
        # Store targets for monitoring
        self.target_long_symbols = long_symbols
        self.target_short_symbols = short_symbols
        
        return {
            'action': 'rebalance',
            'target_positions': target_positions,
            'long_symbols': long_symbols,
            'short_symbols': short_symbols,
            'timestamp': timestamp,
            'portfolio_value': current_portfolio_value,
            'factor_scores': self.factor.calculate(data, timestamp)
        }
    
    def execute_rebalance(self, data: pd.DataFrame, timestamp: pd.Timestamp, 
                         signals: Dict[str, Any]) -> None:
        """
        Execute portfolio rebalancing based on generated signals.
        
        Args:
            data: Multi-symbol OHLCV data
            timestamp: Current timestamp
            signals: Trading signals from generate_signals()
        """
        if signals['action'] != 'rebalance':
            return
        
        # Close all existing positions first
        closed_trades = self.close_all_positions(data, timestamp)
        
        if self.rebalance_count == 0:
            print(f"Initial rebalance at {timestamp}")
        else:
            print(f"Rebalance #{self.rebalance_count} at {timestamp}: Closed {len(closed_trades)} positions")
        
        # Open new positions based on signals
        target_positions = signals['target_positions']
        successful_positions = 0
        
        for symbol, position_info in target_positions.items():
            try:
                position = self.open_position(
                    symbol=symbol,
                    position_type=position_info['position_type'],
                    size=position_info['size'],
                    entry_price=position_info['price'],
                    entry_time=timestamp
                )
                successful_positions += 1
                
            except Exception as e:
                print(f"Warning: Failed to open position for {symbol}: {e}")
                continue
        
        print(f"Opened {successful_positions} new positions (target: {len(target_positions)})")
        print(f"Long: {len(signals['long_symbols'])}, Short: {len(signals['short_symbols'])}")
        
        self.rebalance_count += 1
        
        # Update equity curve
        self.update_equity_curve(data, timestamp)
    
    def get_current_exposures(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate current portfolio exposures.
        
        Args:
            data: Multi-symbol OHLCV data
            timestamp: Current timestamp
            
        Returns:
            Dictionary with exposure metrics
        """
        current_data = data.loc[data.index.get_level_values('open_time') == timestamp]
        
        long_exposure = 0.0
        short_exposure = 0.0
        gross_exposure = 0.0
        
        for symbol, position in self.positions.items():
            try:
                symbol_data = current_data.loc[
                    current_data.index.get_level_values('symbol') == symbol
                ]
                if len(symbol_data) > 0:
                    current_price = symbol_data['close'].iloc[0]
                    notional = position.size * current_price
                    
                    if position.position_type == PositionType.LONG:
                        long_exposure += notional
                    elif position.position_type == PositionType.SHORT:
                        short_exposure += notional
                    
                    gross_exposure += notional
                    
            except Exception as e:
                print(f"Warning: Could not calculate exposure for {symbol}: {e}")
                continue
        
        portfolio_value = self.calculate_portfolio_value(data, timestamp)
        
        return {
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': long_exposure - short_exposure,
            'gross_exposure': gross_exposure,
            'long_exposure_pct': (long_exposure / portfolio_value) * 100 if portfolio_value > 0 else 0,
            'short_exposure_pct': (short_exposure / portfolio_value) * 100 if portfolio_value > 0 else 0,
            'net_exposure_pct': ((long_exposure - short_exposure) / portfolio_value) * 100 if portfolio_value > 0 else 0
        }
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get summary of current positions.
        
        Returns:
            Dictionary with position summary
        """
        long_positions = [p for p in self.positions.values() if p.position_type == PositionType.LONG]
        short_positions = [p for p in self.positions.values() if p.position_type == PositionType.SHORT]
        
        return {
            'total_positions': len(self.positions),
            'long_positions': len(long_positions),
            'short_positions': len(short_positions),
            'target_long': self.top_n,
            'target_short': self.bottom_n,
            'rebalance_count': self.rebalance_count,
            'long_symbols': [p.symbol for p in long_positions],
            'short_symbols': [p.symbol for p in short_positions]
        }
    
    def analyze_factor_performance(self, data: pd.DataFrame, 
                                 start_time: pd.Timestamp, 
                                 end_time: pd.Timestamp) -> Dict[str, Any]:
        """
        Analyze factor performance over a time period.
        
        Args:
            data: Multi-symbol OHLCV data
            start_time: Analysis start time
            end_time: Analysis end time
            
        Returns:
            Dictionary with factor analysis
        """
        # Get factor scores at start
        start_scores = self.factor.calculate(data, start_time)
        
        # Calculate returns for each symbol from start to end
        start_data = data.loc[data.index.get_level_values('open_time') == start_time]
        end_data = data.loc[data.index.get_level_values('open_time') == end_time]
        
        symbol_returns = {}
        
        for symbol in start_scores.index:
            try:
                start_price_data = start_data.loc[start_data.index.get_level_values('symbol') == symbol]
                end_price_data = end_data.loc[end_data.index.get_level_values('symbol') == symbol]
                
                if len(start_price_data) > 0 and len(end_price_data) > 0:
                    start_price = start_price_data['close'].iloc[0]
                    end_price = end_price_data['close'].iloc[0]
                    
                    if start_price != 0:
                        symbol_returns[symbol] = (end_price - start_price) / start_price
                        
            except Exception as e:
                print(f"Warning: Could not calculate return for {symbol}: {e}")
                continue
        
        # Analyze correlation between factor scores and subsequent returns
        factor_return_pairs = []
        for symbol in start_scores.index:
            if symbol in symbol_returns:
                factor_return_pairs.append((start_scores[symbol], symbol_returns[symbol]))
        
        if len(factor_return_pairs) > 1:
            factor_scores_array = np.array([pair[0] for pair in factor_return_pairs])
            returns_array = np.array([pair[1] for pair in factor_return_pairs])
            
            correlation = np.corrcoef(factor_scores_array, returns_array)[0, 1]
        else:
            correlation = 0.0
        
        # Analyze long/short basket performance
        long_returns = [symbol_returns[s] for s in self.target_long_symbols if s in symbol_returns]
        short_returns = [symbol_returns[s] for s in self.target_short_symbols if s in symbol_returns]
        
        return {
            'factor_return_correlation': correlation,
            'long_basket_avg_return': np.mean(long_returns) if long_returns else 0.0,
            'short_basket_avg_return': np.mean(short_returns) if short_returns else 0.0,
            'long_basket_size': len(long_returns),
            'short_basket_size': len(short_returns),
            'spread_return': (np.mean(long_returns) - np.mean(short_returns)) if long_returns and short_returns else 0.0
        }
    
    def __str__(self) -> str:
        return (f"NeutralStrategy(factor={self.factor.name}, "
                f"long={self.top_n}, short={self.bottom_n}, "
                f"rebalances={self.rebalance_count})")
    
    def __repr__(self) -> str:
        return self.__str__()