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

from .base_strategy import BaseStrategy, PositionType
from .factors.base_factor import BaseFactor

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

        # Factor effectiveness tracking
        self.factor_history: List[Dict[str, Any]] = []  # Historical factor data for effectiveness analysis
        self.last_rebalance_data: Optional[Dict[str, Any]] = None  # Previous rebalance data for comparison
        
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
                    # Use open price for actual execution price (not close, which is future information)
                    current_price = symbol_data['open'].iloc[0]
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
                    # Use open price for actual execution price (not close, which is future information)
                    current_price = symbol_data['open'].iloc[0]
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

        # Display factor effectiveness analysis from previous period
        self._display_factor_effectiveness(data, timestamp)

        # Close all existing positions first
        closed_trades = self.close_all_positions(data, timestamp)

        if self.rebalance_count == 0:
            print(f"Initial rebalance at {timestamp}")
        else:
            print(f"Rebalance #{self.rebalance_count} at {timestamp}: Closed {len(closed_trades)} positions")

        # Get factor scores and calculate returns for display
        # factor_scores = signals.get('factor_scores', {})
        # symbol_returns = self._calculate_period_returns(data, timestamp, closed_trades)

        # Update equity curve after closing positions but before opening new ones
        self.update_equity_curve(data, timestamp)

        # Open new positions based on signals
        target_positions = signals['target_positions']
        successful_positions = 0

        for symbol, position_info in target_positions.items():
            try:
                self.open_position(
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

        # Record current period data for next factor effectiveness analysis
        self._record_current_period_data(data, timestamp, signals)

        self.rebalance_count += 1

    def _display_factor_effectiveness(self, data: pd.DataFrame, current_timestamp: pd.Timestamp) -> None:
        """
        Display factor effectiveness analysis comparing previous predictions with actual performance.

        Args:
            data: Multi-symbol OHLCV data
            current_timestamp: Current timestamp
        """
        if not self.last_rebalance_data:
            return

        last_data = self.last_rebalance_data
        print("\n" + "="*80)
        print(f"FACTOR EFFECTIVENESS ANALYSIS")
        print(f"Period: {last_data['timestamp']} -> {current_timestamp}")
        print("="*80)

        # Get current price data
        current_data = data.loc[data.index.get_level_values('open_time') == current_timestamp]

        all_results = []
        correct_predictions = 0
        total_predictions = 0

        print(f"\n{'Symbol':<12} {'Factor':<10} {'Predicted':<10} {'Entry Price':<12} {'Exit Price':<12} {'Return':<10} {'Result':<8}")
        print("-" * 90)

        # Analyze each symbol from previous rebalance
        for symbol_info in last_data['symbols']:
            symbol = symbol_info['symbol']
            factor_score = symbol_info['factor_score']
            last_price = symbol_info['price']
            position_type = symbol_info['position_type']

            try:
                # Get current price
                symbol_data = current_data.loc[
                    current_data.index.get_level_values('symbol') == symbol
                ]

                if len(symbol_data) > 0:
                    # Use open price for actual execution price (exit at next candle's open)
                    current_price = symbol_data['open'].iloc[0]

                    # Calculate actual return
                    if last_price != 0:
                        price_return = (current_price - last_price) / last_price

                        # Determine predictions and results
                        if position_type == "LONG":
                            predicted = "UP"
                            strategy_return = price_return  # Long benefits from price increase
                        else:  # SHORT
                            predicted = "DOWN"
                            strategy_return = -price_return  # Short benefits from price decrease

                        # Check if prediction was correct
                        actual_direction = "UP" if price_return > 0 else "DOWN" if price_return < 0 else "FLAT"
                        is_correct = (predicted == "UP" and price_return > 0) or (predicted == "DOWN" and price_return < 0)

                        result_text = "CORRECT" if is_correct else "WRONG"
                        if is_correct:
                            correct_predictions += 1
                        total_predictions += 1

                        # Store for correlation analysis
                        all_results.append({
                            'symbol': symbol,
                            'factor_score': factor_score,
                            'entry_price': last_price,
                            'exit_price': current_price,
                            'price_return': price_return,
                            'strategy_return': strategy_return,
                            'predicted': predicted,
                            'actual': actual_direction,
                            'correct': is_correct
                        })

                        print(f"{symbol:<12} {factor_score:>8.4f}  {predicted:<10} ${last_price:>10,.2f} ${current_price:>10,.2f} {price_return:>8.2%}  {result_text:<8}")

            except Exception as e:
                print(f"{symbol:<12} {'ERROR':<10} Could not calculate: {e}")
                continue

        # Calculate and display statistics
        if all_results:
            accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

            # Calculate average returns by prediction type
            long_results = [r for r in all_results if r['predicted'] == 'UP']
            short_results = [r for r in all_results if r['predicted'] == 'DOWN']

            long_avg_return = np.mean([r['strategy_return'] for r in long_results]) if long_results else 0.0
            short_avg_return = np.mean([r['strategy_return'] for r in short_results]) if short_results else 0.0

            # Calculate strategy spread (divide by 2 since long and short each use 50% capital)
            strategy_spread = (long_avg_return + short_avg_return) / 2.0

            print("-" * 90)
            print(f"STATISTICS:")
            print(f"  Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
            print(f"  Long Basket Avg Return: {long_avg_return:+.2%}")
            print(f"  Short Basket Avg Return: {short_avg_return:+.2%}")
            print(f"  Strategy Spread: {strategy_spread:+.2%}")

            # Store analysis results in history
            analysis_result = {
                'period_start': last_data['timestamp'],
                'period_end': current_timestamp,
                'accuracy': accuracy,
                'total_symbols': total_predictions,
                'correct_predictions': correct_predictions,
                'long_avg_return': long_avg_return,
                'short_avg_return': short_avg_return,
                'strategy_spread': strategy_spread,
                'details': all_results
            }
            self.factor_history.append(analysis_result)

        print("="*80)

    def _record_current_period_data(self, data: pd.DataFrame, timestamp: pd.Timestamp,
                                  signals: Dict[str, Any]) -> None:
        """
        Record current period data for future factor effectiveness analysis.

        Args:
            data: Multi-symbol OHLCV data
            timestamp: Current timestamp
            signals: Trading signals containing target positions
        """
        current_data = data.loc[data.index.get_level_values('open_time') == timestamp]
        factor_scores = signals.get('factor_scores', {})

        symbols_data = []

        # Record data for all symbols in target positions
        for symbol, position_info in signals.get('target_positions', {}).items():
            try:
                symbol_data = current_data.loc[
                    current_data.index.get_level_values('symbol') == symbol
                ]

                if len(symbol_data) > 0:
                    # Use open price for actual execution price (open-to-open return calculation)
                    current_price = symbol_data['open'].iloc[0]
                    factor_score = factor_scores.get(symbol, 0.0)
                    position_type = "LONG" if position_info['position_type'] == PositionType.LONG else "SHORT"

                    symbols_data.append({
                        'symbol': symbol,
                        'price': current_price,
                        'factor_score': factor_score,
                        'position_type': position_type
                    })

            except Exception as e:
                print(f"Warning: Could not record data for {symbol}: {e}")
                continue

        # Store current period data for next rebalance analysis
        self.last_rebalance_data = {
            'timestamp': timestamp,
            'symbols': symbols_data,
            'rebalance_count': self.rebalance_count
        }

    def get_factor_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of factor performance across all periods.

        Returns:
            Dictionary with factor performance metrics
        """
        if not self.factor_history:
            return {
                'total_periods': 0,
                'avg_accuracy': 0.0,
                'total_predictions': 0
            }

        accuracies = [analysis['accuracy'] for analysis in self.factor_history]
        total_predictions = sum(analysis['total_symbols'] for analysis in self.factor_history)
        total_correct = sum(analysis['correct_predictions'] for analysis in self.factor_history)

        return {
            'total_periods': len(self.factor_history),
            'avg_accuracy': np.mean(accuracies),
            'overall_accuracy': (total_correct / total_predictions * 100) if total_predictions > 0 else 0.0,
            'total_predictions': total_predictions,
            'total_correct': total_correct,
            'best_accuracy': max(accuracies),
            'worst_accuracy': min(accuracies)
        }
    
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
            'spread_return': (np.mean(long_returns) + np.mean(short_returns)) / 2 if long_returns and short_returns else 0.0
        }
    
    def __str__(self) -> str:
        return (f"NeutralStrategy(factor={self.factor.name}, "
                f"long={self.top_n}, short={self.bottom_n}, "
                f"rebalances={self.rebalance_count})")
    
    def __repr__(self) -> str:
        return self.__str__()