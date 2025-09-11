"""
Returns Factor Implementation

Factor based on price returns over various lookback periods.
Higher returns indicate long candidates, lower returns indicate short candidates.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Optional

# Add parent directories to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)

neural_strategy_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, neural_strategy_root)

from .base_factor import BaseFactor


class ReturnsFactor(BaseFactor):
    """
    Calculate returns-based factor for symbol ranking.
    
    This factor calculates price returns over a specified lookback period
    and ranks symbols accordingly. Positive returns indicate upward momentum
    (long candidates), negative returns indicate downward momentum (short candidates).
    """
    
    def __init__(self, lookback_periods: int = 240, name: Optional[str] = None):
        """
        Initialize returns factor.
        
        Args:
            lookback_periods: Number of periods to look back for returns calculation
                             (e.g., 240 for 4-hour returns if using 1-minute data)
            name: Custom name for the factor (auto-generated if None)
        """
        if name is None:
            # Convert periods to human-readable format
            if lookback_periods >= 1440:  # >= 1 day
                days = lookback_periods // 1440
                name = f"Returns_{days}D"
            elif lookback_periods >= 60:  # >= 1 hour
                hours = lookback_periods // 60
                name = f"Returns_{hours}H"
            else:
                name = f"Returns_{lookback_periods}M"
        
        super().__init__(name, lookback_periods)
    
    def calculate(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        """
        Calculate returns factor for all symbols at given timestamp.
        
        Args:
            data: Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
            timestamp: Current timestamp for factor calculation
            
        Returns:
            pd.Series: Returns factor values indexed by symbol
        """
        if not self.validate_data(data, timestamp):
            return pd.Series(dtype=float)
        
        # Get current prices for all symbols
        current_data = data.loc[data.index.get_level_values('open_time') == timestamp]
        if len(current_data) == 0:
            return pd.Series(dtype=float)
        
        current_prices = current_data['close']
        
        # Get historical data for lookback calculation
        historical_data = self.get_historical_data(data, timestamp, self.lookback_periods + 1)
        
        returns = {}
        
        for symbol in current_prices.index.get_level_values('symbol'):
            try:
                # Get symbol-specific historical data
                symbol_data = historical_data.loc[
                    historical_data.index.get_level_values('symbol') == symbol
                ]
                
                if len(symbol_data) < 2:
                    continue  # Need at least 2 data points
                
                # Sort by timestamp to ensure correct ordering
                symbol_data = symbol_data.sort_index(level='open_time')
                
                # Get current and historical prices
                current_price = current_prices.loc[
                    current_prices.index.get_level_values('symbol') == symbol
                ].iloc[0]
                
                # Get price from lookback_periods ago
                if len(symbol_data) >= self.lookback_periods + 1:
                    historical_price = symbol_data['close'].iloc[-(self.lookback_periods + 1)]
                else:
                    historical_price = symbol_data['close'].iloc[0]
                
                # Calculate returns
                if historical_price != 0:
                    returns[symbol] = (current_price - historical_price) / historical_price
                else:
                    returns[symbol] = 0.0
                    
            except Exception as e:
                print(f"Warning: Could not calculate returns for {symbol}: {e}")
                continue
        
        return pd.Series(returns, name=f'{self.name}_factor')
    
    def calculate_log_returns(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        """
        Calculate log returns instead of simple returns.
        
        Args:
            data: Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
            timestamp: Current timestamp for factor calculation
            
        Returns:
            pd.Series: Log returns factor values indexed by symbol
        """
        if not self.validate_data(data, timestamp):
            return pd.Series(dtype=float)
        
        # Get current prices for all symbols
        current_data = data.loc[data.index.get_level_values('open_time') == timestamp]
        if len(current_data) == 0:
            return pd.Series(dtype=float)
        
        current_prices = current_data['close']
        
        # Get historical data for lookback calculation
        historical_data = self.get_historical_data(data, timestamp, self.lookback_periods + 1)
        
        log_returns = {}
        
        for symbol in current_prices.index.get_level_values('symbol'):
            try:
                # Get symbol-specific historical data
                symbol_data = historical_data.loc[
                    historical_data.index.get_level_values('symbol') == symbol
                ]
                
                if len(symbol_data) < 2:
                    continue
                
                # Sort by timestamp
                symbol_data = symbol_data.sort_index(level='open_time')
                
                # Get current and historical prices
                current_price = current_prices.loc[
                    current_prices.index.get_level_values('symbol') == symbol
                ].iloc[0]
                
                # Get price from lookback_periods ago
                if len(symbol_data) >= self.lookback_periods + 1:
                    historical_price = symbol_data['close'].iloc[-(self.lookback_periods + 1)]
                else:
                    historical_price = symbol_data['close'].iloc[0]
                
                # Calculate log returns
                if historical_price > 0 and current_price > 0:
                    log_returns[symbol] = np.log(current_price / historical_price)
                else:
                    log_returns[symbol] = 0.0
                    
            except Exception as e:
                print(f"Warning: Could not calculate log returns for {symbol}: {e}")
                continue
        
        return pd.Series(log_returns, name=f'{self.name}_log_returns')


class MomentumFactor(BaseFactor):
    """
    Momentum factor based on price momentum over multiple timeframes.
    
    Combines short-term and long-term momentum to identify trending symbols.
    """
    
    def __init__(self, short_period: int = 60, long_period: int = 240, 
                 name: Optional[str] = None):
        """
        Initialize momentum factor.
        
        Args:
            short_period: Short-term momentum period (e.g., 60 minutes)
            long_period: Long-term momentum period (e.g., 240 minutes)
            name: Custom name for the factor
        """
        self.short_period = short_period
        self.long_period = long_period
        
        if name is None:
            short_h = short_period // 60
            long_h = long_period // 60
            name = f"Momentum_{short_h}H_{long_h}H"
        
        super().__init__(name, max(short_period, long_period))
    
    def calculate(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        """
        Calculate momentum factor combining short and long-term momentum.
        
        Args:
            data: Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
            timestamp: Current timestamp for factor calculation
            
        Returns:
            pd.Series: Momentum factor values indexed by symbol
        """
        # Create returns factors for short and long periods
        short_returns = ReturnsFactor(self.short_period, "short_momentum")
        long_returns = ReturnsFactor(self.long_period, "long_momentum")
        
        # Calculate both momentum components
        short_momentum = short_returns.calculate(data, timestamp)
        long_momentum = long_returns.calculate(data, timestamp)
        
        # Combine momentum (weighted average with more weight on short-term)
        momentum_scores = {}
        
        for symbol in short_momentum.index:
            if symbol in long_momentum.index:
                # Weight short-term momentum more heavily (0.7 vs 0.3)
                momentum_scores[symbol] = (
                    0.7 * short_momentum[symbol] + 
                    0.3 * long_momentum[symbol]
                )
        
        return pd.Series(momentum_scores, name=f'{self.name}_factor')

class OHLCfigureFactor(BaseFactor):
    """
        Generated by processing the ohlc figure by neural networks
    """
    def __init__(self, timeframes: list = [3, 15, 60], 
                 model: ,
                 name: Optional[str] = None):
        
        self.timeframes = timeframes
        
        if name is None:
            short_h = short_period // 60
            long_h = long_period // 60
            name = f"Momentum_{short_h}H_{long_h}H"
        
        super().__init__(name, max(short_period, long_period))
    
    def calculate(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        """
        Calculate momentum factor combining short and long-term momentum.
        
        Args:
            data: Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
            timestamp: Current timestamp for factor calculation
            
        Returns:
            pd.Series: Momentum factor values indexed by symbol
        """
        # Create returns factors for short and long periods
        short_returns = ReturnsFactor(self.short_period, "short_momentum")
        long_returns = ReturnsFactor(self.long_period, "long_momentum")
        
        # Calculate both momentum components
        short_momentum = short_returns.calculate(data, timestamp)
        long_momentum = long_returns.calculate(data, timestamp)
        
        # Combine momentum (weighted average with more weight on short-term)
        momentum_scores = {}
        
        for symbol in short_momentum.index:
            if symbol in long_momentum.index:
                # Weight short-term momentum more heavily (0.7 vs 0.3)
                momentum_scores[symbol] = (
                    0.7 * short_momentum[symbol] + 
                    0.3 * long_momentum[symbol]
                )
        
        return pd.Series(momentum_scores, name=f'{self.name}_factor')


class VolatilityAdjustedReturnsFactor(BaseFactor):
    """
    Returns factor adjusted for volatility (Sharpe-like ratio).
    
    Calculates returns divided by volatility to identify symbols with
    consistent directional movement rather than just high absolute returns.
    """
    
    def __init__(self, lookback_periods: int = 240, volatility_periods: int = 240,
                 name: Optional[str] = None):
        """
        Initialize volatility-adjusted returns factor.
        
        Args:
            lookback_periods: Periods for returns calculation
            volatility_periods: Periods for volatility calculation
            name: Custom name for the factor
        """
        self.volatility_periods = volatility_periods
        
        if name is None:
            hours = lookback_periods // 60
            name = f"VolAdjReturns_{hours}H"
        
        super().__init__(name, max(lookback_periods, volatility_periods))
    
    def calculate(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        """
        Calculate volatility-adjusted returns factor.
        
        Args:
            data: Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
            timestamp: Current timestamp for factor calculation
            
        Returns:
            pd.Series: Volatility-adjusted returns factor values
        """
        if not self.validate_data(data, timestamp):
            return pd.Series(dtype=float)
        
        # Calculate returns
        returns_factor = ReturnsFactor(self.lookback_periods)
        returns = returns_factor.calculate(data, timestamp)
        
        # Calculate volatility for each symbol
        historical_data = self.get_historical_data(data, timestamp, self.volatility_periods)
        
        vol_adj_returns = {}
        
        for symbol in returns.index:
            try:
                # Get symbol-specific historical data
                symbol_data = historical_data.loc[
                    historical_data.index.get_level_values('symbol') == symbol
                ]
                
                if len(symbol_data) < 2:
                    continue
                
                # Calculate rolling returns for volatility
                symbol_data = symbol_data.sort_index(level='open_time')
                prices = symbol_data['close']
                rolling_returns = prices.pct_change().dropna()
                
                if len(rolling_returns) < 2:
                    continue
                
                # Calculate volatility (standard deviation of returns)
                volatility = rolling_returns.std()
                
                if volatility != 0 and not np.isnan(volatility):
                    vol_adj_returns[symbol] = returns[symbol] / volatility
                else:
                    vol_adj_returns[symbol] = 0.0
                    
            except Exception as e:
                print(f"Warning: Could not calculate vol-adjusted returns for {symbol}: {e}")
                continue
        
        return pd.Series(vol_adj_returns, name=f'{self.name}_factor')


