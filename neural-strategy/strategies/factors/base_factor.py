"""
Base Factor Class

Abstract base class for all factors used in trading strategies.
Provides standardized interface for factor calculation and ranking.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List


class BaseFactor(ABC):
    """
    Abstract base class for all factors used in neutral trading strategies.
    
    Factors are used to rank symbols for long/short position selection.
    Higher factor values indicate symbols suitable for long positions,
    lower factor values indicate symbols suitable for short positions.
    """
    
    def __init__(self, name: str, lookback_periods: int = 1):
        """
        Initialize factor with basic parameters.
        
        Args:
            name: Factor name for identification
            lookback_periods: Number of periods to look back for calculation
        """
        self.name = name
        self.lookback_periods = lookback_periods
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        """
        Calculate factor values for all symbols at a given timestamp.
        
        Args:
            data: Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
            timestamp: Current timestamp for factor calculation
            
        Returns:
            pd.Series: Factor values indexed by symbol, higher values = long candidates
        """
        pass
    
    def rank_symbols(self, data: pd.DataFrame, timestamp: pd.Timestamp, 
                    top_n: int, bottom_n: int) -> Dict[str, List[str]]:
        """
        Rank symbols and return top N for long and bottom N for short positions.
        
        Args:
            data: Multi-symbol OHLCV data
            timestamp: Current timestamp
            top_n: Number of top symbols for long positions
            bottom_n: Number of bottom symbols for short positions
            
        Returns:
            Dict with 'long' and 'short' lists of symbols
        """
        factor_values = self.calculate(data, timestamp)
        
        # Remove NaN values
        factor_values = factor_values.dropna()
        
        # Sort in descending order (high values = long candidates)
        sorted_symbols = factor_values.sort_values(ascending=False)
        
        long_symbols = sorted_symbols.head(top_n).index.tolist()
        short_symbols = sorted_symbols.tail(bottom_n).index.tolist()
        
        return {
            'long': long_symbols,
            'short': short_symbols
        }
    
    def get_historical_data(self, data: pd.DataFrame, timestamp: pd.Timestamp, 
                           periods: int) -> pd.DataFrame:
        """
        Get historical data for the specified number of periods before timestamp.
        
        Args:
            data: Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
            timestamp: Current timestamp
            periods: Number of periods to look back
            
        Returns:
            Historical data subset
        """
        # Get all timestamps up to current timestamp
        available_times = data.index.get_level_values('open_time').unique()
        available_times = available_times[available_times <= timestamp].sort_values()
        
        if len(available_times) < periods:
            # Return all available data if not enough history
            end_time = available_times[-1] if len(available_times) > 0 else timestamp
            start_time = available_times[0] if len(available_times) > 0 else timestamp
        else:
            # Get the last 'periods' timestamps
            end_time = available_times[-1]
            start_time = available_times[-periods]
        
        return data.loc[start_time:end_time]
    
    def validate_data(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> bool:
        """
        Validate that sufficient data is available for factor calculation.
        
        Args:
            data: Multi-symbol OHLCV data
            timestamp: Current timestamp
            
        Returns:
            True if sufficient data available, False otherwise
        """
        historical_data = self.get_historical_data(data, timestamp, self.lookback_periods)
        
        # Check if we have enough historical data
        if len(historical_data) < self.lookback_periods:
            return False
            
        # Check if we have data for the current timestamp
        current_data = data.loc[data.index.get_level_values('open_time') == timestamp]
        if len(current_data) == 0:
            return False
            
        return True
    
    def __str__(self) -> str:
        return f"{self.name}(lookback_periods={self.lookback_periods})"
    
    def __repr__(self) -> str:
        return self.__str__()