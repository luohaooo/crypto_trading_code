"""
Base Factor Class

Abstract base class for all factors used in trading strategies.
Provides standardized interface for factor calculation and ranking.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseFactor(ABC):
    """
    Abstract base class for all factors used in neutral trading strategies.
    
    Factors are used to rank symbols for long/short position selection.
    Higher factor values indicate symbols suitable for long positions,
    lower factor values indicate symbols suitable for short positions.
    """
    
    def __init__(self, name: str):
        """
        Initialize factor with basic parameters.
        
        Args:
            name: Factor name for identification
            lookback_periods: Number of periods to look back for calculation
        """
        self.name = name
    
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
    