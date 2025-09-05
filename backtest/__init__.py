"""
Market-neutral backtesting framework for cryptocurrency trading.

This module provides comprehensive backtesting capabilities for market-neutral
strategies including:
- Factor-based long/short portfolio construction
- Real-time performance tracking and visualization
- Memory-efficient processing of large datasets
- Advanced performance metrics and risk analytics
"""

__version__ = "1.0.0"
__author__ = "Crypto Trading Framework"

from .core.engine import BacktestEngine
from .strategies.long_short_factor import LongShortFactorStrategy
from .factors.returns_factor import ReturnsFactor

__all__ = [
    'BacktestEngine',
    'LongShortFactorStrategy', 
    'ReturnsFactor'
]