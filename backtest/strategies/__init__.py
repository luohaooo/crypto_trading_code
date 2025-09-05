"""
Market-neutral trading strategies for the backtesting framework.

This package contains strategy implementations for market-neutral trading:
- LongShortFactorStrategy: Core long/short factor-based strategy
- BaseStrategy: Abstract base class for all strategies
"""

from .long_short_factor import LongShortFactorStrategy

__all__ = ['LongShortFactorStrategy']