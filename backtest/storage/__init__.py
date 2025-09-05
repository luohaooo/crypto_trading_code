"""
Data storage and persistence for backtest results.

This package handles:
- Trade log storage and retrieval
- Performance metrics storage
- Result archiving and export
"""

from .trade_logger import TradeLogger

__all__ = ['TradeLogger']