"""
Core components for the backtesting framework.

This package contains the main backtesting engine and supporting components:
- BacktestEngine: Main orchestrator for strategy backtesting
- Portfolio: Position tracking and P&L calculation
- Rebalancer: Market-neutral portfolio rebalancing logic
"""

from .engine import BacktestEngine
from .portfolio import Portfolio
from .rebalancer import Rebalancer

__all__ = ['BacktestEngine', 'Portfolio', 'Rebalancer']