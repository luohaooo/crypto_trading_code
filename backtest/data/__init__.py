"""
Data management components for the backtesting framework.

This package contains modules for efficient data loading and memory management:
- DataManager: Coordinates multi-symbol data loading
- MemoryManager: Handles memory optimization and cleanup
"""

from .memory_manager import MemoryManager
from .data_manager import DataManager

__all__ = ['MemoryManager', 'DataManager']