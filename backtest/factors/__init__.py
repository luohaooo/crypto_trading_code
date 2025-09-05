"""
Factor computation system for the backtesting framework.

This package contains factor implementations and caching infrastructure:
- BaseFactor: Abstract base class for all factors
- ReturnsFactor: Returns-based factors (momentum, mean reversion)
- FactorCacheManager: Multi-tier caching system for factor results
"""

from .base_factor import BaseFactor
from .returns_factor import ReturnsFactor, VolatilityFactor, MomentumFactor
from .cache_manager import FactorCacheManager

__all__ = [
    'BaseFactor',
    'ReturnsFactor', 
    'VolatilityFactor',
    'MomentumFactor',
    'FactorCacheManager'
]