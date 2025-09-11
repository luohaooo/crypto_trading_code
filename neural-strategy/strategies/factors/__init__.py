"""
Factors Package

Contains all factor implementations for ranking and scoring symbols
in trading strategies.
"""

from .base_factor import BaseFactor
from .returns_factor import ReturnsFactor, MomentumFactor, VolatilityAdjustedReturnsFactor
from .ohlc_figure_factor import OHLCFigureFactor

__all__ = [
    'BaseFactor',
    'ReturnsFactor', 
    'MomentumFactor',
    'VolatilityAdjustedReturnsFactor',
    'OHLCFigureFactor'
]