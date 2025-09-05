from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from ..factors.base_factor import BaseFactor
from ..factors.returns_factor import ReturnsFactor
from ..factors.cache_manager import FactorCacheManager


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, name: str, factors: List[BaseFactor]):
        self.name = name
        self.factors = factors
    
    @abstractmethod
    def generate_signals(self, 
                        symbol_data: Dict[str, Any],
                        date: datetime) -> Dict[str, float]:
        """Generate trading signals for symbols."""
        pass


class LongShortFactorStrategy(BaseStrategy):
    """
    Market-neutral long/short factor strategy.
    
    This strategy:
    1. Computes factor scores for all symbols
    2. Ranks symbols by factor values
    3. Goes long top-N symbols
    4. Goes short bottom-N symbols
    5. Rebalances at specified intervals
    """
    
    def __init__(self, 
                 factor_config: Dict[str, Any],
                 cache_manager: Optional[FactorCacheManager] = None):
        """
        Initialize long/short factor strategy.
        
        Args:
            factor_config: Configuration for factor computation
            cache_manager: Optional cache manager for factor caching
        """
        # Initialize factor based on configuration
        factor_type = factor_config.get('type', 'returns')
        
        if factor_type == 'returns':
            factor = ReturnsFactor(cache_manager)
        else:
            # Default to returns factor
            factor = ReturnsFactor(cache_manager)
        
        super().__init__('long_short_factor', [factor])
        
        self.factor_config = factor_config
        self.long_count = factor_config.get('long_count', 25)
        self.short_count = factor_config.get('short_count', 25)
        self.rebalance_frequency = factor_config.get('rebalance_frequency', '1D')
        
    def generate_signals(self, 
                        symbol_data: Dict[str, Any],
                        date: datetime) -> Dict[str, float]:
        """
        Generate factor-based trading signals.
        
        Args:
            symbol_data: Dictionary mapping symbols to price DataFrames
            date: Date to generate signals for
            
        Returns:
            Dictionary mapping symbols to factor scores
        """
        factor_scores = {}
        
        # Use the first (and primary) factor
        primary_factor = self.factors[0]
        
        for symbol, data in symbol_data.items():
            try:
                # Compute factor score with caching
                score = primary_factor.compute_with_cache(
                    symbol=symbol,
                    data=data,
                    date=date,
                    params=self.factor_config.get('params', {})
                )
                factor_scores[symbol] = score
                
            except Exception as e:
                print(f"Error computing factor for {symbol}: {e}")
                factor_scores[symbol] = 0.0
        
        return factor_scores
    
    def rank_symbols(self, factor_scores: Dict[str, float]) -> List[tuple]:
        """
        Rank symbols by factor scores.
        
        Args:
            factor_scores: Dictionary mapping symbols to factor scores
            
        Returns:
            List of (symbol, score) tuples sorted by score descending
        """
        return sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
    
    def get_long_symbols(self, ranked_symbols: List[tuple]) -> List[str]:
        """
        Get symbols for long positions (top performers).
        
        Args:
            ranked_symbols: List of (symbol, score) tuples sorted by score
            
        Returns:
            List of symbols to go long
        """
        return [symbol for symbol, score in ranked_symbols[:self.long_count]]
    
    def get_short_symbols(self, ranked_symbols: List[tuple]) -> List[str]:
        """
        Get symbols for short positions (bottom performers).
        
        Args:
            ranked_symbols: List of (symbol, score) tuples sorted by score
            
        Returns:
            List of symbols to go short
        """
        return [symbol for symbol, score in ranked_symbols[-self.short_count:]]
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """
        Get strategy configuration.
        
        Returns:
            Dictionary with strategy parameters
        """
        return {
            'name': self.name,
            'factor_config': self.factor_config,
            'long_count': self.long_count,
            'short_count': self.short_count,
            'rebalance_frequency': self.rebalance_frequency,
            'factor_types': [factor.name for factor in self.factors]
        }
    
    def validate_config(self) -> bool:
        """
        Validate strategy configuration.
        
        Returns:
            True if configuration is valid
        """
        if self.long_count <= 0 or self.short_count <= 0:
            return False
        
        if not self.factors:
            return False
        
        if self.rebalance_frequency not in ['1D', '1W', '1M', '3M']:
            return False
        
        return True
    
    def get_required_symbols_count(self) -> int:
        """
        Get minimum number of symbols required for strategy.
        
        Returns:
            Minimum symbols needed
        """
        return self.long_count + self.short_count
    
    def estimate_turnover(self, 
                         current_longs: List[str], 
                         current_shorts: List[str],
                         new_longs: List[str], 
                         new_shorts: List[str]) -> float:
        """
        Estimate portfolio turnover from position changes.
        
        Args:
            current_longs: Current long positions
            current_shorts: Current short positions
            new_longs: New long positions
            new_shorts: New short positions
            
        Returns:
            Turnover rate (0.0 to 1.0)
        """
        current_positions = set(current_longs + current_shorts)
        new_positions = set(new_longs + new_shorts)
        
        if not current_positions:
            return 1.0  # 100% turnover if starting from empty
        
        unchanged_positions = len(current_positions & new_positions)
        total_positions = len(current_positions | new_positions)
        
        return 1.0 - (unchanged_positions / max(total_positions, 1))