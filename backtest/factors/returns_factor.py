from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .base_factor import BaseFactor


class ReturnsFactor(BaseFactor):
    """
    Returns-based factor implementation.
    
    Computes various return metrics including simple returns, log returns,
    and rolling returns over different time horizons.
    """
    
    def __init__(self, cache_manager=None):
        default_params = {
            'lookback_hours': 96,  # 4 days default
            'return_type': 'simple',  # 'simple', 'log'
            'price_col': 'close'
        }
        super().__init__('returns', cache_manager, default_params)
    
    def compute(self, 
               symbol: str,
               data: pd.DataFrame, 
               date: datetime,
               params: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute returns factor for the specified lookback period.
        
        Args:
            symbol: Symbol identifier
            data: Price data DataFrame with 'open_time' and price columns
            date: Target date for return computation
            params: Parameters including 'lookback_hours', 'return_type', 'price_col'
            
        Returns:
            Return value over the lookback period
        """
        final_params = {**self.default_params, **(params or {})}
        lookback_hours = final_params['lookback_hours']
        return_type = final_params['return_type']
        price_col = final_params['price_col']
        
        # Define the lookback period
        start_time = date - timedelta(hours=lookback_hours)
        
        # Filter data to the lookback period
        mask = (data['open_time'] >= start_time) & (data['open_time'] <= date)
        period_data = data[mask].copy().sort_values('open_time')
        
        if len(period_data) < 2:
            return 0.0
        
        # Get start and end prices
        start_price = period_data[price_col].iloc[0]
        end_price = period_data[price_col].iloc[-1]
        
        if start_price == 0 or pd.isna(start_price) or pd.isna(end_price):
            return 0.0
        
        # Calculate return based on type
        if return_type == 'log':
            return np.log(end_price / start_price)
        else:  # simple return
            return (end_price - start_price) / start_price
    
    def get_required_columns(self) -> list:
        """Required columns for returns computation."""
        return ['open_time', 'close', 'open', 'high', 'low']  # Support multiple price columns
    
    def get_minimum_periods(self, params: Optional[Dict[str, Any]] = None) -> int:
        """Minimum periods needed for computation."""
        final_params = {**self.default_params, **(params or {})}
        lookback_hours = final_params['lookback_hours']
        # Assume minute data, so need at least lookback_hours * 60 periods ideally
        # But require minimum of 2 periods for any return calculation
        return max(2, lookback_hours // 10)  # Conservative estimate
    
    def get_description(self) -> str:
        return "Returns factor: Computes price returns over specified lookback periods"
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            'default_params': self.default_params,
            'parameter_descriptions': {
                'lookback_hours': {
                    'description': 'Number of hours to look back for return calculation',
                    'type': 'int',
                    'min_value': 1,
                    'max_value': 8760,  # 1 year
                    'default': 96
                },
                'return_type': {
                    'description': 'Type of return calculation',
                    'type': 'str',
                    'options': ['simple', 'log'],
                    'default': 'simple'
                },
                'price_col': {
                    'description': 'Price column to use for return calculation',
                    'type': 'str',
                    'options': ['close', 'open', 'high', 'low'],
                    'default': 'close'
                }
            }
        }


class VolatilityFactor(BaseFactor):
    """
    Volatility-based factor implementation.
    
    Computes rolling volatility measures including standard deviation
    of returns and realized volatility.
    """
    
    def __init__(self, cache_manager=None):
        default_params = {
            'lookback_hours': 168,  # 1 week default
            'volatility_type': 'returns_std',  # 'returns_std', 'price_std', 'realized_vol'
            'annualize': True
        }
        super().__init__('volatility', cache_manager, default_params)
    
    def compute(self, 
               symbol: str,
               data: pd.DataFrame, 
               date: datetime,
               params: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute volatility factor for the specified lookback period.
        
        Args:
            symbol: Symbol identifier
            data: Price data DataFrame
            date: Target date for volatility computation
            params: Parameters including 'lookback_hours', 'volatility_type'
            
        Returns:
            Volatility measure over the lookback period
        """
        final_params = {**self.default_params, **(params or {})}
        lookback_hours = final_params['lookback_hours']
        volatility_type = final_params['volatility_type']
        annualize = final_params['annualize']
        
        # Define the lookback period
        start_time = date - timedelta(hours=lookback_hours)
        
        # Filter data to the lookback period
        mask = (data['open_time'] >= start_time) & (data['open_time'] <= date)
        period_data = data[mask].copy().sort_values('open_time')
        
        if len(period_data) < 10:  # Need sufficient data for volatility
            return 0.0
        
        try:
            if volatility_type == 'returns_std':
                # Calculate returns and their standard deviation
                returns = period_data['close'].pct_change().dropna()
                if len(returns) < 5:
                    return 0.0
                vol = returns.std()
                
                # Annualize if requested (assume minute data)
                if annualize:
                    vol = vol * np.sqrt(525600)  # minutes in a year
                    
            elif volatility_type == 'price_std':
                # Standard deviation of prices (normalized by mean)
                vol = period_data['close'].std() / period_data['close'].mean()
                
            elif volatility_type == 'realized_vol':
                # Realized volatility using high-low estimator
                hl_ratio = (period_data['high'] / period_data['low']).apply(np.log)
                vol = hl_ratio.std()
                
                if annualize:
                    vol = vol * np.sqrt(525600)
            else:
                return 0.0
            
            return float(vol) if not pd.isna(vol) else 0.0
            
        except Exception:
            return 0.0
    
    def get_required_columns(self) -> list:
        """Required columns for volatility computation."""
        return ['open_time', 'open', 'high', 'low', 'close']
    
    def get_minimum_periods(self, params: Optional[Dict[str, Any]] = None) -> int:
        """Minimum periods needed for computation."""
        return 10  # Need at least 10 periods for meaningful volatility
    
    def get_description(self) -> str:
        return "Volatility factor: Computes rolling volatility measures over specified periods"


class MomentumFactor(BaseFactor):
    """
    Momentum factor implementation.
    
    Computes momentum indicators including price momentum,
    return momentum, and momentum oscillators.
    """
    
    def __init__(self, cache_manager=None):
        default_params = {
            'short_period_hours': 24,   # 1 day
            'long_period_hours': 168,   # 1 week
            'momentum_type': 'price_ratio'  # 'price_ratio', 'return_diff', 'roc'
        }
        super().__init__('momentum', cache_manager, default_params)
    
    def compute(self, 
               symbol: str,
               data: pd.DataFrame, 
               date: datetime,
               params: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute momentum factor comparing short and long periods.
        
        Args:
            symbol: Symbol identifier
            data: Price data DataFrame
            date: Target date for momentum computation
            params: Parameters including period definitions
            
        Returns:
            Momentum value
        """
        final_params = {**self.default_params, **(params or {})}
        short_hours = final_params['short_period_hours']
        long_hours = final_params['long_period_hours']
        momentum_type = final_params['momentum_type']
        
        # Get short period data
        short_start = date - timedelta(hours=short_hours)
        short_mask = (data['open_time'] >= short_start) & (data['open_time'] <= date)
        short_data = data[short_mask]
        
        # Get long period data
        long_start = date - timedelta(hours=long_hours)
        long_mask = (data['open_time'] >= long_start) & (data['open_time'] <= date)
        long_data = data[long_mask]
        
        if len(short_data) < 2 or len(long_data) < 10:
            return 0.0
        
        try:
            if momentum_type == 'price_ratio':
                # Ratio of average prices
                short_avg = short_data['close'].mean()
                long_avg = long_data['close'].mean()
                
                if long_avg == 0:
                    return 0.0
                    
                return (short_avg / long_avg) - 1
                
            elif momentum_type == 'return_diff':
                # Difference in average returns
                short_returns = short_data['close'].pct_change().mean()
                long_returns = long_data['close'].pct_change().mean()
                
                return short_returns - long_returns
                
            elif momentum_type == 'roc':
                # Rate of change
                current_price = data[data['open_time'] <= date]['close'].iloc[-1]
                past_price = long_data['close'].iloc[0]
                
                if past_price == 0:
                    return 0.0
                    
                return (current_price / past_price) - 1
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def get_required_columns(self) -> list:
        """Required columns for momentum computation."""
        return ['open_time', 'close']
    
    def get_minimum_periods(self, params: Optional[Dict[str, Any]] = None) -> int:
        """Minimum periods needed for computation."""
        final_params = {**self.default_params, **(params or {})}
        return max(10, final_params['long_period_hours'] // 10)
    
    def get_description(self) -> str:
        return "Momentum factor: Computes momentum indicators comparing short and long periods"