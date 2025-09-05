from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from .cache_manager import FactorCacheManager


class BaseFactor(ABC):
    """
    Abstract base class for all factor implementations.
    
    Defines the common interface and functionality for factor computation,
    including caching integration and parameter validation.
    """
    
    def __init__(self, 
                 name: str,
                 cache_manager: Optional[FactorCacheManager] = None,
                 default_params: Optional[Dict[str, Any]] = None):
        """
        Initialize base factor.
        
        Args:
            name: Factor name identifier
            cache_manager: Optional cache manager for result caching
            default_params: Default parameters for factor computation
        """
        self.name = name
        self.cache_manager = cache_manager
        self.default_params = default_params or {}
    
    @abstractmethod
    def compute(self, 
               symbol: str,
               data: pd.DataFrame, 
               date: datetime,
               params: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute factor value for a symbol at a specific date.
        
        Args:
            symbol: Symbol identifier
            data: Price/volume data DataFrame
            date: Date for factor computation
            params: Optional parameters for computation
            
        Returns:
            Factor value as float
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> list:
        """
        Get list of required DataFrame columns for this factor.
        
        Returns:
            List of required column names
        """
        pass
    
    @abstractmethod
    def get_minimum_periods(self, params: Optional[Dict[str, Any]] = None) -> int:
        """
        Get minimum number of data periods required for computation.
        
        Args:
            params: Optional computation parameters
            
        Returns:
            Minimum number of periods needed
        """
        pass
    
    def compute_with_cache(self, 
                          symbol: str,
                          data: pd.DataFrame,
                          date: datetime,
                          params: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute factor value with caching support.
        
        Args:
            symbol: Symbol identifier
            data: Price/volume data DataFrame
            date: Date for factor computation
            params: Optional parameters for computation
            
        Returns:
            Factor value as float
        """
        # Merge with default parameters
        final_params = {**self.default_params, **(params or {})}
        
        # Try cache first if available
        if self.cache_manager:
            cached_value = self.cache_manager.get(
                symbol, self.name, date, final_params
            )
            if cached_value is not None:
                return cached_value
        
        # Validate data requirements
        if not self.validate_data(data, final_params):
            return 0.0
        
        # Compute factor value
        try:
            value = self.compute(symbol, data, date, final_params)
            
            # Cache the result if cache manager available
            if self.cache_manager and value is not None:
                self.cache_manager.set(
                    symbol, self.name, date, value, final_params
                )
            
            return value
            
        except Exception as e:
            print(f"Error computing {self.name} for {symbol}: {e}")
            return 0.0
    
    def validate_data(self, 
                     data: pd.DataFrame, 
                     params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate that data meets requirements for factor computation.
        
        Args:
            data: Price/volume data DataFrame
            params: Optional computation parameters
            
        Returns:
            True if data is valid for computation
        """
        if data is None or len(data) == 0:
            return False
        
        # Check required columns
        required_cols = self.get_required_columns()
        if not all(col in data.columns for col in required_cols):
            return False
        
        # Check minimum periods
        min_periods = self.get_minimum_periods(params)
        if len(data) < min_periods:
            return False
        
        # Check for excessive missing data
        for col in required_cols:
            if data[col].isna().sum() / len(data) > 0.2:  # More than 20% missing
                return False
        
        return True
    
    def get_description(self) -> str:
        """
        Get human-readable description of the factor.
        
        Returns:
            Factor description string
        """
        return f"Factor: {self.name}"
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about factor parameters.
        
        Returns:
            Dictionary describing each parameter
        """
        return {
            'default_params': self.default_params,
            'parameter_descriptions': {}
        }