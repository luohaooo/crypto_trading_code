import psutil
import gc
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import pandas as pd


class MemoryManager:
    """
    Memory management for efficient backtesting operations.
    
    Handles chunked data loading, memory monitoring, and automatic cleanup
    to prevent out-of-memory issues during large-scale backtesting.
    """
    
    def __init__(self, max_memory_gb: float = 8.0, cleanup_threshold: float = 0.8):
        """
        Initialize memory manager.
        
        Args:
            max_memory_gb: Maximum memory usage in GB
            cleanup_threshold: Memory usage threshold to trigger cleanup (0.8 = 80%)
        """
        self.max_memory_gb = max_memory_gb
        self.cleanup_threshold = cleanup_threshold
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        
        # Data cache for loaded symbols
        self.symbol_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Memory tracking
        self.logger = logging.getLogger(__name__)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage info
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'used_gb': memory_info.rss / (1024 ** 3),
            'available_gb': (psutil.virtual_memory().available) / (1024 ** 3),
            'usage_pct': (memory_info.rss / self.max_memory_bytes) * 100,
            'cache_entries': len(self.symbol_cache)
        }
    
    def should_cleanup(self) -> bool:
        """
        Check if memory cleanup is needed.
        
        Returns:
            True if cleanup should be performed
        """
        usage = self.get_memory_usage()
        return usage['usage_pct'] > (self.cleanup_threshold * 100)
    
    def cleanup_expired_data(self, max_age_hours: int = 24):
        """
        Clean up expired data from memory cache.
        
        Args:
            max_age_hours: Maximum age of cached data in hours
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if timestamp < cutoff_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.symbol_cache:
                del self.symbol_cache[key]
                del self.cache_timestamps[key]
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            gc.collect()  # Force garbage collection
    
    def cleanup_memory_pressure(self):
        """
        Perform aggressive memory cleanup when under memory pressure.
        """
        # Remove oldest 50% of cached data
        sorted_items = sorted(
            self.cache_timestamps.items(), 
            key=lambda x: x[1]
        )
        
        cleanup_count = len(sorted_items) // 2
        
        for key, _ in sorted_items[:cleanup_count]:
            if key in self.symbol_cache:
                del self.symbol_cache[key]
                del self.cache_timestamps[key]
        
        gc.collect()
        self.logger.warning(f"Memory pressure cleanup: removed {cleanup_count} cache entries")
    
    def get_cached_data(self, symbol: str, date_key: str) -> Optional[pd.DataFrame]:
        """
        Get cached data for a symbol and date range.
        
        Args:
            symbol: Symbol to get data for
            date_key: Date range key for the data
            
        Returns:
            Cached DataFrame or None if not found
        """
        cache_key = f"{symbol}_{date_key}"
        
        if cache_key in self.symbol_cache:
            # Update access timestamp
            self.cache_timestamps[cache_key] = datetime.now()
            return self.symbol_cache[cache_key].get('data')
        
        return None
    
    def cache_data(self, symbol: str, date_key: str, data: pd.DataFrame):
        """
        Cache data for a symbol and date range.
        
        Args:
            symbol: Symbol to cache data for
            date_key: Date range key for the data
            data: DataFrame to cache
        """
        # Check memory before caching
        if self.should_cleanup():
            self.cleanup_memory_pressure()
        
        cache_key = f"{symbol}_{date_key}"
        
        self.symbol_cache[cache_key] = {
            'data': data.copy(),
            'size_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        self.cache_timestamps[cache_key] = datetime.now()
    
    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get detailed cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_size_mb = sum(
            cache_data.get('size_mb', 0) 
            for cache_data in self.symbol_cache.values()
        )
        
        memory_usage = self.get_memory_usage()
        
        return {
            'memory_usage': memory_usage,
            'cache_entries': len(self.symbol_cache),
            'cache_size_mb': total_size_mb,
            'cache_size_gb': total_size_mb / 1024,
            'oldest_entry': min(self.cache_timestamps.values()) if self.cache_timestamps else None,
            'newest_entry': max(self.cache_timestamps.values()) if self.cache_timestamps else None
        }
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        df_optimized = df.copy()
        
        # Optimize numeric columns
        for col in df_optimized.select_dtypes(include=['int64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Convert object columns to category if appropriate
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if df_optimized[col].nunique() < len(df_optimized) * 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
        
        return df_optimized