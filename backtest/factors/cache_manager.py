import redis
import pickle
import hashlib
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set key-value pair with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend using dictionary."""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            # Check TTL
            if 'expires_at' in entry and datetime.now() > entry['expires_at']:
                del self.cache[key]
                return None
            return entry['value']
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        try:
            entry = {'value': value}
            if ttl_seconds:
                entry['expires_at'] = datetime.now() + timedelta(seconds=ttl_seconds)
            self.cache[key] = entry
            return True
        except Exception as e:
            self.logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def exists(self, key: str) -> bool:
        return key in self.cache


class RedisCacheBackend(CacheBackend):
    """Redis cache backend."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
            self.redis_client.ping()  # Test connection
            self.available = True
        except Exception as e:
            logging.warning(f"Redis not available, falling back to memory cache: {e}")
            self.available = False
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        if not self.available:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        if not self.available:
            return False
        
        try:
            data = pickle.dumps(value)
            if ttl_seconds:
                return self.redis_client.setex(key, ttl_seconds, data)
            else:
                return self.redis_client.set(key, data)
        except Exception as e:
            self.logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        if not self.available:
            return False
        
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            self.logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        if not self.available:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            self.logger.error(f"Failed to check existence of key {key}: {e}")
            return False


class FactorCacheManager:
    """
    Multi-tier caching system for factor computation results.
    
    Uses a combination of Redis (L1) and in-memory (L2) caching for
    optimal performance and reliability.
    """
    
    def __init__(self, 
                 use_redis: bool = True,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 default_ttl_hours: int = 24):
        """
        Initialize factor cache manager.
        
        Args:
            use_redis: Whether to use Redis as L1 cache
            redis_host: Redis host
            redis_port: Redis port
            default_ttl_hours: Default time-to-live in hours
        """
        self.default_ttl_seconds = default_ttl_hours * 3600
        
        # Initialize cache backends
        if use_redis:
            self.l1_cache = RedisCacheBackend(redis_host, redis_port)
        else:
            self.l1_cache = MemoryCacheBackend()
        
        self.l2_cache = MemoryCacheBackend()  # Always use memory for L2
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'l1_hits': 0,
            'l2_hits': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _make_cache_key(self, 
                       symbol: str, 
                       factor_name: str, 
                       date: datetime, 
                       params: Optional[Dict] = None) -> str:
        """
        Create a unique cache key for factor results.
        
        Args:
            symbol: Symbol name
            factor_name: Factor name
            date: Date for factor computation
            params: Optional factor parameters
            
        Returns:
            Unique cache key string
        """
        key_parts = [symbol, factor_name, date.strftime('%Y%m%d')]
        
        if params:
            # Create deterministic hash of parameters
            params_str = str(sorted(params.items()))
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            key_parts.append(params_hash)
        
        return ':'.join(['factor'] + key_parts)
    
    def get(self, 
           symbol: str, 
           factor_name: str, 
           date: datetime,
           params: Optional[Dict] = None) -> Optional[float]:
        """
        Get cached factor value.
        
        Args:
            symbol: Symbol name
            factor_name: Factor name  
            date: Date for factor computation
            params: Optional factor parameters
            
        Returns:
            Cached factor value or None if not found
        """
        cache_key = self._make_cache_key(symbol, factor_name, date, params)
        
        # Try L2 cache first (fastest)
        value = self.l2_cache.get(cache_key)
        if value is not None:
            self.stats['hits'] += 1
            self.stats['l2_hits'] += 1
            return value
        
        # Try L1 cache
        value = self.l1_cache.get(cache_key)
        if value is not None:
            # Promote to L2 cache
            self.l2_cache.set(cache_key, value, ttl_seconds=3600)  # 1 hour in L2
            self.stats['hits'] += 1
            self.stats['l1_hits'] += 1
            return value
        
        self.stats['misses'] += 1
        return None
    
    def set(self, 
           symbol: str, 
           factor_name: str, 
           date: datetime,
           value: float,
           params: Optional[Dict] = None,
           ttl_hours: Optional[int] = None) -> bool:
        """
        Set cached factor value.
        
        Args:
            symbol: Symbol name
            factor_name: Factor name
            date: Date for factor computation  
            value: Factor value to cache
            params: Optional factor parameters
            ttl_hours: Time-to-live in hours
            
        Returns:
            True if successfully cached
        """
        cache_key = self._make_cache_key(symbol, factor_name, date, params)
        ttl_seconds = (ttl_hours or (self.default_ttl_seconds // 3600)) * 3600
        
        # Set in both caches
        l1_success = self.l1_cache.set(cache_key, value, ttl_seconds)
        l2_success = self.l2_cache.set(cache_key, value, min(ttl_seconds, 3600))
        
        if l1_success or l2_success:
            self.stats['sets'] += 1
        
        return l1_success or l2_success
    
    def delete(self, 
              symbol: str, 
              factor_name: str, 
              date: datetime,
              params: Optional[Dict] = None) -> bool:
        """
        Delete cached factor value.
        
        Args:
            symbol: Symbol name
            factor_name: Factor name
            date: Date for factor computation
            params: Optional factor parameters
            
        Returns:
            True if successfully deleted
        """
        cache_key = self._make_cache_key(symbol, factor_name, date, params)
        
        l1_deleted = self.l1_cache.delete(cache_key)
        l2_deleted = self.l2_cache.delete(cache_key)
        
        return l1_deleted or l2_deleted
    
    def exists(self, 
              symbol: str, 
              factor_name: str, 
              date: datetime,
              params: Optional[Dict] = None) -> bool:
        """
        Check if factor value is cached.
        
        Args:
            symbol: Symbol name
            factor_name: Factor name
            date: Date for factor computation
            params: Optional factor parameters
            
        Returns:
            True if value is cached
        """
        cache_key = self._make_cache_key(symbol, factor_name, date, params)
        return self.l1_cache.exists(cache_key) or self.l2_cache.exists(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate_pct': hit_rate,
            'total_requests': total_requests
        }
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries (mainly affects memory cache).
        
        Returns:
            Number of entries cleared
        """
        # For memory cache, expired entries are cleared automatically on access
        # For Redis, TTL handles expiration automatically
        return 0
    
    def warm_cache(self, 
                  symbol_factors: List[Tuple[str, str, datetime, Dict]],
                  compute_function: callable) -> int:
        """
        Warm the cache by pre-computing factors.
        
        Args:
            symbol_factors: List of (symbol, factor_name, date, params) tuples
            compute_function: Function to compute factor values
            
        Returns:
            Number of factors successfully cached
        """
        cached_count = 0
        
        for symbol, factor_name, date, params in symbol_factors:
            try:
                if not self.exists(symbol, factor_name, date, params):
                    value = compute_function(symbol, factor_name, date, params)
                    if value is not None:
                        self.set(symbol, factor_name, date, value, params)
                        cached_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to warm cache for {symbol}/{factor_name}: {e}")
        
        return cached_count