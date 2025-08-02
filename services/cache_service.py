"""
Cache service for RAG system.

This module provides a centralized, pluggable caching layer for expensive
computations like embeddings, retrieval results, and LLM responses.
"""

import os
import time
import logging
import hashlib
import pickle
import threading
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from collections import OrderedDict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache service."""
    backend: str = "memory"  # memory, redis, disk
    ttl: int = 3600  # Time to live in seconds
    max_size: int = 1000  # Maximum number of items
    redis_url: str = "redis://localhost:6379"
    disk_path: str = "./cache"
    enable_logging: bool = True


class CacheBackend:
    """Abstract base class for cache backends."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError
    
    def invalidate(self, key: str) -> bool:
        """Remove key from cache."""
        raise NotImplementedError
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        raise NotImplementedError


class MemoryCacheBackend(CacheBackend):
    """In-memory cache using OrderedDict with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if self._is_expired(key):
                self._remove_key(key)
                return None
            
            # Move to end (LRU)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self.lock:
            # Remove if exists
            if key in self.cache:
                self._remove_key(key)
            
            # Check size limit
            if len(self.cache) >= self.max_size:
                # Remove oldest item
                oldest_key = next(iter(self.cache))
                self._remove_key(oldest_key)
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = time.time()
            return True
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self.lock:
            if key not in self.cache:
                return False
            return not self._is_expired(key)
    
    def invalidate(self, key: str) -> bool:
        """Remove key from cache."""
        with self.lock:
            return self._remove_key(key)
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            return True
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl
    
    def _remove_key(self, key: str) -> bool:
        """Remove key from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
            return True
        return False


class RedisCacheBackend(CacheBackend):
    """Redis-based cache backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis_url = redis_url
        self.ttl = ttl
        self._redis = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis."""
        try:
            import redis
            self._redis = redis.from_url(self.redis_url)
            # Test connection
            self._redis.ping()
            logger.info("Connected to Redis cache")
        except ImportError:
            logger.error("Redis not available. Install with: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            if not self._redis:
                return None
            
            data = self._redis.get(key)
            if data is None:
                return None
            
            return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            if not self._redis:
                return False
            
            data = pickle.dumps(value)
            ttl = ttl or self.ttl
            return self._redis.setex(key, ttl, data)
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            if not self._redis:
                return False
            return bool(self._redis.exists(key))
        except Exception as e:
            logger.warning(f"Redis exists failed: {e}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """Remove key from Redis."""
        try:
            if not self._redis:
                return False
            return bool(self._redis.delete(key))
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries (flush all keys)."""
        try:
            if not self._redis:
                return False
            self._redis.flushdb()
            return True
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")
            return False


class DiskCacheBackend(CacheBackend):
    """Disk-based cache using SQLite."""
    
    def __init__(self, cache_dir: str = "./cache", ttl: int = 3600):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.lock = threading.RLock()
        self._ensure_cache_dir()
        self._init_db()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _init_db(self):
        """Initialize SQLite database."""
        import sqlite3
        db_path = os.path.join(self.cache_dir, "cache.db")
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    ttl INTEGER
                )
            """)
            conn.commit()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            import sqlite3
            db_path = os.path.join(self.cache_dir, "cache.db")
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT value, timestamp, ttl FROM cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                value_data, timestamp, ttl = row
                
                # Check expiration
                if time.time() - timestamp > ttl:
                    self.invalidate(key)
                    return None
                
                return pickle.loads(value_data)
        except Exception as e:
            logger.warning(f"Disk cache get failed: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in disk cache."""
        try:
            import sqlite3
            db_path = os.path.join(self.cache_dir, "cache.db")
            
            value_data = pickle.dumps(value)
            timestamp = time.time()
            ttl = ttl or self.ttl
            
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, timestamp, ttl) VALUES (?, ?, ?, ?)",
                    (key, value_data, timestamp, ttl)
                )
                conn.commit()
            return True
        except Exception as e:
            logger.warning(f"Disk cache set failed: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in disk cache."""
        try:
            import sqlite3
            db_path = os.path.join(self.cache_dir, "cache.db")
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT timestamp, ttl FROM cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row is None:
                    return False
                
                timestamp, ttl = row
                return time.time() - timestamp <= ttl
        except Exception as e:
            logger.warning(f"Disk cache exists failed: {e}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """Remove key from disk cache."""
        try:
            import sqlite3
            db_path = os.path.join(self.cache_dir, "cache.db")
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.warning(f"Disk cache invalidate failed: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            import sqlite3
            db_path = os.path.join(self.cache_dir, "cache.db")
            
            with sqlite3.connect(db_path) as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()
            return True
        except Exception as e:
            logger.warning(f"Disk cache clear failed: {e}")
            return False


class CacheService:
    """Main cache service that manages different backends."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.backend = self._initialize_backend()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'invalidations': 0
        }
    
    def _initialize_backend(self) -> CacheBackend:
        """Initialize cache backend based on configuration."""
        if self.config.backend == "redis":
            return RedisCacheBackend(self.config.redis_url, self.config.ttl)
        elif self.config.backend == "disk":
            return DiskCacheBackend(self.config.disk_path, self.config.ttl)
        else:
            return MemoryCacheBackend(self.config.max_size, self.config.ttl)
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create a string representation
        key_parts = []
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                # For complex objects, use hash
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:16])
        
        for key, value in sorted(kwargs.items()):
            if isinstance(value, (str, int, float, bool)):
                key_parts.append(f"{key}:{value}")
            else:
                key_parts.append(f"{key}:{hashlib.md5(str(value).encode()).hexdigest()[:16]}")
        
        key_string = "|".join(key_parts)
        
        # Hash the final key to ensure reasonable length
        return hashlib.blake2b(key_string.encode(), digest_size=16).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.backend.exists(key):
            value = self.backend.get(key)
            if value is not None:
                self.stats['hits'] += 1
                if self.config.enable_logging:
                    logger.debug(f"Cache hit: {key}")
                return value
        
        self.stats['misses'] += 1
        if self.config.enable_logging:
            logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        success = self.backend.set(key, value, ttl)
        if success:
            self.stats['sets'] += 1
            if self.config.enable_logging:
                logger.debug(f"Cache set: {key}")
        return success
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.backend.exists(key)
    
    def invalidate(self, key: str) -> bool:
        """Remove key from cache."""
        success = self.backend.invalidate(key)
        if success:
            self.stats['invalidations'] += 1
            if self.config.enable_logging:
                logger.debug(f"Cache invalidate: {key}")
        return success
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        return self.backend.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'invalidations': self.stats['invalidations'],
            'hit_rate': round(hit_rate, 2),
            'total_requests': total_requests
        }
    
    # Convenience methods for RAG-specific caching
    
    def get_embedding(self, text: str, model_name: str) -> Optional[Any]:
        """Get cached embedding."""
        key = self._generate_key("embedding", text, model_name)
        return self.get(key)
    
    def set_embedding(self, text: str, model_name: str, embedding: Any, ttl: Optional[int] = None) -> bool:
        """Set cached embedding."""
        key = self._generate_key("embedding", text, model_name)
        return self.set(key, embedding, ttl)
    
    def get_retrieval(self, query: str, filters: Dict[str, Any]) -> Optional[Any]:
        """Get cached retrieval results."""
        key = self._generate_key("retrieval", query, filters)
        return self.get(key)
    
    def set_retrieval(self, query: str, filters: Dict[str, Any], results: Any, ttl: Optional[int] = None) -> bool:
        """Set cached retrieval results."""
        key = self._generate_key("retrieval", query, filters)
        return self.set(key, results, ttl)
    
    def get_rerank(self, query: str, documents: List[str]) -> Optional[Any]:
        """Get cached rerank results."""
        key = self._generate_key("rerank", query, documents)
        return self.get(key)
    
    def set_rerank(self, query: str, documents: List[str], results: Any, ttl: Optional[int] = None) -> bool:
        """Set cached rerank results."""
        key = self._generate_key("rerank", query, documents)
        return self.set(key, results, ttl)
    
    def get_answer(self, query: str, context_hash: str) -> Optional[Any]:
        """Get cached LLM answer."""
        key = self._generate_key("answer", query, context_hash)
        return self.get(key)
    
    def set_answer(self, query: str, context_hash: str, answer: Any, ttl: Optional[int] = None) -> bool:
        """Set cached LLM answer."""
        key = self._generate_key("answer", query, context_hash)
        return self.set(key, answer, ttl)


# Global cache service instance
_cache_service = None


def get_cache_service(config: Optional[CacheConfig] = None) -> CacheService:
    """Get global cache service instance."""
    global _cache_service
    
    if _cache_service is None:
        # Use environment variables for configuration
        backend = os.getenv("CACHE_BACKEND", "memory")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        disk_path = os.getenv("CACHE_DISK_PATH", "./cache")
        ttl = int(os.getenv("CACHE_TTL", "3600"))
        max_size = int(os.getenv("CACHE_MAX_SIZE", "1000"))
        
        config = config or CacheConfig(
            backend=backend,
            redis_url=redis_url,
            disk_path=disk_path,
            ttl=ttl,
            max_size=max_size
        )
        
        _cache_service = CacheService(config)
    
    return _cache_service


def cacheable(ttl: int = 3600):
    """Decorator to cache function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_service = get_cache_service()
            key = cache_service._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_service.get(key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            cache_service.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator


def test_cache_service():
    """Test cache service functionality."""
    print("Testing cache service...")
    
    # Test memory cache
    config = CacheConfig(backend="memory", ttl=60)
    cache = CacheService(config)
    
    # Test basic operations
    cache.set("test_key", "test_value")
    assert cache.get("test_key") == "test_value"
    assert cache.exists("test_key")
    
    # Test invalidation
    cache.invalidate("test_key")
    assert not cache.exists("test_key")
    
    # Test RAG-specific methods
    cache.set_embedding("test text", "test_model", [1.0, 2.0, 3.0])
    embedding = cache.get_embedding("test text", "test_model")
    assert embedding == [1.0, 2.0, 3.0]
    
    # Test decorator
    @cacheable(ttl=60)
    def test_function(x, y):
        return x + y
    
    result1 = test_function(1, 2)
    result2 = test_function(1, 2)
    assert result1 == result2 == 3
    
    print("Cache service test completed successfully!")


if __name__ == "__main__":
    test_cache_service()
