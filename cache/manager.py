"""
Redis cache manager for Supply Chain Risk Intelligence System
Provides high-performance caching with TTL and advanced features
"""

import os
import json
import logging
import asyncio
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import pickle
import hashlib
from contextlib import asynccontextmanager

try:
    import redis.asyncio as redis
    import redis as sync_redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Redis not available. Install with: pip install redis")

logger = logging.getLogger(__name__)

class CacheManager:
    """Advanced Redis cache manager with fallback to in-memory caching"""
    
    def __init__(self, redis_url: Optional[str] = None, fallback_to_memory: bool = True):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.fallback_to_memory = fallback_to_memory
        self.memory_cache = {} if fallback_to_memory else None
        self.memory_cache_ttl = {} if fallback_to_memory else None
        self.redis_client = None
        self.sync_redis_client = None
        self.connected = False
        
        # Cache configuration
        self.default_ttl = int(os.getenv('CACHE_DEFAULT_TTL', '3600'))  # 1 hour
        self.max_memory_items = int(os.getenv('CACHE_MAX_MEMORY_ITEMS', '1000'))
        
        # Initialize Redis connection
        if REDIS_AVAILABLE:
            self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connections"""
        try:
            # Async client
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # We'll handle encoding ourselves
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Sync client for synchronous operations
            self.sync_redis_client = sync_redis.from_url(
                self.redis_url,
                decode_responses=False,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.sync_redis_client.ping()
            self.connected = True
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            if not self.fallback_to_memory:
                raise
            logger.info("Falling back to in-memory cache")
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            # Try JSON first for better readability in Redis
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                return json.dumps(value).encode('utf-8')
            else:
                # Fall back to pickle for complex objects
                return pickle.dumps(value)
        except Exception:
            return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate namespaced cache key"""
        return f"scr:{namespace}:{key}"
    
    def _cleanup_memory_cache(self):
        """Clean up expired items from memory cache"""
        if not self.memory_cache_ttl:
            return
        
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, expiry in self.memory_cache_ttl.items()
            if expiry and current_time > expiry
        ]
        
        for key in expired_keys:
            self.memory_cache.pop(key, None)
            self.memory_cache_ttl.pop(key, None)
        
        # Limit memory cache size
        if len(self.memory_cache) > self.max_memory_items:
            # Remove oldest items (simple LRU approximation)
            items_to_remove = len(self.memory_cache) - self.max_memory_items
            for key in list(self.memory_cache.keys())[:items_to_remove]:
                self.memory_cache.pop(key, None)
                self.memory_cache_ttl.pop(key, None)
    
    async def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL"""
        cache_key = self._generate_key(namespace, key)
        ttl = ttl or self.default_ttl
        
        # Try Redis first
        if self.connected and self.redis_client:
            try:
                serialized_value = self._serialize_value(value)
                await self.redis_client.setex(cache_key, ttl, serialized_value)
                return True
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
                if not self.fallback_to_memory:
                    return False
        
        # Fallback to memory cache
        if self.memory_cache is not None:
            self._cleanup_memory_cache()
            self.memory_cache[cache_key] = value
            self.memory_cache_ttl[cache_key] = datetime.utcnow() + timedelta(seconds=ttl)
            return True
        
        return False
    
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get a value from cache"""
        cache_key = self._generate_key(namespace, key)
        
        # Try Redis first
        if self.connected and self.redis_client:
            try:
                data = await self.redis_client.get(cache_key)
                if data is not None:
                    return self._deserialize_value(data)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Fallback to memory cache
        if self.memory_cache is not None:
            self._cleanup_memory_cache()
            return self.memory_cache.get(cache_key)
        
        return None
    
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete a value from cache"""
        cache_key = self._generate_key(namespace, key)
        
        # Try Redis first
        if self.connected and self.redis_client:
            try:
                result = await self.redis_client.delete(cache_key)
                return bool(result)
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        # Fallback to memory cache
        if self.memory_cache is not None:
            removed = cache_key in self.memory_cache
            self.memory_cache.pop(cache_key, None)
            self.memory_cache_ttl.pop(cache_key, None)
            return removed
        
        return False
    
    async def exists(self, namespace: str, key: str) -> bool:
        """Check if a key exists in cache"""
        cache_key = self._generate_key(namespace, key)
        
        # Try Redis first
        if self.connected and self.redis_client:
            try:
                result = await self.redis_client.exists(cache_key)
                return bool(result)
            except Exception as e:
                logger.warning(f"Redis exists failed: {e}")
        
        # Fallback to memory cache
        if self.memory_cache is not None:
            self._cleanup_memory_cache()
            return cache_key in self.memory_cache
        
        return False
    
    async def increment(self, namespace: str, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment a counter in cache"""
        cache_key = self._generate_key(namespace, key)
        ttl = ttl or self.default_ttl
        
        # Try Redis first
        if self.connected and self.redis_client:
            try:
                # Use Redis pipeline for atomic operation
                async with self.redis_client.pipeline() as pipe:
                    await pipe.incr(cache_key, amount)
                    await pipe.expire(cache_key, ttl)
                    results = await pipe.execute()
                    return results[0]
            except Exception as e:
                logger.warning(f"Redis increment failed: {e}")
        
        # Fallback to memory cache
        if self.memory_cache is not None:
            self._cleanup_memory_cache()
            current_value = self.memory_cache.get(cache_key, 0)
            new_value = current_value + amount
            self.memory_cache[cache_key] = new_value
            self.memory_cache_ttl[cache_key] = datetime.utcnow() + timedelta(seconds=ttl)
            return new_value
        
        return amount
    
    async def get_keys(self, namespace: str, pattern: str = "*") -> List[str]:
        """Get all keys matching a pattern"""
        pattern_key = self._generate_key(namespace, pattern)
        
        # Try Redis first
        if self.connected and self.redis_client:
            try:
                keys = await self.redis_client.keys(pattern_key)
                return [key.decode('utf-8') for key in keys]
            except Exception as e:
                logger.warning(f"Redis keys failed: {e}")
        
        # Fallback to memory cache
        if self.memory_cache is not None:
            self._cleanup_memory_cache()
            prefix = self._generate_key(namespace, "")
            pattern_suffix = pattern.replace("*", "")
            
            matching_keys = []
            for key in self.memory_cache.keys():
                if key.startswith(prefix):
                    if pattern == "*" or pattern_suffix in key:
                        matching_keys.append(key)
            
            return matching_keys
        
        return []
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace"""
        keys = await self.get_keys(namespace)
        count = 0
        
        for key in keys:
            # Extract the actual key part
            key_parts = key.split(":", 2)
            if len(key_parts) >= 3:
                actual_key = key_parts[2]
                if await self.delete(namespace, actual_key):
                    count += 1
        
        return count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'connected_to_redis': self.connected,
            'memory_cache_enabled': self.memory_cache is not None,
            'default_ttl': self.default_ttl
        }
        
        # Redis stats
        if self.connected and self.redis_client:
            try:
                info = await self.redis_client.info()
                stats.update({
                    'redis_used_memory': info.get('used_memory', 0),
                    'redis_connected_clients': info.get('connected_clients', 0),
                    'redis_total_commands_processed': info.get('total_commands_processed', 0),
                    'redis_keyspace_hits': info.get('keyspace_hits', 0),
                    'redis_keyspace_misses': info.get('keyspace_misses', 0)
                })
                
                # Calculate hit rate
                hits = stats.get('redis_keyspace_hits', 0)
                misses = stats.get('redis_keyspace_misses', 0)
                total = hits + misses
                stats['redis_hit_rate'] = hits / total if total > 0 else 0
                
            except Exception as e:
                logger.warning(f"Failed to get Redis stats: {e}")
        
        # Memory cache stats
        if self.memory_cache is not None:
            self._cleanup_memory_cache()
            stats.update({
                'memory_cache_items': len(self.memory_cache),
                'memory_cache_max_items': self.max_memory_items
            })
        
        return stats
    
    # High-level cache patterns
    async def cache_result(self, func, namespace: str, key: str, ttl: Optional[int] = None, *args, **kwargs):
        """Cache the result of a function call"""
        # Generate a unique key based on function and arguments
        func_name = f"{func.__name__}"
        args_hash = hashlib.md5(str((args, kwargs)).encode()).hexdigest()[:8]
        cache_key = f"{key}:{func_name}:{args_hash}"
        
        # Try to get from cache first
        cached_result = await self.get(namespace, cache_key)
        if cached_result is not None:
            return cached_result
        
        # Execute function and cache result
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        await self.set(namespace, cache_key, result, ttl)
        return result
    
    async def get_or_set(self, namespace: str, key: str, value_func, ttl: Optional[int] = None):
        """Get value from cache or set it using a function"""
        cached_value = await self.get(namespace, key)
        if cached_value is not None:
            return cached_value
        
        # Generate value
        if asyncio.iscoroutinefunction(value_func):
            value = await value_func()
        else:
            value = value_func()
        
        await self.set(namespace, key, value, ttl)
        return value
    
    async def close(self):
        """Close Redis connections"""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.sync_redis_client:
            self.sync_redis_client.close()

# Global cache manager instance
cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager()
    return cache_manager

async def get_async_cache_manager() -> CacheManager:
    """Get global cache manager instance (async)"""
    return get_cache_manager()
