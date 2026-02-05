"""
Redis caching service for performance optimization
"""
import json
import logging
import pickle
import time
from typing import Any, Optional, Dict, List, Union, Callable
from datetime import datetime, timedelta
import redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError
import hashlib

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class RedisCacheService:
    """Service for Redis caching operations"""
    
    def __init__(self):
        self.redis_client = None
        self.connection_pool = None
        self.default_ttl = settings.REDIS_DEFAULT_TTL
        self.max_retries = settings.REDIS_MAX_RETRIES
        self.retry_delay = settings.REDIS_RETRY_DELAY
        self.key_prefix = settings.REDIS_KEY_PREFIX
        self.serialization_method = settings.REDIS_SERIALIZATION_METHOD
        self.compression_enabled = settings.REDIS_COMPRESSION_ENABLED
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Redis connection"""
        try:
            # Create connection pool
            self.connection_pool = redis.ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                retry_on_timeout=True,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT,
                health_check_interval=settings.REDIS_HEALTH_CHECK_INTERVAL
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=False  # We'll handle encoding/decoding ourselves
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {str(e)}")
            self.redis_client = None
    
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return True
        except Exception:
            pass
        return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        try:
            if not self.is_connected():
                return default
            
            full_key = self._build_key(key)
            cached_value = self.redis_client.get(full_key)
            
            if cached_value is None:
                return default
            
            return self._deserialize(cached_value)
            
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {str(e)}")
            return default
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set value in cache"""
        try:
            if not self.is_connected():
                return False
            
            full_key = self._build_key(key)
            serialized_value = self._serialize(value)
            
            if ttl is None:
                ttl = self.default_ttl
            
            # Set with options
            if nx:
                result = self.redis_client.set(full_key, serialized_value, ex=ttl, nx=True)
            elif xx:
                result = self.redis_client.set(full_key, serialized_value, ex=ttl, xx=True)
            else:
                result = self.redis_client.set(full_key, serialized_value, ex=ttl)
            
            return result is True
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if not self.is_connected():
                return False
            
            full_key = self._build_key(key)
            result = self.redis_client.delete(full_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {str(e)}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if not self.is_connected():
                return False
            
            full_key = self._build_key(key)
            return self.redis_client.exists(full_key) > 0
            
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {str(e)}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key"""
        try:
            if not self.is_connected():
                return False
            
            full_key = self._build_key(key)
            return self.redis_client.expire(full_key, ttl)
            
        except Exception as e:
            logger.error(f"Error setting expiration for key {key}: {str(e)}")
            return False
    
    def ttl(self, key: str) -> int:
        """Get TTL for key"""
        try:
            if not self.is_connected():
                return -1
            
            full_key = self._build_key(key)
            return self.redis_client.ttl(full_key)
            
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {str(e)}")
            return -1
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment numeric value in cache"""
        try:
            if not self.is_connected():
                return None
            
            full_key = self._build_key(key)
            return self.redis_client.incrby(full_key, amount)
            
        except Exception as e:
            logger.error(f"Error incrementing cache key {key}: {str(e)}")
            return None
    
    def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """Decrement numeric value in cache"""
        try:
            if not self.is_connected():
                return None
            
            full_key = self._build_key(key)
            return self.redis_client.decrby(full_key, amount)
            
        except Exception as e:
            logger.error(f"Error decrementing cache key {key}: {str(e)}")
            return None
    
    def get_or_set(
        self, 
        key: str, 
        func: Callable, 
        ttl: Optional[int] = None,
        *args, 
        **kwargs
    ) -> Any:
        """Get value from cache or set it using function"""
        try:
            # Try to get from cache first
            cached_value = self.get(key)
            if cached_value is not None:
                return cached_value
            
            # Execute function to get value
            value = func(*args, **kwargs)
            
            # Set in cache
            self.set(key, value, ttl)
            
            return value
            
        except Exception as e:
            logger.error(f"Error in get_or_set for key {key}: {str(e)}")
            # Fallback to function execution
            try:
                return func(*args, **kwargs)
            except Exception as func_error:
                logger.error(f"Function execution failed for key {key}: {str(func_error)}")
                raise
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        try:
            if not self.is_connected():
                return {}
            
            full_keys = [self._build_key(key) for key in keys]
            values = self.redis_client.mget(full_keys)
            
            result = {}
            for i, value in enumerate(values):
                if value is not None:
                    result[keys[i]] = self._deserialize(value)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting multiple cache keys: {str(e)}")
            return {}
    
    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache"""
        try:
            if not self.is_connected():
                return False
            
            if ttl is None:
                ttl = self.default_ttl
            
            pipe = self.redis_client.pipeline()
            for key, value in mapping.items():
                full_key = self._build_key(key)
                serialized_value = self._serialize(value)
                pipe.set(full_key, serialized_value, ex=ttl)
            
            pipe.execute()
            return True
            
        except Exception as e:
            logger.error(f"Error setting multiple cache keys: {str(e)}")
            return False
    
    def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys from cache"""
        try:
            if not self.is_connected():
                return 0
            
            full_keys = [self._build_key(key) for key in keys]
            return self.redis_client.delete(*full_keys)
            
        except Exception as e:
            logger.error(f"Error deleting multiple cache keys: {str(e)}")
            return 0
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        try:
            if not self.is_connected():
                return 0
            
            full_pattern = self._build_key(pattern)
            keys = self.redis_client.keys(full_pattern)
            
            if keys:
                return self.redis_client.delete(*keys)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error clearing cache pattern {pattern}: {str(e)}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.is_connected():
                return {"error": "Redis not connected"}
            
            info = self.redis_client.info()
            
            return {
                "connected": True,
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory_human", "0B"),
                "used_memory_peak": info.get("used_memory_peak_human", "0B"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"error": str(e)}
    
    def _build_key(self, key: str) -> str:
        """Build full cache key with prefix"""
        return f"{self.key_prefix}:{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            if self.serialization_method == "json":
                return json.dumps(value, default=str).encode('utf-8')
            elif self.serialization_method == "pickle":
                return pickle.dumps(value)
            else:
                # Default to JSON
                return json.dumps(value, default=str).encode('utf-8')
        except Exception as e:
            logger.error(f"Error serializing value: {str(e)}")
            raise
    
    def _deserialize(self, value: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            if self.serialization_method == "json":
                return json.loads(value.decode('utf-8'))
            elif self.serialization_method == "pickle":
                return pickle.loads(value)
            else:
                # Default to JSON
                return json.loads(value.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error deserializing value: {str(e)}")
            raise
    
    def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
        """Calculate cache hit rate"""
        try:
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses
            
            if total == 0:
                return 0.0
            
            return (hits / total) * 100
        except Exception:
            return 0.0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connection"""
        try:
            start_time = time.time()
            
            if not self.is_connected():
                return {
                    "healthy": False,
                    "error": "Redis not connected",
                    "response_time_ms": 0
                }
            
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            test_value = "test_value"
            
            # Test set
            set_result = self.set(test_key, test_value, ttl=10)
            if not set_result:
                return {
                    "healthy": False,
                    "error": "Failed to set test key",
                    "response_time_ms": 0
                }
            
            # Test get
            get_result = self.get(test_key)
            if get_result != test_value:
                return {
                    "healthy": False,
                    "error": "Failed to get test key",
                    "response_time_ms": 0
                }
            
            # Test delete
            delete_result = self.delete(test_key)
            if not delete_result:
                return {
                    "healthy": False,
                    "error": "Failed to delete test key",
                    "response_time_ms": 0
                }
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "healthy": True,
                "response_time_ms": response_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return {
                "healthy": False,
                "error": str(e),
                "response_time_ms": 0
            }

# Global cache instance
_cache_service = None

def get_cache_service() -> RedisCacheService:
    """Get global cache service instance"""
    global _cache_service
    if _cache_service is None:
        _cache_service = RedisCacheService()
    return _cache_service

def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
    
    # Add keyword arguments
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (str, int, float, bool)):
            key_parts.append(f"{key}:{value}")
        else:
            key_parts.append(f"{key}:{hashlib.md5(str(value).encode()).hexdigest()[:8]}")
    
    return ":".join(key_parts)

def cached(ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_service = get_cache_service()
            
            # Generate cache key
            if key_func:
                cache_key_str = key_func(*args, **kwargs)
            else:
                cache_key_str = cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_service.get(cache_key_str)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_service.set(cache_key_str, result, ttl)
            
            return result
        
        return wrapper
    return decorator
