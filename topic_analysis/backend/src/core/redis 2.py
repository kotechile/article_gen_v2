"""
Redis configuration and connection management for TrendTap
"""

import redis
import json
import os
import pickle
import hashlib
from typing import Any, Optional, Union, List, Dict, Callable
import structlog
import time
from functools import wraps

logger = structlog.get_logger()

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Create Redis connection pool
redis_pool = redis.ConnectionPool.from_url(
    REDIS_URL,
    password=REDIS_PASSWORD,
    db=REDIS_DB,
    decode_responses=True,
    max_connections=20
)

# Create Redis client
redis_client = redis.Redis(connection_pool=redis_pool)

class RedisCache:
    """
    Redis cache wrapper with common operations
    """
    
    def __init__(self, client: redis.Redis = redis_client):
        self.client = client
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error("Redis get error", key=key, error=str(e))
            return None
    
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            serialized_value = json.dumps(value)
            return self.client.set(key, serialized_value, ex=expire)
        except Exception as e:
            logger.error("Redis set error", key=key, error=str(e))
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error("Redis delete error", key=key, error=str(e))
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error("Redis exists error", key=key, error=str(e))
            return False
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key"""
        try:
            return self.client.expire(key, seconds)
        except Exception as e:
            logger.error("Redis expire error", key=key, error=str(e))
            return False
    
    def ttl(self, key: str) -> int:
        """Get time to live for key"""
        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.error("Redis ttl error", key=key, error=str(e))
            return -1
    
    def flushdb(self) -> bool:
        """Flush current database"""
        try:
            return self.client.flushdb()
        except Exception as e:
            logger.error("Redis flushdb error", error=str(e))
            return False
    
    def get_or_set(self, key: str, func: Callable, expire: Optional[int] = None) -> Any:
        """Get value from cache or set it using function"""
        value = self.get(key)
        if value is not None:
            return value
        
        # Generate value using function
        value = func()
        self.set(key, value, expire)
        return value
    
    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values from cache"""
        try:
            values = self.client.mget(keys)
            result = []
            for value in values:
                if value:
                    result.append(json.loads(value))
                else:
                    result.append(None)
            return result
        except Exception as e:
            logger.error("Redis mget error", keys=keys, error=str(e))
            return [None] * len(keys)
    
    def mset(self, mapping: Dict[str, Any], expire: Optional[int] = None) -> bool:
        """Set multiple key-value pairs"""
        try:
            serialized_mapping = {k: json.dumps(v) for k, v in mapping.items()}
            result = self.client.mset(serialized_mapping)
            
            if expire and result:
                for key in mapping.keys():
                    self.client.expire(key, expire)
            
            return result
        except Exception as e:
            logger.error("Redis mset error", error=str(e))
            return False
    
    def increment(self, key: str, amount: int = 1, expire: Optional[int] = None) -> int:
        """Increment a counter"""
        try:
            result = self.client.incr(key, amount)
            if expire:
                self.client.expire(key, expire)
            return result
        except Exception as e:
            logger.error("Redis increment error", key=key, error=str(e))
            return 0
    
    def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement a counter"""
        try:
            return self.client.decr(key, amount)
        except Exception as e:
            logger.error("Redis decrement error", key=key, error=str(e))
            return 0
    
    def set_hash(self, name: str, mapping: Dict[str, Any], expire: Optional[int] = None) -> bool:
        """Set hash field"""
        try:
            serialized_mapping = {k: json.dumps(v) for k, v in mapping.items()}
            result = self.client.hset(name, mapping=serialized_mapping)
            if expire:
                self.client.expire(name, expire)
            return result
        except Exception as e:
            logger.error("Redis hset error", name=name, error=str(e))
            return False
    
    def get_hash(self, name: str, field: str) -> Optional[Any]:
        """Get hash field value"""
        try:
            value = self.client.hget(name, field)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error("Redis hget error", name=name, field=field, error=str(e))
            return None
    
    def get_all_hash(self, name: str) -> Dict[str, Any]:
        """Get all hash fields"""
        try:
            values = self.client.hgetall(name)
            result = {}
            for field, value in values.items():
                result[field] = json.loads(value)
            return result
        except Exception as e:
            logger.error("Redis hgetall error", name=name, error=str(e))
            return {}
    
    def delete_hash_field(self, name: str, field: str) -> bool:
        """Delete hash field"""
        try:
            return bool(self.client.hdel(name, field))
        except Exception as e:
            logger.error("Redis hdel error", name=name, field=field, error=str(e))
            return False
    
    def add_to_set(self, key: str, *values: Any) -> int:
        """Add values to set"""
        try:
            serialized_values = [json.dumps(v) for v in values]
            return self.client.sadd(key, *serialized_values)
        except Exception as e:
            logger.error("Redis sadd error", key=key, error=str(e))
            return 0
    
    def get_set_members(self, key: str) -> List[Any]:
        """Get all set members"""
        try:
            values = self.client.smembers(key)
            return [json.loads(v) for v in values]
        except Exception as e:
            logger.error("Redis smembers error", key=key, error=str(e))
            return []
    
    def is_in_set(self, key: str, value: Any) -> bool:
        """Check if value is in set"""
        try:
            serialized_value = json.dumps(value)
            return bool(self.client.sismember(key, serialized_value))
        except Exception as e:
            logger.error("Redis sismember error", key=key, error=str(e))
            return False
    
    def list_push(self, key: str, *values: Any, left: bool = False) -> int:
        """Push values to list"""
        try:
            serialized_values = [json.dumps(v) for v in values]
            if left:
                return self.client.lpush(key, *serialized_values)
            else:
                return self.client.rpush(key, *serialized_values)
        except Exception as e:
            logger.error("Redis list push error", key=key, error=str(e))
            return 0
    
    def list_pop(self, key: str, left: bool = False) -> Optional[Any]:
        """Pop value from list"""
        try:
            if left:
                value = self.client.lpop(key)
            else:
                value = self.client.rpop(key)
            
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error("Redis list pop error", key=key, error=str(e))
            return None
    
    def list_range(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get list range"""
        try:
            values = self.client.lrange(key, start, end)
            return [json.loads(v) for v in values]
        except Exception as e:
            logger.error("Redis lrange error", key=key, error=str(e))
            return []
    
    def get_keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        try:
            return self.client.keys(pattern)
        except Exception as e:
            logger.error("Redis keys error", pattern=pattern, error=str(e))
            return []
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern"""
        try:
            keys = self.get_keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error("Redis delete pattern error", pattern=pattern, error=str(e))
            return 0

# Global cache instance
cache = RedisCache()

def get_redis_client() -> redis.Redis:
    """Get Redis client instance"""
    return redis_client

def check_redis_connection() -> bool:
    """Check if Redis connection is healthy"""
    try:
        redis_client.ping()
        logger.info("Redis connection healthy")
        return True
    except Exception as e:
        logger.error("Redis connection failed", error=str(e))
        return False

def get_redis_info() -> dict:
    """Get Redis server information"""
    try:
        info = redis_client.info()
        return {
            "version": info.get("redis_version"),
            "uptime": info.get("uptime_in_seconds"),
            "connected_clients": info.get("connected_clients"),
            "used_memory": info.get("used_memory_human"),
            "keyspace": info.get("db0", {}).get("keys", 0)
        }
    except Exception as e:
        logger.error("Redis info error", error=str(e))
        return {}

# Cache decorators
def cached(expire: int = 3600, key_prefix: str = ""):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{key_prefix}:{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            cache_key = f"cache:{key_hash}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug("Cache hit", function=func.__name__, key=cache_key)
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, expire)
            logger.debug("Cache miss", function=func.__name__, key=cache_key)
            return result
        return wrapper
    return decorator

def cache_invalidate(pattern: str):
    """Decorator to invalidate cache on function call"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            cache.delete_pattern(pattern)
            logger.debug("Cache invalidated", pattern=pattern)
            return result
        return wrapper
    return decorator

# Cache key patterns
class CacheKeys:
    """Common cache key patterns"""
    
    USER_SESSION = "user:session:{user_id}"
    AFFILIATE_RESEARCH = "affiliate:research:{research_id}"
    TREND_ANALYSIS = "trend:analysis:{analysis_id}"
    KEYWORD_DATA = "keyword:data:{keyword_id}"
    CONTENT_IDEAS = "content:ideas:{content_id}"
    SOFTWARE_SOLUTIONS = "software:solutions:{solution_id}"
    CALENDAR_ENTRIES = "calendar:entries:{user_id}"
    API_RATE_LIMIT = "rate_limit:{user_id}:{endpoint}"
    JWT_BLACKLIST = "jwt:blacklist:{token_hash}"
    TREND_DATA = "trend:data:{keyword}:{geo}"
    AFFILIATE_PROGRAMS = "affiliate:programs:{niche}"
    
    @staticmethod
    def format(key_pattern: str, **kwargs) -> str:
        """Format cache key with parameters"""
        return key_pattern.format(**kwargs)

# Cache utilities
class CacheManager:
    """Advanced cache management utilities"""
    
    def __init__(self, cache_instance: RedisCache = cache):
        self.cache = cache_instance
    
    def cache_user_data(self, user_id: str, data: Dict[str, Any], expire: int = 3600):
        """Cache user-specific data"""
        key = CacheKeys.format(CacheKeys.USER_SESSION, user_id=user_id)
        self.cache.set_hash(key, data, expire)
    
    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get cached user data"""
        key = CacheKeys.format(CacheKeys.USER_SESSION, user_id=user_id)
        return self.cache.get_all_hash(key)
    
    def invalidate_user_cache(self, user_id: str):
        """Invalidate all user-related cache"""
        patterns = [
            f"user:session:{user_id}",
            f"affiliate:research:*{user_id}*",
            f"trend:analysis:*{user_id}*",
            f"keyword:data:*{user_id}*",
            f"content:ideas:*{user_id}*",
            f"software:solutions:*{user_id}*",
            f"calendar:entries:{user_id}"
        ]
        
        for pattern in patterns:
            self.cache.delete_pattern(pattern)
    
    def cache_api_response(self, endpoint: str, params: Dict[str, Any], response: Any, expire: int = 1800):
        """Cache API response"""
        param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        key_hash = hashlib.md5(f"{endpoint}:{param_str}".encode()).hexdigest()
        cache_key = f"api:{endpoint}:{key_hash}"
        self.cache.set(cache_key, response, expire)
    
    def get_cached_api_response(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached API response"""
        param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        key_hash = hashlib.md5(f"{endpoint}:{param_str}".encode()).hexdigest()
        cache_key = f"api:{endpoint}:{key_hash}"
        return self.cache.get(cache_key)
    
    def cache_trend_data(self, keyword: str, geo: str, data: Any, expire: int = 3600):
        """Cache trend analysis data"""
        key = CacheKeys.format(CacheKeys.TREND_DATA, keyword=keyword, geo=geo)
        self.cache.set(key, data, expire)
    
    def get_cached_trend_data(self, keyword: str, geo: str) -> Optional[Any]:
        """Get cached trend analysis data"""
        key = CacheKeys.format(CacheKeys.TREND_DATA, keyword=keyword, geo=geo)
        return self.cache.get(key)
    
    def cache_affiliate_programs(self, niche: str, programs: List[Dict[str, Any]], expire: int = 1800):
        """Cache affiliate programs data"""
        key = CacheKeys.format(CacheKeys.AFFILIATE_PROGRAMS, niche=niche)
        self.cache.set(key, programs, expire)
    
    def get_cached_affiliate_programs(self, niche: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached affiliate programs data"""
        key = CacheKeys.format(CacheKeys.AFFILIATE_PROGRAMS, niche=niche)
        return self.cache.get(key)
    
    def set_rate_limit(self, user_id: str, endpoint: str, limit: int, window: int):
        """Set rate limit for user and endpoint"""
        key = CacheKeys.format(CacheKeys.API_RATE_LIMIT, user_id=user_id, endpoint=endpoint)
        self.cache.increment(key, 1, window)
    
    def check_rate_limit(self, user_id: str, endpoint: str, limit: int) -> bool:
        """Check if user has exceeded rate limit"""
        key = CacheKeys.format(CacheKeys.API_RATE_LIMIT, user_id=user_id, endpoint=endpoint)
        current_count = self.cache.get(key) or 0
        return current_count < limit
    
    def blacklist_jwt_token(self, token_hash: str, expire: int = 86400):
        """Add JWT token to blacklist"""
        key = CacheKeys.format(CacheKeys.JWT_BLACKLIST, token_hash=token_hash)
        self.cache.set(key, True, expire)
    
    def is_jwt_blacklisted(self, token_hash: str) -> bool:
        """Check if JWT token is blacklisted"""
        key = CacheKeys.format(CacheKeys.JWT_BLACKLIST, token_hash=token_hash)
        return self.cache.exists(key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        info = get_redis_info()
        return {
            "redis_info": info,
            "total_keys": len(self.cache.get_keys()),
            "memory_usage": info.get("used_memory", "unknown"),
            "connected_clients": info.get("connected_clients", 0)
        }

# Global cache manager
cache_manager = CacheManager()