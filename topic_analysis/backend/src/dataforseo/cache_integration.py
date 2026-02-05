"""
Cache integration for DataForSEO features

Handles Redis caching for API responses and computed data
to improve performance and reduce API costs.
"""

import asyncio
import json
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
import hashlib

import redis.asyncio as redis
from redis.exceptions import RedisError

from ..core.config import settings
from ..models.trend_data import TrendData
from ..models.keyword_data import KeywordData
from ..models.subtopic_data import SubtopicData

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages Redis caching for DataForSEO features"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.cache_ttl = {
            'trend_data': 86400,      # 24 hours
            'keyword_data': 21600,    # 6 hours
            'suggestions': 3600,      # 1 hour
            'api_errors': 300,        # 5 minutes
            'computed_data': 1800     # 30 minutes
        }
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            # Fallback to no caching
            self.redis_client = None
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis cache connection closed")
    
    def _generate_cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate cache key from parameters"""
        # Sort parameters for consistent key generation
        sorted_params = sorted(params.items())
        param_string = json.dumps(sorted_params, sort_keys=True)
        
        # Create hash for long parameter strings
        if len(param_string) > 100:
            param_hash = hashlib.md5(param_string.encode()).hexdigest()
            return f"dataforseo:{prefix}:{param_hash}"
        else:
            return f"dataforseo:{prefix}:{param_string}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.redis_client:
            return False
        
        try:
            serialized_value = json.dumps(value, default=str)
            if ttl:
                await self.redis_client.setex(key, ttl, serialized_value)
            else:
                await self.redis_client.set(key, serialized_value)
            return True
        except (RedisError, TypeError) as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.delete(key)
            return True
        except RedisError as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    async def get_trend_data(self, subtopics: List[str], location: str, time_range: str) -> Optional[List[TrendData]]:
        """Get cached trend data"""
        cache_key = self._generate_cache_key(
            "trend_data",
            {
                "subtopics": sorted(subtopics),
                "location": location,
                "time_range": time_range
            }
        )
        
        cached_data = await self.get(cache_key)
        if cached_data:
            try:
                return [TrendData(**item) for item in cached_data]
            except Exception as e:
                logger.warning(f"Error deserializing cached trend data: {e}")
                await self.delete(cache_key)
        
        return None
    
    async def set_trend_data(self, subtopics: List[str], location: str, time_range: str, data: List[TrendData]) -> bool:
        """Cache trend data"""
        cache_key = self._generate_cache_key(
            "trend_data",
            {
                "subtopics": sorted(subtopics),
                "location": location,
                "time_range": time_range
            }
        )
        
        serialized_data = [item.dict() for item in data]
        return await self.set(cache_key, serialized_data, self.cache_ttl['trend_data'])
    
    async def get_keyword_data(self, seed_keywords: List[str], filters: Dict[str, Any]) -> Optional[List[KeywordData]]:
        """Get cached keyword data"""
        cache_key = self._generate_cache_key(
            "keyword_data",
            {
                "seed_keywords": sorted(seed_keywords),
                "filters": filters
            }
        )
        
        cached_data = await self.get(cache_key)
        if cached_data:
            try:
                return [KeywordData(**item) for item in cached_data]
            except Exception as e:
                logger.warning(f"Error deserializing cached keyword data: {e}")
                await self.delete(cache_key)
        
        return None
    
    async def set_keyword_data(self, seed_keywords: List[str], filters: Dict[str, Any], data: List[KeywordData]) -> bool:
        """Cache keyword data"""
        cache_key = self._generate_cache_key(
            "keyword_data",
            {
                "seed_keywords": sorted(seed_keywords),
                "filters": filters
            }
        )
        
        serialized_data = [item.dict() for item in data]
        return await self.set(cache_key, serialized_data, self.cache_ttl['keyword_data'])
    
    async def get_suggestions(self, base_subtopics: List[str], location: str) -> Optional[List[SubtopicData]]:
        """Get cached suggestions"""
        cache_key = self._generate_cache_key(
            "suggestions",
            {
                "base_subtopics": sorted(base_subtopics),
                "location": location
            }
        )
        
        cached_data = await self.get(cache_key)
        if cached_data:
            try:
                return [SubtopicData(**item) for item in cached_data]
            except Exception as e:
                logger.warning(f"Error deserializing cached suggestions: {e}")
                await self.delete(cache_key)
        
        return None
    
    async def set_suggestions(self, base_subtopics: List[str], location: str, data: List[SubtopicData]) -> bool:
        """Cache suggestions"""
        cache_key = self._generate_cache_key(
            "suggestions",
            {
                "base_subtopics": sorted(base_subtopics),
                "location": location
            }
        )
        
        serialized_data = [item.dict() for item in data]
        return await self.set(cache_key, serialized_data, self.cache_ttl['suggestions'])
    
    async def get_api_error(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached API error"""
        cache_key = self._generate_cache_key(
            "api_errors",
            {
                "endpoint": endpoint,
                "params": params
            }
        )
        
        return await self.get(cache_key)
    
    async def set_api_error(self, endpoint: str, params: Dict[str, Any], error_data: Dict[str, Any]) -> bool:
        """Cache API error"""
        cache_key = self._generate_cache_key(
            "api_errors",
            {
                "endpoint": endpoint,
                "params": params
            }
        )
        
        return await self.set(cache_key, error_data, self.cache_ttl['api_errors'])
    
    async def get_computed_data(self, computation_type: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached computed data"""
        cache_key = self._generate_cache_key(
            "computed_data",
            {
                "type": computation_type,
                "params": params
            }
        )
        
        return await self.get(cache_key)
    
    async def set_computed_data(self, computation_type: str, params: Dict[str, Any], data: Any) -> bool:
        """Cache computed data"""
        cache_key = self._generate_cache_key(
            "computed_data",
            {
                "type": computation_type,
                "params": params
            }
        )
        
        return await self.set(cache_key, data, self.cache_ttl['computed_data'])
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = await self.redis_client.keys(f"dataforseo:{pattern}")
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        except RedisError as e:
            logger.warning(f"Cache invalidation error for pattern {pattern}: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {"status": "disabled"}
        
        try:
            info = await self.redis_client.info()
            return {
                "status": "active",
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except RedisError as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}

# Global instance
cache_manager = CacheManager()
