"""
Search cache utility for Linkup search results.

Provides query normalization, cache key generation, and TTL management
for caching expensive Linkup API calls.
"""

import logging
import hashlib
from typing import Optional, Dict, Any
import sys
import os

# Add parent directory to path to import cache utilities
# Try multiple import paths for different execution contexts
try:
    from app.utils.cache import CacheManager, CacheKeys
except ImportError:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
        from app.utils.cache import CacheManager, CacheKeys
    except ImportError:
        # Fallback: create a minimal CacheManager interface if not available
        class CacheManager:
            def __init__(self, redis_url: str = None):
                pass
            def get(self, key: str):
                return None
            def set(self, key: str, value: Any, ttl: int = None):
                return False

logger = logging.getLogger(__name__)


class SearchCache:
    """
    Cache utility for Linkup search queries.
    
    Provides:
    - Query normalization (lowercase, strip, normalize whitespace)
    - Cache key generation from query parameters
    - TTL management (different TTLs for standard vs deep searches)
    - Hit/miss metrics logging
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None, enabled: bool = True):
        """
        Initialize SearchCache.
        
        Args:
            cache_manager: CacheManager instance (creates new one if None)
            enabled: Whether caching is enabled (default: True)
        """
        self.cache_manager = cache_manager or CacheManager()
        self.enabled = enabled
        self.hit_count = 0
        self.miss_count = 0
        
    def normalize_query(self, query: str) -> str:
        """
        Normalize search query for consistent caching.
        
        Args:
            query: Original search query
            
        Returns:
            Normalized query string
        """
        if not query:
            return ""
        
        # Lowercase, strip, and collapse multiple spaces
        normalized = " ".join(query.lower().strip().split())
        return normalized
    
    def generate_cache_key(
        self,
        query: str,
        depth: str = "standard",
        date_range: Optional[str] = None,
        site: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> str:
        """
        Generate cache key from search parameters.
        
        Args:
            query: Search query
            depth: Search depth ('standard' or 'deep')
            date_range: Optional date range filter
            site: Optional site filter
            max_results: Optional max results limit
            
        Returns:
            Cache key string
        """
        # Normalize query
        normalized_query = self.normalize_query(query)
        
        # Create key components
        components = [
            "linkup",
            normalized_query,
            depth or "standard",
            date_range or "any",
            site or "all",
            str(max_results or 10)
        ]
        
        # Create hash for consistent key length and avoid special chars
        key_string = ":".join(components)
        key_hash = hashlib.md5(key_string.encode('utf-8')).hexdigest()
        
        # Use readable prefix with hash
        return f"linkup_search:{depth}:{key_hash}"
    
    def get(
        self,
        query: str,
        depth: str = "standard",
        date_range: Optional[str] = None,
        site: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached search result.
        
        Args:
            query: Search query
            depth: Search depth
            date_range: Optional date range filter
            site: Optional site filter
            max_results: Optional max results limit
            
        Returns:
            Cached result dict if found, None otherwise
        """
        if not self.enabled:
            return None
            
        try:
            cache_key = self.generate_cache_key(query, depth, date_range, site, max_results)
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result:
                self.hit_count += 1
                logger.debug(f"Cache HIT for query: '{query[:50]}...' (key: {cache_key})")
                return cached_result
            else:
                self.miss_count += 1
                logger.debug(f"Cache MISS for query: '{query[:50]}...' (key: {cache_key})")
                return None
                
        except Exception as e:
            logger.warning(f"Cache get error: {str(e)}, treating as miss")
            self.miss_count += 1
            return None
    
    def set(
        self,
        query: str,
        result: Dict[str, Any],
        depth: str = "standard",
        ttl_standard: int = 21600,  # 6 hours
        ttl_deep: int = 86400,  # 24 hours
        date_range: Optional[str] = None,
        site: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> bool:
        """
        Cache search result.
        
        Args:
            query: Search query
            result: Search result to cache (must be JSON-serializable)
            depth: Search depth ('standard' or 'deep')
            ttl_standard: TTL in seconds for standard searches (default: 6 hours)
            ttl_deep: TTL in seconds for deep searches (default: 24 hours)
            date_range: Optional date range filter
            site: Optional site filter
            max_results: Optional max results limit
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            cache_key = self.generate_cache_key(query, depth, date_range, site, max_results)
            ttl = ttl_deep if depth == "deep" else ttl_standard
            
            success = self.cache_manager.set(cache_key, result, ttl=ttl)
            
            if success:
                logger.debug(f"Cached result for query: '{query[:50]}...' (key: {cache_key}, TTL: {ttl}s)")
            else:
                logger.warning(f"Failed to cache result for query: '{query[:50]}...'")
            
            return success
            
        except Exception as e:
            logger.warning(f"Cache set error: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with hit count, miss count, and hit rate
        """
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0.0
        
        return {
            "enabled": self.enabled,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": total_requests,
            "hit_rate": hit_rate
        }
    
    def reset_stats(self):
        """Reset cache statistics counters."""
        self.hit_count = 0
        self.miss_count = 0
        logger.debug("Cache statistics reset")

