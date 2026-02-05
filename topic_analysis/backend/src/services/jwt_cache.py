"""
JWT validation caching service for performance optimization
"""
import logging
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json

from .redis_cache import get_cache_service
from .jwt_service import JWTService
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class JWTCacheService:
    """Service for JWT validation caching"""
    
    def __init__(self):
        self.cache_service = get_cache_service()
        self.jwt_service = JWTService()
        self.default_ttl = settings.JWT_CACHE_TTL
        self.max_cache_size = settings.JWT_CACHE_MAX_SIZE
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    def validate_token_cached(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate JWT token with caching"""
        try:
            self.cache_stats["total_requests"] += 1
            
            # Generate cache key for token
            cache_key = self._generate_token_cache_key(token)
            
            # Try to get from cache first
            cached_result = self.cache_service.get(cache_key)
            if cached_result is not None:
                self.cache_stats["hits"] += 1
                logger.debug(f"JWT validation cache hit for token: {token[:10]}...")
                return cached_result["valid"], cached_result.get("payload")
            
            # Cache miss - validate token
            self.cache_stats["misses"] += 1
            logger.debug(f"JWT validation cache miss for token: {token[:10]}...")
            
            # Validate token using JWT service
            is_valid, payload = self.jwt_service.validate_token(token)
            
            # Cache the result
            cache_data = {
                "valid": is_valid,
                "payload": payload,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Set TTL based on token expiration
            ttl = self._calculate_token_ttl(payload)
            self.cache_service.set(cache_key, cache_data, ttl)
            
            return is_valid, payload
            
        except Exception as e:
            logger.error(f"Error in JWT validation caching: {str(e)}")
            # Fallback to direct validation
            return self.jwt_service.validate_token(token)
    
    def get_user_from_token_cached(self, token: str) -> Optional[Dict[str, Any]]:
        """Get user information from token with caching"""
        try:
            # Generate cache key for user info
            cache_key = self._generate_user_cache_key(token)
            
            # Try to get from cache first
            cached_user = self.cache_service.get(cache_key)
            if cached_user is not None:
                self.cache_stats["hits"] += 1
                logger.debug(f"JWT user cache hit for token: {token[:10]}...")
                return cached_user
            
            # Cache miss - get user info
            self.cache_stats["misses"] += 1
            logger.debug(f"JWT user cache miss for token: {token[:10]}...")
            
            # Validate token and get payload
            is_valid, payload = self.jwt_service.validate_token(token)
            if not is_valid or not payload:
                return None
            
            # Extract user information
            user_info = {
                "user_id": payload.get("user_id"),
                "email": payload.get("email"),
                "username": payload.get("username"),
                "is_admin": payload.get("is_admin", False),
                "exp": payload.get("exp"),
                "iat": payload.get("iat"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Cache user info
            ttl = self._calculate_token_ttl(payload)
            self.cache_service.set(cache_key, user_info, ttl)
            
            return user_info
            
        except Exception as e:
            logger.error(f"Error in JWT user caching: {str(e)}")
            # Fallback to direct validation
            is_valid, payload = self.jwt_service.validate_token(token)
            if not is_valid or not payload:
                return None
            
            return {
                "user_id": payload.get("user_id"),
                "email": payload.get("email"),
                "username": payload.get("username"),
                "is_admin": payload.get("is_admin", False),
                "exp": payload.get("exp"),
                "iat": payload.get("iat")
            }
    
    def invalidate_token_cache(self, token: str) -> bool:
        """Invalidate cached token data"""
        try:
            # Generate cache keys
            token_cache_key = self._generate_token_cache_key(token)
            user_cache_key = self._generate_user_cache_key(token)
            
            # Delete from cache
            deleted_count = self.cache_service.delete_many([token_cache_key, user_cache_key])
            
            logger.info(f"Invalidated JWT cache for token: {token[:10]}...")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error invalidating JWT cache: {str(e)}")
            return False
    
    def invalidate_user_cache(self, user_id: int) -> bool:
        """Invalidate all cached data for a user"""
        try:
            # Clear all cache entries for this user
            pattern = f"jwt:user:{user_id}:*"
            deleted_count = self.cache_service.clear_pattern(pattern)
            
            logger.info(f"Invalidated JWT cache for user {user_id}: {deleted_count} entries")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error invalidating user JWT cache: {str(e)}")
            return False
    
    def preload_user_tokens(self, user_id: int, tokens: list) -> int:
        """Preload user tokens into cache"""
        try:
            preloaded_count = 0
            
            for token in tokens:
                # Validate and cache token
                is_valid, payload = self.jwt_service.validate_token(token)
                if is_valid and payload:
                    # Cache token validation
                    token_cache_key = self._generate_token_cache_key(token)
                    cache_data = {
                        "valid": True,
                        "payload": payload,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    ttl = self._calculate_token_ttl(payload)
                    self.cache_service.set(token_cache_key, cache_data, ttl)
                    
                    # Cache user info
                    user_cache_key = self._generate_user_cache_key(token)
                    user_info = {
                        "user_id": payload.get("user_id"),
                        "email": payload.get("email"),
                        "username": payload.get("username"),
                        "is_admin": payload.get("is_admin", False),
                        "exp": payload.get("exp"),
                        "iat": payload.get("iat"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    self.cache_service.set(user_cache_key, user_info, ttl)
                    
                    preloaded_count += 1
            
            logger.info(f"Preloaded {preloaded_count} tokens for user {user_id}")
            return preloaded_count
            
        except Exception as e:
            logger.error(f"Error preloading user tokens: {str(e)}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get JWT cache statistics"""
        try:
            total_requests = self.cache_stats["total_requests"]
            hits = self.cache_stats["hits"]
            misses = self.cache_stats["misses"]
            
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "total_requests": total_requests,
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate": hit_rate,
                "evictions": self.cache_stats["evictions"],
                "cache_size": self._get_cache_size(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting JWT cache stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_cache(self) -> bool:
        """Clear all JWT cache data"""
        try:
            # Clear all JWT-related cache entries
            pattern = "jwt:*"
            deleted_count = self.cache_service.clear_pattern(pattern)
            
            # Reset stats
            self.cache_stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "total_requests": 0
            }
            
            logger.info(f"Cleared JWT cache: {deleted_count} entries")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing JWT cache: {str(e)}")
            return False
    
    def _generate_token_cache_key(self, token: str) -> str:
        """Generate cache key for token validation"""
        # Use hash of token for consistent key length
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
        return f"jwt:token:{token_hash}"
    
    def _generate_user_cache_key(self, token: str) -> str:
        """Generate cache key for user information"""
        # Use hash of token for consistent key length
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
        return f"jwt:user:{token_hash}"
    
    def _calculate_token_ttl(self, payload: Optional[Dict[str, Any]]) -> int:
        """Calculate TTL for token based on expiration"""
        try:
            if not payload or "exp" not in payload:
                return self.default_ttl
            
            exp_timestamp = payload["exp"]
            current_timestamp = int(time.time())
            
            # Calculate remaining time until expiration
            remaining_seconds = exp_timestamp - current_timestamp
            
            # Use remaining time or default TTL, whichever is smaller
            return min(remaining_seconds, self.default_ttl)
            
        except Exception as e:
            logger.error(f"Error calculating token TTL: {str(e)}")
            return self.default_ttl
    
    def _get_cache_size(self) -> int:
        """Get current cache size"""
        try:
            # Count JWT-related cache entries
            pattern = "jwt:*"
            keys = self.cache_service.redis_client.keys(f"{self.cache_service.key_prefix}:{pattern}")
            return len(keys)
        except Exception:
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on JWT cache"""
        try:
            start_time = time.time()
            
            # Test cache operations
            test_token = "test_token_123"
            test_cache_key = self._generate_token_cache_key(test_token)
            
            # Test set
            test_data = {
                "valid": True,
                "payload": {"user_id": 1, "exp": int(time.time()) + 3600},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            set_result = self.cache_service.set(test_cache_key, test_data, ttl=10)
            if not set_result:
                return {
                    "healthy": False,
                    "error": "Failed to set test cache data",
                    "response_time_ms": 0
                }
            
            # Test get
            get_result = self.cache_service.get(test_cache_key)
            if get_result is None:
                return {
                    "healthy": False,
                    "error": "Failed to get test cache data",
                    "response_time_ms": 0
                }
            
            # Test delete
            delete_result = self.cache_service.delete(test_cache_key)
            if not delete_result:
                return {
                    "healthy": False,
                    "error": "Failed to delete test cache data",
                    "response_time_ms": 0
                }
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "healthy": True,
                "response_time_ms": response_time,
                "cache_stats": self.get_cache_stats(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"JWT cache health check failed: {str(e)}")
            return {
                "healthy": False,
                "error": str(e),
                "response_time_ms": 0
            }

# Global JWT cache instance
_jwt_cache_service = None

def get_jwt_cache_service() -> JWTCacheService:
    """Get global JWT cache service instance"""
    global _jwt_cache_service
    if _jwt_cache_service is None:
        _jwt_cache_service = JWTCacheService()
    return _jwt_cache_service

def cached_jwt_validation(ttl: Optional[int] = None):
    """Decorator for caching JWT validation results"""
    def decorator(func):
        def wrapper(token: str, *args, **kwargs):
            jwt_cache = get_jwt_cache_service()
            
            # Try to get from cache first
            cache_key = jwt_cache._generate_token_cache_key(token)
            cached_result = jwt_cache.cache_service.get(cache_key)
            
            if cached_result is not None:
                return cached_result["valid"], cached_result.get("payload")
            
            # Execute function and cache result
            is_valid, payload = func(token, *args, **kwargs)
            
            # Cache the result
            cache_data = {
                "valid": is_valid,
                "payload": payload,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Calculate TTL for caching
            cache_ttl = ttl if ttl is not None else jwt_cache._calculate_token_ttl(payload)
            
            jwt_cache.cache_service.set(cache_key, cache_data, cache_ttl)
            
            return is_valid, payload
        
        return wrapper
    return decorator
