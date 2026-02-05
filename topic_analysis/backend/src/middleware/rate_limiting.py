"""
Rate Limiting Middleware
Implements various rate limiting algorithms and strategies
"""

import time
import hashlib
from typing import Dict, Any, Optional, List
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import structlog

from ..core.redis import cache_manager
from ..core.config import settings

logger = structlog.get_logger()

class RateLimiter:
    """Rate limiter with multiple algorithms"""
    
    def __init__(self):
        self.redis = cache_manager.cache
    
    def get_client_identifier(self, request: Request) -> str:
        """Get unique identifier for the client"""
        # Try to get user ID from request state (set by auth middleware)
        if hasattr(request.state, 'user_id'):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host
        return f"ip:{client_ip}"
    
    def sliding_window_limit(
        self,
        identifier: str,
        window_size: int,
        max_requests: int,
        endpoint: str = "global"
    ) -> bool:
        """
        Sliding window rate limiting
        More accurate but more resource intensive
        """
        now = time.time()
        window_start = now - window_size
        
        # Create sorted set key
        key = f"rate_limit:sliding:{identifier}:{endpoint}"
        
        # Remove expired entries
        self.redis.client.zremrangebyscore(key, 0, window_start)
        
        # Count current requests in window
        current_requests = self.redis.client.zcard(key)
        
        if current_requests >= max_requests:
            return False
        
        # Add current request
        self.redis.client.zadd(key, {str(now): now})
        self.redis.client.expire(key, window_size)
        
        return True
    
    def fixed_window_limit(
        self,
        identifier: str,
        window_size: int,
        max_requests: int,
        endpoint: str = "global"
    ) -> bool:
        """
        Fixed window rate limiting
        Simpler but can have burst issues at window boundaries
        """
        now = time.time()
        window = int(now // window_size)
        
        key = f"rate_limit:fixed:{identifier}:{endpoint}:{window}"
        
        # Get current count
        current_requests = self.redis.get(key) or 0
        
        if current_requests >= max_requests:
            return False
        
        # Increment counter
        self.redis.increment(key, 1, window_size)
        
        return True
    
    def token_bucket_limit(
        self,
        identifier: str,
        bucket_size: int,
        refill_rate: float,
        endpoint: str = "global"
    ) -> bool:
        """
        Token bucket rate limiting
        Allows burst traffic up to bucket size
        """
        now = time.time()
        key = f"rate_limit:bucket:{identifier}:{endpoint}"
        
        # Get bucket state
        bucket_data = self.redis.get(key)
        
        if bucket_data:
            tokens, last_refill = bucket_data
        else:
            tokens = bucket_size
            last_refill = now
        
        # Calculate tokens to add based on time passed
        time_passed = now - last_refill
        tokens_to_add = time_passed * refill_rate
        tokens = min(bucket_size, tokens + tokens_to_add)
        
        if tokens < 1:
            return False
        
        # Consume a token
        tokens -= 1
        
        # Update bucket state
        self.redis.set(key, [tokens, now], 3600)  # 1 hour expiry
        
        return True
    
    def adaptive_limit(
        self,
        identifier: str,
        base_limit: int,
        window_size: int,
        endpoint: str = "global"
    ) -> bool:
        """
        Adaptive rate limiting based on system load
        """
        # Get system load (simplified)
        system_load = self._get_system_load()
        
        # Adjust limit based on load
        if system_load > 0.8:  # High load
            adjusted_limit = int(base_limit * 0.5)
        elif system_load > 0.6:  # Medium load
            adjusted_limit = int(base_limit * 0.7)
        else:  # Low load
            adjusted_limit = base_limit
        
        return self.sliding_window_limit(identifier, window_size, adjusted_limit, endpoint)
    
    def _get_system_load(self) -> float:
        """Get current system load (simplified)"""
        try:
            # In a real implementation, you would get actual system metrics
            # For now, return a mock value
            return 0.3
        except Exception:
            return 0.5
    
    def check_rate_limit(
        self,
        request: Request,
        algorithm: str = "sliding_window",
        **kwargs
    ) -> bool:
        """Check rate limit using specified algorithm"""
        identifier = self.get_client_identifier(request)
        endpoint = request.url.path
        
        try:
            if algorithm == "sliding_window":
                return self.sliding_window_limit(
                    identifier,
                    kwargs.get("window_size", 60),
                    kwargs.get("max_requests", 100),
                    endpoint
                )
            elif algorithm == "fixed_window":
                return self.fixed_window_limit(
                    identifier,
                    kwargs.get("window_size", 60),
                    kwargs.get("max_requests", 100),
                    endpoint
                )
            elif algorithm == "token_bucket":
                return self.token_bucket_limit(
                    identifier,
                    kwargs.get("bucket_size", 100),
                    kwargs.get("refill_rate", 1.0),
                    endpoint
                )
            elif algorithm == "adaptive":
                return self.adaptive_limit(
                    identifier,
                    kwargs.get("base_limit", 100),
                    kwargs.get("window_size", 60),
                    endpoint
                )
            else:
                logger.warning("Unknown rate limiting algorithm", algorithm=algorithm)
                return True
                
        except Exception as e:
            logger.error("Rate limiting error", error=str(e))
            # Fail open - allow request if rate limiting fails
            return True

class RateLimitConfig:
    """Rate limiting configuration for different endpoints"""
    
    def __init__(self):
        self.configs = {
            # Authentication endpoints
            "/api/auth/login": {
                "algorithm": "fixed_window",
                "window_size": 300,  # 5 minutes
                "max_requests": 5
            },
            "/api/auth/register": {
                "algorithm": "fixed_window",
                "window_size": 3600,  # 1 hour
                "max_requests": 3
            },
            "/api/auth/forgot-password": {
                "algorithm": "fixed_window",
                "window_size": 3600,  # 1 hour
                "max_requests": 3
            },
            
            # API endpoints
            "/api/affiliate/research": {
                "algorithm": "sliding_window",
                "window_size": 60,
                "max_requests": 10
            },
            "/api/trends/analyze": {
                "algorithm": "sliding_window",
                "window_size": 60,
                "max_requests": 20
            },
            "/api/keywords/upload": {
                "algorithm": "sliding_window",
                "window_size": 300,  # 5 minutes
                "max_requests": 5
            },
            "/api/content/generate": {
                "algorithm": "token_bucket",
                "bucket_size": 10,
                "refill_rate": 0.1  # 1 request per 10 seconds
            },
            "/api/software/generate": {
                "algorithm": "token_bucket",
                "bucket_size": 5,
                "refill_rate": 0.05  # 1 request per 20 seconds
            },
            
            # Export endpoints
            "/api/export/": {
                "algorithm": "fixed_window",
                "window_size": 3600,  # 1 hour
                "max_requests": 20
            },
            
            # Default configuration
            "default": {
                "algorithm": "sliding_window",
                "window_size": 60,
                "max_requests": 100
            }
        }
    
    def get_config(self, path: str) -> Dict[str, Any]:
        """Get rate limiting configuration for a path"""
        # Check for exact match
        if path in self.configs:
            return self.configs[path]
        
        # Check for prefix match
        for config_path, config in self.configs.items():
            if path.startswith(config_path):
                return config
        
        # Return default configuration
        return self.configs["default"]

# Global instances
rate_limiter = RateLimiter()
rate_limit_config = RateLimitConfig()

# Middleware function
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    try:
        # Get rate limiting configuration for this endpoint
        config = rate_limit_config.get_config(request.url.path)
        
        # Check rate limit
        if not rate_limiter.check_rate_limit(request, **config):
            logger.warning(
                "Rate limit exceeded",
                path=request.url.path,
                client=request.client.host
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": config.get("window_size", 60)
                },
                headers={
                    "Retry-After": str(config.get("window_size", 60))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(config.get("max_requests", 100))
        response.headers["X-RateLimit-Window"] = str(config.get("window_size", 60))
        
        return response
        
    except Exception as e:
        logger.error("Rate limiting middleware error", error=str(e))
        # Fail open - process request if middleware fails
        return await call_next(request)

# Rate limiting decorators
def rate_limit(algorithm: str = "sliding_window", **kwargs):
    """Decorator for rate limiting specific endpoints"""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            if not rate_limiter.check_rate_limit(request, algorithm, **kwargs):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# Utility functions
def get_rate_limit_status(identifier: str, endpoint: str = "global") -> Dict[str, Any]:
    """Get current rate limit status for an identifier"""
    try:
        # Get sliding window status
        sliding_key = f"rate_limit:sliding:{identifier}:{endpoint}"
        current_requests = rate_limiter.redis.client.zcard(sliding_key)
        
        # Get fixed window status
        now = time.time()
        window = int(now // 60)  # 1 minute windows
        fixed_key = f"rate_limit:fixed:{identifier}:{endpoint}:{window}"
        fixed_requests = rate_limiter.redis.get(fixed_key) or 0
        
        # Get token bucket status
        bucket_key = f"rate_limit:bucket:{identifier}:{endpoint}"
        bucket_data = rate_limiter.redis.get(bucket_key)
        bucket_tokens = bucket_data[0] if bucket_data else 0
        
        return {
            "identifier": identifier,
            "endpoint": endpoint,
            "sliding_window_requests": current_requests,
            "fixed_window_requests": fixed_requests,
            "token_bucket_tokens": bucket_tokens,
            "timestamp": now
        }
        
    except Exception as e:
        logger.error("Error getting rate limit status", error=str(e))
        return {}

def reset_rate_limit(identifier: str, endpoint: str = "global") -> bool:
    """Reset rate limit for an identifier"""
    try:
        patterns = [
            f"rate_limit:sliding:{identifier}:{endpoint}",
            f"rate_limit:fixed:{identifier}:{endpoint}:*",
            f"rate_limit:bucket:{identifier}:{endpoint}"
        ]
        
        for pattern in patterns:
            rate_limiter.redis.delete_pattern(pattern)
        
        logger.info("Rate limit reset", identifier=identifier, endpoint=endpoint)
        return True
        
    except Exception as e:
        logger.error("Error resetting rate limit", error=str(e))
        return False

def get_rate_limit_stats() -> Dict[str, Any]:
    """Get overall rate limiting statistics"""
    try:
        # Get all rate limit keys
        sliding_keys = rate_limiter.redis.get_keys("rate_limit:sliding:*")
        fixed_keys = rate_limiter.redis.get_keys("rate_limit:fixed:*")
        bucket_keys = rate_limiter.redis.get_keys("rate_limit:bucket:*")
        
        return {
            "sliding_window_entries": len(sliding_keys),
            "fixed_window_entries": len(fixed_keys),
            "token_bucket_entries": len(bucket_keys),
            "total_rate_limit_entries": len(sliding_keys) + len(fixed_keys) + len(bucket_keys),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error("Error getting rate limit stats", error=str(e))
        return {}