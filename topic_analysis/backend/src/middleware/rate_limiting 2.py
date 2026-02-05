"""
Rate limiting middleware for Google Autocomplete requests.
"""
import time
from typing import Dict, Optional
from collections import defaultdict, deque
from fastapi import HTTPException, Request
import asyncio

class RateLimiter:
    """Rate limiter for Google Autocomplete requests."""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Client identifier (IP address or user ID)
            
        Returns:
            True if request is allowed, False otherwise
        """
        async with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # Remove old requests outside time window
            while client_requests and client_requests[0] <= now - self.time_window:
                client_requests.popleft()
            
            # Check if we're under the limit
            if len(client_requests) < self.max_requests:
                client_requests.append(now)
                return True
            
            return False
    
    async def get_retry_after(self, client_id: str) -> int:
        """
        Get seconds until next request is allowed.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Seconds until next request is allowed
        """
        async with self.lock:
            client_requests = self.requests[client_id]
            if not client_requests:
                return 0
            
            oldest_request = client_requests[0]
            return max(0, int(oldest_request + self.time_window - time.time()))

# Global rate limiter instance
google_autocomplete_limiter = RateLimiter(max_requests=10, time_window=60)

async def check_rate_limit(request: Request) -> None:
    """
    Check rate limit for Google Autocomplete requests.
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    # Get client identifier (IP address)
    client_id = request.client.host if request.client else "unknown"
    
    if not await google_autocomplete_limiter.is_allowed(client_id):
        retry_after = await google_autocomplete_limiter.get_retry_after(client_id)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": "Too many Google Autocomplete requests. Please try again later.",
                "retry_after": retry_after
            }
        )

def create_rate_limit_middleware(limiter: RateLimiter):
    """
    Create rate limiting middleware for FastAPI.
    
    Args:
        limiter: Rate limiter instance
        
    Returns:
        Middleware function
    """
    async def rate_limit_middleware(request: Request, call_next):
        # Only apply rate limiting to Google Autocomplete endpoints
        if request.url.path.startswith("/api/enhanced-topics/autocomplete"):
            await check_rate_limit(request)
        
        response = await call_next(request)
        return response
    
    return rate_limit_middleware