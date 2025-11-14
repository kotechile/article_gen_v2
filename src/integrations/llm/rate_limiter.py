"""
Rate limiter for LLM requests.

This module provides rate limiting functionality to prevent
exceeding API rate limits for LLM providers.
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import deque
import time


logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for LLM requests.
    
    This rate limiter implements token bucket algorithm
    to control request rate and prevent exceeding API limits.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
        window_size: int = 60
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_hour: Maximum requests per hour
            burst_size: Maximum burst size
            window_size: Time window in seconds
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.window_size = window_size
        
        # Token bucket for minute rate limiting
        self.minute_tokens = requests_per_minute
        self.minute_last_refill = time.time()
        
        # Token bucket for hour rate limiting
        self.hour_tokens = requests_per_hour
        self.hour_last_refill = time.time()
        
        # Request history for sliding window
        self.request_history = deque()
        
        # Statistics
        self.total_requests = 0
        self.blocked_requests = 0
        
        logger.info(f"RateLimiter initialized: {requests_per_minute}/min, {requests_per_hour}/hour")
    
    async def wait_if_needed(self):
        """
        Wait if rate limit would be exceeded.
        
        This method blocks until it's safe to make a request
        without exceeding rate limits.
        """
        current_time = time.time()
        
        # Refill tokens based on time passed
        self._refill_tokens(current_time)
        
        # Check if we can make a request
        if not self._can_make_request():
            # Calculate wait time
            wait_time = self._calculate_wait_time(current_time)
            
            if wait_time > 0:
                logger.debug(f"Rate limit exceeded, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Record the request
        self._record_request(current_time)
    
    def _refill_tokens(self, current_time: float):
        """Refill tokens based on time passed."""
        # Refill minute tokens
        time_passed = current_time - self.minute_last_refill
        tokens_to_add = time_passed * (self.requests_per_minute / 60)
        self.minute_tokens = min(self.requests_per_minute, self.minute_tokens + tokens_to_add)
        self.minute_last_refill = current_time
        
        # Refill hour tokens
        time_passed = current_time - self.hour_last_refill
        tokens_to_add = time_passed * (self.requests_per_hour / 3600)
        self.hour_tokens = min(self.requests_per_hour, self.hour_tokens + tokens_to_add)
        self.hour_last_refill = current_time
    
    def _can_make_request(self) -> bool:
        """Check if we can make a request without exceeding limits."""
        # Check minute rate limit
        if self.minute_tokens < 1:
            return False
        
        # Check hour rate limit
        if self.hour_tokens < 1:
            return False
        
        # Check burst limit
        current_time = time.time()
        recent_requests = [
            req_time for req_time in self.request_history
            if current_time - req_time < 1.0  # Last 1 second
        ]
        
        if len(recent_requests) >= self.burst_size:
            return False
        
        return True
    
    def _calculate_wait_time(self, current_time: float) -> float:
        """Calculate how long to wait before making a request."""
        wait_times = []
        
        # Wait for minute tokens
        if self.minute_tokens < 1:
            tokens_needed = 1 - self.minute_tokens
            time_needed = tokens_needed / (self.requests_per_minute / 60)
            wait_times.append(time_needed)
        
        # Wait for hour tokens
        if self.hour_tokens < 1:
            tokens_needed = 1 - self.hour_tokens
            time_needed = tokens_needed / (self.requests_per_hour / 3600)
            wait_times.append(time_needed)
        
        # Wait for burst limit
        recent_requests = [
            req_time for req_time in self.request_history
            if current_time - req_time < 1.0
        ]
        
        if len(recent_requests) >= self.burst_size:
            oldest_request = min(recent_requests)
            wait_time = 1.0 - (current_time - oldest_request)
            wait_times.append(wait_time)
        
        return max(wait_times) if wait_times else 0.0
    
    def _record_request(self, current_time: float):
        """Record a request in the history."""
        self.request_history.append(current_time)
        self.total_requests += 1
        
        # Clean old requests from history
        cutoff_time = current_time - self.window_size
        while self.request_history and self.request_history[0] < cutoff_time:
            self.request_history.popleft()
        
        # Consume tokens
        self.minute_tokens -= 1
        self.hour_tokens -= 1
    
    def get_stats(self) -> Dict[str, any]:
        """Get rate limiter statistics."""
        current_time = time.time()
        
        # Calculate current rates
        recent_requests = [
            req_time for req_time in self.request_history
            if current_time - req_time < 60  # Last minute
        ]
        
        requests_per_minute = len(recent_requests)
        
        recent_requests_hour = [
            req_time for req_time in self.request_history
            if current_time - req_time < 3600  # Last hour
        ]
        
        requests_per_hour = len(recent_requests_hour)
        
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "current_requests_per_minute": requests_per_minute,
            "current_requests_per_hour": requests_per_hour,
            "minute_tokens_remaining": max(0, self.minute_tokens),
            "hour_tokens_remaining": max(0, self.hour_tokens),
            "requests_per_minute_limit": self.requests_per_minute,
            "requests_per_hour_limit": self.requests_per_hour,
            "burst_size": self.burst_size
        }
    
    def update_limits(
        self,
        requests_per_minute: Optional[int] = None,
        requests_per_hour: Optional[int] = None,
        burst_size: Optional[int] = None
    ):
        """
        Update rate limits.
        
        Args:
            requests_per_minute: New requests per minute limit
            requests_per_hour: New requests per hour limit
            burst_size: New burst size limit
        """
        if requests_per_minute is not None:
            self.requests_per_minute = requests_per_minute
            self.minute_tokens = min(self.minute_tokens, requests_per_minute)
        
        if requests_per_hour is not None:
            self.requests_per_hour = requests_per_hour
            self.hour_tokens = min(self.hour_tokens, requests_per_hour)
        
        if burst_size is not None:
            self.burst_size = burst_size
        
        logger.info(f"Rate limits updated: {self.get_stats()}")


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts limits based on response times.
    
    This rate limiter monitors response times and adjusts rate limits
    to optimize throughput while staying within API limits.
    """
    
    def __init__(
        self,
        initial_requests_per_minute: int = 60,
        min_requests_per_minute: int = 10,
        max_requests_per_minute: int = 300,
        target_response_time: float = 2.0,
        adjustment_factor: float = 0.1,
        **kwargs
    ):
        """
        Initialize adaptive rate limiter.
        
        Args:
            initial_requests_per_minute: Initial rate limit
            min_requests_per_minute: Minimum rate limit
            max_requests_per_minute: Maximum rate limit
            target_response_time: Target response time in seconds
            adjustment_factor: Rate adjustment factor
            **kwargs: Additional rate limiter arguments
        """
        super().__init__(
            requests_per_minute=initial_requests_per_minute,
            **kwargs
        )
        
        self.initial_requests_per_minute = initial_requests_per_minute
        self.min_requests_per_minute = min_requests_per_minute
        self.max_requests_per_minute = max_requests_per_minute
        self.target_response_time = target_response_time
        self.adjustment_factor = adjustment_factor
        
        # Response time tracking
        self.response_times = deque(maxlen=100)  # Keep last 100 response times
        self.last_adjustment = time.time()
        self.adjustment_interval = 60  # Adjust every 60 seconds
        
        logger.info(f"AdaptiveRateLimiter initialized with target response time: {target_response_time}s")
    
    def record_response_time(self, response_time: float):
        """Record response time for adaptive adjustment."""
        self.response_times.append(response_time)
        
        # Check if it's time to adjust
        current_time = time.time()
        if current_time - self.last_adjustment >= self.adjustment_interval:
            self._adjust_rate_limit()
            self.last_adjustment = current_time
    
    def _adjust_rate_limit(self):
        """Adjust rate limit based on response times."""
        if len(self.response_times) < 10:  # Need at least 10 samples
            return
        
        # Calculate average response time
        avg_response_time = sum(self.response_times) / len(self.response_times)
        
        # Calculate adjustment
        if avg_response_time > self.target_response_time * 1.2:
            # Response time is too high, reduce rate
            adjustment = -self.adjustment_factor
        elif avg_response_time < self.target_response_time * 0.8:
            # Response time is good, increase rate
            adjustment = self.adjustment_factor
        else:
            # Response time is acceptable, no adjustment
            return
        
        # Calculate new rate limit
        current_rate = self.requests_per_minute
        new_rate = current_rate * (1 + adjustment)
        
        # Apply limits
        new_rate = max(self.min_requests_per_minute, new_rate)
        new_rate = min(self.max_requests_per_minute, new_rate)
        
        # Update rate limit
        if new_rate != current_rate:
            self.update_limits(requests_per_minute=int(new_rate))
            logger.info(
                f"Rate limit adjusted from {current_rate} to {int(new_rate)} "
                f"(avg response time: {avg_response_time:.2f}s)"
            )
    
    def get_adaptive_stats(self) -> Dict[str, any]:
        """Get adaptive rate limiter statistics."""
        stats = self.get_stats()
        
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            stats["average_response_time"] = avg_response_time
            stats["target_response_time"] = self.target_response_time
            stats["response_time_ratio"] = avg_response_time / self.target_response_time
        
        return stats
