"""
Retry handler for LLM requests.

This module provides retry logic with exponential backoff
for handling transient failures in LLM requests.
"""

import asyncio
import logging
from typing import Callable, Any, Optional
from datetime import datetime, timedelta
import random

from ...core.models.errors import LLMError


logger = logging.getLogger(__name__)


class RetryHandler:
    """
    Retry handler with exponential backoff for LLM requests.
    
    This handler implements intelligent retry logic with:
    - Exponential backoff
    - Jitter to prevent thundering herd
    - Retryable error detection
    - Maximum retry limits
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
        retryable_errors: Optional[list] = None
    ):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            backoff_multiplier: Backoff multiplier
            jitter: Whether to add jitter
            retryable_errors: List of retryable error types
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        self.retryable_errors = retryable_errors or [
            "RateLimitError",
            "TimeoutError", 
            "ConnectionError",
            "ServiceUnavailableError",
            "APIError"
        ]
        
        logger.info(f"RetryHandler initialized with max_retries: {max_retries}")
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            LLMError: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - return result
                if attempt > 0:
                    logger.info(f"Function succeeded after {attempt} retries")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if error is retryable
                if not self._is_retryable_error(e):
                    logger.error(f"Non-retryable error: {str(e)}")
                    raise e
                
                # Check if we've exhausted retries
                if attempt >= self.max_retries:
                    logger.error(f"All retries exhausted. Last error: {str(e)}")
                    break
                
                # Calculate delay for next retry
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries exhausted
        if isinstance(last_exception, LLMError):
            raise last_exception
        else:
            raise LLMError(
                message=f"All retries exhausted. Last error: {str(last_exception)}",
                retryable=False
            )
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Check if error is retryable.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is retryable
        """
        error_type = type(error).__name__
        
        # Check if error type is in retryable list
        if error_type in self.retryable_errors:
            return True
        
        # Check if it's an LLMError with retryable flag
        if isinstance(error, LLMError):
            return error.retryable
        
        # Check for specific error patterns
        error_message = str(error).lower()
        retryable_patterns = [
            "rate limit",
            "timeout",
            "connection",
            "service unavailable",
            "temporary",
            "retry"
        ]
        
        return any(pattern in error_message for pattern in retryable_patterns)
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Calculate exponential backoff
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            jitter = random.uniform(-jitter_range, jitter_range)
            delay += jitter
        
        # Ensure delay is positive
        return max(0.1, delay)
    
    def get_retry_stats(self) -> dict:
        """Get retry handler statistics."""
        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "backoff_multiplier": self.backoff_multiplier,
            "jitter_enabled": self.jitter,
            "retryable_errors": self.retryable_errors
        }
    
    def update_config(
        self,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        backoff_multiplier: Optional[float] = None,
        jitter: Optional[bool] = None
    ):
        """
        Update retry handler configuration.
        
        Args:
            max_retries: New max retries
            base_delay: New base delay
            max_delay: New max delay
            backoff_multiplier: New backoff multiplier
            jitter: New jitter setting
        """
        if max_retries is not None:
            self.max_retries = max_retries
        if base_delay is not None:
            self.base_delay = base_delay
        if max_delay is not None:
            self.max_delay = max_delay
        if backoff_multiplier is not None:
            self.backoff_multiplier = backoff_multiplier
        if jitter is not None:
            self.jitter = jitter
        
        logger.info(f"RetryHandler configuration updated: {self.get_retry_stats()}")


class CircuitBreaker:
    """
    Circuit breaker for LLM requests.
    
    This circuit breaker prevents cascading failures by
    temporarily stopping requests when error rate is high.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Time to wait before trying again
            expected_exception: Exception type to track
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        logger.info(f"CircuitBreaker initialized with threshold: {failure_threshold}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moved to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker moved to CLOSED state")
            
            return result
            
        except self.expected_exception as e:
            self._record_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def _record_failure(self):
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def get_state(self) -> dict:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout
        }
