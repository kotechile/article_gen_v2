"""
Performance metrics tracking for enhanced topics functionality.
"""
import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    operation: str
    duration: float
    timestamp: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OperationStats:
    """Statistics for a specific operation"""
    operation: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_duration: float = 0.0
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=100))

class PerformanceTracker:
    """Performance metrics tracker for enhanced topics"""
    
    def __init__(self, max_metrics: int = 1000):
        """
        Initialize performance tracker.
        
        Args:
            max_metrics: Maximum number of metrics to store in memory
        """
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.operation_stats: Dict[str, OperationStats] = defaultdict(
            lambda: OperationStats(operation="")
        )
        self.lock = asyncio.Lock()
    
    async def record_metric(
        self,
        operation: str,
        duration: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a performance metric.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation was successful
            metadata: Additional metadata
        """
        async with self.lock:
            metric = PerformanceMetric(
                operation=operation,
                duration=duration,
                timestamp=time.time(),
                success=success,
                metadata=metadata or {}
            )
            
            self.metrics.append(metric)
            
            # Update operation stats
            stats = self.operation_stats[operation]
            if not stats.operation:
                stats.operation = operation
            
            stats.total_requests += 1
            if success:
                stats.successful_requests += 1
            else:
                stats.failed_requests += 1
            
            stats.total_duration += duration
            stats.min_duration = min(stats.min_duration, duration)
            stats.max_duration = max(stats.max_duration, duration)
            stats.avg_duration = stats.total_duration / stats.total_requests
            stats.recent_durations.append(duration)
    
    async def get_operation_stats(self, operation: str) -> Optional[OperationStats]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Operation statistics or None if not found
        """
        async with self.lock:
            return self.operation_stats.get(operation)
    
    async def get_all_stats(self) -> Dict[str, OperationStats]:
        """
        Get statistics for all operations.
        
        Returns:
            Dictionary of operation statistics
        """
        async with self.lock:
            return dict(self.operation_stats)
    
    async def get_recent_metrics(self, limit: int = 100) -> List[PerformanceMetric]:
        """
        Get recent performance metrics.
        
        Args:
            limit: Maximum number of metrics to return
            
        Returns:
            List of recent metrics
        """
        async with self.lock:
            return list(self.metrics)[-limit:]
    
    async def get_health_score(self) -> float:
        """
        Calculate overall health score based on success rates and performance.
        
        Returns:
            Health score between 0 and 1 (1 = perfect health)
        """
        async with self.lock:
            if not self.operation_stats:
                return 1.0
            
            total_requests = sum(stats.total_requests for stats in self.operation_stats.values())
            total_successful = sum(stats.successful_requests for stats in self.operation_stats.values())
            
            if total_requests == 0:
                return 1.0
            
            success_rate = total_successful / total_requests
            
            # Calculate performance score based on average response times
            performance_scores = []
            for stats in self.operation_stats.values():
                if stats.avg_duration > 0:
                    # Normalize performance (lower is better, max 10 seconds)
                    perf_score = max(0, 1 - (stats.avg_duration / 10.0))
                    performance_scores.append(perf_score)
            
            avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 1.0
            
            # Combine success rate and performance
            return (success_rate * 0.7 + avg_performance * 0.3)
    
    async def clear_metrics(self) -> None:
        """Clear all metrics and statistics."""
        async with self.lock:
            self.metrics.clear()
            self.operation_stats.clear()

# Global performance tracker instance
performance_tracker = PerformanceTracker()

class PerformanceMonitor:
    """Context manager for monitoring operation performance"""
    
    def __init__(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize performance monitor.
        
        Args:
            operation: Operation name
            metadata: Additional metadata
        """
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None
        self.success = False
    
    async def __aenter__(self):
        """Start monitoring."""
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End monitoring and record metric."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            success = exc_type is None
            
            await performance_tracker.record_metric(
                operation=self.operation,
                duration=duration,
                success=success,
                metadata=self.metadata
            )
    
    def mark_success(self):
        """Mark operation as successful."""
        self.success = True

def monitor_performance(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for monitoring function performance.
    
    Args:
        operation: Operation name
        metadata: Additional metadata
        
    Returns:
        Decorated function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with PerformanceMonitor(operation, metadata):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# Convenience functions for common operations
async def record_autocomplete_performance(duration: float, success: bool, query: str):
    """Record autocomplete performance."""
    await performance_tracker.record_metric(
        operation="autocomplete",
        duration=duration,
        success=success,
        metadata={"query": query}
    )

async def record_decomposition_performance(duration: float, success: bool, query: str, method: str):
    """Record topic decomposition performance."""
    await performance_tracker.record_metric(
        operation="decomposition",
        duration=duration,
        success=success,
        metadata={"query": query, "method": method}
    )

async def record_llm_performance(duration: float, success: bool, query: str, provider: str):
    """Record LLM performance."""
    await performance_tracker.record_metric(
        operation="llm",
        duration=duration,
        success=success,
        metadata={"query": query, "provider": provider}
    )

