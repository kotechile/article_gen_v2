"""
Performance optimization utilities for DataForSEO features

Provides performance monitoring, optimization, and tuning
capabilities for the DataForSEO integration.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import statistics
from dataclasses import dataclass

from .cache_integration import cache_manager
from .database import dataforseo_repository

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    operation: str
    duration_ms: float
    success: bool
    timestamp: datetime
    metadata: Dict[str, Any] = None

class PerformanceMonitor:
    """Monitors and tracks performance metrics"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.max_metrics = 10000  # Keep last 10k metrics
        self.performance_thresholds = {
            'api_call': 5000,      # 5 seconds
            'cache_operation': 100, # 100ms
            'database_operation': 1000, # 1 second
            'data_processing': 500   # 500ms
        }
    
    def record_metric(self, operation: str, duration_ms: float, success: bool, metadata: Dict[str, Any] = None):
        """Record a performance metric"""
        metric = PerformanceMetrics(
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Keep only recent metrics
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
        
        # Log slow operations
        threshold = self.performance_thresholds.get(operation, 1000)
        if duration_ms > threshold:
            logger.warning(f"Slow operation detected: {operation} took {duration_ms:.2f}ms")
    
    def get_metrics_summary(self, operation: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics summary"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter metrics
        filtered_metrics = [
            m for m in self.metrics 
            if m.timestamp >= cutoff_time and (operation is None or m.operation == operation)
        ]
        
        if not filtered_metrics:
            return {"error": "No metrics found"}
        
        # Calculate statistics
        durations = [m.duration_ms for m in filtered_metrics]
        success_count = sum(1 for m in filtered_metrics if m.success)
        total_count = len(filtered_metrics)
        
        return {
            "operation": operation or "all",
            "total_operations": total_count,
            "success_count": success_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "avg_duration_ms": statistics.mean(durations),
            "median_duration_ms": statistics.median(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "p95_duration_ms": self._percentile(durations, 95),
            "p99_duration_ms": self._percentile(durations, 99),
            "time_range_hours": hours
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_slow_operations(self, threshold_ms: float = 1000) -> List[PerformanceMetrics]:
        """Get operations slower than threshold"""
        return [m for m in self.metrics if m.duration_ms > threshold_ms]
    
    def get_failed_operations(self) -> List[PerformanceMetrics]:
        """Get failed operations"""
        return [m for m in self.metrics if not m.success]

class PerformanceOptimizer:
    """Optimizes performance based on metrics and patterns"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.optimization_rules = {
            'cache_miss_rate': 0.3,  # If cache miss rate > 30%, increase TTL
            'api_error_rate': 0.1,   # If API error rate > 10%, implement backoff
            'slow_operation_rate': 0.2  # If slow operation rate > 20%, optimize
        }
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and suggest optimizations"""
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "recommendations": [],
            "metrics": {}
        }
        
        # Analyze cache performance
        cache_metrics = self.monitor.get_metrics_summary("cache_operation")
        if cache_metrics.get("success_rate", 1) < 0.9:
            analysis["recommendations"].append({
                "type": "cache",
                "priority": "high",
                "message": "Cache operations failing frequently",
                "suggestion": "Check Redis connection and configuration"
            })
        
        # Analyze API performance
        api_metrics = self.monitor.get_metrics_summary("api_call")
        if api_metrics.get("avg_duration_ms", 0) > 5000:
            analysis["recommendations"].append({
                "type": "api",
                "priority": "high",
                "message": "API calls taking too long",
                "suggestion": "Implement request timeout and retry logic"
            })
        
        # Analyze database performance
        db_metrics = self.monitor.get_metrics_summary("database_operation")
        if db_metrics.get("avg_duration_ms", 0) > 1000:
            analysis["recommendations"].append({
                "type": "database",
                "priority": "medium",
                "message": "Database operations slow",
                "suggestion": "Check indexes and query optimization"
            })
        
        analysis["metrics"] = {
            "cache": cache_metrics,
            "api": api_metrics,
            "database": db_metrics
        }
        
        return analysis
    
    async def optimize_cache_ttl(self) -> Dict[str, Any]:
        """Optimize cache TTL based on usage patterns"""
        try:
            # Get cache statistics
            cache_stats = await cache_manager.get_cache_stats()
            
            # Analyze cache hit/miss patterns
            cache_metrics = self.monitor.get_metrics_summary("cache_operation")
            
            # Calculate optimal TTL based on data freshness requirements
            trend_data_ttl = 86400  # 24 hours
            keyword_data_ttl = 21600  # 6 hours
            suggestions_ttl = 3600  # 1 hour
            
            # Adjust TTL based on performance
            if cache_metrics.get("success_rate", 1) < 0.8:
                # Reduce TTL if cache is failing
                trend_data_ttl = int(trend_data_ttl * 0.5)
                keyword_data_ttl = int(keyword_data_ttl * 0.5)
                suggestions_ttl = int(suggestions_ttl * 0.5)
            
            return {
                "status": "optimized",
                "recommendations": {
                    "trend_data_ttl": trend_data_ttl,
                    "keyword_data_ttl": keyword_data_ttl,
                    "suggestions_ttl": suggestions_ttl
                },
                "current_stats": cache_stats
            }
            
        except Exception as e:
            logger.error(f"Error optimizing cache TTL: {e}")
            return {"status": "error", "error": str(e)}
    
    async def optimize_database_queries(self) -> Dict[str, Any]:
        """Optimize database queries based on performance metrics"""
        try:
            # Get database performance metrics
            db_metrics = self.monitor.get_metrics_summary("database_operation")
            
            recommendations = []
            
            # Check for slow queries
            if db_metrics.get("p95_duration_ms", 0) > 2000:
                recommendations.append({
                    "type": "index_optimization",
                    "message": "Consider adding indexes for frequently queried columns",
                    "priority": "high"
                })
            
            # Check for high error rate
            if db_metrics.get("success_rate", 1) < 0.95:
                recommendations.append({
                    "type": "connection_pool",
                    "message": "Consider increasing database connection pool size",
                    "priority": "medium"
                })
            
            return {
                "status": "analyzed",
                "recommendations": recommendations,
                "metrics": db_metrics
            }
            
        except Exception as e:
            logger.error(f"Error optimizing database queries: {e}")
            return {"status": "error", "error": str(e)}
    
    async def optimize_api_requests(self) -> Dict[str, Any]:
        """Optimize API requests based on performance patterns"""
        try:
            # Get API performance metrics
            api_metrics = self.monitor.get_metrics_summary("api_call")
            
            recommendations = []
            
            # Check for high latency
            if api_metrics.get("avg_duration_ms", 0) > 3000:
                recommendations.append({
                    "type": "timeout_optimization",
                    "message": "Consider reducing API timeout or implementing parallel requests",
                    "priority": "high"
                })
            
            # Check for high error rate
            if api_metrics.get("success_rate", 1) < 0.9:
                recommendations.append({
                    "type": "retry_logic",
                    "message": "Implement exponential backoff for failed requests",
                    "priority": "high"
                })
            
            # Check for rate limiting
            if api_metrics.get("total_operations", 0) > 1000:
                recommendations.append({
                    "type": "rate_limiting",
                    "message": "Implement request rate limiting to avoid API quotas",
                    "priority": "medium"
                })
            
            return {
                "status": "analyzed",
                "recommendations": recommendations,
                "metrics": api_metrics
            }
            
        except Exception as e:
            logger.error(f"Error optimizing API requests: {e}")
            return {"status": "error", "error": str(e)}

class PerformanceProfiler:
    """Profiles performance of specific operations"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
    
    def profile_operation(self, operation_name: str):
        """Decorator to profile operation performance"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                metadata = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    metadata["error"] = str(e)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    self.monitor.record_metric(operation_name, duration_ms, success, metadata)
            
            return wrapper
        return decorator
    
    async def profile_batch_operations(self, operations: List[Tuple[str, callable, tuple, dict]]) -> Dict[str, Any]:
        """Profile a batch of operations"""
        results = []
        total_start_time = time.time()
        
        for operation_name, func, args, kwargs in operations:
            start_time = time.time()
            success = True
            metadata = {
                "function": func.__name__,
                "batch_operation": True
            }
            
            try:
                result = await func(*args, **kwargs)
                results.append({
                    "operation": operation_name,
                    "result": result,
                    "success": True
                })
            except Exception as e:
                success = False
                metadata["error"] = str(e)
                results.append({
                    "operation": operation_name,
                    "error": str(e),
                    "success": False
                })
            finally:
                duration_ms = (time.time() - start_time) * 1000
                self.monitor.record_metric(operation_name, duration_ms, success, metadata)
        
        total_duration_ms = (time.time() - total_start_time) * 1000
        
        return {
            "total_duration_ms": total_duration_ms,
            "operation_count": len(operations),
            "avg_duration_ms": total_duration_ms / len(operations) if operations else 0,
            "results": results
        }

# Global instances
performance_monitor = PerformanceMonitor()
performance_optimizer = PerformanceOptimizer()
performance_profiler = PerformanceProfiler()
