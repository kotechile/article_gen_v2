"""
API response time monitoring service
"""
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import threading
import json

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class APIMonitoringService:
    """Service for API response time monitoring"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_thread = None
        self.response_times = defaultdict(lambda: deque(maxlen=1000))  # Keep last 1000 requests per endpoint
        self.endpoint_stats = defaultdict(lambda: {
            "total_requests": 0,
            "total_response_time": 0.0,
            "min_response_time": float('inf'),
            "max_response_time": 0.0,
            "error_count": 0,
            "success_count": 0,
            "last_request": None
        })
        self.global_stats = {
            "total_requests": 0,
            "total_response_time": 0.0,
            "min_response_time": float('inf'),
            "max_response_time": 0.0,
            "error_count": 0,
            "success_count": 0,
            "average_response_time": 0.0,
            "p95_response_time": 0.0,
            "p99_response_time": 0.0
        }
        self.alert_thresholds = {
            "response_time_ms": settings.API_RESPONSE_TIME_THRESHOLD,
            "error_rate_percent": settings.API_ERROR_RATE_THRESHOLD,
            "requests_per_second": settings.API_RPS_THRESHOLD
        }
        self.alerts = []
        self.monitoring_interval = settings.API_MONITORING_INTERVAL
    
    def start_monitoring(self) -> bool:
        """Start API monitoring"""
        try:
            if self.monitoring_active:
                return False
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("API monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting API monitoring: {str(e)}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop API monitoring"""
        try:
            if not self.monitoring_active:
                return False
            
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            logger.info("API monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping API monitoring: {str(e)}")
            return False
    
    def record_request(
        self, 
        endpoint: str, 
        method: str, 
        response_time_ms: float, 
        status_code: int,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None
    ):
        """Record API request metrics"""
        try:
            full_endpoint = f"{method} {endpoint}"
            current_time = datetime.utcnow()
            
            # Record response time
            self.response_times[full_endpoint].append(response_time_ms)
            
            # Update endpoint stats
            stats = self.endpoint_stats[full_endpoint]
            stats["total_requests"] += 1
            stats["total_response_time"] += response_time_ms
            stats["min_response_time"] = min(stats["min_response_time"], response_time_ms)
            stats["max_response_time"] = max(stats["max_response_time"], response_time_ms)
            stats["last_request"] = current_time.isoformat()
            
            if 200 <= status_code < 400:
                stats["success_count"] += 1
            else:
                stats["error_count"] += 1
            
            # Update global stats
            self.global_stats["total_requests"] += 1
            self.global_stats["total_response_time"] += response_time_ms
            self.global_stats["min_response_time"] = min(self.global_stats["min_response_time"], response_time_ms)
            self.global_stats["max_response_time"] = max(self.global_stats["max_response_time"], response_time_ms)
            
            if 200 <= status_code < 400:
                self.global_stats["success_count"] += 1
            else:
                self.global_stats["error_count"] += 1
            
            # Update average response time
            self.global_stats["average_response_time"] = (
                self.global_stats["total_response_time"] / self.global_stats["total_requests"]
            )
            
            # Check for alerts
            self._check_alerts(full_endpoint, response_time_ms, status_code)
            
        except Exception as e:
            logger.error(f"Error recording request metrics: {str(e)}")
    
    def get_endpoint_stats(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for specific endpoint or all endpoints"""
        try:
            if endpoint:
                # Get stats for specific endpoint
                if endpoint not in self.endpoint_stats:
                    return {"error": "Endpoint not found"}
                
                stats = self.endpoint_stats[endpoint]
                response_times = list(self.response_times[endpoint])
                
                return {
                    "endpoint": endpoint,
                    "total_requests": stats["total_requests"],
                    "average_response_time": stats["total_response_time"] / stats["total_requests"] if stats["total_requests"] > 0 else 0,
                    "min_response_time": stats["min_response_time"] if stats["min_response_time"] != float('inf') else 0,
                    "max_response_time": stats["max_response_time"],
                    "success_count": stats["success_count"],
                    "error_count": stats["error_count"],
                    "success_rate": (stats["success_count"] / stats["total_requests"] * 100) if stats["total_requests"] > 0 else 0,
                    "p95_response_time": self._calculate_percentile(response_times, 95),
                    "p99_response_time": self._calculate_percentile(response_times, 99),
                    "last_request": stats["last_request"]
                }
            else:
                # Get stats for all endpoints
                all_endpoints = {}
                for ep, stats in self.endpoint_stats.items():
                    response_times = list(self.response_times[ep])
                    all_endpoints[ep] = {
                        "total_requests": stats["total_requests"],
                        "average_response_time": stats["total_response_time"] / stats["total_requests"] if stats["total_requests"] > 0 else 0,
                        "min_response_time": stats["min_response_time"] if stats["min_response_time"] != float('inf') else 0,
                        "max_response_time": stats["max_response_time"],
                        "success_count": stats["success_count"],
                        "error_count": stats["error_count"],
                        "success_rate": (stats["success_count"] / stats["total_requests"] * 100) if stats["total_requests"] > 0 else 0,
                        "p95_response_time": self._calculate_percentile(response_times, 95),
                        "p99_response_time": self._calculate_percentile(response_times, 99),
                        "last_request": stats["last_request"]
                    }
                
                return all_endpoints
                
        except Exception as e:
            logger.error(f"Error getting endpoint stats: {str(e)}")
            return {"error": str(e)}
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global API statistics"""
        try:
            # Calculate percentiles from all response times
            all_response_times = []
            for times in self.response_times.values():
                all_response_times.extend(times)
            
            self.global_stats["p95_response_time"] = self._calculate_percentile(all_response_times, 95)
            self.global_stats["p99_response_time"] = self._calculate_percentile(all_response_times, 99)
            
            return {
                "global_stats": self.global_stats,
                "monitoring_active": self.monitoring_active,
                "total_endpoints": len(self.endpoint_stats),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting global stats: {str(e)}")
            return {"error": str(e)}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            endpoint_stats = self.get_endpoint_stats()
            global_stats = self.get_global_stats()
            
            # Identify slow endpoints
            slow_endpoints = []
            for endpoint, stats in endpoint_stats.items():
                if isinstance(stats, dict) and "average_response_time" in stats:
                    if stats["average_response_time"] > self.alert_thresholds["response_time_ms"]:
                        slow_endpoints.append({
                            "endpoint": endpoint,
                            "average_response_time": stats["average_response_time"],
                            "p95_response_time": stats.get("p95_response_time", 0),
                            "p99_response_time": stats.get("p99_response_time", 0)
                        })
            
            # Identify high error rate endpoints
            high_error_endpoints = []
            for endpoint, stats in endpoint_stats.items():
                if isinstance(stats, dict) and "success_rate" in stats:
                    if stats["success_rate"] < (100 - self.alert_thresholds["error_rate_percent"]):
                        high_error_endpoints.append({
                            "endpoint": endpoint,
                            "success_rate": stats["success_rate"],
                            "error_count": stats.get("error_count", 0),
                            "total_requests": stats.get("total_requests", 0)
                        })
            
            # Generate recommendations
            recommendations = self._generate_recommendations(slow_endpoints, high_error_endpoints)
            
            return {
                "summary": {
                    "total_requests": self.global_stats["total_requests"],
                    "average_response_time": self.global_stats["average_response_time"],
                    "p95_response_time": self.global_stats["p95_response_time"],
                    "p99_response_time": self.global_stats["p99_response_time"],
                    "success_rate": (self.global_stats["success_count"] / self.global_stats["total_requests"] * 100) if self.global_stats["total_requests"] > 0 else 0,
                    "total_endpoints": len(self.endpoint_stats)
                },
                "slow_endpoints": slow_endpoints,
                "high_error_endpoints": high_error_endpoints,
                "recommendations": recommendations,
                "alerts": self.alerts[-10:],  # Last 10 alerts
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        try:
            if not values:
                return 0.0
            
            sorted_values = sorted(values)
            index = int((percentile / 100) * len(sorted_values))
            if index >= len(sorted_values):
                index = len(sorted_values) - 1
            
            return sorted_values[index]
            
        except Exception:
            return 0.0
    
    def _check_alerts(self, endpoint: str, response_time_ms: float, status_code: int):
        """Check for alert conditions"""
        try:
            current_time = datetime.utcnow()
            
            # Check response time alert
            if response_time_ms > self.alert_thresholds["response_time_ms"]:
                alert = {
                    "type": "high_response_time",
                    "endpoint": endpoint,
                    "value": response_time_ms,
                    "threshold": self.alert_thresholds["response_time_ms"],
                    "timestamp": current_time.isoformat(),
                    "severity": "warning" if response_time_ms < self.alert_thresholds["response_time_ms"] * 2 else "critical"
                }
                self.alerts.append(alert)
                logger.warning(f"High response time alert: {endpoint} - {response_time_ms}ms")
            
            # Check error rate alert
            if status_code >= 400:
                stats = self.endpoint_stats[endpoint]
                error_rate = (stats["error_count"] / stats["total_requests"] * 100) if stats["total_requests"] > 0 else 0
                
                if error_rate > self.alert_thresholds["error_rate_percent"]:
                    alert = {
                        "type": "high_error_rate",
                        "endpoint": endpoint,
                        "value": error_rate,
                        "threshold": self.alert_thresholds["error_rate_percent"],
                        "timestamp": current_time.isoformat(),
                        "severity": "warning" if error_rate < self.alert_thresholds["error_rate_percent"] * 2 else "critical"
                    }
                    self.alerts.append(alert)
                    logger.warning(f"High error rate alert: {endpoint} - {error_rate}%")
            
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
    
    def _generate_recommendations(self, slow_endpoints: List[Dict], high_error_endpoints: List[Dict]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        try:
            # Recommendations for slow endpoints
            if slow_endpoints:
                recommendations.append(f"Optimize {len(slow_endpoints)} slow endpoints for better performance")
                for endpoint in slow_endpoints[:3]:  # Top 3 slowest
                    recommendations.append(f"Consider caching for {endpoint['endpoint']} (avg: {endpoint['average_response_time']:.1f}ms)")
            
            # Recommendations for high error endpoints
            if high_error_endpoints:
                recommendations.append(f"Investigate {len(high_error_endpoints)} endpoints with high error rates")
                for endpoint in high_error_endpoints[:3]:  # Top 3 highest error rate
                    recommendations.append(f"Review error handling for {endpoint['endpoint']} (success rate: {endpoint['success_rate']:.1f}%)")
            
            # General recommendations
            if self.global_stats["average_response_time"] > 500:  # 500ms
                recommendations.append("Consider implementing database query optimization")
            
            if self.global_stats["p95_response_time"] > 1000:  # 1 second
                recommendations.append("Consider implementing response caching")
            
            if len(self.endpoint_stats) > 50:
                recommendations.append("Consider implementing API rate limiting")
            
            if not recommendations:
                recommendations.append("API performance is within acceptable limits")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("Error generating recommendations")
        
        return recommendations
    
    def _monitoring_loop(self):
        """Monitoring loop for API metrics"""
        while self.monitoring_active:
            try:
                # Clean up old alerts (keep last 100)
                if len(self.alerts) > 100:
                    self.alerts = self.alerts[-100:]
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def clear_stats(self):
        """Clear all monitoring statistics"""
        try:
            self.response_times.clear()
            self.endpoint_stats.clear()
            self.global_stats = {
                "total_requests": 0,
                "total_response_time": 0.0,
                "min_response_time": float('inf'),
                "max_response_time": 0.0,
                "error_count": 0,
                "success_count": 0,
                "average_response_time": 0.0,
                "p95_response_time": 0.0,
                "p99_response_time": 0.0
            }
            self.alerts.clear()
            
            logger.info("API monitoring statistics cleared")
            
        except Exception as e:
            logger.error(f"Error clearing stats: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on API monitoring"""
        try:
            start_time = time.time()
            
            # Test monitoring functionality
            test_endpoint = "GET /health"
            test_response_time = 10.0  # 10ms
            test_status_code = 200
            
            self.record_request(test_endpoint, "GET", test_response_time, test_status_code)
            
            # Verify recording worked
            stats = self.get_endpoint_stats(test_endpoint)
            if "error" in stats:
                return {
                    "healthy": False,
                    "error": "Failed to record test request",
                    "response_time_ms": 0
                }
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "healthy": True,
                "response_time_ms": response_time,
                "monitoring_active": self.monitoring_active,
                "total_endpoints": len(self.endpoint_stats),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"API monitoring health check failed: {str(e)}")
            return {
                "healthy": False,
                "error": str(e),
                "response_time_ms": 0
            }

# Global API monitoring instance
_api_monitoring_service = None

def get_api_monitoring_service() -> APIMonitoringService:
    """Get global API monitoring service instance"""
    global _api_monitoring_service
    if _api_monitoring_service is None:
        _api_monitoring_service = APIMonitoringService()
    return _api_monitoring_service

def monitor_api_request(endpoint: str, method: str, response_time_ms: float, status_code: int, user_id: Optional[int] = None, ip_address: Optional[str] = None):
    """Monitor API request (convenience function)"""
    monitoring_service = get_api_monitoring_service()
    monitoring_service.record_request(endpoint, method, response_time_ms, status_code, user_id, ip_address)

def api_monitoring_middleware(func):
    """Decorator for API monitoring"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            response_time = (time.time() - start_time) * 1000
            
            # Extract endpoint and method from function
            endpoint = getattr(func, '__name__', 'unknown')
            method = 'GET'  # Default, could be extracted from request
            
            # Monitor the request
            monitor_api_request(endpoint, method, response_time, 200)
            
            return result
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            # Extract endpoint and method from function
            endpoint = getattr(func, '__name__', 'unknown')
            method = 'GET'  # Default, could be extracted from request
            
            # Monitor the request with error status
            monitor_api_request(endpoint, method, response_time, 500)
            
            raise
    
    return wrapper
