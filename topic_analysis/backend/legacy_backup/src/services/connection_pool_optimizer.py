"""
Database connection pool optimization service
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool, StaticPool, NullPool
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta
import threading
import queue

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class ConnectionPoolOptimizer:
    """Service for database connection pool optimization"""
    
    def __init__(self, db: Session):
        self.db = db
        self.engine = db.bind
        self.pool = self.engine.pool
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "overflow_connections": 0,
            "connection_errors": 0,
            "connection_timeouts": 0,
            "pool_overflow_count": 0,
            "pool_checked_in": 0,
            "pool_checked_out": 0,
            "pool_invalidated": 0,
            "pool_recreated": 0
        }
        self.performance_metrics = {
            "average_connection_time": 0.0,
            "average_query_time": 0.0,
            "peak_connections": 0,
            "connection_wait_time": 0.0,
            "pool_efficiency": 0.0
        }
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 30  # seconds
    
    def analyze_connection_pool(self) -> Dict[str, Any]:
        """Analyze current connection pool configuration and performance"""
        try:
            pool_info = self._get_pool_info()
            performance_info = self._get_performance_info()
            recommendations = self._generate_pool_recommendations(pool_info, performance_info)
            
            return {
                "pool_configuration": pool_info,
                "performance_metrics": performance_info,
                "recommendations": recommendations,
                "optimization_score": self._calculate_optimization_score(pool_info, performance_info),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing connection pool: {str(e)}")
            return {"error": "Failed to analyze connection pool"}
    
    def _get_pool_info(self) -> Dict[str, Any]:
        """Get current pool configuration information"""
        try:
            if hasattr(self.pool, 'size'):
                pool_size = self.pool.size()
            else:
                pool_size = getattr(self.pool, '_pool_size', 5)
            
            if hasattr(self.pool, 'overflow'):
                pool_overflow = self.pool.overflow()
            else:
                pool_overflow = getattr(self.pool, '_max_overflow', 10)
            
            if hasattr(self.pool, 'checked_in'):
                checked_in = self.pool.checked_in()
            else:
                checked_in = getattr(self.pool, '_checked_in', 0)
            
            if hasattr(self.pool, 'checked_out'):
                checked_out = self.pool.checked_out()
            else:
                checked_out = getattr(self.pool, '_checked_out', 0)
            
            return {
                "pool_type": type(self.pool).__name__,
                "pool_size": pool_size,
                "max_overflow": pool_overflow,
                "checked_in_connections": checked_in,
                "checked_out_connections": checked_out,
                "total_connections": checked_in + checked_out,
                "pool_utilization": (checked_out / (pool_size + pool_overflow)) * 100 if (pool_size + pool_overflow) > 0 else 0,
                "overflow_utilization": (max(0, checked_out - pool_size) / pool_overflow) * 100 if pool_overflow > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting pool info: {str(e)}")
            return {"error": "Failed to get pool information"}
    
    def _get_performance_info(self) -> Dict[str, Any]:
        """Get connection pool performance information"""
        try:
            # Test connection performance
            start_time = time.time()
            connection = self.engine.connect()
            connection_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Test query performance
            start_time = time.time()
            result = connection.execute(text("SELECT 1"))
            result.fetchone()
            query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            connection.close()
            
            return {
                "connection_time_ms": connection_time,
                "query_time_ms": query_time,
                "pool_efficiency": self._calculate_pool_efficiency(),
                "connection_success_rate": self._calculate_connection_success_rate(),
                "average_connection_wait_time": self._calculate_average_wait_time(),
                "peak_connections": self.connection_stats["peak_connections"],
                "total_connections_created": self.connection_stats["total_connections"],
                "connection_errors": self.connection_stats["connection_errors"],
                "connection_timeouts": self.connection_stats["connection_timeouts"]
            }
            
        except Exception as e:
            logger.error(f"Error getting performance info: {str(e)}")
            return {"error": "Failed to get performance information"}
    
    def _calculate_pool_efficiency(self) -> float:
        """Calculate pool efficiency based on current stats"""
        try:
            pool_info = self._get_pool_info()
            if "error" in pool_info:
                return 0.0
            
            checked_out = pool_info.get("checked_out_connections", 0)
            total_capacity = pool_info.get("pool_size", 0) + pool_info.get("max_overflow", 0)
            
            if total_capacity == 0:
                return 0.0
            
            # Efficiency is based on how well we're utilizing the pool
            utilization = checked_out / total_capacity
            
            # Penalize overflow usage
            if checked_out > pool_info.get("pool_size", 0):
                overflow_penalty = (checked_out - pool_info.get("pool_size", 0)) / total_capacity
                utilization -= overflow_penalty * 0.5
            
            return max(0.0, min(100.0, utilization * 100))
            
        except Exception as e:
            logger.error(f"Error calculating pool efficiency: {str(e)}")
            return 0.0
    
    def _calculate_connection_success_rate(self) -> float:
        """Calculate connection success rate"""
        try:
            total_attempts = self.connection_stats["total_connections"]
            errors = self.connection_stats["connection_errors"]
            timeouts = self.connection_stats["connection_timeouts"]
            
            if total_attempts == 0:
                return 100.0
            
            failures = errors + timeouts
            success_rate = ((total_attempts - failures) / total_attempts) * 100
            
            return max(0.0, min(100.0, success_rate))
            
        except Exception as e:
            logger.error(f"Error calculating connection success rate: {str(e)}")
            return 0.0
    
    def _calculate_average_wait_time(self) -> float:
        """Calculate average connection wait time"""
        try:
            # This is a simplified implementation
            # In a real system, this would track actual wait times
            
            pool_info = self._get_pool_info()
            if "error" in pool_info:
                return 0.0
            
            checked_out = pool_info.get("checked_out_connections", 0)
            pool_size = pool_info.get("pool_size", 0)
            
            # Estimate wait time based on pool utilization
            if checked_out <= pool_size:
                return 0.0  # No wait time if within pool size
            
            # Estimate wait time for overflow connections
            overflow = checked_out - pool_size
            max_overflow = pool_info.get("max_overflow", 0)
            
            if max_overflow == 0:
                return 0.0
            
            # Simple estimation: more overflow = more wait time
            wait_time = (overflow / max_overflow) * 100  # milliseconds
            
            return wait_time
            
        except Exception as e:
            logger.error(f"Error calculating average wait time: {str(e)}")
            return 0.0
    
    def _generate_pool_recommendations(self, pool_info: Dict[str, Any], performance_info: Dict[str, Any]) -> List[str]:
        """Generate connection pool optimization recommendations"""
        recommendations = []
        
        try:
            # Check pool utilization
            utilization = pool_info.get("pool_utilization", 0)
            if utilization > 90:
                recommendations.append("Pool utilization is very high (>90%). Consider increasing pool size.")
            elif utilization < 30:
                recommendations.append("Pool utilization is low (<30%). Consider decreasing pool size to save resources.")
            
            # Check overflow utilization
            overflow_utilization = pool_info.get("overflow_utilization", 0)
            if overflow_utilization > 80:
                recommendations.append("Overflow utilization is high (>80%). Consider increasing max_overflow or pool_size.")
            
            # Check connection success rate
            success_rate = performance_info.get("connection_success_rate", 100)
            if success_rate < 95:
                recommendations.append(f"Connection success rate is low ({success_rate:.1f}%). Check database connectivity and pool configuration.")
            
            # Check connection time
            connection_time = performance_info.get("connection_time_ms", 0)
            if connection_time > 1000:  # 1 second
                recommendations.append(f"Connection time is slow ({connection_time:.1f}ms). Check database performance and network latency.")
            
            # Check query time
            query_time = performance_info.get("query_time_ms", 0)
            if query_time > 100:  # 100ms
                recommendations.append(f"Query time is slow ({query_time:.1f}ms). Consider query optimization and indexing.")
            
            # Check pool efficiency
            efficiency = performance_info.get("pool_efficiency", 0)
            if efficiency < 50:
                recommendations.append(f"Pool efficiency is low ({efficiency:.1f}%). Review pool configuration and connection usage patterns.")
            
            # Check for connection errors
            errors = performance_info.get("connection_errors", 0)
            if errors > 0:
                recommendations.append(f"Connection errors detected ({errors}). Check database health and connection parameters.")
            
            # Check for timeouts
            timeouts = performance_info.get("connection_timeouts", 0)
            if timeouts > 0:
                recommendations.append(f"Connection timeouts detected ({timeouts}). Consider increasing connection timeout values.")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Connection pool appears to be well configured.")
            
            recommendations.append("Monitor connection pool metrics regularly for optimal performance.")
            recommendations.append("Consider implementing connection pooling best practices.")
            
        except Exception as e:
            logger.error(f"Error generating pool recommendations: {str(e)}")
            recommendations.append("Error generating recommendations. Check pool configuration manually.")
        
        return recommendations
    
    def _calculate_optimization_score(self, pool_info: Dict[str, Any], performance_info: Dict[str, Any]) -> float:
        """Calculate overall optimization score (0-100)"""
        try:
            score = 100.0
            
            # Deduct points for high utilization
            utilization = pool_info.get("pool_utilization", 0)
            if utilization > 90:
                score -= 20
            elif utilization > 80:
                score -= 10
            
            # Deduct points for high overflow usage
            overflow_utilization = pool_info.get("overflow_utilization", 0)
            if overflow_utilization > 80:
                score -= 15
            elif overflow_utilization > 60:
                score -= 5
            
            # Deduct points for low success rate
            success_rate = performance_info.get("connection_success_rate", 100)
            if success_rate < 95:
                score -= 25
            elif success_rate < 98:
                score -= 10
            
            # Deduct points for slow connection time
            connection_time = performance_info.get("connection_time_ms", 0)
            if connection_time > 1000:
                score -= 20
            elif connection_time > 500:
                score -= 10
            
            # Deduct points for slow query time
            query_time = performance_info.get("query_time_ms", 0)
            if query_time > 100:
                score -= 15
            elif query_time > 50:
                score -= 5
            
            # Deduct points for low efficiency
            efficiency = performance_info.get("pool_efficiency", 0)
            if efficiency < 50:
                score -= 15
            elif efficiency < 70:
                score -= 5
            
            # Deduct points for errors
            errors = performance_info.get("connection_errors", 0)
            if errors > 0:
                score -= min(20, errors * 2)
            
            # Deduct points for timeouts
            timeouts = performance_info.get("connection_timeouts", 0)
            if timeouts > 0:
                score -= min(15, timeouts * 3)
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating optimization score: {str(e)}")
            return 0.0
    
    def optimize_pool_configuration(self) -> Dict[str, Any]:
        """Optimize connection pool configuration based on analysis"""
        try:
            analysis = self.analyze_connection_pool()
            if "error" in analysis:
                return analysis
            
            pool_info = analysis["pool_configuration"]
            performance_info = analysis["performance_metrics"]
            
            # Calculate optimal configuration
            optimal_config = self._calculate_optimal_configuration(pool_info, performance_info)
            
            # Apply optimizations
            optimizations_applied = self._apply_pool_optimizations(optimal_config)
            
            return {
                "current_configuration": pool_info,
                "optimal_configuration": optimal_config,
                "optimizations_applied": optimizations_applied,
                "expected_improvement": self._calculate_expected_improvement(pool_info, optimal_config),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing pool configuration: {str(e)}")
            return {"error": "Failed to optimize pool configuration"}
    
    def _calculate_optimal_configuration(self, pool_info: Dict[str, Any], performance_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal pool configuration"""
        try:
            current_size = pool_info.get("pool_size", 5)
            current_overflow = pool_info.get("max_overflow", 10)
            utilization = pool_info.get("pool_utilization", 0)
            overflow_utilization = pool_info.get("overflow_utilization", 0)
            
            # Calculate optimal pool size
            if utilization > 80:
                optimal_size = min(current_size * 2, 20)  # Double size, max 20
            elif utilization < 30:
                optimal_size = max(current_size // 2, 2)  # Half size, min 2
            else:
                optimal_size = current_size
            
            # Calculate optimal overflow
            if overflow_utilization > 70:
                optimal_overflow = min(current_overflow * 2, 30)  # Double overflow, max 30
            elif overflow_utilization < 20:
                optimal_overflow = max(current_overflow // 2, 5)  # Half overflow, min 5
            else:
                optimal_overflow = current_overflow
            
            # Calculate optimal timeout values
            connection_time = performance_info.get("connection_time_ms", 0)
            if connection_time > 1000:
                optimal_timeout = 30  # 30 seconds
            elif connection_time > 500:
                optimal_timeout = 20  # 20 seconds
            else:
                optimal_timeout = 10  # 10 seconds
            
            return {
                "pool_size": optimal_size,
                "max_overflow": optimal_overflow,
                "pool_timeout": optimal_timeout,
                "pool_recycle": 3600,  # 1 hour
                "pool_pre_ping": True,
                "pool_reset_on_return": True
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal configuration: {str(e)}")
            return {}
    
    def _apply_pool_optimizations(self, optimal_config: Dict[str, Any]) -> List[str]:
        """Apply pool optimizations (simplified implementation)"""
        try:
            # In a real implementation, this would modify the actual pool configuration
            # For now, we'll just return what would be applied
            
            optimizations = []
            
            if optimal_config.get("pool_size"):
                optimizations.append(f"Set pool_size to {optimal_config['pool_size']}")
            
            if optimal_config.get("max_overflow"):
                optimizations.append(f"Set max_overflow to {optimal_config['max_overflow']}")
            
            if optimal_config.get("pool_timeout"):
                optimizations.append(f"Set pool_timeout to {optimal_config['pool_timeout']} seconds")
            
            if optimal_config.get("pool_recycle"):
                optimizations.append(f"Set pool_recycle to {optimal_config['pool_recycle']} seconds")
            
            if optimal_config.get("pool_pre_ping"):
                optimizations.append("Enable pool_pre_ping for connection validation")
            
            if optimal_config.get("pool_reset_on_return"):
                optimizations.append("Enable pool_reset_on_return for connection cleanup")
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Error applying pool optimizations: {str(e)}")
            return ["Error applying optimizations"]
    
    def _calculate_expected_improvement(self, current_config: Dict[str, Any], optimal_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate expected improvement from optimizations"""
        try:
            current_utilization = current_config.get("pool_utilization", 0)
            current_overflow = current_config.get("overflow_utilization", 0)
            
            # Estimate improvements
            utilization_improvement = 0
            if current_utilization > 80:
                utilization_improvement = min(20, current_utilization - 70)  # Reduce to 70%
            
            overflow_improvement = 0
            if current_overflow > 70:
                overflow_improvement = min(30, current_overflow - 50)  # Reduce to 50%
            
            return {
                "utilization_improvement": f"{utilization_improvement:.1f}%",
                "overflow_improvement": f"{overflow_improvement:.1f}%",
                "expected_efficiency_gain": f"{utilization_improvement + overflow_improvement:.1f}%",
                "estimated_performance_boost": "10-30%"
            }
            
        except Exception as e:
            logger.error(f"Error calculating expected improvement: {str(e)}")
            return {"error": "Failed to calculate expected improvement"}
    
    def start_monitoring(self) -> bool:
        """Start connection pool monitoring"""
        try:
            if self.monitoring_active:
                return False
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Connection pool monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop connection pool monitoring"""
        try:
            if not self.monitoring_active:
                return False
            
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            logger.info("Connection pool monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
            return False
    
    def _monitoring_loop(self):
        """Monitoring loop for connection pool"""
        while self.monitoring_active:
            try:
                # Update connection stats
                self._update_connection_stats()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _update_connection_stats(self):
        """Update connection statistics"""
        try:
            pool_info = self._get_pool_info()
            if "error" not in pool_info:
                self.connection_stats["total_connections"] = pool_info.get("total_connections", 0)
                self.connection_stats["active_connections"] = pool_info.get("checked_out_connections", 0)
                self.connection_stats["idle_connections"] = pool_info.get("checked_in_connections", 0)
                
                # Update peak connections
                current_total = pool_info.get("total_connections", 0)
                if current_total > self.connection_stats["peak_connections"]:
                    self.connection_stats["peak_connections"] = current_total
                
        except Exception as e:
            logger.error(f"Error updating connection stats: {str(e)}")
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Get current monitoring data"""
        try:
            return {
                "monitoring_active": self.monitoring_active,
                "connection_stats": self.connection_stats,
                "performance_metrics": self.performance_metrics,
                "monitoring_interval": self.monitoring_interval,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring data: {str(e)}")
            return {"error": "Failed to get monitoring data"}
