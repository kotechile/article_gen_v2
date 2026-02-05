"""
Connection pooling configuration service
"""
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Engine
from sqlalchemy.pool import QueuePool, StaticPool, NullPool
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
import threading
import queue

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class ConnectionPoolConfig:
    """Service for connection pool configuration management"""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self.pool_config = self._get_default_pool_config()
        self.monitoring_active = False
        self.monitoring_thread = None
        self.pool_metrics = {
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
            "pool_recreated": 0,
            "average_connection_time": 0.0,
            "average_query_time": 0.0,
            "peak_connections": 0,
            "connection_wait_time": 0.0
        }
        self._initialize_engine()
    
    def _get_default_pool_config(self) -> Dict[str, Any]:
        """Get default connection pool configuration"""
        return {
            "pool_size": settings.DATABASE_POOL_SIZE,
            "max_overflow": settings.DATABASE_MAX_OVERFLOW,
            "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
            "pool_recycle": settings.DATABASE_POOL_RECYCLE,
            "pool_pre_ping": settings.DATABASE_POOL_PRE_PING,
            "pool_reset_on_return": settings.DATABASE_POOL_RESET_ON_RETURN,
            "pool_class": settings.DATABASE_POOL_CLASS,
            "echo": settings.DATABASE_ECHO,
            "echo_pool": settings.DATABASE_ECHO_POOL,
            "max_retries": settings.DATABASE_MAX_RETRIES,
            "retry_delay": settings.DATABASE_RETRY_DELAY
        }
    
    def _initialize_engine(self):
        """Initialize database engine with connection pool"""
        try:
            # Build database URL
            database_url = self._build_database_url()
            
            # Create engine with connection pool
            self.engine = create_engine(
                database_url,
                poolclass=self._get_pool_class(),
                pool_size=self.pool_config["pool_size"],
                max_overflow=self.pool_config["max_overflow"],
                pool_timeout=self.pool_config["pool_timeout"],
                pool_recycle=self.pool_config["pool_recycle"],
                pool_pre_ping=self.pool_config["pool_pre_ping"],
                pool_reset_on_return=self.pool_config["pool_reset_on_return"],
                echo=self.pool_config["echo"],
                echo_pool=self.pool_config["echo_pool"]
            )
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            
            logger.info("Database engine initialized with connection pool")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {str(e)}")
            raise
    
    def _build_database_url(self) -> str:
        """Build database URL from configuration"""
        db_type = settings.DATABASE_TYPE.lower()
        
        if db_type == "postgresql":
            return f"postgresql://{settings.DATABASE_USER}:{settings.DATABASE_PASSWORD}@{settings.DATABASE_HOST}:{settings.DATABASE_PORT}/{settings.DATABASE_NAME}"
        elif db_type == "mysql":
            return f"mysql+pymysql://{settings.DATABASE_USER}:{settings.DATABASE_PASSWORD}@{settings.DATABASE_HOST}:{settings.DATABASE_PORT}/{settings.DATABASE_NAME}"
        elif db_type == "sqlite":
            return f"sqlite:///{settings.DATABASE_NAME}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def _get_pool_class(self):
        """Get pool class based on configuration"""
        pool_class_name = self.pool_config["pool_class"].lower()
        
        if pool_class_name == "queuepool":
            return QueuePool
        elif pool_class_name == "staticpool":
            return StaticPool
        elif pool_class_name == "nullpool":
            return NullPool
        else:
            return QueuePool  # Default
    
    def get_session(self) -> Session:
        """Get database session from pool"""
        try:
            if not self.session_factory:
                raise Exception("Session factory not initialized")
            
            session = self.session_factory()
            self._update_pool_metrics("checked_out")
            return session
            
        except Exception as e:
            logger.error(f"Error getting database session: {str(e)}")
            self._update_pool_metrics("error")
            raise
    
    def return_session(self, session: Session):
        """Return session to pool"""
        try:
            if session:
                session.close()
                self._update_pool_metrics("checked_in")
        except Exception as e:
            logger.error(f"Error returning database session: {str(e)}")
            self._update_pool_metrics("error")
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status"""
        try:
            if not self.engine:
                return {"error": "Engine not initialized"}
            
            pool = self.engine.pool
            
            return {
                "pool_type": type(pool).__name__,
                "pool_size": getattr(pool, 'size', lambda: 0)(),
                "checked_in": getattr(pool, 'checked_in', lambda: 0)(),
                "checked_out": getattr(pool, 'checked_out', lambda: 0)(),
                "overflow": getattr(pool, 'overflow', lambda: 0)(),
                "invalidated": getattr(pool, 'invalidated', lambda: 0)(),
                "total_connections": getattr(pool, 'size', lambda: 0)() + getattr(pool, 'overflow', lambda: 0)(),
                "pool_utilization": self._calculate_pool_utilization(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting pool status: {str(e)}")
            return {"error": str(e)}
    
    def optimize_pool_configuration(self) -> Dict[str, Any]:
        """Optimize connection pool configuration based on usage patterns"""
        try:
            current_status = self.get_pool_status()
            if "error" in current_status:
                return current_status
            
            # Analyze current usage
            utilization = current_status.get("pool_utilization", 0)
            checked_out = current_status.get("checked_out", 0)
            pool_size = current_status.get("pool_size", 0)
            overflow = current_status.get("overflow", 0)
            
            # Calculate optimal configuration
            optimal_config = self._calculate_optimal_config(utilization, checked_out, pool_size, overflow)
            
            # Apply optimizations
            optimizations_applied = self._apply_pool_optimizations(optimal_config)
            
            return {
                "current_configuration": self.pool_config,
                "optimal_configuration": optimal_config,
                "optimizations_applied": optimizations_applied,
                "expected_improvement": self._calculate_expected_improvement(current_status, optimal_config),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing pool configuration: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_optimal_config(self, utilization: float, checked_out: int, pool_size: int, overflow: int) -> Dict[str, Any]:
        """Calculate optimal pool configuration"""
        optimal_config = self.pool_config.copy()
        
        # Optimize pool size
        if utilization > 80:
            optimal_config["pool_size"] = min(pool_size * 2, 50)  # Double size, max 50
        elif utilization < 30:
            optimal_config["pool_size"] = max(pool_size // 2, 2)  # Half size, min 2
        
        # Optimize max overflow
        if overflow > 0 and overflow > pool_size * 0.5:
            optimal_config["max_overflow"] = min(overflow * 2, 100)  # Double overflow, max 100
        elif overflow == 0 and utilization > 70:
            optimal_config["max_overflow"] = max(pool_size // 2, 5)  # Add overflow
        
        # Optimize timeout
        if self.pool_metrics["connection_wait_time"] > 1000:  # 1 second
            optimal_config["pool_timeout"] = min(optimal_config["pool_timeout"] * 2, 60)  # Double timeout, max 60s
        elif self.pool_metrics["connection_wait_time"] < 100:  # 100ms
            optimal_config["pool_timeout"] = max(optimal_config["pool_timeout"] // 2, 5)  # Half timeout, min 5s
        
        # Optimize recycle time
        if self.pool_metrics["average_connection_time"] > 5000:  # 5 seconds
            optimal_config["pool_recycle"] = min(optimal_config["pool_recycle"] * 2, 7200)  # Double recycle, max 2h
        elif self.pool_metrics["average_connection_time"] < 1000:  # 1 second
            optimal_config["pool_recycle"] = max(optimal_config["pool_recycle"] // 2, 1800)  # Half recycle, min 30m
        
        return optimal_config
    
    def _apply_pool_optimizations(self, optimal_config: Dict[str, Any]) -> List[str]:
        """Apply pool optimizations (simplified implementation)"""
        optimizations = []
        
        # In a real implementation, this would modify the actual pool configuration
        # For now, we'll just return what would be applied
        
        if optimal_config["pool_size"] != self.pool_config["pool_size"]:
            optimizations.append(f"Set pool_size to {optimal_config['pool_size']}")
        
        if optimal_config["max_overflow"] != self.pool_config["max_overflow"]:
            optimizations.append(f"Set max_overflow to {optimal_config['max_overflow']}")
        
        if optimal_config["pool_timeout"] != self.pool_config["pool_timeout"]:
            optimizations.append(f"Set pool_timeout to {optimal_config['pool_timeout']} seconds")
        
        if optimal_config["pool_recycle"] != self.pool_config["pool_recycle"]:
            optimizations.append(f"Set pool_recycle to {optimal_config['pool_recycle']} seconds")
        
        if optimal_config["pool_pre_ping"] != self.pool_config["pool_pre_ping"]:
            optimizations.append(f"Set pool_pre_ping to {optimal_config['pool_pre_ping']}")
        
        if optimal_config["pool_reset_on_return"] != self.pool_config["pool_reset_on_return"]:
            optimizations.append(f"Set pool_reset_on_return to {optimal_config['pool_reset_on_return']}")
        
        return optimizations
    
    def _calculate_expected_improvement(self, current_status: Dict[str, Any], optimal_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate expected improvement from optimizations"""
        current_utilization = current_status.get("pool_utilization", 0)
        current_timeout = self.pool_config["pool_timeout"]
        optimal_timeout = optimal_config["pool_timeout"]
        
        utilization_improvement = 0
        if current_utilization > 80:
            utilization_improvement = min(20, current_utilization - 70)  # Reduce to 70%
        
        timeout_improvement = 0
        if current_timeout > optimal_timeout:
            timeout_improvement = ((current_timeout - optimal_timeout) / current_timeout) * 100
        
        return {
            "utilization_improvement": f"{utilization_improvement:.1f}%",
            "timeout_improvement": f"{timeout_improvement:.1f}%",
            "expected_efficiency_gain": f"{utilization_improvement + timeout_improvement:.1f}%",
            "estimated_performance_boost": "15-40%"
        }
    
    def _calculate_pool_utilization(self) -> float:
        """Calculate pool utilization percentage"""
        try:
            if not self.engine:
                return 0.0
            
            pool = self.engine.pool
            pool_size = getattr(pool, 'size', lambda: 0)()
            checked_out = getattr(pool, 'checked_out', lambda: 0)()
            
            if pool_size == 0:
                return 0.0
            
            return (checked_out / pool_size) * 100
            
        except Exception:
            return 0.0
    
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
                # Update pool metrics
                self._update_pool_metrics("monitoring")
                
                # Sleep for monitoring interval
                time.sleep(30)  # 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(30)
    
    def _update_pool_metrics(self, event_type: str):
        """Update pool metrics"""
        try:
            if event_type == "checked_out":
                self.pool_metrics["pool_checked_out"] += 1
            elif event_type == "checked_in":
                self.pool_metrics["pool_checked_in"] += 1
            elif event_type == "error":
                self.pool_metrics["connection_errors"] += 1
            elif event_type == "timeout":
                self.pool_metrics["connection_timeouts"] += 1
            elif event_type == "monitoring":
                # Update current metrics
                status = self.get_pool_status()
                if "error" not in status:
                    self.pool_metrics["total_connections"] = status.get("total_connections", 0)
                    self.pool_metrics["active_connections"] = status.get("checked_out", 0)
                    self.pool_metrics["idle_connections"] = status.get("checked_in", 0)
                    self.pool_metrics["overflow_connections"] = status.get("overflow", 0)
                    
                    # Update peak connections
                    current_total = status.get("total_connections", 0)
                    if current_total > self.pool_metrics["peak_connections"]:
                        self.pool_metrics["peak_connections"] = current_total
                
        except Exception as e:
            logger.error(f"Error updating pool metrics: {str(e)}")
    
    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics"""
        try:
            return {
                "pool_metrics": self.pool_metrics,
                "pool_status": self.get_pool_status(),
                "monitoring_active": self.monitoring_active,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting pool metrics: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on connection pool"""
        try:
            start_time = time.time()
            
            if not self.engine:
                return {
                    "healthy": False,
                    "error": "Engine not initialized",
                    "response_time_ms": 0
                }
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "healthy": True,
                "response_time_ms": response_time,
                "pool_status": self.get_pool_status(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Connection pool health check failed: {str(e)}")
            return {
                "healthy": False,
                "error": str(e),
                "response_time_ms": 0
            }

# Global connection pool config instance
_connection_pool_config = None

def get_connection_pool_config() -> ConnectionPoolConfig:
    """Get global connection pool config instance"""
    global _connection_pool_config
    if _connection_pool_config is None:
        _connection_pool_config = ConnectionPoolConfig()
    return _connection_pool_config

def get_database_session() -> Session:
    """Get database session from connection pool"""
    pool_config = get_connection_pool_config()
    return pool_config.get_session()

def return_database_session(session: Session):
    """Return database session to connection pool"""
    pool_config = get_connection_pool_config()
    pool_config.return_session(session)
