"""
Health Monitoring Middleware

This module provides connection health monitoring middleware for Supabase
with automatic health checks and connection status tracking.
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import asyncio
import time
from datetime import datetime, timedelta

from ..core.supabase_client import get_health_status, reset_supabase_connection
from ..core.logging import db_operation_logger, get_logger

logger = get_logger(__name__)

class HealthMonitoringMiddleware:
    """
    Middleware for monitoring Supabase connection health and automatic recovery.
    """
    
    def __init__(self, health_check_interval: int = 30, max_failures: int = 3):
        """
        Initialize the health monitoring middleware.
        
        Args:
            health_check_interval: Health check interval in seconds
            max_failures: Maximum consecutive failures before marking as unhealthy
        """
        self.health_check_interval = health_check_interval
        self.max_failures = max_failures
        self.logger = db_operation_logger
        self._last_health_check = None
        self._consecutive_failures = 0
        self._is_healthy = True
        self._health_status = None
    
    async def check_health(self, force_check: bool = False) -> Dict[str, Any]:
        """
        Check Supabase connection health.
        
        Args:
            force_check: Force health check regardless of interval
            
        Returns:
            Dict containing health status
        """
        current_time = datetime.utcnow()
        
        # Check if we need to perform health check
        if (not force_check and 
            self._last_health_check and 
            (current_time - self._last_health_check).seconds < self.health_check_interval):
            return self._health_status or {"status": "unknown"}
        
        try:
            # Perform health check
            health_status = get_health_status()
            self._last_health_check = current_time
            
            if health_status["status"] == "healthy":
                self._consecutive_failures = 0
                self._is_healthy = True
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.max_failures:
                    self._is_healthy = False
            
            self._health_status = health_status
            
            # Log health status
            self.logger.log_connection_health(
                status=health_status["status"],
                execution_time_ms=health_status.get("execution_time_ms"),
                error_count=self._consecutive_failures
            )
            
            return health_status
            
        except Exception as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.max_failures:
                self._is_healthy = False
            
            logger.error(
                "Health check failed",
                error=str(e),
                consecutive_failures=self._consecutive_failures
            )
            
            return {
                "status": "unhealthy",
                "message": str(e),
                "timestamp": current_time.isoformat(),
                "consecutive_failures": self._consecutive_failures
            }
    
    async def handle_unhealthy_connection(self, request: Request) -> JSONResponse:
        """
        Handle requests when connection is unhealthy.
        
        Args:
            request: FastAPI request object
            
        Returns:
            JSONResponse with error details
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log unhealthy connection
        self.logger.log_connection_health(
            status="unhealthy",
            execution_time_ms=None,
            error_count=self._consecutive_failures
        )
        
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service Unavailable",
                "message": "Database connection is unhealthy",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "consecutive_failures": self._consecutive_failures,
                    "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None
                }
            }
        )
    
    async def attempt_recovery(self) -> bool:
        """
        Attempt to recover from unhealthy connection.
        
        Returns:
            True if recovery was successful
        """
        try:
            # Reset connection
            reset_supabase_connection()
            
            # Wait a bit before checking health
            await asyncio.sleep(2)
            
            # Check health again
            health_status = await self.check_health(force_check=True)
            
            if health_status["status"] == "healthy":
                logger.info("Connection recovery successful")
                return True
            else:
                logger.warning("Connection recovery failed")
                return False
                
        except Exception as e:
            logger.error("Connection recovery error", error=str(e))
            return False
    
    async def monitor_health(self, request: Request, call_next):
        """
        Monitor health before processing request.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        # Check if connection is healthy
        if not self._is_healthy:
            # Attempt recovery
            recovery_successful = await self.attempt_recovery()
            if not recovery_successful:
                return await self.handle_unhealthy_connection(request)
        
        # Perform health check if needed
        health_status = await self.check_health()
        
        if health_status["status"] != "healthy":
            return await self.handle_unhealthy_connection(request)
        
        # Process request
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Log error and check if it's connection-related
            logger.error(
                "Request processing error",
                error=str(e),
                request_id=getattr(request.state, 'request_id', 'unknown')
            )
            
            # If it's a connection error, mark as unhealthy
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                self._is_healthy = False
                self._consecutive_failures += 1
            
            raise
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Get health monitoring metrics.
        
        Returns:
            Dict containing health metrics
        """
        return {
            "is_healthy": self._is_healthy,
            "consecutive_failures": self._consecutive_failures,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
            "health_check_interval": self.health_check_interval,
            "max_failures": self.max_failures,
            "current_status": self._health_status
        }
    
    def is_healthy(self) -> bool:
        """
        Check if connection is currently healthy.
        
        Returns:
            True if connection is healthy
        """
        return self._is_healthy
    
    def get_failure_count(self) -> int:
        """
        Get current consecutive failure count.
        
        Returns:
            Number of consecutive failures
        """
        return self._consecutive_failures
    
    def reset_failures(self) -> None:
        """
        Reset consecutive failure count.
        """
        self._consecutive_failures = 0
        self._is_healthy = True

# Global middleware instance
health_middleware = HealthMonitoringMiddleware(
    health_check_interval=30,
    max_failures=3
)

async def health_monitoring_middleware(request: Request, call_next):
    """
    FastAPI middleware for health monitoring.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware/handler
        
    Returns:
        Response object
    """
    return await health_middleware.monitor_health(request, call_next)
