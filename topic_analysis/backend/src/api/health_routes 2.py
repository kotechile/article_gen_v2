"""
Health Routes

This module provides health check endpoints for database connectivity
and system status monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from datetime import datetime

from ..services.supabase_service import SupabaseService
from ..core.logging import db_operation_logger

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/database")
async def health_check() -> Dict[str, Any]:
    """
    Check database health status.
    
    Returns:
        Dict containing health status and metrics
    """
    try:
        # Initialize service
        supabase_service = SupabaseService()
        
        # Perform health check
        health_status = supabase_service.health_check()
        
        # Log health check
        db_operation_logger.log_connection_health(
            status=health_status["status"],
            execution_time_ms=health_status.get("execution_time_ms"),
            error_count=0
        )
        
        # Return health status
        if health_status["status"] == "healthy":
            return {
                "status": "healthy",
                "database": "supabase",
                "timestamp": datetime.utcnow().isoformat(),
                "tables_available": health_status.get("tables_available", []),
                "execution_time_ms": health_status.get("execution_time_ms")
            }
        else:
            # Return 503 for unhealthy status
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "message": health_status.get("message", "Database connection failed"),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error and return 503
        db_operation_logger.log_connection_health(
            status="unhealthy",
            execution_time_ms=None,
            error_count=1
        )
        
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        )