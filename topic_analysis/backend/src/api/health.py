"""
Health check endpoints for database and system status
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import structlog
from datetime import datetime

from ..database.supabase_client import test_supabase_connection

logger = structlog.get_logger()
router = APIRouter()

@router.get("/health/database")
async def database_health() -> Dict[str, Any]:
    """
    Check database connection health
    
    Returns:
        Dict containing database health status, Supabase status, response time, and details
    """
    try:
        logger.info("Checking database health")
        
        # Test Supabase connection
        health_data = test_supabase_connection()
        
        # Determine overall health status
        if health_data["status"] == "healthy":
            logger.info("Database health check passed", 
                       response_time=health_data.get("response_time_ms", 0))
            return health_data
        else:
            logger.warning("Database health check failed", 
                          status=health_data["status"],
                          error=health_data.get("error"))
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Database unhealthy",
                    "message": health_data.get("error", "Unknown error"),
                    "timestamp": health_data.get("last_check")
                }
            )
            
    except Exception as e:
        logger.error("Database health check error", error=str(e))
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Database health check failed",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.get("/health/system")
async def system_health() -> Dict[str, Any]:
    """
    Check overall system health
    
    Returns:
        Dict containing system health status
    """
    try:
        # Check database health
        db_health = test_supabase_connection()
        
        # Determine overall system status
        if db_health["status"] == "healthy":
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "services": {
                    "database": db_health
                }
            }
        else:
            return {
                "status": "degraded",
                "timestamp": datetime.utcnow().isoformat(),
                "services": {
                    "database": db_health
                }
            }
            
    except Exception as e:
        logger.error("System health check error", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

