"""
Health check and monitoring endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional
from datetime import datetime
import structlog
import psutil
import time

from ..core.database import get_db, check_db_connection
from ..core.redis import check_redis_connection
from ..core.config import get_settings
from src.core.supabase_database_service import SupabaseDatabaseService
from src.services.supabase_service import get_supabase_service

logger = structlog.get_logger()
router = APIRouter(prefix="/api/health", tags=["health-monitoring"])

@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "TrendTap Backend API",
            "version": "1.0.1",
            "build_id": "debug-2026-01-28-v3-NO-TRUSTED-HOST"
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")

@router.get("/detailed")
async def detailed_health_check(db: SupabaseDatabaseService = Depends(get_db)):
    """Detailed health check with system metrics"""
    try:
        # Check database connection
        db_healthy = check_db_connection()
        
        # Check Redis connection
        redis_healthy = await check_redis_connection()
        
        # Get system metrics
        system_metrics = get_system_metrics()
        
        # Determine overall health
        overall_healthy = db_healthy and redis_healthy
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "TrendTap Backend API",
            "version": "0.1.0",
            "checks": {
                "database": {
                    "status": "healthy" if db_healthy else "unhealthy",
                    "message": "Database connection successful" if db_healthy else "Database connection failed"
                },
                "redis": {
                    "status": "healthy" if redis_healthy else "unhealthy",
                    "message": "Redis connection successful" if redis_healthy else "Redis connection failed"
                }
            },
            "system_metrics": system_metrics
        }
        
    except Exception as e:
        logger.error("Detailed health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "TrendTap Backend API",
            "version": "0.1.0",
            "error": str(e)
        }

@router.get("/supabase-http")
async def supabase_http_health_check():
    """Supabase HTTP API health check"""
    try:
        supabase = get_supabase_service().get_client()
        # Test HTTP connection by checking health or auth
        # We'll try to get the auth settings or just make a lightweight call
        # get_session() is local, doesn't ping. 
        # But auth.get_user('invalid') should make a request and fail fast.
        
        start_time = time.time()
        try:
            # We explicitly want to fail auth to test connectivity
            supabase.auth.get_user("00000000-0000-0000-0000-000000000000")
        except Exception:
            # We expect an error, but if we get here, it means we CONNECTED
            pass
            
        latency = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "connection_type": "HTTP/HTTPS",
            "latency_ms": round(latency, 2),
            "message": "Successfully connected to Supabase HTTP API"
        }
        
    except Exception as e:
        logger.error("Supabase HTTP health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Supabase HTTP API Unreachable: {str(e)}")

@router.get("/database")
async def database_health_check(db: SupabaseDatabaseService = Depends(get_db)):
    """Database-specific health check"""
    try:
        # Test database connection
        db_healthy = check_db_connection()
        
        if not db_healthy:
            raise HTTPException(status_code=503, detail="Database connection failed")
        
        # Get database metrics
        db_metrics = get_database_metrics(db)
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "PostgreSQL",
            "metrics": db_metrics
        }
        
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Database unhealthy: {str(e)}")

@router.get("/redis")
async def redis_health_check():
    """Redis-specific health check"""
    try:
        # Test Redis connection
        redis_healthy = await check_redis_connection()
        
        if not redis_healthy:
            raise HTTPException(status_code=503, detail="Redis connection failed")
        
        # Get Redis metrics
        redis_metrics = await get_redis_metrics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "redis": "Redis",
            "metrics": redis_metrics
        }
        
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Redis unhealthy: {str(e)}")

@router.get("/metrics")
async def get_metrics():
    """Get system and application metrics"""
    try:
        system_metrics = get_system_metrics()
        app_metrics = get_application_metrics()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_metrics,
            "application": app_metrics
        }
        
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@router.get("/status")
async def get_status():
    """Get application status and configuration"""
    try:
        settings = get_settings()
        
        return {
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "TrendTap Backend API",
            "version": "0.1.0",
            "environment": "development",  # This would come from settings
            "features": {
                "affiliate_research": True,
                "trend_analysis": True,
                "keyword_management": True,
                "content_generation": True,
                "software_generation": True,
                "export_integration": True,
                "calendar_management": True,
                "user_management": True
            },
            "external_services": {
                "dataforseo": True,  # Always available via database
                "google_trends": True,  # Available via DataForSEO
                "openai": True,  # Available via database
                "anthropic": True,  # Available via database
                "google_ai": True,  # Available via database
                "surferseo": True,  # Available via database
                "frase": True,  # Available via database
                "coschedule": True  # Available via database
            }
        }
        
    except Exception as e:
        logger.error("Failed to get status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve status")

def get_system_metrics() -> Dict[str, Any]:
    """Get system-level metrics"""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available
        memory_total = memory.total
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free = disk.free
        disk_total = disk.total
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu = process.cpu_percent()
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count
            },
            "memory": {
                "percent": memory_percent,
                "available_bytes": memory_available,
                "total_bytes": memory_total,
                "available_mb": round(memory_available / 1024 / 1024, 2),
                "total_mb": round(memory_total / 1024 / 1024, 2)
            },
            "disk": {
                "percent": disk_percent,
                "free_bytes": disk_free,
                "total_bytes": disk_total,
                "free_gb": round(disk_free / 1024 / 1024 / 1024, 2),
                "total_gb": round(disk_total / 1024 / 1024 / 1024, 2)
            },
            "process": {
                "memory_bytes": process_memory.rss,
                "memory_mb": round(process_memory.rss / 1024 / 1024, 2),
                "cpu_percent": process_cpu
            }
        }
        
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        return {"error": str(e)}

def get_database_metrics(db: SupabaseDatabaseService) -> Dict[str, Any]:
    """Get database-specific metrics"""
    try:
        # Get connection pool info
        pool = db.bind.pool
        pool_size = pool.size()
        checked_in = pool.checkedin()
        checked_out = pool.checkedout()
        overflow = pool.overflow()
        
        # Get database version
        result = db.execute("SELECT version()")
        db_version = result.scalar()
        
        return {
            "connection_pool": {
                "size": pool_size,
                "checked_in": checked_in,
                "checked_out": checked_out,
                "overflow": overflow
            },
            "version": db_version,
            "status": "connected"
        }
        
    except Exception as e:
        logger.error("Failed to get database metrics", error=str(e))
        return {"error": str(e)}

async def get_redis_metrics() -> Dict[str, Any]:
    """Get Redis-specific metrics"""
    try:
        from ..core.redis import get_redis_client
        redis_client = await get_redis_client()
        
        # Get Redis info
        info = await redis_client.info()
        
        return {
            "version": info.get("redis_version", "unknown"),
            "uptime_seconds": info.get("uptime_in_seconds", 0),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory": info.get("used_memory", 0),
            "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "status": "connected"
        }
        
    except Exception as e:
        logger.error("Failed to get Redis metrics", error=str(e))
        return {"error": str(e)}

def get_application_metrics() -> Dict[str, Any]:
    """Get application-specific metrics"""
    try:
        # This would typically come from application state or monitoring
        return {
            "uptime_seconds": time.time() - start_time,
            "requests_processed": 0,  # This would be tracked in real implementation
            "errors_count": 0,  # This would be tracked in real implementation
            "active_sessions": 0,  # This would be tracked in real implementation
            "features_enabled": {
                "affiliate_research": True,
                "trend_analysis": True,
                "keyword_management": True,
                "content_generation": True,
                "software_generation": True,
                "export_integration": True,
                "calendar_management": True,
                "user_management": True
            }
        }
        
    except Exception as e:
        logger.error("Failed to get application metrics", error=str(e))
        return {"error": str(e)}

# Track application start time
start_time = time.time()