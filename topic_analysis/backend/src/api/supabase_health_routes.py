"""
Supabase Health Check Routes
Simple API routes that use Supabase SDK instead of SQLAlchemy
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import structlog

from ..core.supabase_database_service import get_supabase_db, SupabaseDatabaseService
from src.core.supabase_database_service import SupabaseDatabaseService

logger = structlog.get_logger()
router = APIRouter(prefix="/api/health", tags=["health"])

@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "TrendTap API",
        "database": "supabase",
        "timestamp": "2024-01-01T00:00:00Z"
    }

@router.get("/database")
async def database_health(db: SupabaseDatabaseService = Depends(get_supabase_db)):
    """Check Supabase database health"""
    try:
        health_status = db.health_check()
        return health_status
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Database unhealthy")

@router.get("/supabase")
async def supabase_health(db: SupabaseDatabaseService = Depends(get_supabase_db)):
    """Check Supabase connection specifically"""
    try:
        # Test basic Supabase connection
        result = db.client.table("users").select("id").limit(1).execute()
        return {
            "status": "healthy",
            "database": "supabase",
            "connection": "active",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error("Supabase health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Supabase connection failed")
