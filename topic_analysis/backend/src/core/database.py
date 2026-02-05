"""
Database configuration and connection management for TrendTap
This file now uses Supabase SDK exclusively - no PostgreSQL/SQLAlchemy dependencies
"""

from typing import Generator, Optional
import structlog
from contextlib import contextmanager
from datetime import datetime, timedelta

# Import the Supabase-only database service
from .supabase_database_service import (
    SupabaseDatabaseService, 
    get_database_service, 
    get_db as get_supabase_db,
    get_db_session as get_supabase_db_session
)

logger = structlog.get_logger()

# Placeholder Base class for backward compatibility (not used with Supabase)
class Base:
    """Placeholder Base class for backward compatibility with Supabase-only architecture"""
    pass

# Re-export Supabase functions for backward compatibility
get_db = get_supabase_db
get_db_session = get_supabase_db_session

def get_db_session_manual() -> SupabaseDatabaseService:
    """
    Get a database service instance (Supabase doesn't need manual session management)
    """
    return get_database_service()

def close_db_session(session: SupabaseDatabaseService) -> None:
    """
    Close a database session (Supabase client doesn't need explicit cleanup)
    """
    # Supabase client is stateless, no cleanup needed
    pass

def execute_in_transaction(func, *args, **kwargs):
    """
    Execute a function within a database context
    Note: Supabase handles transactions automatically
    """
    with get_db_session() as db:
        try:
            result = func(db, *args, **kwargs)
            return result
        except Exception as e:
            logger.error("Database operation error", error=str(e))
            raise

def init_db() -> None:
    """
    Initialize database (Supabase tables are managed via Supabase dashboard)
    """
    try:
        # Test connection
        db_service = get_database_service()
        health = db_service.health_check()
        
        if health["healthy"]:
            logger.info("Supabase database connection verified")
        else:
            logger.error("Supabase database connection failed", error=health.get("error"))
            raise Exception(f"Database connection failed: {health.get('error')}")
        
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise

def check_db_connection() -> bool:
    """
    Check if database connection is healthy
    """
    try:
        db_service = get_database_service()
        health = db_service.health_check()
        return health["healthy"]
    except Exception as e:
        logger.error("Database connection check failed", error=str(e))
        return False

def get_connection_pool_status() -> dict:
    """
    Get connection status (Supabase manages connections automatically)
    """
    try:
        db_service = get_database_service()
        health = db_service.health_check()
        
        return {
            "status": "connected" if health["healthy"] else "disconnected",
            "supabase_url": health.get("supabase_url", "unknown"),
            "timestamp": health.get("timestamp", "unknown")
        }
    except Exception as e:
        logger.error("Failed to get connection status", error=str(e))
        return {
            "status": "error",
            "error": str(e)
        }

def optimize_database() -> None:
    """
    Optimize database performance (handled by Supabase)
    """
    logger.info("Database optimization is handled automatically by Supabase")

def get_database_stats() -> dict:
    """
    Get database statistics (basic info from Supabase)
    """
    try:
        db_service = get_database_service()
        health = db_service.health_check()
        
        return {
            "status": "connected" if health["healthy"] else "disconnected",
            "supabase_url": health.get("supabase_url", "unknown"),
            "timestamp": health.get("timestamp", "unknown")
        }
    except Exception as e:
        logger.error("Failed to get database stats", error=str(e))
        return {"error": str(e)}

def cleanup_old_data(days: int = 30) -> None:
    """
    Clean up old data using Supabase
    """
    try:
        db_service = get_database_service()
        
        # Clean up old failed records
        cutoff_date = f"{(datetime.utcnow() - timedelta(days=days)).isoformat()}"
        
        # Clean affiliate researches
        db_service.execute_query(
            "affiliate_researches",
            "delete",
            filters={"status": "failed", "created_at": f"lt.{cutoff_date}"}
        )
        
        # Clean trend analyses
        db_service.execute_query(
            "trend_analyses", 
            "delete",
            filters={"status": "failed", "created_at": f"lt.{cutoff_date}"}
        )
        
        # Clean keyword data
        db_service.execute_query(
            "keyword_data",
            "delete", 
            filters={"status": "failed", "created_at": f"lt.{cutoff_date}"}
        )
        
        logger.info(f"Cleaned up old data older than {days} days")
        
    except Exception as e:
        logger.error("Failed to cleanup old data", error=str(e))
        raise