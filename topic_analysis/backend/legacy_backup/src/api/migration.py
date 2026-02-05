"""
Migration API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import structlog

from ..core.database import get_db
from ..services.migration_service import MigrationService
from ..database.migration import MigrationType

logger = structlog.get_logger()
router = APIRouter()

@router.post("/database/migrate")
async def start_migration(
    migration_type: str,
    dry_run: bool = False,
    tables: Optional[List[str]] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Start a database migration
    
    Args:
        migration_type: Type of migration (full, incremental, rollback)
        dry_run: Whether to perform a dry run
        tables: Specific tables to migrate (empty for all)
        db: Database session
        
    Returns:
        Dict containing migration details
    """
    try:
        logger.info("Starting migration", 
                   type=migration_type, 
                   dry_run=dry_run,
                   tables=tables)
        
        # Validate migration type
        try:
            parsed_migration_type = MigrationType(migration_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid migration request",
                    "message": f"Migration type must be one of: {[t.value for t in MigrationType]}",
                    "details": {"migration_type": migration_type},
                    "timestamp": "2024-12-19T10:30:00Z"
                }
            )
        
        # Initialize service
        migration_service = MigrationService(db)
        
        # Start migration
        migration_info = migration_service.start_migration(
            migration_type=parsed_migration_type,
            tables=tables,
            dry_run=dry_run
        )
        
        logger.info("Migration started successfully", 
                   migration_id=migration_info["migration_id"],
                   type=migration_type)
        
        return {
            "migration_id": migration_info["migration_id"],
            "status": migration_info["status"],
            "estimated_duration": "5-10 minutes"  # This would be calculated based on data size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to start migration", 
                    type=migration_type,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to start migration",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.get("/database/migrate/{migration_id}")
async def get_migration_status(
    migration_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get migration status
    
    Args:
        migration_id: Migration ID
        db: Database session
        
    Returns:
        Dict containing migration status
    """
    try:
        logger.info("Getting migration status", migration_id=migration_id)
        
        # Initialize service
        migration_service = MigrationService(db)
        
        # Get migration status
        status = migration_service.get_migration_status(migration_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Migration not found",
                    "message": f"Migration with ID '{migration_id}' does not exist",
                    "details": {"migration_id": migration_id},
                    "timestamp": "2024-12-19T10:30:00Z"
                }
            )
        
        logger.info("Migration status retrieved successfully", 
                   migration_id=migration_id,
                   status=status.get("status"))
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get migration status", 
                    migration_id=migration_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to retrieve migration status",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.post("/database/migrate/{migration_id}/execute")
async def execute_migration(
    migration_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Execute a migration
    
    Args:
        migration_id: Migration ID
        db: Database session
        
    Returns:
        Dict containing execution results
    """
    try:
        logger.info("Executing migration", migration_id=migration_id)
        
        # Initialize service
        migration_service = MigrationService(db)
        
        # Execute migration
        result = migration_service.execute_migration(migration_id)
        
        logger.info("Migration executed", 
                   migration_id=migration_id,
                   status=result.get("status"))
        
        return result
        
    except Exception as e:
        logger.error("Failed to execute migration", 
                    migration_id=migration_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to execute migration",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.post("/database/migrate/{migration_id}/rollback")
async def rollback_migration(
    migration_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Rollback a migration
    
    Args:
        migration_id: Migration ID
        db: Database session
        
    Returns:
        Dict containing rollback results
    """
    try:
        logger.info("Rolling back migration", migration_id=migration_id)
        
        # Initialize service
        migration_service = MigrationService(db)
        
        # Rollback migration
        result = migration_service.rollback_migration(migration_id)
        
        logger.info("Migration rolled back", 
                   migration_id=migration_id,
                   status=result.get("status"))
        
        return result
        
    except Exception as e:
        logger.error("Failed to rollback migration", 
                    migration_id=migration_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to rollback migration",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.get("/database/migrate/history")
async def get_migration_history(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get migration history
    
    Args:
        limit: Maximum number of migrations
        offset: Offset for pagination
        db: Database session
        
    Returns:
        List of migration records
    """
    try:
        logger.info("Getting migration history", limit=limit, offset=offset)
        
        # Initialize service
        migration_service = MigrationService(db)
        
        # Get migration history
        history = migration_service.get_migration_history(limit=limit, offset=offset)
        
        logger.info("Migration history retrieved successfully", 
                   count=len(history))
        
        return history
        
    except Exception as e:
        logger.error("Failed to get migration history", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to retrieve migration history",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.post("/database/migrate/{migration_id}/validate")
async def validate_migration(
    migration_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Validate migration data integrity
    
    Args:
        migration_id: Migration ID
        db: Database session
        
    Returns:
        Dict containing validation results
    """
    try:
        logger.info("Validating migration", migration_id=migration_id)
        
        # Initialize service
        migration_service = MigrationService(db)
        
        # Validate migration
        validation_result = migration_service.validate_migration_data(migration_id)
        
        logger.info("Migration validation completed", 
                   migration_id=migration_id,
                   valid=validation_result.get("valid"))
        
        return validation_result
        
    except Exception as e:
        logger.error("Failed to validate migration", 
                    migration_id=migration_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to validate migration",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.get("/database/migrate/statistics")
async def get_migration_statistics(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get migration statistics
    
    Args:
        db: Database session
        
    Returns:
        Dict containing migration statistics
    """
    try:
        logger.info("Getting migration statistics")
        
        # Initialize service
        migration_service = MigrationService(db)
        
        # Get statistics
        stats = migration_service.get_migration_statistics()
        
        logger.info("Migration statistics retrieved successfully", 
                   total_migrations=stats["total_migrations"],
                   success_rate=stats["success_rate"])
        
        return stats
        
    except Exception as e:
        logger.error("Failed to get migration statistics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to retrieve migration statistics",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

