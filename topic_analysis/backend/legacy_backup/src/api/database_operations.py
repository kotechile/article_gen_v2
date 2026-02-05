"""
Database operations API endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import structlog

from ..core.database import get_db
from ..services.database_operation_service import DatabaseOperationService
from ..models.database_operation import OperationType, OperationStatus

logger = structlog.get_logger()
router = APIRouter()

@router.get("/database/operations")
async def get_database_operations(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of operations to return"),
    operation_type: Optional[str] = Query(None, description="Filter by operation type"),
    status: Optional[str] = Query(None, description="Filter by operation status"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get database operations with optional filters
    
    Args:
        limit: Maximum number of operations to return
        operation_type: Filter by operation type
        status: Filter by operation status
        user_id: Filter by user ID
        db: Database session
        
    Returns:
        Dict containing operations and pagination info
    """
    try:
        logger.info("Getting database operations", 
                   limit=limit, 
                   operation_type=operation_type,
                   status=status,
                   user_id=user_id)
        
        # Initialize service
        operation_service = DatabaseOperationService(db)
        
        # Parse filters
        parsed_operation_type = None
        if operation_type:
            try:
                parsed_operation_type = OperationType(operation_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid operation type: {operation_type}. Must be one of: {[t.value for t in OperationType]}"
                )
        
        parsed_status = None
        if status:
            try:
                parsed_status = OperationStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Must be one of: {[s.value for s in OperationStatus]}"
                )
        
        # Get operations
        operations = operation_service.get_operations(
            limit=limit,
            operation_type=parsed_operation_type,
            status=parsed_status,
            user_id=user_id
        )
        
        # Get total count
        total_count = operation_service.get_operation_count(
            operation_type=parsed_operation_type,
            status=parsed_status,
            user_id=user_id
        )
        
        # Calculate pagination info
        total_pages = (total_count + limit - 1) // limit
        has_next = total_count > limit
        has_previous = False  # We're always starting from page 1 for now
        
        # Convert operations to dict format
        operations_data = [op.to_dict() for op in operations]
        
        response = {
            "operations": operations_data,
            "total_count": total_count,
            "page_info": {
                "page": 1,
                "limit": limit,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_previous": has_previous
            }
        }
        
        logger.info("Database operations retrieved successfully", 
                   count=len(operations_data),
                   total_count=total_count)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get database operations", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to retrieve database operations",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.get("/database/operations/{operation_id}")
async def get_database_operation(
    operation_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get a specific database operation by ID
    
    Args:
        operation_id: Operation ID
        db: Database session
        
    Returns:
        Dict containing operation details
    """
    try:
        logger.info("Getting database operation", operation_id=operation_id)
        
        # Initialize service
        operation_service = DatabaseOperationService(db)
        
        # Get operation
        operation = operation_service.get_operation(operation_id)
        
        if not operation:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Operation not found",
                    "message": f"Operation with ID '{operation_id}' does not exist",
                    "timestamp": "2024-12-19T10:30:00Z"
                }
            )
        
        logger.info("Database operation retrieved successfully", 
                   operation_id=operation_id,
                   status=operation.status.value)
        
        return operation.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get database operation", 
                    operation_id=operation_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to retrieve database operation",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.get("/database/operations/stats/summary")
async def get_operation_statistics(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to look back"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get database operation statistics
    
    Args:
        hours: Number of hours to look back
        db: Database session
        
    Returns:
        Dict containing operation statistics
    """
    try:
        logger.info("Getting operation statistics", hours=hours)
        
        # Initialize service
        operation_service = DatabaseOperationService(db)
        
        # Get statistics
        stats = operation_service.get_operation_stats(hours=hours)
        
        logger.info("Operation statistics retrieved successfully", 
                   total_operations=stats["total_operations"],
                   success_rate=stats["success_rate"])
        
        return stats
        
    except Exception as e:
        logger.error("Failed to get operation statistics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to retrieve operation statistics",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.get("/database/operations/failed/recent")
async def get_recent_failed_operations(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to look back"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get recent failed operations
    
    Args:
        hours: Number of hours to look back
        db: Database session
        
    Returns:
        List of failed operations
    """
    try:
        logger.info("Getting recent failed operations", hours=hours)
        
        # Initialize service
        operation_service = DatabaseOperationService(db)
        
        # Get failed operations
        failed_operations = operation_service.get_failed_operations(hours=hours)
        
        # Convert to dict format
        operations_data = [op.to_dict() for op in failed_operations]
        
        logger.info("Recent failed operations retrieved successfully", 
                   count=len(operations_data))
        
        return operations_data
        
    except Exception as e:
        logger.error("Failed to get recent failed operations", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to retrieve recent failed operations",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.get("/database/operations/slow/recent")
async def get_slow_operations(
    threshold_ms: float = Query(1000, ge=100, le=10000, description="Threshold in milliseconds"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get slow operations above threshold
    
    Args:
        threshold_ms: Threshold in milliseconds
        db: Database session
        
    Returns:
        List of slow operations
    """
    try:
        logger.info("Getting slow operations", threshold_ms=threshold_ms)
        
        # Initialize service
        operation_service = DatabaseOperationService(db)
        
        # Get slow operations
        slow_operations = operation_service.get_slow_operations(threshold_ms=threshold_ms)
        
        # Convert to dict format
        operations_data = [op.to_dict() for op in slow_operations]
        
        logger.info("Slow operations retrieved successfully", 
                   count=len(operations_data),
                   threshold_ms=threshold_ms)
        
        return operations_data
        
    except Exception as e:
        logger.error("Failed to get slow operations", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to retrieve slow operations",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

