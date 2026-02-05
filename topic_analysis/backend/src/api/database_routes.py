"""
Database Routes

This module provides API endpoints for database operations,
including CRUD operations and operation status tracking.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4

from ..services.supabase_service import SupabaseService
from ..services.database_operation_service import DatabaseOperationService
from ..core.logging import db_operation_logger
from ..core.error_handler import DatabaseTimeoutError, DatabaseConnectionError, DatabaseAuthenticationError

router = APIRouter(prefix="/database", tags=["Database Operations"])

def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """
    Extract user ID from authorization header.
    
    Args:
        authorization: Authorization header value
        
    Returns:
        User ID
        
    Raises:
        HTTPException: If authorization is invalid
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized", "message": "Missing or invalid authorization header"}
        )
    
    # Validate the JWT token with Supabase
    token = authorization.replace("Bearer ", "")
    try:
        from ..core.supabase_auth import supabase_auth_service
        user = supabase_auth_service.get_user_by_token(token)
        if user and user.get("id"):
            return user["id"]
        else:
            raise HTTPException(
                status_code=401,
                detail={"error": "Unauthorized", "message": "Invalid token"}
            )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized", "message": "Token validation failed"}
        )

@router.post("/operations")
async def execute_database_operation(
    operation_type: str,
    table_name: str,
    query_data: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = 100,
    offset: Optional[int] = 0,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Execute a database operation.
    
    Args:
        operation_type: Type of operation (create, read, update, delete, real_time)
        table_name: Target table name
        query_data: Optional query data
        filters: Optional filters to apply
        limit: Maximum number of records to return
        offset: Number of records to skip
        user_id: User performing the operation
        
    Returns:
        Dict containing operation result
    """
    try:
        # Initialize services
        supabase_service = SupabaseService()
        operation_service = DatabaseOperationService()
        
        # Generate request ID
        request_id = str(uuid4())
        
        # Create operation record
        operation = operation_service.create_operation(
            client_id="default-client",
            operation_type=operation_type,
            table_name=table_name,
            query_data=query_data,
            user_id=user_id,
            request_id=request_id
        )
        
        # Execute operation based on type
        if operation_type == "create":
            if not query_data:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "Bad Request", "message": "query_data is required for create operations"}
                )
            
            result = supabase_service.create(
                table_name=table_name,
                data=query_data,
                user_id=user_id,
                request_id=request_id
            )
            
            # Update operation status
            operation_service.update_operation_status(
                operation_id=operation.operation_id,
                status="success",
                response_data=result,
                execution_time_ms=operation.execution_time_ms
            )
            
            return {
                "success": True,
                "data": result,
                "operation_id": operation.operation_id,
                "execution_time_ms": operation.execution_time_ms
            }
            
        elif operation_type == "read":
            result = supabase_service.read(
                table_name=table_name,
                filters=filters,
                limit=limit,
                offset=offset,
                user_id=user_id,
                request_id=request_id
            )
            
            # Update operation status
            operation_service.update_operation_status(
                operation_id=operation.operation_id,
                status="success",
                response_data={"count": len(result)},
                execution_time_ms=operation.execution_time_ms
            )
            
            return {
                "success": True,
                "data": result,
                "operation_id": operation.operation_id,
                "execution_time_ms": operation.execution_time_ms
            }
            
        elif operation_type == "update":
            if not query_data or "id" not in query_data:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "Bad Request", "message": "query_data with 'id' is required for update operations"}
                )
            
            record_id = query_data.pop("id")
            result = supabase_service.update(
                table_name=table_name,
                record_id=record_id,
                data=query_data,
                user_id=user_id,
                request_id=request_id
            )
            
            # Update operation status
            operation_service.update_operation_status(
                operation_id=operation.operation_id,
                status="success",
                response_data=result,
                execution_time_ms=operation.execution_time_ms
            )
            
            return {
                "success": True,
                "data": result,
                "operation_id": operation.operation_id,
                "execution_time_ms": operation.execution_time_ms
            }
            
        elif operation_type == "delete":
            if not query_data or "id" not in query_data:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "Bad Request", "message": "query_data with 'id' is required for delete operations"}
                )
            
            record_id = query_data["id"]
            result = supabase_service.delete(
                table_name=table_name,
                record_id=record_id,
                user_id=user_id,
                request_id=request_id
            )
            
            # Update operation status
            operation_service.update_operation_status(
                operation_id=operation.operation_id,
                status="success",
                response_data={"deleted": result},
                execution_time_ms=operation.execution_time_ms
            )
            
            return {
                "success": True,
                "data": {"deleted": result},
                "operation_id": operation.operation_id,
                "execution_time_ms": operation.execution_time_ms
            }
            
        else:
            raise HTTPException(
                status_code=400,
                detail={"error": "Bad Request", "message": f"Invalid operation type: {operation_type}"}
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except DatabaseTimeoutError as e:
        # Update operation status for timeout
        if 'operation' in locals():
            operation_service.update_operation_status(
                operation_id=operation.operation_id,
                status="timeout",
                error_message=str(e),
                execution_time_ms=60000  # 60 seconds timeout
            )
        
        raise HTTPException(
            status_code=408,
            detail={"error": "Request Timeout", "message": str(e)}
        )
    except DatabaseConnectionError as e:
        # Update operation status for connection error
        if 'operation' in locals():
            operation_service.update_operation_status(
                operation_id=operation.operation_id,
                status="error",
                error_message=str(e)
            )
        
        raise HTTPException(
            status_code=503,
            detail={"error": "Service Unavailable", "message": str(e)}
        )
    except DatabaseAuthenticationError as e:
        # Update operation status for auth error
        if 'operation' in locals():
            operation_service.update_operation_status(
                operation_id=operation.operation_id,
                status="error",
                error_message=str(e)
            )
        
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized", "message": str(e)}
        )
    except Exception as e:
        # Update operation status for general error
        if 'operation' in locals():
            operation_service.update_operation_status(
                operation_id=operation.operation_id,
                status="error",
                error_message=str(e)
            )
        
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": str(e)}
        )

@router.get("/operations/{operation_id}")
async def get_operation_status(
    operation_id: str,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the status of a database operation.
    
    Args:
        operation_id: Operation identifier
        user_id: User requesting the status
        
    Returns:
        Dict containing operation status
    """
    try:
        # Initialize service
        operation_service = DatabaseOperationService()
        
        # Get operation
        operation = operation_service.get_operation(operation_id)
        
        if not operation:
            raise HTTPException(
                status_code=404,
                detail={"error": "Not Found", "message": "Operation not found"}
            )
        
        # Check if user has access to this operation
        if operation.user_id and operation.user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail={"error": "Forbidden", "message": "Access denied to this operation"}
            )
        
        return {
            "operation_id": operation.operation_id,
            "status": operation.status.value,
            "created_at": operation.created_at.isoformat(),
            "completed_at": operation.completed_at.isoformat() if operation.completed_at else None,
            "execution_time_ms": operation.execution_time_ms,
            "error_message": operation.error_message
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": str(e)}
        )
