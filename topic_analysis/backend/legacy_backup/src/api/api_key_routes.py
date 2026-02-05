"""
API Key Management API routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import structlog

from ..core.database import get_db
from ..core.auth import get_current_user
from ..services.api_key_service import APIKeyService
from ..models.user import User

logger = structlog.get_logger()
router = APIRouter(prefix="/api/api-keys", tags=["api-keys"])

@router.post("/")
async def create_api_key(
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a new API key for external service integration
    
    Request body:
    - service_name: Name of the service (linkup, semrush, ahrefs, google_trends)
    - api_key: The API key to store
    
    Returns:
    - Created API key data
    """
    try:
        # Validate request
        service_name = request.get("service_name")
        api_key = request.get("api_key")
        
        if not service_name or not api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="service_name and api_key are required"
            )
        
        valid_services = ['linkup', 'semrush', 'ahrefs', 'google_trends']
        if service_name not in valid_services:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid service_name. Must be one of: {valid_services}"
            )
        
        # Create API key
        service = APIKeyService(db)
        result = service.create_api_key(
            user_id=str(current_user.id),
            service_name=service_name,
            api_key=api_key
        )
        
        logger.info("API key created", 
                   user_id=current_user.id, 
                   service_name=service_name)
        
        return {
            "success": True,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create API key", 
                    user_id=current_user.id, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )

@router.get("/")
async def list_api_keys(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all API keys for the current user
    
    Returns:
    - List of user's API keys
    """
    try:
        service = APIKeyService(db)
        api_keys = service.list_user_api_keys(str(current_user.id))
        
        logger.info("API keys listed", 
                   user_id=current_user.id, 
                   count=len(api_keys))
        
        return {
            "success": True,
            "data": {
                "api_keys": api_keys,
                "total": len(api_keys)
            }
        }
        
    except Exception as e:
        logger.error("Failed to list API keys", 
                    user_id=current_user.id, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list API keys"
        )

@router.get("/{service_name}")
async def get_api_key(
    service_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get API key for a specific service
    
    Args:
        service_name: Name of the service
        
    Returns:
    - API key data if found
    """
    try:
        valid_services = ['linkup', 'semrush', 'ahrefs', 'google_trends']
        if service_name not in valid_services:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid service_name. Must be one of: {valid_services}"
            )
        
        service = APIKeyService(db)
        api_key = service.get_api_key(str(current_user.id), service_name)
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No API key found for service: {service_name}"
            )
        
        logger.info("API key retrieved", 
                   user_id=current_user.id, 
                   service_name=service_name)
        
        return {
            "success": True,
            "data": api_key
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get API key", 
                    user_id=current_user.id, 
                    service_name=service_name,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get API key"
        )

@router.put("/{key_id}")
async def update_api_key(
    key_id: str,
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update API key
    
    Args:
        key_id: API key ID
        request: Update data (api_key, is_active)
        
    Returns:
    - Updated API key data
    """
    try:
        service = APIKeyService(db)
        
        # Validate request
        api_key = request.get("api_key")
        is_active = request.get("is_active")
        
        if api_key is None and is_active is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one field (api_key, is_active) must be provided"
            )
        
        result = service.update_api_key(
            key_id=key_id,
            user_id=str(current_user.id),
            api_key=api_key,
            is_active=is_active
        )
        
        logger.info("API key updated", 
                   user_id=current_user.id, 
                   key_id=key_id)
        
        return {
            "success": True,
            "data": result
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update API key", 
                    user_id=current_user.id, 
                    key_id=key_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update API key"
        )

@router.delete("/{key_id}")
async def delete_api_key(
    key_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete API key
    
    Args:
        key_id: API key ID
        
    Returns:
    - Success status
    """
    try:
        service = APIKeyService(db)
        success = service.delete_api_key(key_id, str(current_user.id))
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        logger.info("API key deleted", 
                   user_id=current_user.id, 
                   key_id=key_id)
        
        return {
            "success": True,
            "message": "API key deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete API key", 
                    user_id=current_user.id, 
                    key_id=key_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete API key"
        )

@router.get("/{service_name}/validate")
async def validate_api_key(
    service_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Validate API key for a specific service
    
    Args:
        service_name: Name of the service
        
    Returns:
    - Validation result
    """
    try:
        valid_services = ['linkup', 'semrush', 'ahrefs', 'google_trends']
        if service_name not in valid_services:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid service_name. Must be one of: {valid_services}"
            )
        
        service = APIKeyService(db)
        is_valid = service.validate_api_key(str(current_user.id), service_name)
        
        logger.info("API key validation checked", 
                   user_id=current_user.id, 
                   service_name=service_name,
                   is_valid=is_valid)
        
        return {
            "success": True,
            "data": {
                "service_name": service_name,
                "is_valid": is_valid
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to validate API key", 
                    user_id=current_user.id, 
                    service_name=service_name,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate API key"
        )
