"""
Topic Decomposition API routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import structlog

from ..core.database import get_db
from ..core.auth import get_current_user
from ..services.topic_decomposition_service import TopicDecompositionService
from ..models.user import User

logger = structlog.get_logger()
router = APIRouter(prefix="/api/topics", tags=["topic-decomposition"])

@router.post("/decompose")
async def decompose_topic(
    request: Dict[str, Any],
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Decompose a search query into related subtopics using LLM analysis
    
    Request body:
    - search_query: The search query to decompose
    - max_subtopics: Maximum number of subtopics to generate (default: 10)
    
    Returns:
    - Decomposition results with subtopics and metadata
    """
    try:
        # Validate request
        search_query = request.get("search_query")
        if not search_query or not search_query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="search_query is required and cannot be empty"
            )
        
        max_subtopics = request.get("max_subtopics", 10)
        if not isinstance(max_subtopics, int) or max_subtopics < 1 or max_subtopics > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_subtopics must be an integer between 1 and 20"
            )
        
        # Create service and decompose topic (temporary: no auth required)
        service = TopicDecompositionService(db)
        result = await service.decompose_topic(
            search_query=search_query.strip(),
            user_id="temp_user",  # Temporary user ID for testing
            max_subtopics=max_subtopics
        )
        
        logger.info("Topic decomposition completed", 
                   user_id="temp_user", 
                   search_query=search_query,
                   subtopics_count=len(result.get("subtopics", [])))
        
        return {
            "success": True,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Topic decomposition failed", 
                    user_id="temp_user", 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Topic decomposition failed. Please try again."
        )

@router.get("/decompositions/{decomposition_id}")
async def get_decomposition(
    decomposition_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get topic decomposition by ID
    
    Args:
        decomposition_id: The ID of the decomposition to retrieve
        
    Returns:
        Decomposition data if found and accessible by user
    """
    try:
        service = TopicDecompositionService(db)
        result = await service.get_decomposition(decomposition_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Topic decomposition not found"
            )
        
        # Verify user owns this decomposition
        if result["user_id"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this topic decomposition"
            )
        
        logger.info("Topic decomposition retrieved", 
                   user_id=current_user.id, 
                   decomposition_id=decomposition_id)
        
        return {
            "success": True,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get topic decomposition", 
                    user_id=current_user.id, 
                    decomposition_id=decomposition_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve topic decomposition"
        )

@router.delete("/decompositions/{decomposition_id}")
async def delete_decomposition(
    decomposition_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete topic decomposition by ID
    
    Args:
        decomposition_id: The ID of the decomposition to delete
        
    Returns:
        Success status
    """
    try:
        service = TopicDecompositionService(db)
        
        # First check if decomposition exists and user owns it
        result = await service.get_decomposition(decomposition_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Topic decomposition not found"
            )
        
        if result["user_id"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this topic decomposition"
            )
        
        # Delete the decomposition
        success = await service.delete_decomposition(decomposition_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete topic decomposition"
            )
        
        logger.info("Topic decomposition deleted", 
                   user_id=current_user.id, 
                   decomposition_id=decomposition_id)
        
        return {
            "success": True,
            "message": "Topic decomposition deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete topic decomposition", 
                    user_id=current_user.id, 
                    decomposition_id=decomposition_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete topic decomposition"
        )

@router.get("/decompositions")
async def list_user_decompositions(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List topic decompositions for the current user
    
    Args:
        skip: Number of decompositions to skip
        limit: Maximum number of decompositions to return
        
    Returns:
        List of user's topic decompositions
    """
    try:
        if limit > 100:
            limit = 100
        
        service = TopicDecompositionService(db)
        
        # Get user's decompositions from database
        decompositions = db.query(service.db.query(TopicDecomposition).filter(
            TopicDecomposition.user_id == str(current_user.id)
        ).offset(skip).limit(limit).all())
        
        result = []
        for decomp in decompositions:
            result.append({
                "id": str(decomp.id),
                "search_query": decomp.search_query,
                "subtopics_count": len(decomp.subtopics) if decomp.subtopics else 0,
                "created_at": decomp.created_at.isoformat(),
                "updated_at": decomp.updated_at.isoformat()
            })
        
        logger.info("User decompositions listed", 
                   user_id=current_user.id, 
                   count=len(result))
        
        return {
            "success": True,
            "data": {
                "decompositions": result,
                "total": len(result),
                "skip": skip,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error("Failed to list user decompositions", 
                    user_id=current_user.id, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list topic decompositions"
        )
