"""
Enhanced Workflow API routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import structlog

from ..core.database import get_db
from ..core.auth import get_current_user
from ..services.workflow_session_service import WorkflowSessionService
from ..models.user import User

logger = structlog.get_logger()
router = APIRouter(prefix="/api/workflow", tags=["enhanced-workflow"])

@router.post("/sessions/{session_id}/topic-decomposition")
async def start_topic_decomposition(
    session_id: str,
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Start topic decomposition for a workflow session
    
    Request body:
    - search_query: The search query to decompose
    - max_subtopics: Maximum number of subtopics to generate (default: 10)
    
    Returns:
    - Topic decomposition results
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
        
        # Start topic decomposition
        service = WorkflowSessionService(db)
        result = await service.start_topic_decomposition(
            session_id=session_id,
            user_id=str(current_user.id),
            search_query=search_query.strip(),
            max_subtopics=max_subtopics
        )
        
        logger.info("Topic decomposition started", 
                   user_id=current_user.id, 
                   session_id=session_id,
                   search_query=search_query)
        
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
        logger.error("Failed to start topic decomposition", 
                    user_id=current_user.id, 
                    session_id=session_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start topic decomposition"
        )

@router.get("/sessions/{session_id}/topic-decomposition")
async def get_topic_decomposition(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get topic decomposition for a workflow session
    
    Args:
        session_id: Workflow session ID
        
    Returns:
    - Topic decomposition data
    """
    try:
        service = WorkflowSessionService(db)
        result = await service.get_topic_decomposition(session_id, str(current_user.id))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Topic decomposition not found for this session"
            )
        
        logger.info("Topic decomposition retrieved", 
                   user_id=current_user.id, 
                   session_id=session_id)
        
        return {
            "success": True,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get topic decomposition", 
                    user_id=current_user.id, 
                    session_id=session_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get topic decomposition"
        )

@router.post("/sessions/{session_id}/affiliate-research")
async def start_affiliate_research(
    session_id: str,
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Start affiliate research for selected subtopics
    
    Request body:
    - selected_subtopics: List of subtopic names to research
    
    Returns:
    - Affiliate research results
    """
    try:
        # Validate request
        selected_subtopics = request.get("selected_subtopics")
        if not selected_subtopics or not isinstance(selected_subtopics, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="selected_subtopics is required and must be a list"
            )
        
        if len(selected_subtopics) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one subtopic must be selected"
            )
        
        # Start affiliate research
        service = WorkflowSessionService(db)
        result = await service.start_affiliate_research(
            session_id=session_id,
            user_id=str(current_user.id),
            selected_subtopics=selected_subtopics
        )
        
        logger.info("Affiliate research started", 
                   user_id=current_user.id, 
                   session_id=session_id,
                   selected_subtopics=selected_subtopics)
        
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
        logger.error("Failed to start affiliate research", 
                    user_id=current_user.id, 
                    session_id=session_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start affiliate research"
        )

@router.get("/sessions/{session_id}/complete")
async def get_workflow_complete_data(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get complete workflow data including all steps
    
    Args:
        session_id: Workflow session ID
        
    Returns:
    - Complete workflow data
    """
    try:
        service = WorkflowSessionService(db)
        result = await service.get_workflow_complete_data(session_id, str(current_user.id))
        
        logger.info("Complete workflow data retrieved", 
                   user_id=current_user.id, 
                   session_id=session_id)
        
        return {
            "success": True,
            "data": result
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to get complete workflow data", 
                    user_id=current_user.id, 
                    session_id=session_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get complete workflow data"
        )

@router.post("/sessions")
async def create_workflow_session(
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a new workflow session
    
    Request body:
    - session_name: Name of the workflow session
    - description: Optional description
    
    Returns:
    - Created workflow session data
    """
    try:
        # Validate request
        session_name = request.get("session_name")
        if not session_name or not session_name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_name is required and cannot be empty"
            )
        
        # Create workflow session
        service = WorkflowSessionService(db)
        result = await service.create_session(
            user_id=str(current_user.id),
            session_name=session_name.strip(),
            description=request.get("description")
        )
        
        logger.info("Workflow session created", 
                   user_id=current_user.id, 
                   session_id=result["id"])
        
        return {
            "success": True,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create workflow session", 
                    user_id=current_user.id, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create workflow session"
        )

@router.get("/sessions")
async def list_workflow_sessions(
    status: str = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List workflow sessions for the current user
    
    Query parameters:
    - status: Filter by status (optional)
    - limit: Maximum number of sessions to return
    - offset: Number of sessions to skip
    
    Returns:
    - List of workflow sessions
    """
    try:
        service = WorkflowSessionService(db)
        result = await service.list_sessions(
            user_id=str(current_user.id),
            status=status,
            limit=limit,
            offset=offset
        )
        
        logger.info("Workflow sessions listed", 
                   user_id=current_user.id, 
                   count=len(result["sessions"]))
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error("Failed to list workflow sessions", 
                    user_id=current_user.id, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list workflow sessions"
        )

@router.get("/sessions/{session_id}")
async def get_workflow_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get a workflow session by ID
    
    Args:
        session_id: Workflow session ID
        
    Returns:
    - Workflow session data
    """
    try:
        service = WorkflowSessionService(db)
        result = await service.get_session(session_id, str(current_user.id))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow session not found"
            )
        
        logger.info("Workflow session retrieved", 
                   user_id=current_user.id, 
                   session_id=session_id)
        
        return {
            "success": True,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workflow session", 
                    user_id=current_user.id, 
                    session_id=session_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow session"
        )
