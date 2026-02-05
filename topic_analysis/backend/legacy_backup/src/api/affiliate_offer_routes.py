"""
Affiliate Offer API routes
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import structlog

from ..core.database import get_db
from ..core.auth import get_current_user
from ..services.affiliate_offer_service import AffiliateOfferService
from ..models.affiliate_offer import OfferStatus
from ..models.user import User

logger = structlog.get_logger()
router = APIRouter(prefix="/api/affiliate/offers", tags=["affiliate-offers"])

@router.post("/")
async def create_affiliate_offer(
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a new affiliate offer
    
    Request body:
    - offer_name: Name of the offer (required)
    - offer_description: Description of the offer
    - commission_rate: Commission rate percentage (0-100)
    - access_instructions: Instructions for accessing the offer
    - workflow_session_id: Workflow session ID (required)
    - subtopic_id: Associated subtopic ID
    - linkup_data: Raw data from LinkUp API
    
    Returns:
    - Created affiliate offer data
    """
    try:
        # Validate required fields
        offer_name = request.get("offer_name")
        workflow_session_id = request.get("workflow_session_id")
        
        if not offer_name or not workflow_session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="offer_name and workflow_session_id are required"
            )
        
        # Check user feature access
        if not current_user.can_access_feature("affiliate_research"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Affiliate research feature not available for your subscription tier"
            )
        
        # Create affiliate offer
        service = AffiliateOfferService(db)
        result = service.create_offer(
            user_id=str(current_user.id),
            workflow_session_id=workflow_session_id,
            offer_name=offer_name,
            offer_description=request.get("offer_description"),
            commission_rate=request.get("commission_rate"),
            access_instructions=request.get("access_instructions"),
            subtopic_id=request.get("subtopic_id"),
            linkup_data=request.get("linkup_data")
        )
        
        logger.info("Affiliate offer created", 
                   user_id=current_user.id, 
                   workflow_session_id=workflow_session_id,
                   offer_id=result["id"])
        
        return {
            "success": True,
            "data": result
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create affiliate offer", 
                    user_id=current_user.id, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create affiliate offer"
        )

@router.get("/")
async def list_affiliate_offers(
    workflow_session_id: Optional[str] = Query(None, description="Filter by workflow session ID"),
    status: Optional[str] = Query(None, description="Filter by status (active, inactive, expired)"),
    skip: int = Query(0, ge=0, description="Number of offers to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of offers to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List affiliate offers for the current user
    
    Query parameters:
    - workflow_session_id: Filter by workflow session ID
    - status: Filter by status (active, inactive, expired)
    - skip: Number of offers to skip
    - limit: Maximum number of offers to return
    
    Returns:
    - List of affiliate offers
    """
    try:
        service = AffiliateOfferService(db)
        
        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = OfferStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid status. Must be one of: active, inactive, expired"
                )
        
        if workflow_session_id:
            # List offers for specific workflow session
            offers = service.list_offers_by_workflow(
                workflow_session_id=workflow_session_id,
                user_id=str(current_user.id),
                status=status_filter
            )
        else:
            # List all user offers
            offers = service.list_user_offers(
                user_id=str(current_user.id),
                status=status_filter,
                skip=skip,
                limit=limit
            )
        
        logger.info("Affiliate offers listed", 
                   user_id=current_user.id, 
                   count=len(offers),
                   workflow_session_id=workflow_session_id)
        
        return {
            "success": True,
            "data": {
                "offers": offers,
                "total": len(offers),
                "skip": skip,
                "limit": limit
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list affiliate offers", 
                    user_id=current_user.id, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list affiliate offers"
        )

@router.get("/{offer_id}")
async def get_affiliate_offer(
    offer_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get affiliate offer by ID
    
    Args:
        offer_id: Affiliate offer ID
        
    Returns:
    - Affiliate offer data
    """
    try:
        service = AffiliateOfferService(db)
        offer = service.get_offer(offer_id, str(current_user.id))
        
        if not offer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Affiliate offer not found"
            )
        
        logger.info("Affiliate offer retrieved", 
                   user_id=current_user.id, 
                   offer_id=offer_id)
        
        return {
            "success": True,
            "data": offer
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get affiliate offer", 
                    user_id=current_user.id, 
                    offer_id=offer_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get affiliate offer"
        )

@router.put("/{offer_id}")
async def update_affiliate_offer(
    offer_id: str,
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update affiliate offer
    
    Args:
        offer_id: Affiliate offer ID
        request: Update data
        
    Returns:
    - Updated affiliate offer data
    """
    try:
        service = AffiliateOfferService(db)
        
        # Parse status if provided
        status_value = request.get("status")
        status_filter = None
        if status_value:
            try:
                status_filter = OfferStatus(status_value)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid status. Must be one of: active, inactive, expired"
                )
        
        result = service.update_offer(
            offer_id=offer_id,
            user_id=str(current_user.id),
            offer_name=request.get("offer_name"),
            offer_description=request.get("offer_description"),
            commission_rate=request.get("commission_rate"),
            access_instructions=request.get("access_instructions"),
            status=status_filter,
            linkup_data=request.get("linkup_data")
        )
        
        logger.info("Affiliate offer updated", 
                   user_id=current_user.id, 
                   offer_id=offer_id)
        
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
        logger.error("Failed to update affiliate offer", 
                    user_id=current_user.id, 
                    offer_id=offer_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update affiliate offer"
        )

@router.delete("/{offer_id}")
async def delete_affiliate_offer(
    offer_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete affiliate offer
    
    Args:
        offer_id: Affiliate offer ID
        
    Returns:
    - Success status
    """
    try:
        service = AffiliateOfferService(db)
        success = service.delete_offer(offer_id, str(current_user.id))
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Affiliate offer not found"
            )
        
        logger.info("Affiliate offer deleted", 
                   user_id=current_user.id, 
                   offer_id=offer_id)
        
        return {
            "success": True,
            "message": "Affiliate offer deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete affiliate offer", 
                    user_id=current_user.id, 
                    offer_id=offer_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete affiliate offer"
        )

@router.get("/subtopic/{subtopic_id}")
async def get_offers_by_subtopic(
    subtopic_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get affiliate offers for a specific subtopic
    
    Args:
        subtopic_id: Subtopic ID
        
    Returns:
    - List of affiliate offers for the subtopic
    """
    try:
        service = AffiliateOfferService(db)
        offers = service.get_offers_by_subtopic(subtopic_id, str(current_user.id))
        
        logger.info("Affiliate offers retrieved by subtopic", 
                   user_id=current_user.id, 
                   subtopic_id=subtopic_id,
                   count=len(offers))
        
        return {
            "success": True,
            "data": {
                "subtopic_id": subtopic_id,
                "offers": offers,
                "total": len(offers)
            }
        }
        
    except Exception as e:
        logger.error("Failed to get offers by subtopic", 
                    user_id=current_user.id, 
                    subtopic_id=subtopic_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get affiliate offers by subtopic"
        )

@router.patch("/{offer_id}/status")
async def update_offer_status(
    offer_id: str,
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update affiliate offer status
    
    Args:
        offer_id: Affiliate offer ID
        request: Status update data (status field required)
        
    Returns:
    - Updated affiliate offer data
    """
    try:
        status_value = request.get("status")
        if not status_value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="status is required"
            )
        
        try:
            status_filter = OfferStatus(status_value)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid status. Must be one of: active, inactive, expired"
            )
        
        service = AffiliateOfferService(db)
        result = service.update_offer_status(offer_id, str(current_user.id), status_filter)
        
        logger.info("Affiliate offer status updated", 
                   user_id=current_user.id, 
                   offer_id=offer_id,
                   status=status_value)
        
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
        logger.error("Failed to update offer status", 
                    user_id=current_user.id, 
                    offer_id=offer_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update affiliate offer status"
        )
