"""
Affiliate Offer Service
Handles affiliate offer management and persistence
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog
from sqlalchemy.orm import Session
from ..models.affiliate_offer import AffiliateOffer, OfferStatus

logger = structlog.get_logger()

class AffiliateOfferService:
    """Service for managing affiliate offers"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_offer(
        self,
        user_id: str,
        workflow_session_id: str,
        offer_name: str,
        offer_description: str = None,
        commission_rate: float = None,
        access_instructions: str = None,
        subtopic_id: str = None,
        linkup_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a new affiliate offer
        
        Args:
            user_id: User ID
            workflow_session_id: Workflow session ID
            offer_name: Name of the offer
            offer_description: Description of the offer
            commission_rate: Commission rate percentage
            access_instructions: Instructions for accessing the offer
            subtopic_id: Associated subtopic ID
            linkup_data: Raw data from LinkUp API
            
        Returns:
            Created affiliate offer data
        """
        try:
            # Validate required fields
            if not offer_name or not offer_name.strip():
                raise ValueError("offer_name is required")
            
            # Validate commission rate if provided
            if commission_rate is not None and (commission_rate < 0 or commission_rate > 100):
                raise ValueError("commission_rate must be between 0 and 100")
            
            # Create affiliate offer
            offer = AffiliateOffer(
                user_id=user_id,
                workflow_session_id=workflow_session_id,
                offer_name=offer_name.strip(),
                offer_description=offer_description.strip() if offer_description else None,
                commission_rate=commission_rate,
                access_instructions=access_instructions.strip() if access_instructions else None,
                subtopic_id=subtopic_id,
                linkup_data=linkup_data or {},
                status=OfferStatus.ACTIVE
            )
            
            self.db.add(offer)
            self.db.commit()
            self.db.refresh(offer)
            
            logger.info("Affiliate offer created", 
                       user_id=user_id, 
                       workflow_session_id=workflow_session_id,
                       offer_id=offer.id,
                       offer_name=offer_name)
            
            return offer.to_dict()
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to create affiliate offer", 
                        user_id=user_id, 
                        workflow_session_id=workflow_session_id,
                        error=str(e))
            raise
    
    def get_offer(self, offer_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get affiliate offer by ID
        
        Args:
            offer_id: Offer ID
            user_id: User ID
            
        Returns:
            Affiliate offer data if found and accessible by user
        """
        try:
            offer = self.db.query(AffiliateOffer).filter(
                AffiliateOffer.id == offer_id,
                AffiliateOffer.user_id == user_id
            ).first()
            
            if not offer:
                return None
            
            return offer.to_dict()
            
        except Exception as e:
            logger.error("Failed to get affiliate offer", 
                        offer_id=offer_id, 
                        user_id=user_id,
                        error=str(e))
            raise
    
    def list_offers_by_workflow(
        self, 
        workflow_session_id: str, 
        user_id: str,
        status: Optional[OfferStatus] = None
    ) -> List[Dict[str, Any]]:
        """
        List affiliate offers for a workflow session
        
        Args:
            workflow_session_id: Workflow session ID
            user_id: User ID
            status: Filter by status (optional)
            
        Returns:
            List of affiliate offer data
        """
        try:
            query = self.db.query(AffiliateOffer).filter(
                AffiliateOffer.workflow_session_id == workflow_session_id,
                AffiliateOffer.user_id == user_id
            )
            
            if status:
                query = query.filter(AffiliateOffer.status == status)
            
            offers = query.order_by(AffiliateOffer.created_at.desc()).all()
            
            return [offer.to_dict() for offer in offers]
            
        except Exception as e:
            logger.error("Failed to list affiliate offers", 
                        workflow_session_id=workflow_session_id, 
                        user_id=user_id,
                        error=str(e))
            raise
    
    def list_user_offers(
        self, 
        user_id: str,
        status: Optional[OfferStatus] = None,
        skip: int = 0,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        List all affiliate offers for a user
        
        Args:
            user_id: User ID
            status: Filter by status (optional)
            skip: Number of offers to skip
            limit: Maximum number of offers to return
            
        Returns:
            List of affiliate offer data
        """
        try:
            query = self.db.query(AffiliateOffer).filter(
                AffiliateOffer.user_id == user_id
            )
            
            if status:
                query = query.filter(AffiliateOffer.status == status)
            
            offers = query.order_by(AffiliateOffer.created_at.desc()).offset(skip).limit(limit).all()
            
            return [offer.to_dict() for offer in offers]
            
        except Exception as e:
            logger.error("Failed to list user affiliate offers", 
                        user_id=user_id,
                        error=str(e))
            raise
    
    def update_offer(
        self,
        offer_id: str,
        user_id: str,
        offer_name: str = None,
        offer_description: str = None,
        commission_rate: float = None,
        access_instructions: str = None,
        status: OfferStatus = None,
        linkup_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Update affiliate offer
        
        Args:
            offer_id: Offer ID
            user_id: User ID
            offer_name: New offer name (optional)
            offer_description: New offer description (optional)
            commission_rate: New commission rate (optional)
            access_instructions: New access instructions (optional)
            status: New status (optional)
            linkup_data: New LinkUp data (optional)
            
        Returns:
            Updated affiliate offer data
        """
        try:
            offer = self.db.query(AffiliateOffer).filter(
                AffiliateOffer.id == offer_id,
                AffiliateOffer.user_id == user_id
            ).first()
            
            if not offer:
                raise ValueError("Affiliate offer not found")
            
            # Update fields if provided
            if offer_name is not None:
                if not offer_name.strip():
                    raise ValueError("offer_name cannot be empty")
                offer.offer_name = offer_name.strip()
            
            if offer_description is not None:
                offer.offer_description = offer_description.strip() if offer_description else None
            
            if commission_rate is not None:
                if commission_rate < 0 or commission_rate > 100:
                    raise ValueError("commission_rate must be between 0 and 100")
                offer.commission_rate = commission_rate
            
            if access_instructions is not None:
                offer.access_instructions = access_instructions.strip() if access_instructions else None
            
            if status is not None:
                offer.status = status
            
            if linkup_data is not None:
                offer.linkup_data = linkup_data
            
            offer.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(offer)
            
            logger.info("Affiliate offer updated", 
                       offer_id=offer_id, 
                       user_id=user_id)
            
            return offer.to_dict()
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to update affiliate offer", 
                        offer_id=offer_id, 
                        user_id=user_id,
                        error=str(e))
            raise
    
    def delete_offer(self, offer_id: str, user_id: str) -> bool:
        """
        Delete affiliate offer
        
        Args:
            offer_id: Offer ID
            user_id: User ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            offer = self.db.query(AffiliateOffer).filter(
                AffiliateOffer.id == offer_id,
                AffiliateOffer.user_id == user_id
            ).first()
            
            if not offer:
                return False
            
            self.db.delete(offer)
            self.db.commit()
            
            logger.info("Affiliate offer deleted", 
                       offer_id=offer_id, 
                       user_id=user_id)
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to delete affiliate offer", 
                        offer_id=offer_id, 
                        user_id=user_id,
                        error=str(e))
            raise
    
    def get_offers_by_subtopic(
        self, 
        subtopic_id: str, 
        user_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get affiliate offers for a specific subtopic
        
        Args:
            subtopic_id: Subtopic ID
            user_id: User ID
            
        Returns:
            List of affiliate offer data
        """
        try:
            offers = self.db.query(AffiliateOffer).filter(
                AffiliateOffer.subtopic_id == subtopic_id,
                AffiliateOffer.user_id == user_id,
                AffiliateOffer.status == OfferStatus.ACTIVE
            ).order_by(AffiliateOffer.created_at.desc()).all()
            
            return [offer.to_dict() for offer in offers]
            
        except Exception as e:
            logger.error("Failed to get offers by subtopic", 
                        subtopic_id=subtopic_id, 
                        user_id=user_id,
                        error=str(e))
            raise
    
    def update_offer_status(
        self, 
        offer_id: str, 
        user_id: str, 
        status: OfferStatus
    ) -> Dict[str, Any]:
        """
        Update affiliate offer status
        
        Args:
            offer_id: Offer ID
            user_id: User ID
            status: New status
            
        Returns:
            Updated affiliate offer data
        """
        try:
            offer = self.db.query(AffiliateOffer).filter(
                AffiliateOffer.id == offer_id,
                AffiliateOffer.user_id == user_id
            ).first()
            
            if not offer:
                raise ValueError("Affiliate offer not found")
            
            offer.status = status
            offer.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(offer)
            
            logger.info("Affiliate offer status updated", 
                       offer_id=offer_id, 
                       user_id=user_id,
                       status=status.value)
            
            return offer.to_dict()
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to update offer status", 
                        offer_id=offer_id, 
                        user_id=user_id,
                        error=str(e))
            raise
