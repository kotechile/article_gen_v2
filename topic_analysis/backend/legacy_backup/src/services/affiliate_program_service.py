"""
Affiliate Program Service
Manages affiliate programs in the database
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import structlog
from datetime import datetime

from ..models.affiliate_program import AffiliateProgram

logger = structlog.get_logger()

class AffiliateProgramService:
    def __init__(self, db: Session):
        self.db = db
    
    async def save_programs(self, programs: List[Dict[str, Any]], search_term: str) -> List[AffiliateProgram]:
        """
        Save new affiliate programs to the database
        """
        saved_programs = []
        
        for program_data in programs:
            try:
                # Check if program already exists (by name and affiliate_network)
                existing_program = self.db.query(AffiliateProgram).filter(
                    and_(
                        AffiliateProgram.name == program_data.get("name"),
                        AffiliateProgram.affiliate_network == program_data.get("affiliate_network")
                    )
                ).first()
                
                if existing_program:
                    # Update existing program with new search term
                    if search_term not in (existing_program.search_terms or []):
                        if existing_program.search_terms is None:
                            existing_program.search_terms = []
                        existing_program.search_terms.append(search_term)
                        existing_program.last_used = datetime.utcnow()
                        existing_program.usage_count += 1
                        self.db.commit()
                        saved_programs.append(existing_program)
                        logger.info("Updated existing program with new search term", 
                                   program_id=existing_program.id, 
                                   search_term=search_term)
                else:
                    # Create new program
                    new_program = AffiliateProgram(
                        name=program_data.get("name"),
                        description=program_data.get("description"),
                        commission=program_data.get("commission"),
                        cookie_duration=program_data.get("cookie_duration"),
                        payment_terms=program_data.get("payment_terms"),
                        min_payout=program_data.get("min_payout"),
                        category=program_data.get("category"),
                        rating=program_data.get("rating", 0.0),
                        estimated_earnings=program_data.get("estimated_earnings"),
                        difficulty=program_data.get("difficulty"),
                        affiliate_network=program_data.get("affiliate_network"),
                        tracking_method=program_data.get("tracking_method"),
                        payment_methods=program_data.get("payment_methods"),
                        support_level=program_data.get("support_level"),
                        promotional_materials=program_data.get("promotional_materials"),
                        restrictions=program_data.get("restrictions"),
                        source=program_data.get("source", "web_search"),
                        search_terms=[search_term],
                        is_active=True,
                        is_verified=False
                    )
                    
                    self.db.add(new_program)
                    self.db.commit()
                    self.db.refresh(new_program)
                    saved_programs.append(new_program)
                    
                    logger.info("Saved new affiliate program", 
                               program_id=new_program.id, 
                               name=new_program.name,
                               search_term=search_term)
                    
            except Exception as e:
                logger.error("Failed to save program", 
                           program_name=program_data.get("name"), 
                           error=str(e))
                self.db.rollback()
                continue
        
        return saved_programs
    
    def get_programs_by_search_term(self, search_term: str, limit: int = 10) -> List[AffiliateProgram]:
        """
        Get programs that were found for a specific search term
        """
        try:
            programs = self.db.query(AffiliateProgram).filter(
                and_(
                    AffiliateProgram.is_active == True,
                    AffiliateProgram.search_terms.contains([search_term])
                )
            ).order_by(AffiliateProgram.usage_count.desc()).limit(limit).all()
            
            return programs
        except Exception as e:
            logger.error("Failed to get programs by search term", 
                        search_term=search_term, error=str(e))
            return []
    
    def get_programs_by_category(self, category: str, limit: int = 10) -> List[AffiliateProgram]:
        """
        Get programs by category
        """
        try:
            programs = self.db.query(AffiliateProgram).filter(
                and_(
                    AffiliateProgram.is_active == True,
                    AffiliateProgram.category.ilike(f"%{category}%")
                )
            ).order_by(AffiliateProgram.usage_count.desc()).limit(limit).all()
            
            return programs
        except Exception as e:
            logger.error("Failed to get programs by category", 
                        category=category, error=str(e))
            return []
    
    def search_programs(self, search_term: str, limit: int = 10) -> List[AffiliateProgram]:
        """
        Search programs by name, description, or category
        """
        try:
            search_pattern = f"%{search_term}%"
            programs = self.db.query(AffiliateProgram).filter(
                and_(
                    AffiliateProgram.is_active == True,
                    or_(
                        AffiliateProgram.name.ilike(search_pattern),
                        AffiliateProgram.description.ilike(search_pattern),
                        AffiliateProgram.category.ilike(search_pattern)
                    )
                )
            ).order_by(AffiliateProgram.usage_count.desc()).limit(limit).all()
            
            return programs
        except Exception as e:
            logger.error("Failed to search programs", 
                        search_term=search_term, error=str(e))
            return []
    
    def get_all_programs(self, limit: int = 50) -> List[AffiliateProgram]:
        """
        Get all active programs
        """
        try:
            programs = self.db.query(AffiliateProgram).filter(
                AffiliateProgram.is_active == True
            ).order_by(AffiliateProgram.usage_count.desc()).limit(limit).all()
            
            return programs
        except Exception as e:
            logger.error("Failed to get all programs", error=str(e))
            return []
    
    def update_program_usage(self, program_id: int):
        """
        Update program usage statistics
        """
        try:
            program = self.db.query(AffiliateProgram).filter(
                AffiliateProgram.id == program_id
            ).first()
            
            if program:
                program.last_used = datetime.utcnow()
                program.usage_count += 1
                self.db.commit()
                
        except Exception as e:
            logger.error("Failed to update program usage", 
                        program_id=program_id, error=str(e))
    
    def verify_program(self, program_id: int, verified: bool = True):
        """
        Mark a program as verified or unverified
        """
        try:
            program = self.db.query(AffiliateProgram).filter(
                AffiliateProgram.id == program_id
            ).first()
            
            if program:
                program.is_verified = verified
                self.db.commit()
                
        except Exception as e:
            logger.error("Failed to verify program", 
                        program_id=program_id, error=str(e))


