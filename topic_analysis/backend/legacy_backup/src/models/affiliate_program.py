"""
Affiliate Program Model
Stores affiliate programs discovered through web search or manual addition
"""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, JSON
from sqlalchemy.sql import func
from ..core.database import Base

class AffiliateProgram(Base):
    __tablename__ = "affiliate_programs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    commission = Column(String(50))  # e.g., "5-15%"
    cookie_duration = Column(String(50))  # e.g., "30 days"
    payment_terms = Column(String(50))  # e.g., "Net 30"
    min_payout = Column(String(50))  # e.g., "$50"
    category = Column(String(100), index=True)
    rating = Column(Float, default=0.0)
    estimated_earnings = Column(String(100))  # e.g., "$200-800/month"
    difficulty = Column(String(50))  # e.g., "Easy", "Medium", "Hard"
    affiliate_network = Column(String(100), index=True)
    tracking_method = Column(String(100))
    payment_methods = Column(JSON)  # List of payment methods
    support_level = Column(String(50))
    promotional_materials = Column(JSON)  # List of available materials
    restrictions = Column(Text)
    
    # Source tracking
    source = Column(String(50), default="web_search")  # "web_search", "manual", "database"
    search_terms = Column(JSON)  # List of search terms that found this program
    discovery_date = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)  # Whether program details have been verified
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<AffiliateProgram(id={self.id}, name='{self.name}', category='{self.category}')>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "commission": self.commission,
            "cookie_duration": self.cookie_duration,
            "payment_terms": self.payment_terms,
            "min_payout": self.min_payout,
            "category": self.category,
            "rating": self.rating,
            "estimated_earnings": self.estimated_earnings,
            "difficulty": self.difficulty,
            "affiliate_network": self.affiliate_network,
            "tracking_method": self.tracking_method,
            "payment_methods": self.payment_methods or [],
            "support_level": self.support_level,
            "promotional_materials": self.promotional_materials or [],
            "restrictions": self.restrictions,
            "source": self.source,
            "search_terms": self.search_terms or [],
            "discovery_date": self.discovery_date.isoformat() if self.discovery_date else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "is_active": self.is_active,
            "is_verified": self.is_verified
        }


