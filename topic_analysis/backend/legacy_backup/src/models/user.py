"""
User model for TrendTap
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base


class UserRole(PyEnum):
    """User role enumeration"""
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"


class SubscriptionTier(PyEnum):
    """Subscription tier enumeration"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    subscription_tier = Column(Enum(SubscriptionTier), default=SubscriptionTier.FREE, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # User preferences
    preferences = Column(Text, nullable=True)  # JSON string for user preferences
    timezone = Column(String(50), default="UTC", nullable=False)
    language = Column(String(10), default="en", nullable=False)
    
    # Subscription details
    subscription_start_date = Column(DateTime(timezone=True), nullable=True)
    subscription_end_date = Column(DateTime(timezone=True), nullable=True)
    subscription_status = Column(String(50), default="active", nullable=False)
    
    # Usage tracking
    api_calls_count = Column(Integer, default=0, nullable=False)
    last_api_call = Column(DateTime(timezone=True), nullable=True)
    
    # Supabase integration fields
    supabase_user_id = Column(String(255), nullable=True, unique=True, index=True)
    profile_data = Column(Text, nullable=True)  # JSON string for profile data
    
    # Relationships
    affiliate_researches = relationship("AffiliateResearch", back_populates="user")
    trend_analyses = relationship("TrendAnalysis", back_populates="user")
    keyword_data = relationship("KeywordData", back_populates="user")
    content_ideas = relationship("ContentIdeas", back_populates="user")
    software_solutions = relationship("SoftwareSolutions", back_populates="user")
    content_calendar = relationship("ContentCalendar", back_populates="user")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    password_resets = relationship("PasswordReset", back_populates="user", cascade="all, delete-orphan")
    authentication_logs = relationship("AuthenticationLog", back_populates="user", cascade="all, delete-orphan")
    workflow_sessions = relationship("WorkflowSessions", back_populates="user", cascade="all, delete-orphan")
    trend_selections = relationship("TrendSelections", back_populates="user", cascade="all, delete-orphan")
    keyword_clusters = relationship("KeywordClusters", back_populates="user", cascade="all, delete-orphan")
    topic_decompositions = relationship("TopicDecomposition", back_populates="user", cascade="all, delete-orphan")
    affiliate_offers = relationship("AffiliateOffer", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    content_ideas = relationship("ContentIdea", back_populates="user", cascade="all, delete-orphan")
    keyword_clusters = relationship("KeywordCluster", back_populates="user", cascade="all, delete-orphan")
    external_tool_results = relationship("ExternalToolResult", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role.value}')>"
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            "id": self.id,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "role": self.role.value,
            "subscription_tier": self.subscription_tier.value,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "timezone": self.timezone,
            "language": self.language,
            "subscription_status": self.subscription_status,
            "api_calls_count": self.api_calls_count,
            "last_api_call": self.last_api_call.isoformat() if self.last_api_call else None
        }
    
    def is_admin(self):
        """Check if user is admin"""
        return self.role == UserRole.ADMIN
    
    def is_moderator(self):
        """Check if user is moderator"""
        return self.role == UserRole.MODERATOR
    
    def can_access_feature(self, feature: str) -> bool:
        """Check if user can access a specific feature based on subscription tier"""
        feature_access = {
            "affiliate_research": {
                SubscriptionTier.FREE: 5,  # 5 researches per month
                SubscriptionTier.BASIC: 50,
                SubscriptionTier.PREMIUM: 200,
                SubscriptionTier.ENTERPRISE: -1  # Unlimited
            },
            "trend_analysis": {
                SubscriptionTier.FREE: 10,
                SubscriptionTier.BASIC: 100,
                SubscriptionTier.PREMIUM: 500,
                SubscriptionTier.ENTERPRISE: -1
            },
            "content_generation": {
                SubscriptionTier.FREE: 20,
                SubscriptionTier.BASIC: 200,
                SubscriptionTier.PREMIUM: 1000,
                SubscriptionTier.ENTERPRISE: -1
            },
            "software_generation": {
                SubscriptionTier.FREE: 5,
                SubscriptionTier.BASIC: 50,
                SubscriptionTier.PREMIUM: 200,
                SubscriptionTier.ENTERPRISE: -1
            },
            "export": {
                SubscriptionTier.FREE: 10,
                SubscriptionTier.BASIC: 100,
                SubscriptionTier.PREMIUM: 500,
                SubscriptionTier.ENTERPRISE: -1
            }
        }
        
        if feature not in feature_access:
            return False
        
        limit = feature_access[feature].get(self.subscription_tier, 0)
        return limit == -1 or limit > 0  # -1 means unlimited
    
    def get_feature_limit(self, feature: str) -> int:
        """Get feature limit for user's subscription tier"""
        feature_access = {
            "affiliate_research": {
                SubscriptionTier.FREE: 5,
                SubscriptionTier.BASIC: 50,
                SubscriptionTier.PREMIUM: 200,
                SubscriptionTier.ENTERPRISE: -1
            },
            "trend_analysis": {
                SubscriptionTier.FREE: 10,
                SubscriptionTier.BASIC: 100,
                SubscriptionTier.PREMIUM: 500,
                SubscriptionTier.ENTERPRISE: -1
            },
            "content_generation": {
                SubscriptionTier.FREE: 20,
                SubscriptionTier.BASIC: 200,
                SubscriptionTier.PREMIUM: 1000,
                SubscriptionTier.ENTERPRISE: -1
            },
            "software_generation": {
                SubscriptionTier.FREE: 5,
                SubscriptionTier.BASIC: 50,
                SubscriptionTier.PREMIUM: 200,
                SubscriptionTier.ENTERPRISE: -1
            },
            "export": {
                SubscriptionTier.FREE: 10,
                SubscriptionTier.BASIC: 100,
                SubscriptionTier.PREMIUM: 500,
                SubscriptionTier.ENTERPRISE: -1
            }
        }
        
        if feature not in feature_access:
            return 0
        
        return feature_access[feature].get(self.subscription_tier, 0)
    
    # Relationships
    export_templates = relationship("ExportTemplate", back_populates="user")