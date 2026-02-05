"""
UserService for user management and authentication
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog
from ..core.database import get_db
from ..core.redis import cache
from ..core.config import get_settings
from ..models.user import User, UserRole, SubscriptionTier
from ..models.affiliate_research import AffiliateResearch
from ..models.trend_analysis import TrendAnalysis
from ..models.content_ideas import ContentIdeas
from ..models.software_solutions import SoftwareSolutions
from ..models.content_calendar import ContentCalendar

logger = structlog.get_logger()
settings = get_settings()

class UserService:
    """Service for user management and authentication"""
    
    def __init__(self):
        self.jwt_secret = settings.jwt_secret
        self.jwt_algorithm = settings.jwt_algorithm
        self.jwt_expiration_hours = settings.jwt_expiration_hours
        
        # User limits by subscription tier
        self.subscription_limits = {
            SubscriptionTier.FREE: {
                "affiliate_researches": 5,
                "trend_analyses": 10,
                "content_ideas": 20,
                "software_solutions": 10,
                "calendar_entries": 50
            },
            SubscriptionTier.BASIC: {
                "affiliate_researches": 25,
                "trend_analyses": 50,
                "content_ideas": 100,
                "software_solutions": 50,
                "calendar_entries": 250
            },
            SubscriptionTier.PRO: {
                "affiliate_researches": 100,
                "trend_analyses": 200,
                "content_ideas": 500,
                "software_solutions": 200,
                "calendar_entries": 1000
            },
            SubscriptionTier.ENTERPRISE: {
                "affiliate_researches": -1,  # Unlimited
                "trend_analyses": -1,
                "content_ideas": -1,
                "software_solutions": -1,
                "calendar_entries": -1
            }
        }
    
    async def create_user(self, username: str, email: str, password: str, 
                         subscription_tier: SubscriptionTier = SubscriptionTier.FREE) -> Dict[str, Any]:
        """Create new user"""
        try:
            db = next(get_db())
            
            # Check if user already exists
            existing_user = db.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                raise ValueError("User with this username or email already exists")
            
            # Hash password
            hashed_password = self._hash_password(password)
            
            # Create user
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                subscription_tier=subscription_tier,
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            # Generate JWT token
            token = self._generate_jwt_token(user)
            
            logger.info("User created", user_id=user.id, username=username, email=email)
            
            return {
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "subscription_tier": user.subscription_tier.value,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "token": token
            }
            
        except Exception as e:
            logger.error("Failed to create user", username=username, email=email, error=str(e))
            raise
    
    async def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user"""
        try:
            db = next(get_db())
            
            # Find user by username or email
            user = db.query(User).filter(
                (User.username == username) | (User.email == username)
            ).first()
            
            if not user or not self._verify_password(password, user.hashed_password):
                raise ValueError("Invalid credentials")
            
            if not user.is_active:
                raise ValueError("User account is disabled")
            
            # Generate JWT token
            token = self._generate_jwt_token(user)
            
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            
            logger.info("User authenticated", user_id=user.id, username=user.username)
            
            return {
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "subscription_tier": user.subscription_tier.value,
                "is_active": user.is_active,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "token": token
            }
            
        except Exception as e:
            logger.error("Failed to authenticate user", username=username, error=str(e))
            raise
    
    async def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Get user profile"""
        try:
            db = next(get_db())
            user = db.get_User_by_id(User.id == user_id)
            
            if not user:
                raise ValueError("User not found")
            
            # Get user statistics
            stats = await self._get_user_statistics(user_id)
            
            # Get subscription limits
            limits = self.subscription_limits.get(user.subscription_tier, {})
            
            return {
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "subscription_tier": user.subscription_tier.value,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "statistics": stats,
                "limits": limits
            }
            
        except Exception as e:
            logger.error("Failed to get user profile", user_id=user_id, error=str(e))
            raise
    
    async def update_user_profile(self, user_id: int, username: Optional[str] = None,
                                email: Optional[str] = None, password: Optional[str] = None) -> bool:
        """Update user profile"""
        try:
            db = next(get_db())
            user = db.get_User_by_id(User.id == user_id)
            
            if not user:
                raise ValueError("User not found")
            
            # Update fields
            if username:
                # Check if username is already taken
                existing_user = db.get_User_by_id(
                    User.username == username,
                    User.id != user_id
                )
                if existing_user:
                    raise ValueError("Username already taken")
                user.username = username
            
            if email:
                # Check if email is already taken
                existing_user = db.get_User_by_id(
                    User.email == email,
                    User.id != user_id
                )
                if existing_user:
                    raise ValueError("Email already taken")
                user.email = email
            
            if password:
                user.hashed_password = self._hash_password(password)
            
            user.updated_at = datetime.utcnow()
            
            db.commit()
            
            logger.info("User profile updated", user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("Failed to update user profile", user_id=user_id, error=str(e))
            raise
    
    async def upgrade_subscription(self, user_id: int, new_tier: SubscriptionTier) -> bool:
        """Upgrade user subscription"""
        try:
            db = next(get_db())
            user = db.get_User_by_id(User.id == user_id)
            
            if not user:
                raise ValueError("User not found")
            
            # Check if upgrade is valid
            current_tier_value = self._get_tier_value(user.subscription_tier)
            new_tier_value = self._get_tier_value(new_tier)
            
            if new_tier_value <= current_tier_value:
                raise ValueError("New subscription tier must be higher than current tier")
            
            # Update subscription
            user.subscription_tier = new_tier
            user.updated_at = datetime.utcnow()
            
            db.commit()
            
            logger.info("Subscription upgraded", user_id=user_id, new_tier=new_tier.value)
            return True
            
        except Exception as e:
            logger.error("Failed to upgrade subscription", user_id=user_id, error=str(e))
            raise
    
    async def deactivate_user(self, user_id: int) -> bool:
        """Deactivate user account"""
        try:
            db = next(get_db())
            user = db.get_User_by_id(User.id == user_id)
            
            if not user:
                raise ValueError("User not found")
            
            user.is_active = False
            user.updated_at = datetime.utcnow()
            
            db.commit()
            
            logger.info("User deactivated", user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("Failed to deactivate user", user_id=user_id, error=str(e))
            raise
    
    async def check_user_limits(self, user_id: int, resource_type: str) -> bool:
        """Check if user has reached limits for resource type"""
        try:
            db = next(get_db())
            user = db.get_User_by_id(User.id == user_id)
            
            if not user:
                raise ValueError("User not found")
            
            # Get limits for user's subscription tier
            limits = self.subscription_limits.get(user.subscription_tier, {})
            limit = limits.get(resource_type, -1)
            
            # Unlimited access
            if limit == -1:
                return True
            
            # Check current usage
            current_count = await self._get_resource_count(user_id, resource_type)
            
            return current_count < limit
            
        except Exception as e:
            logger.error("Failed to check user limits", user_id=user_id, resource_type=resource_type, error=str(e))
            raise
    
    async def get_user_dashboard(self, user_id: int) -> Dict[str, Any]:
        """Get user dashboard data"""
        try:
            db = next(get_db())
            user = db.get_User_by_id(User.id == user_id)
            
            if not user:
                raise ValueError("User not found")
            
            # Get recent activity
            recent_activity = await self._get_recent_activity(user_id)
            
            # Get statistics
            statistics = await self._get_user_statistics(user_id)
            
            # Get upcoming reminders
            upcoming_reminders = await self._get_upcoming_reminders(user_id)
            
            # Get subscription info
            subscription_info = {
                "tier": user.subscription_tier.value,
                "limits": self.subscription_limits.get(user.subscription_tier, {}),
                "usage": await self._get_current_usage(user_id)
            }
            
            return {
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "subscription_tier": user.subscription_tier.value,
                    "is_active": user.is_active
                },
                "recent_activity": recent_activity,
                "statistics": statistics,
                "upcoming_reminders": upcoming_reminders,
                "subscription": subscription_info
            }
            
        except Exception as e:
            logger.error("Failed to get user dashboard", user_id=user_id, error=str(e))
            raise
    
    async def _get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """Get user statistics"""
        try:
            db = next(get_db())
            
            # Count different resource types
            affiliate_researches_count = db.query(AffiliateResearch).filter(AffiliateResearch.user_id == user_id).count()
            trend_analyses_count = db.query(TrendAnalysis).filter(TrendAnalysis.user_id == user_id).count()
            content_ideas_count = db.query(ContentIdeas).filter(ContentIdeas.user_id == user_id).count()
            software_solutions_count = db.query(SoftwareSolutions).filter(SoftwareSolutions.user_id == user_id).count()
            calendar_entries_count = db.query(ContentCalendar).filter(ContentCalendar.user_id == user_id).count()
            
            return {
                "affiliate_researches": affiliate_researches_count,
                "trend_analyses": trend_analyses_count,
                "content_ideas": content_ideas_count,
                "software_solutions": software_solutions_count,
                "calendar_entries": calendar_entries_count,
                "total_activities": affiliate_researches_count + trend_analyses_count + content_ideas_count + software_solutions_count + calendar_entries_count
            }
            
        except Exception as e:
            logger.error("Failed to get user statistics", user_id=user_id, error=str(e))
            raise
    
    async def _get_recent_activity(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent user activity"""
        try:
            db = next(get_db())
            
            # Get recent affiliate researches
            recent_researches = db.query(AffiliateResearch).filter(
                AffiliateResearch.user_id == user_id
            ).order_by(AffiliateResearch.created_at.desc()).limit(limit).all()
            
            # Get recent trend analyses
            recent_analyses = db.query(TrendAnalysis).filter(
                TrendAnalysis.user_id == user_id
            ).order_by(TrendAnalysis.created_at.desc()).limit(limit).all()
            
            # Get recent content ideas
            recent_content = db.query(ContentIdeas).filter(
                ContentIdeas.user_id == user_id
            ).order_by(ContentIdeas.created_at.desc()).limit(limit).all()
            
            # Get recent software solutions
            recent_software = db.query(SoftwareSolutions).filter(
                SoftwareSolutions.user_id == user_id
            ).order_by(SoftwareSolutions.created_at.desc()).limit(limit).all()
            
            # Combine and sort by date
            activities = []
            
            for research in recent_researches:
                activities.append({
                    "type": "affiliate_research",
                    "id": research.id,
                    "title": f"Research: {research.niche}",
                    "created_at": research.created_at.isoformat(),
                    "status": research.status.value
                })
            
            for analysis in recent_analyses:
                activities.append({
                    "type": "trend_analysis",
                    "id": analysis.id,
                    "title": f"Analysis: {analysis.keyword}",
                    "created_at": analysis.created_at.isoformat(),
                    "status": analysis.status.value
                })
            
            for content in recent_content:
                activities.append({
                    "type": "content_idea",
                    "id": content.id,
                    "title": f"Content: {content.title or 'Untitled'}",
                    "created_at": content.created_at.isoformat(),
                    "status": content.status.value
                })
            
            for software in recent_software:
                activities.append({
                    "type": "software_solution",
                    "id": software.id,
                    "title": f"Software: {software.name or 'Untitled'}",
                    "created_at": software.created_at.isoformat(),
                    "status": software.status.value
                })
            
            # Sort by created_at and limit
            activities.sort(key=lambda x: x["created_at"], reverse=True)
            return activities[:limit]
            
        except Exception as e:
            logger.error("Failed to get recent activity", user_id=user_id, error=str(e))
            raise
    
    async def _get_upcoming_reminders(self, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Get upcoming reminders"""
        try:
            db = next(get_db())
            now = datetime.utcnow()
            reminder_time = now + timedelta(days=7)  # Next 7 days
            
            # Get upcoming calendar entries
            upcoming_entries = db.query(ContentCalendar).filter(
                ContentCalendar.user_id == user_id,
                ContentCalendar.scheduled_date >= now,
                ContentCalendar.scheduled_date <= reminder_time,
                ContentCalendar.status == CalendarStatus.SCHEDULED
            ).order_by(ContentCalendar.scheduled_date).limit(limit).all()
            
            reminders = []
            for entry in upcoming_entries:
                reminders.append({
                    "id": entry.id,
                    "type": entry.item_type.value,
                    "title": entry.title or "Scheduled Item",
                    "scheduled_date": entry.scheduled_date.isoformat(),
                    "notes": entry.notes
                })
            
            return reminders
            
        except Exception as e:
            logger.error("Failed to get upcoming reminders", user_id=user_id, error=str(e))
            raise
    
    async def _get_current_usage(self, user_id: int) -> Dict[str, int]:
        """Get current usage for all resource types"""
        try:
            db = next(get_db())
            
            return {
                "affiliate_researches": db.query(AffiliateResearch).filter(AffiliateResearch.user_id == user_id).count(),
                "trend_analyses": db.query(TrendAnalysis).filter(TrendAnalysis.user_id == user_id).count(),
                "content_ideas": db.query(ContentIdeas).filter(ContentIdeas.user_id == user_id).count(),
                "software_solutions": db.query(SoftwareSolutions).filter(SoftwareSolutions.user_id == user_id).count(),
                "calendar_entries": db.query(ContentCalendar).filter(ContentCalendar.user_id == user_id).count()
            }
            
        except Exception as e:
            logger.error("Failed to get current usage", user_id=user_id, error=str(e))
            raise
    
    async def _get_resource_count(self, user_id: int, resource_type: str) -> int:
        """Get count of specific resource type for user"""
        try:
            db = next(get_db())
            
            if resource_type == "affiliate_researches":
                return db.query(AffiliateResearch).filter(AffiliateResearch.user_id == user_id).count()
            elif resource_type == "trend_analyses":
                return db.query(TrendAnalysis).filter(TrendAnalysis.user_id == user_id).count()
            elif resource_type == "content_ideas":
                return db.query(ContentIdeas).filter(ContentIdeas.user_id == user_id).count()
            elif resource_type == "software_solutions":
                return db.query(SoftwareSolutions).filter(SoftwareSolutions.user_id == user_id).count()
            elif resource_type == "calendar_entries":
                return db.query(ContentCalendar).filter(ContentCalendar.user_id == user_id).count()
            else:
                return 0
                
        except Exception as e:
            logger.error("Failed to get resource count", user_id=user_id, resource_type=resource_type, error=str(e))
            raise
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def _generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user"""
        import jwt
        from datetime import datetime, timedelta
        
        payload = {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "subscription_tier": user.subscription_tier.value,
            "exp": datetime.utcnow() + timedelta(hours=self.jwt_expiration_hours)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _get_tier_value(self, tier: SubscriptionTier) -> int:
        """Get numeric value for subscription tier"""
        tier_values = {
            SubscriptionTier.FREE: 0,
            SubscriptionTier.BASIC: 1,
            SubscriptionTier.PRO: 2,
            SubscriptionTier.ENTERPRISE: 3
        }
        return tier_values.get(tier, 0)