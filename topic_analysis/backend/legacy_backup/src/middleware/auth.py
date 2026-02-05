"""
Authentication and Authorization Middleware
Handles JWT token validation, user authentication, and role-based access control
"""

import jwt
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import structlog

from ..core.config import settings
from ..core.database import get_db
from ..core.redis import cache_manager
from ..models.user import User, UserRole

logger = structlog.get_logger()

# JWT Configuration
JWT_SECRET = settings.secret_key
JWT_ALGORITHM = settings.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes

# Security scheme
security = HTTPBearer()

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.secret_key = JWT_SECRET
        self.algorithm = JWT_ALGORITHM
        self.access_token_expire_minutes = ACCESS_TOKEN_EXPIRE_MINUTES
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            # Check if token is blacklisted
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            if cache_manager.is_jwt_blacklisted(token_hash):
                logger.warning("Attempted to use blacklisted token", token_hash=token_hash)
                return None
            
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning("Invalid token", error=str(e))
            return None
    
    def blacklist_token(self, token: str) -> bool:
        """Add token to blacklist"""
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            cache_manager.blacklist_jwt_token(token_hash)
            logger.info("Token blacklisted", token_hash=token_hash)
            return True
        except Exception as e:
            logger.error("Failed to blacklist token", error=str(e))
            return False
    
    def get_user_from_token(self, token: str, db: Session) -> Optional[User]:
        """Get user from JWT token"""
        payload = self.verify_token(token)
        if not payload:
            return None
        
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            return None
        
        return user

# Global auth manager
auth_manager = AuthManager()

# Dependency functions
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    
    user = auth_manager.get_user_from_token(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

# Role-based access control
class RoleChecker:
    """Role-based access control checker"""
    
    def __init__(self, allowed_roles: List[UserRole]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted"
            )
        return current_user

# Permission-based access control
class PermissionChecker:
    """Permission-based access control checker"""
    
    def __init__(self, required_permission: str):
        self.required_permission = required_permission
    
    def __call__(self, current_user: User = Depends(get_current_active_user)) -> User:
        # Check if user has required permission
        user_permissions = current_user.feature_access.get("permissions", [])
        
        if self.required_permission not in user_permissions:
            # Check role-based permissions
            role_permissions = self._get_role_permissions(current_user.role)
            if self.required_permission not in role_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{self.required_permission}' required"
                )
        
        return current_user
    
    def _get_role_permissions(self, role: UserRole) -> List[str]:
        """Get permissions for a role"""
        role_permissions = {
            UserRole.ADMIN: [
                "read:all", "write:all", "delete:all", "admin:all",
                "affiliate:research", "trend:analysis", "keyword:management",
                "content:generation", "software:generation", "export:all",
                "calendar:management", "user:management"
            ],
            UserRole.USER: [
                "read:own", "write:own", "affiliate:research", "trend:analysis",
                "keyword:management", "content:generation", "software:generation",
                "export:basic", "calendar:management"
            ],
            UserRole.USER: [
                "read:own", "write:own", "affiliate:research", "trend:analysis",
                "keyword:management", "content:generation", "export:basic"
            ]
        }
        return role_permissions.get(role, [])

# Resource ownership checker
class ResourceOwnershipChecker:
    """Check if user owns a resource"""
    
    def __init__(self, resource_type: str, resource_id_param: str = "id"):
        self.resource_type = resource_type
        self.resource_id_param = resource_id_param
    
    def __call__(
        self,
        request: Request,
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        # Get resource ID from request
        resource_id = request.path_params.get(self.resource_id_param)
        if not resource_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resource ID not found"
            )
        
        # Check ownership based on resource type
        if not self._check_ownership(current_user, resource_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't own this resource"
            )
        
        return current_user
    
    def _check_ownership(self, user: User, resource_id: str) -> bool:
        """Check if user owns the resource"""
        # Admin users can access all resources
        if user.role == UserRole.ADMIN:
            return True
        
        # For now, implement basic ownership check
        # In a real implementation, you would query the database
        # to check if the resource belongs to the user
        return True

# Subscription tier checker
class SubscriptionTierChecker:
    """Check if user has required subscription tier"""
    
    def __init__(self, required_tier: str):
        self.required_tier = required_tier
    
    def __call__(self, current_user: User = Depends(get_current_active_user)) -> User:
        tier_hierarchy = {
            "free": 0,
            "basic": 1,
            "pro": 2,
            "enterprise": 3
        }
        
        user_tier_level = tier_hierarchy.get(current_user.subscription_tier.value, 0)
        required_tier_level = tier_hierarchy.get(self.required_tier, 0)
        
        if user_tier_level < required_tier_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Subscription tier '{self.required_tier}' required"
            )
        
        return current_user

# Rate limiting checker
class RateLimitChecker:
    """Check if user has exceeded rate limits"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
    
    def __call__(
        self,
        request: Request,
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        endpoint = request.url.path
        user_id = current_user.id
        
        # Check rate limit
        if not cache_manager.check_rate_limit(user_id, endpoint, self.requests_per_minute):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Set rate limit
        cache_manager.set_rate_limit(user_id, endpoint, self.requests_per_minute, 60)
        
        return current_user

# Common dependency combinations
require_admin = RoleChecker([UserRole.ADMIN])
require_premium = RoleChecker([UserRole.ADMIN, UserRole.USER])
require_user = RoleChecker([UserRole.ADMIN, UserRole.USER, UserRole.USER])

# Permission dependencies
require_affiliate_research = PermissionChecker("affiliate:research")
require_trend_analysis = PermissionChecker("trend:analysis")
require_keyword_management = PermissionChecker("keyword:management")
require_content_generation = PermissionChecker("content:generation")
require_software_generation = PermissionChecker("software:generation")
require_export_access = PermissionChecker("export:all")
require_calendar_management = PermissionChecker("calendar:management")
require_user_management = PermissionChecker("user:management")

# Subscription dependencies
require_basic_tier = SubscriptionTierChecker("basic")
require_pro_tier = SubscriptionTierChecker("pro")
require_enterprise_tier = SubscriptionTierChecker("enterprise")

# Rate limiting dependencies
rate_limit_60_per_minute = RateLimitChecker(60)
rate_limit_100_per_minute = RateLimitChecker(100)
rate_limit_200_per_minute = RateLimitChecker(200)

# Utility functions
def get_user_permissions(user: User) -> List[str]:
    """Get all permissions for a user"""
    role_permissions = {
        UserRole.ADMIN: [
            "read:all", "write:all", "delete:all", "admin:all",
            "affiliate:research", "trend:analysis", "keyword:management",
            "content:generation", "software:generation", "export:all",
            "calendar:management", "user:management"
        ],
        UserRole.USER: [
            "read:own", "write:own", "affiliate:research", "trend:analysis",
            "keyword:management", "content:generation", "software:generation",
            "export:basic", "calendar:management"
        ],
        UserRole.USER: [
            "read:own", "write:own", "affiliate:research", "trend:analysis",
            "keyword:management", "content:generation", "export:basic"
        ]
    }
    
    base_permissions = role_permissions.get(user.role, [])
    user_permissions = user.feature_access.get("permissions", [])
    
    return list(set(base_permissions + user_permissions))

def has_permission(user: User, permission: str) -> bool:
    """Check if user has a specific permission"""
    permissions = get_user_permissions(user)
    return permission in permissions

def is_admin(user: User) -> bool:
    """Check if user is admin"""
    return user.role == UserRole.ADMIN

def is_premium_user(user: User) -> bool:
    """Check if user is premium or admin"""
    return user.role in [UserRole.ADMIN, UserRole.USER]

def can_access_resource(user: User, resource_user_id: str) -> bool:
    """Check if user can access a resource"""
    if is_admin(user):
        return True
    return user.id == resource_user_id