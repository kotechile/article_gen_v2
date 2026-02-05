"""
Supabase Authentication Service
Handles user authentication using Supabase Auth
"""

from typing import Dict, Any, Optional
from supabase import create_client, Client
import structlog
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
from src.core.config import get_settings

logger = structlog.get_logger()

# Get Supabase configuration from Settings
settings = get_settings()

# Validate required auth configuration
if not settings.supabase_url:
    raise ValueError("SUPABASE_URL environment variable is required for authentication")

# Note: Auth operations need anon key, not service role key
# We need to get anon key from environment directly as it's not in Settings
import os
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase_auth: Optional[Client] = None

if not SUPABASE_ANON_KEY:
    logger.warning("SUPABASE_ANON_KEY environment variable not found. Authentication will fail.")
else:
    # Create Supabase client for authentication (uses anon key)
    try:
        supabase_auth = create_client(settings.supabase_url, SUPABASE_ANON_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase Auth client: {e}")

# Security scheme
security = HTTPBearer()

class SupabaseAuthService:
    """Authentication service using Supabase Auth"""
    
    def __init__(self):
        self.client = supabase_auth
        logger.info("Supabase authentication service initialized")
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and get user info"""
        try:
            if not self.client:
                logger.error("Supabase client not initialized (missing SUPABASE_ANON_KEY)")
                return None

            # Verify the token with Supabase
            response = self.client.auth.get_user(token)
            if response.user:
                return {
                    "id": response.user.id,
                    "email": response.user.email,
                    "user_metadata": response.user.user_metadata,
                    "app_metadata": response.user.app_metadata
                }
            return None
        except Exception as e:
            logger.error("Token verification failed", error=str(e))
            return None
    
    def get_user_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Get user information from token"""
        return self.verify_token(token)

# Create global instance
supabase_auth_service = SupabaseAuthService()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user from JWT token"""
    try:
        token = credentials.credentials
        user = supabase_auth_service.get_user_by_token(token)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
    except Exception as e:
        logger.error("Authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """Get current user if authenticated, otherwise return None"""
    try:
        if not credentials:
            return None
        
        token = credentials.credentials
        user = supabase_auth_service.get_user_by_token(token)
        return user
    except Exception as e:
        logger.error("Optional authentication failed", error=str(e))
        return None
