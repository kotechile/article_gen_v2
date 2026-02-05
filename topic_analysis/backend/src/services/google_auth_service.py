"""
Google OAuth Authentication Service

This service handles Google OAuth authentication integration with Supabase Auth.
It manages user creation, profile updates, and session management for Google-authenticated users.
"""

import structlog
from typing import Dict, Any, Optional
from datetime import datetime
from ..core.supabase_database import get_supabase_db

logger = structlog.get_logger()

class GoogleAuthService:
    """Service for handling Google OAuth authentication"""
    
    def __init__(self):
        self.db = get_supabase_db()
    
    async def create_or_update_user_from_google(self, google_user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update user from Google OAuth data
        
        Args:
            google_user_data: User data from Google OAuth
            
        Returns:
            Dict containing user information
        """
        try:
            user_id = google_user_data.get('id')
            email = google_user_data.get('email')
            name = google_user_data.get('name', '')
            avatar_url = google_user_data.get('picture', '')
            
            if not user_id or not email:
                raise ValueError("Missing required user data from Google")
            
            # Check if user already exists
            existing_user = await self._get_user_by_google_id(user_id)
            
            if existing_user:
                # Update existing user
                updated_user = await self._update_user_profile(
                    existing_user['id'],
                    {
                        'name': name,
                        'avatar_url': avatar_url,
                        'last_login': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat()
                    }
                )
                logger.info("Updated existing Google user", user_id=user_id, email=email)
                return updated_user
            else:
                # Create new user
                new_user = await self._create_google_user({
                    'google_id': user_id,
                    'email': email,
                    'name': name,
                    'avatar_url': avatar_url,
                    'provider': 'google',
                    'is_active': True,
                    'role': 'user',
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat()
                })
                logger.info("Created new Google user", user_id=user_id, email=email)
                return new_user
                
        except Exception as e:
            logger.error("Error creating/updating Google user", error=str(e), user_data=google_user_data)
            raise
    
    async def _get_user_by_google_id(self, google_id: str) -> Optional[Dict[str, Any]]:
        """Get user by Google ID"""
        try:
            # This would query the users table for a user with the given google_id
            # For now, we'll use a simple approach since Supabase handles the user creation
            return None
        except Exception as e:
            logger.error("Error getting user by Google ID", google_id=google_id, error=str(e))
            return None
    
    async def _create_google_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Google user"""
        try:
            # Supabase Auth automatically creates users, so we just return the data
            # In a real implementation, you might want to store additional user data
            return {
                'id': user_data.get('google_id'),
                'email': user_data.get('email'),
                'name': user_data.get('name'),
                'avatar_url': user_data.get('avatar_url'),
                'provider': 'google',
                'is_active': True,
                'role': 'user',
                'created_at': user_data.get('created_at'),
                'updated_at': user_data.get('updated_at')
            }
        except Exception as e:
            logger.error("Error creating Google user", error=str(e), user_data=user_data)
            raise
    
    async def _update_user_profile(self, user_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile"""
        try:
            # In a real implementation, you would update the user in the database
            # For now, we'll just return the update data
            return {
                'id': user_id,
                **update_data
            }
        except Exception as e:
            logger.error("Error updating user profile", user_id=user_id, error=str(e))
            raise
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by ID"""
        try:
            # This would query the users table
            # For now, return a mock response
            return {
                'id': user_id,
                'email': 'user@example.com',
                'name': 'Google User',
                'avatar_url': '',
                'provider': 'google',
                'is_active': True,
                'role': 'user'
            }
        except Exception as e:
            logger.error("Error getting user profile", user_id=user_id, error=str(e))
            return None
    
    async def validate_google_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate Google OAuth token
        
        Args:
            token: Google OAuth token
            
        Returns:
            User data if token is valid, None otherwise
        """
        try:
            # In a real implementation, you would validate the token with Google
            # For now, we'll return a mock response
            return {
                'id': 'google_user_123',
                'email': 'user@example.com',
                'name': 'Google User',
                'picture': 'https://example.com/avatar.jpg',
                'verified_email': True
            }
        except Exception as e:
            logger.error("Error validating Google token", error=str(e))
            return None

# Global service instance
google_auth_service = GoogleAuthService()

def get_google_auth_service() -> GoogleAuthService:
    """Get Google Auth service instance"""
    return google_auth_service

