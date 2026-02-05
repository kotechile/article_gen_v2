"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
UserSession model for JWT session management.
"""
import uuid

class UserSession:
    """Simple data class for UserSession - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def deactivate(self):
        """Deactivate the session"""
        self.is_active = False
    
    def is_expired(self):
        """Check if session is expired"""
        from datetime import datetime
        if hasattr(self, 'expires_at'):
            return self.expires_at < datetime.utcnow()
        return False
