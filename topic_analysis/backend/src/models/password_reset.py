"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
PasswordReset model for password reset functionality.
"""
import uuid
from datetime import datetime

class PasswordReset:
    """Simple data class for PasswordReset - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def create_reset_token(cls, user_id, token, expires_in_hours=1):
        """Create a password reset record"""
        from datetime import timedelta
        return cls(
            user_id=user_id,
            token=token,
            expires_at=datetime.utcnow() + timedelta(hours=expires_in_hours),
            is_used=False
        )
    
    def is_expired(self):
        """Check if reset token is expired"""
        if hasattr(self, 'expires_at'):
            return self.expires_at < datetime.utcnow()
        return False
    
    def mark_as_used(self):
        """Mark reset token as used"""
        self.is_used = True
