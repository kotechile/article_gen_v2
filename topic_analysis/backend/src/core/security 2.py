"""
Security utilities and dependencies
"""

from fastapi import HTTPException, Depends
from typing import Optional

def require_admin():
    """Simple admin requirement - for now, always allow"""
    # TODO: Implement proper admin authentication
    return True

def get_current_user():
    """Get current user - requires proper authentication"""
    raise NotImplementedError("Use Supabase authentication instead")

