"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
Idea Burst Session model for managing idea generation sessions
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

# Base = declarative_base()  # Disabled for Supabase-only

class IdeaBurstSession:
    """Simple data class for IdeaBurstSession - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class SelectedIdea:
    """Simple data class for SelectedIdea - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
