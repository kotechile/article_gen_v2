"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
Content opportunity model for SEO content ideas
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

# Base = declarative_base()  # Disabled for Supabase-only

class ContentOpportunity:
    """Simple data class for ContentOpportunity - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
