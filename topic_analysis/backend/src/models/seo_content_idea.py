"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
SEO Content Idea model for enhanced content ideas
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

# Base = declarative_base()  # Disabled for Supabase-only

class SEOContentIdea:
    """Simple data class for SEOContentIdea - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
