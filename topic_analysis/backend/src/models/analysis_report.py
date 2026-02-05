"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
Analysis report model for keyword analysis results
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid

# Base = declarative_base()  # Disabled for Supabase-only

class KeywordAnalysisReport:
    """Simple data class for KeywordAnalysisReport - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
