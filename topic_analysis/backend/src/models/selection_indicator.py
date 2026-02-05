"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
Selection Indicator model for idea selection criteria
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

# Base = declarative_base()  # Disabled for Supabase-only

class SelectionIndicator:
    """Simple data class for SelectionIndicator - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class IndicatorTemplate:
    """Simple data class for IndicatorTemplate - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
