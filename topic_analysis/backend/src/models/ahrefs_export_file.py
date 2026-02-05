"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
Ahrefs Export File model for Ahrefs TSV file processing
"""

from datetime import datetime
from typing import Dict, Any, Optional
import uuid

# Base = declarative_base()  # Disabled for Supabase-only

class AhrefsExportFile:
    """Simple data class for AhrefsExportFile - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
