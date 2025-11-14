"""
API endpoints for Content Generator V2.

This module contains all the REST API endpoints for the system.
"""

from .research import research_bp
from .health import health_bp

__all__ = [
    'research_bp',
    'health_bp'
]
