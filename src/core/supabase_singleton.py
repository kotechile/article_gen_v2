"""
Centralized Supabase client singleton for backend operations.

This module provides a single, shared Supabase client instance that:
- Uses Settings class for configuration
- Implements lazy initialization
- Validates required credentials before creating client
- Never falls back to anon key (fails fast if service role key missing)
"""

from supabase import create_client, Client
from typing import Optional
from src.core.config import get_settings
import logging

logger = logging.getLogger(__name__)
_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """
    Get the singleton Supabase client instance.
    
    The client is created lazily on first access and reused for all subsequent calls.
    Uses service role key for backend operations (bypasses RLS).
    
    Returns:
        Client: The Supabase client instance
        
    Raises:
        ValueError: If required Supabase credentials are missing
    """
    global _supabase_client
    if _supabase_client is None:
        settings = get_settings()
        if not settings.supabase_url:
            raise ValueError("SUPABASE_URL environment variable is required")
        if not settings.supabase_service_role_key:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable is required")
        
        _supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key
        )
        logger.info("Supabase client initialized successfully")
    
    return _supabase_client


def reset_supabase_client() -> None:
    """
    Reset the singleton client instance.
    
    This is primarily useful for testing purposes.
    """
    global _supabase_client
    _supabase_client = None
    logger.info("Supabase client reset")
