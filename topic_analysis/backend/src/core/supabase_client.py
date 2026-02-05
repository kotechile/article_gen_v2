"""
Supabase Client Initialization and Management

This module provides centralized Supabase client initialization and configuration
for the trend analysis platform backend, replacing direct PostgreSQL connections
with managed Supabase database operations.

Note: This module now uses the singleton pattern from supabase_singleton.py
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from supabase import Client
from src.core.supabase_singleton import get_supabase_client as get_singleton_client

# Configure logging
logger = logging.getLogger(__name__)

class SupabaseClientManager:
    """
    Manages Supabase client instances with connection pooling and error handling.
    Now uses the centralized singleton for client access.
    """
    
    def __init__(self):
        self._connection_status = "disconnected"
        self._error_count = 0
        self._retry_count = 0
        self._last_used: Optional[datetime] = None
        self._max_retries = 5
        self._max_errors = 10
        
    def get_client(self) -> Client:
        """
        Get Supabase client instance from singleton.
        
        Returns:
            Client: Configured Supabase client
            
        Raises:
            ValueError: If environment variables are missing
            ConnectionError: If client initialization fails
        """
        try:
            client = get_singleton_client()
            self._connection_status = "connected"
            self._error_count = 0
            self._last_used = datetime.utcnow()
            return client
        except Exception as e:
            self._connection_status = "error"
            self._error_count += 1
            if self._error_count >= self._max_errors:
                raise ConnectionError("Max error count exceeded. Client disabled.")
            raise ConnectionError(f"Supabase client access failed: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check Supabase connection health.
        
        Returns:
            Dict containing health status and metrics
        """
        try:
            # Get client from singleton
            client = get_singleton_client()
            
            # Test connection with simple query
            start_time = datetime.utcnow()
            result = client.table("users").select("id").limit(1).execute()
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self._connection_status = "connected"
            self._last_used = datetime.utcnow()
            
            return {
                "status": "healthy",
                "database": "supabase",
                "timestamp": datetime.utcnow().isoformat(),
                "execution_time_ms": round(execution_time, 2),
                "connection_status": self._connection_status,
                "error_count": self._error_count,
                "retry_count": self._retry_count
            }
            
        except Exception as e:
            self._connection_status = "error"
            self._error_count += 1
            logger.error("Health check failed", error=str(e))
            
            return {
                "status": "unhealthy",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "connection_status": self._connection_status,
                "error_count": self._error_count
            }
    
    def reset_connection(self) -> None:
        """
        Reset connection state and attempt reconnection.
        """
        self._client = None
        self._connection_status = "disconnected"
        self._error_count = 0
        self._retry_count = 0
        logger.info("Connection reset requested")
    
    @property
    def connection_status(self) -> str:
        """Get current connection status."""
        return self._connection_status
    
    @property
    def error_count(self) -> int:
        """Get current error count."""
        return self._error_count
    
    @property
    def last_used(self) -> Optional[datetime]:
        """Get last successful operation timestamp."""
        return self._last_used

# Global client manager instance
_client_manager = SupabaseClientManager()

def get_supabase_client() -> Client:
    """
    Get configured Supabase client instance.
    
    This is the main entry point for database operations.
    
    Returns:
        Client: Configured Supabase client
        
    Raises:
        ConnectionError: If client is unavailable
    """
    return _client_manager.get_client()

def get_health_status() -> Dict[str, Any]:
    """
    Get Supabase connection health status.
    
    Returns:
        Dict containing health metrics and status
    """
    return _client_manager.health_check()

def reset_supabase_connection() -> None:
    """
    Reset Supabase connection and attempt reconnection.
    """
    _client_manager.reset_connection()

# Convenience function for backward compatibility
def create_supabase_client() -> Client:
    """
    Get Supabase client instance (uses singleton).
    
    Note: This now uses the singleton pattern. Use get_supabase_client() 
    for the same behavior.
    
    Returns:
        Client: Supabase client instance
    """
    return get_singleton_client()