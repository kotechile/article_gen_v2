"""
SupabaseConnectionManager for connection pooling and session management
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import structlog
import asyncio
from threading import Lock
from contextlib import asynccontextmanager

from ..database.supabase_client import get_supabase_client, test_supabase_connection
from ..models.authentication_context import AuthenticationContext
from ..models.supabase_client import ConnectionStatus
from src.core.supabase_database_service import SupabaseDatabaseService

logger = structlog.get_logger()

class SupabaseConnectionManager:
    """Manager for Supabase connections and sessions"""
    
    def __init__(self):
        self.client = get_supabase_client()
        self.connection_status = ConnectionStatus.CONNECTED
        self.last_health_check = datetime.utcnow()
        self.health_check_interval = timedelta(minutes=5)
        self.connection_lock = Lock()
        self.active_sessions = {}
        self.session_cleanup_interval = timedelta(hours=1)
        self.last_session_cleanup = datetime.utcnow()
        
        # Connection pool settings
        self.max_connections = 10
        self.connection_timeout = 30  # seconds
        self.retry_attempts = 3
        self.retry_delay = 1  # seconds
        
        # Performance metrics
        self.connection_metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "connection_errors": 0,
            "last_connection_time": None,
            "average_connection_time_ms": 0
        }
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status
        
        Returns:
            Dict containing connection status
        """
        try:
            with self.connection_lock:
                return {
                    "status": self.connection_status.value,
                    "last_health_check": self.last_health_check.isoformat(),
                    "active_sessions": len(self.active_sessions),
                    "metrics": self.connection_metrics.copy()
                }
        except Exception as e:
            logger.error("Failed to get connection status", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "last_health_check": self.last_health_check.isoformat()
            }
    
    def check_connection_health(self) -> Dict[str, Any]:
        """
        Check connection health
        
        Returns:
            Dict containing health check results
        """
        try:
            start_time = datetime.utcnow()
            
            # Test connection
            health_result = test_supabase_connection()
            
            # Update connection status
            with self.connection_lock:
                self.last_health_check = datetime.utcnow()
                if health_result["status"] == "healthy":
                    self.connection_status = ConnectionStatus.CONNECTED
                else:
                    self.connection_status = ConnectionStatus.ERROR
                
                # Update metrics
                connection_time = (self.last_health_check - start_time).total_seconds() * 1000
                self.connection_metrics["last_connection_time"] = connection_time
                
                # Calculate average connection time
                if self.connection_metrics["average_connection_time_ms"] == 0:
                    self.connection_metrics["average_connection_time_ms"] = connection_time
                else:
                    self.connection_metrics["average_connection_time_ms"] = (
                        self.connection_metrics["average_connection_time_ms"] + connection_time
                    ) / 2
            
            logger.info("Connection health check completed", 
                       status=health_result["status"],
                       response_time_ms=connection_time)
            
            return health_result
            
        except Exception as e:
            logger.error("Connection health check failed", error=str(e))
            
            with self.connection_lock:
                self.connection_status = ConnectionStatus.ERROR
                self.connection_metrics["connection_errors"] += 1
            
            return {
                "status": "unhealthy",
                "supabase_status": "error",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    def create_session(self, user_id: str, access_token: str, refresh_token: str, 
                      expires_at: datetime) -> str:
        """
        Create a new authentication session
        
        Args:
            user_id: User ID
            access_token: Supabase access token
            refresh_token: Supabase refresh token
            expires_at: Token expiration time
            
        Returns:
            Session ID
        """
        try:
            session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": expires_at,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "last_used": datetime.utcnow()
            }
            
            with self.connection_lock:
                self.active_sessions[session_id] = session_data
            
            logger.info("Session created", 
                       session_id=session_id,
                       user_id=user_id,
                       expires_at=expires_at.isoformat())
            
            return session_id
            
        except Exception as e:
            logger.error("Failed to create session", 
                        user_id=user_id,
                        error=str(e))
            raise
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session by ID
        
        Args:
            session_id: SupabaseDatabaseService ID
            
        Returns:
            Session data or None
        """
        try:
            with self.connection_lock:
                session = self.active_sessions.get(session_id)
                
                if session and session["is_active"]:
                    # Update last used time
                    session["last_used"] = datetime.utcnow()
                    return session.copy()
                
                return None
                
        except Exception as e:
            logger.error("Failed to get session", 
                        session_id=session_id,
                        error=str(e))
            return None
    
    def refresh_session(self, session_id: str, access_token: str, 
                       refresh_token: str, expires_at: datetime) -> bool:
        """
        Refresh session tokens
        
        Args:
            session_id: SupabaseDatabaseService ID
            access_token: New access token
            refresh_token: New refresh token
            expires_at: New expiration time
            
        Returns:
            True if successful
        """
        try:
            with self.connection_lock:
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    session["access_token"] = access_token
                    session["refresh_token"] = refresh_token
                    session["expires_at"] = expires_at
                    session["last_used"] = datetime.utcnow()
                    
                    logger.info("Session refreshed", 
                               session_id=session_id,
                               expires_at=expires_at.isoformat())
                    
                    return True
                
                return False
                
        except Exception as e:
            logger.error("Failed to refresh session", 
                        session_id=session_id,
                        error=str(e))
            return False
    
    def revoke_session(self, session_id: str) -> bool:
        """
        Revoke a session
        
        Args:
            session_id: SupabaseDatabaseService ID
            
        Returns:
            True if successful
        """
        try:
            with self.connection_lock:
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]["is_active"] = False
                    del self.active_sessions[session_id]
                    
                    logger.info("Session revoked", session_id=session_id)
                    return True
                
                return False
                
        except Exception as e:
            logger.error("Failed to revoke session", 
                        session_id=session_id,
                        error=str(e))
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            current_time = datetime.utcnow()
            expired_sessions = []
            
            with self.connection_lock:
                for session_id, session in self.active_sessions.items():
                    if (session["expires_at"] and 
                        session["expires_at"] < current_time):
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
            
            if expired_sessions:
                logger.info("Cleaned up expired sessions", 
                           count=len(expired_sessions))
            
            return len(expired_sessions)
            
        except Exception as e:
            logger.error("Failed to cleanup expired sessions", error=str(e))
            return 0
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all active sessions
        
        Returns:
            List of active session data
        """
        try:
            with self.connection_lock:
                active_sessions = []
                for session_id, session in self.active_sessions.items():
                    if session["is_active"]:
                        session_data = session.copy()
                        session_data["session_id"] = session_id
                        active_sessions.append(session_data)
                
                return active_sessions
                
        except Exception as e:
            logger.error("Failed to get active sessions", error=str(e))
            return []
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get session statistics
        
        Returns:
            Dict containing session statistics
        """
        try:
            with self.connection_lock:
                total_sessions = len(self.active_sessions)
                active_sessions = len([s for s in self.active_sessions.values() if s["is_active"]])
                expired_sessions = 0
                
                current_time = datetime.utcnow()
                for session in self.active_sessions.values():
                    if (session["expires_at"] and 
                        session["expires_at"] < current_time):
                        expired_sessions += 1
                
                return {
                    "total_sessions": total_sessions,
                    "active_sessions": active_sessions,
                    "expired_sessions": expired_sessions,
                    "connection_status": self.connection_status.value,
                    "last_health_check": self.last_health_check.isoformat(),
                    "metrics": self.connection_metrics.copy()
                }
                
        except Exception as e:
            logger.error("Failed to get session statistics", error=str(e))
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "expired_sessions": 0,
                "connection_status": "error",
                "error": str(e)
            }
    
    def should_cleanup_sessions(self) -> bool:
        """
        Check if session cleanup is needed
        
        Returns:
            True if cleanup is needed
        """
        return (datetime.utcnow() - self.last_session_cleanup) > self.session_cleanup_interval
    
    def should_check_health(self) -> bool:
        """
        Check if health check is needed
        
        Returns:
            True if health check is needed
        """
        return (datetime.utcnow() - self.last_health_check) > self.health_check_interval
    
    def update_connection_metrics(self, success: bool, connection_time_ms: float):
        """
        Update connection metrics
        
        Args:
            success: Whether connection was successful
            connection_time_ms: Connection time in milliseconds
        """
        try:
            with self.connection_lock:
                self.connection_metrics["total_connections"] += 1
                
                if success:
                    self.connection_metrics["active_connections"] += 1
                else:
                    self.connection_metrics["failed_connections"] += 1
                
                # Update average connection time
                if self.connection_metrics["average_connection_time_ms"] == 0:
                    self.connection_metrics["average_connection_time_ms"] = connection_time_ms
                else:
                    self.connection_metrics["average_connection_time_ms"] = (
                        self.connection_metrics["average_connection_time_ms"] + connection_time_ms
                    ) / 2
                
        except Exception as e:
            logger.error("Failed to update connection metrics", error=str(e))
    
    @asynccontextmanager
    async def get_managed_connection(self):
        """
        Get a managed connection with automatic cleanup
        
        Yields:
            Supabase client
        """
        connection_start = datetime.utcnow()
        success = False
        
        try:
            # Check if we need to verify connection health
            if self.should_check_health():
                health_result = self.check_connection_health()
                if health_result["status"] != "healthy":
                    raise Exception(f"Connection unhealthy: {health_result.get('error', 'Unknown error')}")
            
            yield self.client
            success = True
            
        except Exception as e:
            logger.error("Managed connection failed", error=str(e))
            raise
        finally:
            # Update metrics
            connection_time = (datetime.utcnow() - connection_start).total_seconds() * 1000
            self.update_connection_metrics(success, connection_time)
            
            # Cleanup expired sessions if needed
            if self.should_cleanup_sessions():
                self.cleanup_expired_sessions()
                self.last_session_cleanup = datetime.utcnow()

