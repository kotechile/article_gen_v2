"""
AuthenticationService

This module provides service layer for user authentication and session management
using Supabase Auth.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..core.logging import db_operation_logger
from ..core.error_handler import safe_database_operation
from ..models.auth_context import AuthenticationContext
from .supabase_service import SupabaseService
from src.core.supabase_database_service import SupabaseDatabaseService

class AuthenticationService:
    """
    Service for managing user authentication and sessions.
    
    This service handles user login, logout, token management,
    and session tracking using Supabase Auth.
    """
    
    def __init__(self):
        """Initialize the authentication service."""
        self.supabase_service = SupabaseService()
        self.logger = db_operation_logger
    
    def create_session(self, user_id: str, access_token: str, refresh_token: str,
                      expires_at: datetime, ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None,
                      permissions: Optional[Dict[str, Any]] = None) -> AuthenticationContext:
        """
        Create a new authentication session.
        
        Args:
            user_id: Supabase user ID
            access_token: JWT access token
            refresh_token: JWT refresh token
            expires_at: Token expiration timestamp
            ip_address: Client IP address
            user_agent: Client user agent
            permissions: User permissions and roles
            
        Returns:
            AuthenticationContext instance
        """
        session = AuthenticationContext.create_session(
            user_id=user_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=permissions
        )
        
        # Store session in database
        session_data = session.to_dict()
        created_session = self.supabase_service.create(
            table_name="auth_contexts",
            data=session_data,
            user_id=user_id
        )
        
        return AuthenticationContext(**created_session)
    
    def get_session(self, session_id: str) -> Optional[AuthenticationContext]:
        """
        Get an authentication session by ID.
        
        Args:
            session_id: SupabaseDatabaseService identifier
            
        Returns:
            AuthenticationContext instance or None
        """
        try:
            sessions = self.supabase_service.read(
                table_name="auth_contexts",
                filters={"session_id": session_id},
                limit=1
            )
            
            if sessions:
                return AuthenticationContext(**sessions[0])
            return None
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id=session_id,
                error_message=str(e),
                error_type="get_session_error"
            )
            return None
    
    def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[AuthenticationContext]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: User identifier
            active_only: Whether to return only active sessions
            
        Returns:
            List of AuthenticationContext instances
        """
        try:
            filters = {"user_id": user_id}
            if active_only:
                filters["is_active"] = True
            
            sessions = self.supabase_service.read(
                table_name="auth_contexts",
                filters=filters,
                order_by="last_activity",
                order_desc=True
            )
            
            return [AuthenticationContext(**session) for session in sessions]
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id="user_sessions",
                error_message=str(e),
                error_type="get_user_sessions_error"
            )
            return []
    
    def update_session_activity(self, session_id: str, ip_address: Optional[str] = None,
                               user_agent: Optional[str] = None) -> bool:
        """
        Update session activity timestamp.
        
        Args:
            session_id: SupabaseDatabaseService identifier
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            True if update was successful
        """
        try:
            update_data = {
                "last_activity": datetime.utcnow().isoformat()
            }
            
            if ip_address:
                update_data["ip_address"] = ip_address
            if user_agent:
                update_data["user_agent"] = user_agent
            
            result = self.supabase_service.update(
                table_name="auth_contexts",
                record_id=session_id,
                data=update_data
            )
            
            return result is not None
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id=session_id,
                error_message=str(e),
                error_type="update_activity_error"
            )
            return False
    
    def refresh_tokens(self, session_id: str, new_access_token: str,
                      new_refresh_token: str, expires_at: datetime) -> bool:
        """
        Refresh authentication tokens.
        
        Args:
            session_id: SupabaseDatabaseService identifier
            new_access_token: New access token
            new_refresh_token: New refresh token
            expires_at: New expiration timestamp
            
        Returns:
            True if refresh was successful
        """
        try:
            update_data = {
                "access_token": new_access_token,
                "refresh_token": new_refresh_token,
                "expires_at": expires_at.isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            }
            
            result = self.supabase_service.update(
                table_name="auth_contexts",
                record_id=session_id,
                data=update_data
            )
            
            return result is not None
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id=session_id,
                error_message=str(e),
                error_type="refresh_tokens_error"
            )
            return False
    
    def logout_session(self, session_id: str) -> bool:
        """
        Logout a specific session.
        
        Args:
            session_id: SupabaseDatabaseService identifier
            
        Returns:
            True if logout was successful
        """
        try:
            update_data = {
                "is_active": False,
                "last_activity": datetime.utcnow().isoformat()
            }
            
            result = self.supabase_service.update(
                table_name="auth_contexts",
                record_id=session_id,
                data=update_data
            )
            
            return result is not None
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id=session_id,
                error_message=str(e),
                error_type="logout_error"
            )
            return False
    
    def logout_user(self, user_id: str) -> int:
        """
        Logout all sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of sessions logged out
        """
        try:
            # Get all active sessions for user
            sessions = self.get_user_sessions(user_id, active_only=True)
            
            logged_out_count = 0
            for session in sessions:
                if self.logout_session(session.session_id):
                    logged_out_count += 1
            
            return logged_out_count
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id="logout_user",
                error_message=str(e),
                error_type="logout_user_error"
            )
            return 0
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            current_time = datetime.utcnow().isoformat()
            
            # Get expired sessions
            expired_sessions = self.supabase_service.read(
                table_name="auth_contexts",
                filters={
                    "is_active": True,
                    "expires_at": current_time
                },
                limit=1000
            )
            
            # Deactivate expired sessions
            cleaned_count = 0
            for session in expired_sessions:
                try:
                    self.supabase_service.update(
                        table_name="auth_contexts",
                        record_id=session["id"],
                        data={"is_active": False}
                    )
                    cleaned_count += 1
                except Exception:
                    continue
            
            return cleaned_count
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id="cleanup_sessions",
                error_message=str(e),
                error_type="cleanup_sessions_error"
            )
            return 0
    
    def get_session_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get session metrics for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dict containing session metrics
        """
        try:
            # Get sessions from the last N hours
            cutoff_time = datetime.utcnow().replace(hour=datetime.utcnow().hour - hours)
            
            sessions = self.supabase_service.read(
                table_name="auth_contexts",
                filters={"created_at": cutoff_time.isoformat()},
                limit=1000
            )
            
            if not sessions:
                return {
                    "total_sessions": 0,
                    "active_sessions": 0,
                    "expired_sessions": 0,
                    "sessions_by_user": {}
                }
            
            # Calculate metrics
            total_sessions = len(sessions)
            active_sessions = len([s for s in sessions if s.get("is_active")])
            expired_sessions = len([s for s in sessions if s.get("expires_at") < datetime.utcnow().isoformat()])
            
            # Group by user
            sessions_by_user = {}
            for session in sessions:
                user_id = session.get("user_id", "unknown")
                sessions_by_user[user_id] = sessions_by_user.get(user_id, 0) + 1
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "expired_sessions": expired_sessions,
                "sessions_by_user": sessions_by_user
            }
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id="session_metrics",
                error_message=str(e),
                error_type="get_session_metrics_error"
            )
            return {}
    
    def validate_session(self, session_id: str) -> bool:
        """
        Validate if a session is still valid.
        
        Args:
            session_id: SupabaseDatabaseService identifier
            
        Returns:
            True if session is valid
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            return session.is_valid()
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id=session_id,
                error_message=str(e),
                error_type="validate_session_error"
            )
            return False