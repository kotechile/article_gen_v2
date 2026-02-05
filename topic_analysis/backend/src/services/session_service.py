"""
Session management service for handling user sessions and JWT tokens.
"""
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from src.core.config import settings
from src.core.database import get_db_context
from src.models.user import User
from src.models.user_session import UserSession
from src.services.jwt_service import jwt_service

logger = logging.getLogger(__name__)

class SessionService:
    """Service for managing user sessions and JWT tokens."""
    
    def __init__(self):
        self.session_timeout_hours = settings.session_timeout_hours
        self.jwt_expiration_minutes = settings.jwt_expiration_minutes
        self.jwt_refresh_expiration_days = settings.jwt_refresh_expiration_days
    
    def create_session(
        self, 
        user: User, 
        device_info: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new user session with JWT tokens."""
        try:
            with get_db_context() as db:
                # Generate JWT tokens
                access_token = jwt_service.create_access_token(
                    data={"sub": str(user.id), "email": user.email, "role": user.role.value}
                )
                refresh_token = jwt_service.create_refresh_token(
                    data={"sub": str(user.id)}
                )
                
                # Generate unique token identifiers
                access_token_jti = str(uuid.uuid4())
                refresh_token_jti = str(uuid.uuid4())
                
                # Calculate expiration times
                access_expires = datetime.utcnow() + timedelta(minutes=self.jwt_expiration_minutes)
                refresh_expires = datetime.utcnow() + timedelta(days=self.jwt_refresh_expiration_days)
                
                # Create session record
                session = UserSession(
                    user_id=user.id,
                    token_jti=access_token_jti,
                    refresh_token=refresh_token_jti,
                    device_info=device_info,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    expires_at=refresh_expires
                )
                
                db.add(session)
                db.commit()
                db.refresh(session)
                
                logger.info(f"Session created for user {user.email} with ID {session.id}")
                
                return {
                    "session_id": str(session.id),
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "bearer",
                    "expires_in": self.jwt_expiration_minutes * 60,
                    "refresh_expires_in": self.jwt_refresh_expiration_days * 24 * 60 * 60
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Database error creating session: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise
    
    def validate_session(self, token_jti: str) -> Optional[UserSession]:
        """Validate a session by token JTI."""
        try:
            with get_db_context() as db:
                session = db.get_UserSession_by_id(
                    UserSession.token_jti == token_jti,
                    UserSession.is_active == True
                )
                
                if not session:
                    logger.warning(f"Session not found for token JTI: {token_jti}")
                    return None
                
                if session.is_expired():
                    logger.warning(f"Session expired for token JTI: {token_jti}")
                    session.deactivate()
                    db.commit()
                    return None
                
                # Update last accessed time
                session.update_last_accessed()
                db.commit()
                
                return session
                
        except SQLAlchemyError as e:
            logger.error(f"Database error validating session: {e}")
            return None
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None
    
    def refresh_session(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """Refresh a session using refresh token."""
        try:
            with get_db_context() as db:
                # Find session by refresh token
                session = db.get_UserSession_by_id(
                    UserSession.refresh_token == refresh_token,
                    UserSession.is_active == True
                )
                
                if not session:
                    logger.warning(f"Session not found for refresh token")
                    return None
                
                if session.is_expired():
                    logger.warning(f"Session expired for refresh token")
                    session.deactivate()
                    db.commit()
                    return None
                
                # Get user
                user = db.get_User_by_id(User.id == session.user_id)
                if not user or not user.is_active:
                    logger.warning(f"User not found or inactive for session {session.id}")
                    session.deactivate()
                    db.commit()
                    return None
                
                # Generate new tokens
                new_access_token = jwt_service.create_access_token(
                    data={"sub": str(user.id), "email": user.email, "role": user.role.value}
                )
                new_refresh_token = jwt_service.create_refresh_token(
                    data={"sub": str(user.id)}
                )
                
                # Update session with new tokens
                session.token_jti = str(uuid.uuid4())
                session.refresh_token = str(uuid.uuid4())
                session.update_last_accessed()
                
                db.commit()
                
                logger.info(f"Session refreshed for user {user.email}")
                
                return {
                    "session_id": str(session.id),
                    "access_token": new_access_token,
                    "refresh_token": new_refresh_token,
                    "token_type": "bearer",
                    "expires_in": self.jwt_expiration_minutes * 60,
                    "refresh_expires_in": self.jwt_refresh_expiration_days * 24 * 60 * 60
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Database error refreshing session: {e}")
            return None
        except Exception as e:
            logger.error(f"Error refreshing session: {e}")
            return None
    
    def revoke_session(self, token_jti: str) -> bool:
        """Revoke a session by token JTI."""
        try:
            with get_db_context() as db:
                session = db.get_UserSession_by_id(
                    UserSession.token_jti == token_jti
                )
                
                if not session:
                    logger.warning(f"Session not found for token JTI: {token_jti}")
                    return False
                
                session.deactivate()
                db.commit()
                
                logger.info(f"Session revoked for token JTI: {token_jti}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Database error revoking session: {e}")
            return False
        except Exception as e:
            logger.error(f"Error revoking session: {e}")
            return False
    
    def revoke_user_sessions(self, user_id: uuid.UUID) -> int:
        """Revoke all sessions for a user."""
        try:
            with get_db_context() as db:
                sessions = db.query(UserSession).filter(
                    UserSession.user_id == user_id,
                    UserSession.is_active == True
                ).all()
                
                revoked_count = 0
                for session in sessions:
                    session.deactivate()
                    revoked_count += 1
                
                db.commit()
                
                logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
                return revoked_count
                
        except SQLAlchemyError as e:
            logger.error(f"Database error revoking user sessions: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error revoking user sessions: {e}")
            return 0
    
    def get_user_sessions(self, user_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Get all active sessions for a user."""
        try:
            with get_db_context() as db:
                sessions = db.query(UserSession).filter(
                    UserSession.user_id == user_id,
                    UserSession.is_active == True
                ).order_by(UserSession.last_accessed.desc()).all()
                
                return [
                    {
                        "id": str(session.id),
                        "device_info": session.device_info,
                        "ip_address": session.ip_address,
                        "user_agent": session.user_agent,
                        "created_at": session.created_at.isoformat(),
                        "last_accessed": session.last_accessed.isoformat(),
                        "expires_at": session.expires_at.isoformat(),
                        "is_expired": session.is_expired()
                    }
                    for session in sessions
                ]
                
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user sessions: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            with get_db_context() as db:
                expired_sessions = db.query(UserSession).filter(
                    UserSession.is_active == True,
                    UserSession.expires_at < datetime.utcnow()
                ).all()
                
                cleaned_count = 0
                for session in expired_sessions:
                    session.deactivate()
                    cleaned_count += 1
                
                db.commit()
                
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} expired sessions")
                
                return cleaned_count
                
        except SQLAlchemyError as e:
            logger.error(f"Database error cleaning up expired sessions: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            with get_db_context() as db:
                total_sessions = db.query(UserSession).count()
                active_sessions = db.query(UserSession).filter(
                    UserSession.is_active == True
                ).count()
                expired_sessions = db.query(UserSession).filter(
                    UserSession.is_active == True,
                    UserSession.expires_at < datetime.utcnow()
                ).count()
                
                # Get sessions by user
                user_session_counts = db.query(
                    UserSession.user_id,
                    db.func.count(UserSession.id).label('session_count')
                ).filter(
                    UserSession.is_active == True
                ).group_by(UserSession.user_id).all()
                
                return {
                    "total_sessions": total_sessions,
                    "active_sessions": active_sessions,
                    "expired_sessions": expired_sessions,
                    "users_with_sessions": len(user_session_counts),
                    "average_sessions_per_user": (
                        active_sessions / len(user_session_counts) 
                        if user_session_counts else 0
                    )
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Database error getting session stats: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {"error": str(e)}
    
    def validate_and_get_user(self, token_jti: str) -> Optional[User]:
        """Validate session and return associated user."""
        session = self.validate_session(token_jti)
        if not session:
            return None
        
        try:
            with get_db_context() as db:
                user = db.get_User_by_id(User.id == session.user_id)
                return user if user and user.is_active else None
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user from session: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting user from session: {e}")
            return None

# Global session service instance
session_service = SessionService()
