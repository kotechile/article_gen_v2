"""
Authentication service for user registration, login, logout, and session management.
"""
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
import logging

from src.models.user import User, UserRole
from src.models.user_session import UserSession
from src.models.password_reset import PasswordReset
from src.models.authentication_log import AuthenticationLog, AuthenticationEventType
from src.schemas.auth_schemas import (
    UserRegistrationRequest, LoginRequest, PasswordResetRequest, 
    PasswordResetConfirmRequest, TokenData
)
from src.schemas.user_schemas import UserResponse, ChangePasswordRequest
from src.services.password_service import password_service
from src.services.jwt_service import jwt_service
from src.services.email_service import email_service
from src.core.supabase_database_service import SupabaseDatabaseService

logger = logging.getLogger(__name__)

class AuthenticationService:
    """Service for authentication operations."""
    
    def __init__(self):
        self.password_service = password_service
        self.jwt_service = jwt_service
        self.email_service = email_service
    
    def register_user(self, db: SupabaseDatabaseService, user_data: UserRegistrationRequest, 
                     ip_address: str = None, user_agent: str = None) -> Tuple[bool, str, Optional[UserResponse]]:
        """Register a new user."""
        try:
            # Check if user already exists
            existing_user = db.get_User_by_id(User.email == user_data.email)
            if existing_user:
                return False, "User with this email already exists", None
            
            # Validate password strength
            is_valid, message = self.password_service.validate_password_strength(user_data.password)
            if not is_valid:
                return False, message, None
            
            # Hash password
            hashed_password = self.password_service.hash_password(user_data.password)
            
            # Generate email verification token
            verification_token = self.email_service.generate_verification_token()
            verification_expires = datetime.utcnow() + timedelta(hours=24)
            
            # Create user
            user = User(
                email=user_data.email,
                password_hash=hashed_password,
                first_name=user_data.first_name,
                last_name=user_data.last_name,
                role=UserRole.USER,
                email_verification_token=verification_token,
                email_verification_expires=verification_expires
            )
            
            db.add(user)
            db.flush()  # Flush to get the user ID
            
            # Log registration event
            auth_log = AuthenticationLog.log_event(
                event_type=AuthenticationEventType.LOGIN_SUCCESS,  # Registration is a success event
                success=True,
                user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent,
                event_metadata={"action": "user_registration"}
            )
            db.add(auth_log)
            
            # Send verification email
            email_sent = self.email_service.send_verification_email(
                user.email, 
                verification_token, 
                user.full_name
            )
            
            if not email_sent:
                logger.warning(f"Failed to send verification email to {user.email}")
            
            db.commit()
            
            # Return user response (without sensitive data)
            user_response = UserResponse.from_orm(user)
            return True, "User registered successfully. Please check your email for verification.", user_response
            
        except IntegrityError:
            db.rollback()
            return False, "User with this email already exists", None
        except Exception as e:
            db.rollback()
            logger.error(f"Registration failed: {e}")
            return False, "Registration failed", None
    
    def verify_email(self, db: SupabaseDatabaseService, token: str, ip_address: str = None, user_agent: str = None) -> Tuple[bool, str]:
        """Verify user email with token."""
        try:
            # Find user by verification token
            user = db.query(User).filter(
                User.email_verification_token == token,
                User.email_verification_expires > datetime.utcnow()
            ).first()
            
            if not user:
                return False, "Invalid or expired verification token"
            
            # Update user as verified
            user.is_verified = True
            user.email_verification_token = None
            user.email_verification_expires = None
            
            # Log verification event
            auth_log = AuthenticationLog.log_email_verified(
                user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            db.add(auth_log)
            
            db.commit()
            
            # Send welcome email
            self.email_service.send_welcome_email(user.email, user.full_name)
            
            return True, "Email verified successfully"
            
        except Exception as e:
            db.rollback()
            logger.error(f"Email verification failed: {e}")
            return False, "Email verification failed"
    
    def login_user(self, db: SupabaseDatabaseService, login_data: LoginRequest, 
                  ip_address: str = None, user_agent: str = None, 
                  device_info: Dict[str, Any] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Authenticate user login."""
        try:
            # Find user by email
            user = db.get_User_by_id(User.email == login_data.email)
            
            if not user:
                # Log failed attempt
                auth_log = AuthenticationLog.log_login_failed(
                    email=login_data.email,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    error_message="User not found"
                )
                db.add(auth_log)
                db.commit()
                return False, "Invalid credentials", None
            
            # Check if user is active
            if not user.is_active:
                auth_log = AuthenticationLog.log_login_failed(
                    email=login_data.email,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    error_message="Account deactivated"
                )
                db.add(auth_log)
                db.commit()
                return False, "Account is deactivated", None
            
            # Check if account is locked
            if user.is_locked():
                auth_log = AuthenticationLog.log_login_failed(
                    email=login_data.email,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    error_message="Account locked"
                )
                db.add(auth_log)
                db.commit()
                return False, "Account is temporarily locked due to too many failed attempts", None
            
            # Verify password
            if not self.password_service.verify_password(login_data.password, user.password_hash):
                # Increment failed attempts
                user.increment_failed_attempts()
                
                # Log failed attempt
                auth_log = AuthenticationLog.log_login_failed(
                    email=login_data.email,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    error_message="Invalid password"
                )
                db.add(auth_log)
                db.commit()
                return False, "Invalid credentials", None
            
            # Reset failed attempts on successful login
            user.reset_failed_attempts()
            user.last_login = datetime.utcnow()
            
            # Create access and refresh tokens
            access_token = self.jwt_service.create_access_token(
                str(user.id), user.email, user.role.value
            )
            refresh_token = self.jwt_service.create_refresh_token(str(user.id))
            
            # Create user session
            session = UserSession(
                user_id=user.id,
                token_jti=self.jwt_service.extract_jti(access_token),
                refresh_token=refresh_token,
                device_info=device_info,
                ip_address=ip_address,
                user_agent=user_agent,
                expires_at=datetime.utcnow() + timedelta(days=7)  # 7 days for refresh token
            )
            db.add(session)
            
            # Log successful login
            auth_log = AuthenticationLog.log_login_success(
                user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent,
                device_info=device_info
            )
            db.add(auth_log)
            
            db.commit()
            
            # Prepare response
            user_response = UserResponse.from_orm(user)
            response_data = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": self.jwt_service.access_token_expire_minutes * 60,
                "user": user_response
            }
            
            return True, "Login successful", response_data
            
        except Exception as e:
            db.rollback()
            logger.error(f"Login failed: {e}")
            return False, "Login failed", None
    
    def refresh_token(self, db: SupabaseDatabaseService, refresh_token: str, 
                     ip_address: str = None, user_agent: str = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Refresh access token using refresh token."""
        try:
            # Verify refresh token
            token_data = self.jwt_service.verify_token(refresh_token)
            if not token_data or not self.jwt_service.is_refresh_token(refresh_token):
                return False, "Invalid refresh token", None
            
            # Find user session
            session = db.get_UserSession_by_id(
                UserSession.refresh_token == refresh_token,
                UserSession.is_active == True
            )
            
            if not session or session.is_expired():
                return False, "Invalid or expired refresh token", None
            
            # Find user
            user = db.get_User_by_id(User.id == session.user_id)
            if not user or not user.is_active:
                return False, "User not found or inactive", None
            
            # Create new access token
            new_access_token = self.jwt_service.create_access_token(
                str(user.id), user.email, user.role.value
            )
            
            # Update session with new JTI
            session.token_jti = self.jwt_service.extract_jti(new_access_token)
            session.update_last_accessed()
            
            db.commit()
            
            # Prepare response
            user_response = UserResponse.from_orm(user)
            response_data = {
                "access_token": new_access_token,
                "refresh_token": refresh_token,  # Keep same refresh token
                "token_type": "bearer",
                "expires_in": self.jwt_service.access_token_expire_minutes * 60,
                "user": user_response
            }
            
            return True, "Token refreshed successfully", response_data
            
        except Exception as e:
            db.rollback()
            logger.error(f"Token refresh failed: {e}")
            return False, "Token refresh failed", None
    
    def logout_user(self, db: SupabaseDatabaseService, access_token: str, 
                   ip_address: str = None, user_agent: str = None) -> Tuple[bool, str]:
        """Logout user and invalidate session with token blacklisting."""
        try:
            # Extract JTI from token
            jti = self.jwt_service.extract_jti(access_token)
            if not jti:
                return False, "Invalid token"
            
            # Find and deactivate session
            session = db.get_UserSession_by_id(
                UserSession.token_jti == jti,
                UserSession.is_active == True
            )
            
            if session:
                # Blacklist the access token
                access_blacklisted = self.jwt_service.blacklist_token(
                    access_token, 
                    reason="logout"
                )
                
                if not access_blacklisted:
                    logger.warning(f"Failed to blacklist access token {jti}")
                
                # Also blacklist the refresh token if it exists
                if session.refresh_token:
                    refresh_blacklisted = self.jwt_service.blacklist_token(
                        session.refresh_token,
                        reason="logout"
                    )
                    if not refresh_blacklisted:
                        logger.warning(f"Failed to blacklist refresh token for session {jti}")
                
                # Deactivate session in database
                session.deactivate()
                
                # Log logout event
                auth_log = AuthenticationLog.log_logout(
                    user_id=session.user_id,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                db.add(auth_log)
            else:
                # Even if session not found in DB, blacklist the token
                self.jwt_service.blacklist_token(access_token, reason="logout")
            
            db.commit()
            return True, "Logout successful"
            
        except Exception as e:
            db.rollback()
            logger.error(f"Logout failed: {e}")
            return False, "Logout failed"
    
    def blacklist_all_user_tokens(self, db: SupabaseDatabaseService, user_id: str, 
                                 reason: str = "security", ip_address: str = None, 
                                 user_agent: str = None) -> Tuple[bool, str, int]:
        """
        Blacklist all tokens for a user (for security purposes).
        
        Args:
            db: Database session
            user_id: ID of the user
            reason: Reason for blacklisting
            ip_address: IP address of the request
            user_agent: User agent of the request
            
        Returns:
            Tuple of (success, message, number_of_tokens_blacklisted)
        """
        try:
            # Blacklist all user tokens in Redis
            blacklisted_count = self.jwt_service.blacklist_user_tokens(user_id, reason)
            
            # Deactivate all active sessions in database
            active_sessions = db.query(UserSession).filter(
                UserSession.user_id == user_id,
                UserSession.is_active == True
            ).all()
            
            for session in active_sessions:
                session.deactivate()
            
            # Log security event
            auth_log = AuthenticationLog.log_security_event(
                user_id=user_id,
                event_type=AuthenticationEventType.TOKEN_BLACKLISTED,
                ip_address=ip_address,
                user_agent=user_agent,
                details=f"All tokens blacklisted: {reason}"
            )
            db.add(auth_log)
            
            db.commit()
            
            logger.info(f"Blacklisted {blacklisted_count} tokens for user {user_id}")
            return True, f"All tokens blacklisted successfully", blacklisted_count
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to blacklist user tokens: {e}")
            return False, "Failed to blacklist user tokens", 0
    
    def request_password_reset(self, db: SupabaseDatabaseService, reset_data: PasswordResetRequest, 
                              ip_address: str = None, user_agent: str = None) -> Tuple[bool, str]:
        """Request password reset."""
        try:
            # Find user by email
            user = db.get_User_by_id(User.email == reset_data.email)
            
            if not user:
                # Don't reveal if user exists or not
                return True, "If the email exists, a password reset link has been sent"
            
            # Generate reset token
            reset_token = self.email_service.generate_reset_token()
            expires_at = datetime.utcnow() + timedelta(hours=1)
            
            # Create password reset record
            password_reset = PasswordReset.create_reset_token(
                user_id=user.id,
                token=reset_token,
                expires_in_hours=1
            )
            db.add(password_reset)
            
            # Log reset request
            auth_log = AuthenticationLog.log_password_reset_requested(
                user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            db.add(auth_log)
            
            # Send reset email
            email_sent = self.email_service.send_password_reset_email(
                user.email, 
                reset_token, 
                user.full_name
            )
            
            if not email_sent:
                logger.warning(f"Failed to send password reset email to {user.email}")
            
            db.commit()
            return True, "If the email exists, a password reset link has been sent"
            
        except Exception as e:
            db.rollback()
            logger.error(f"Password reset request failed: {e}")
            return False, "Password reset request failed"
    
    def confirm_password_reset(self, db: SupabaseDatabaseService, reset_data: PasswordResetConfirmRequest, 
                              ip_address: str = None, user_agent: str = None) -> Tuple[bool, str]:
        """Confirm password reset with new password."""
        try:
            # Find valid reset token
            password_reset = db.get_PasswordReset_by_id(
                PasswordReset.token == reset_data.token,
                PasswordReset.is_used == False
            )
            
            if not password_reset or password_reset.is_expired():
                return False, "Invalid or expired reset token"
            
            # Find user
            user = db.get_User_by_id(User.id == password_reset.user_id)
            if not user or not user.is_active:
                return False, "User not found or inactive"
            
            # Validate new password strength
            is_valid, message = self.password_service.validate_password_strength(reset_data.new_password)
            if not is_valid:
                return False, message
            
            # Hash new password
            new_password_hash = self.password_service.hash_password(reset_data.new_password)
            
            # Update user password
            user.password_hash = new_password_hash
            user.reset_failed_attempts()  # Reset any failed attempts
            
            # Mark reset token as used
            password_reset.mark_as_used()
            
            # Log reset completion
            auth_log = AuthenticationLog.log_password_reset_used(
                user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            db.add(auth_log)
            
            db.commit()
            return True, "Password reset successfully"
            
        except Exception as e:
            db.rollback()
            logger.error(f"Password reset confirmation failed: {e}")
            return False, "Password reset confirmation failed"
    
    def change_password(self, db: SupabaseDatabaseService, user_id: UUID, change_data: ChangePasswordRequest, 
                       ip_address: str = None, user_agent: str = None) -> Tuple[bool, str]:
        """Change user password."""
        try:
            # Find user
            user = db.get_User_by_id(User.id == user_id)
            if not user:
                return False, "User not found"
            
            # Verify current password
            if not self.password_service.verify_password(change_data.current_password, user.password_hash):
                return False, "Current password is incorrect"
            
            # Validate new password strength
            is_valid, message = self.password_service.validate_password_strength(change_data.new_password)
            if not is_valid:
                return False, message
            
            # Hash new password
            new_password_hash = self.password_service.hash_password(change_data.new_password)
            
            # Update password
            user.password_hash = new_password_hash
            user.reset_failed_attempts()
            
            # Log password change
            auth_log = AuthenticationLog.log_event(
                event_type=AuthenticationEventType.PASSWORD_RESET_USED,  # Reuse this event type
                success=True,
                user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent,
                event_metadata={"action": "password_changed"}
            )
            db.add(auth_log)
            
            db.commit()
            return True, "Password changed successfully"
            
        except Exception as e:
            db.rollback()
            logger.error(f"Password change failed: {e}")
            return False, "Password change failed"
    
    def get_user_by_id(self, db: SupabaseDatabaseService, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        return db.get_User_by_id(User.id == user_id)
    
    def get_user_by_email(self, db: SupabaseDatabaseService, email: str) -> Optional[User]:
        """Get user by email."""
        return db.get_User_by_id(User.email == email)
    
    def verify_access_token(self, token: str) -> Optional[TokenData]:
        """Verify access token and return token data."""
        return self.jwt_service.verify_token(token)
    
    def cleanup_expired_sessions(self, db: SupabaseDatabaseService) -> int:
        """Clean up expired sessions."""
        try:
            expired_sessions = db.query(UserSession).filter(
                UserSession.expires_at < datetime.utcnow()
            ).all()
            
            count = len(expired_sessions)
            for session in expired_sessions:
                session.deactivate()
            
            db.commit()
            return count
        except Exception as e:
            db.rollback()
            logger.error(f"Session cleanup failed: {e}")
            return 0

# Create singleton instance
auth_service = AuthenticationService()
