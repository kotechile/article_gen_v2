"""
Account lockout service for managing failed login attempts and account security
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
import secrets
import hashlib
import logging

from ..models.account_lockout import (
    AccountLockout, 
    FailedLoginAttempt, 
    SecurityEvent, 
    PasswordBreach, 
    SuspiciousActivity,
    LockoutReason
)
# from ..models.user import User  # Commented out for testing
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class AccountLockoutService:
    """Service for managing account lockouts and security events"""
    
    def __init__(self, db: Session):
        self.db = db
        self.max_failed_attempts = settings.MAX_FAILED_LOGIN_ATTEMPTS
        self.lockout_duration = settings.ACCOUNT_LOCKOUT_DURATION
        self.failed_attempt_window = settings.FAILED_ATTEMPT_WINDOW
        self.suspicious_activity_threshold = settings.SUSPICIOUS_ACTIVITY_THRESHOLD
    
    def record_failed_login(
        self, 
        email: str, 
        ip_address: str, 
        user_agent: Optional[str] = None,
        user_id: Optional[int] = None,
        failure_reason: Optional[str] = None
    ) -> FailedLoginAttempt:
        """Record a failed login attempt"""
        attempt = FailedLoginAttempt(
            user_id=user_id,
            email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            failure_reason=failure_reason,
            is_suspicious=self._is_suspicious_attempt(email, ip_address)
        )
        
        self.db.add(attempt)
        self.db.commit()
        self.db.refresh(attempt)
        
        # Check if account should be locked
        if user_id:
            self._check_and_lock_account(user_id, ip_address, user_agent)
        
        # Log security event
        self._log_security_event(
            user_id=user_id,
            event_type="failed_login",
            event_description=f"Failed login attempt for {email} from {ip_address}",
            ip_address=ip_address,
            user_agent=user_agent,
            severity="medium"
        )
        
        return attempt
    
    def _is_suspicious_attempt(self, email: str, ip_address: str) -> bool:
        """Check if login attempt is suspicious"""
        # Check for multiple failed attempts from same IP
        recent_attempts = self.db.query(FailedLoginAttempt).filter(
            and_(
                FailedLoginAttempt.ip_address == ip_address,
                FailedLoginAttempt.attempt_time > datetime.utcnow() - timedelta(hours=1)
            )
        ).count()
        
        if recent_attempts > 10:  # More than 10 attempts in 1 hour
            return True
        
        # Check for attempts from different countries (simplified)
        # In real implementation, you'd use geolocation service
        return False
    
    def _check_and_lock_account(self, user_id: int, ip_address: str, user_agent: Optional[str]) -> None:
        """Check if account should be locked based on failed attempts"""
        # Count recent failed attempts
        recent_attempts = self.db.query(FailedLoginAttempt).filter(
            and_(
                FailedLoginAttempt.user_id == user_id,
                FailedLoginAttempt.attempt_time > datetime.utcnow() - timedelta(minutes=self.failed_attempt_window)
            )
        ).count()
        
        if recent_attempts >= self.max_failed_attempts:
            self.lock_account(
                user_id=user_id,
                reason=LockoutReason.FAILED_LOGIN,
                duration_minutes=self.lockout_duration,
                ip_address=ip_address,
                user_agent=user_agent,
                reason_description=f"Account locked due to {recent_attempts} failed login attempts"
            )
    
    def lock_account(
        self,
        user_id: int,
        reason: LockoutReason,
        duration_minutes: Optional[int] = None,
        is_permanent: bool = False,
        locked_by_admin: Optional[int] = None,
        reason_description: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AccountLockout:
        """Lock an account"""
        # Unlock any existing active lockouts
        self.unlock_account(user_id, "New lockout applied")
        
        # Calculate lockout duration
        locked_until = None
        if not is_permanent and duration_minutes:
            locked_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        
        # Generate unlock token
        unlock_token = secrets.token_urlsafe(32)
        
        lockout = AccountLockout(
            user_id=user_id,
            reason=reason.value,
            locked_until=locked_until,
            is_permanent=is_permanent,
            unlock_token=unlock_token,
            locked_by_admin=locked_by_admin,
            reason_description=reason_description,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db.add(lockout)
        self.db.commit()
        self.db.refresh(lockout)
        
        # Log security event
        self._log_security_event(
            user_id=user_id,
            event_type="account_locked",
            event_description=f"Account locked: {reason_description or reason.value}",
            ip_address=ip_address,
            user_agent=user_agent,
            severity="high"
        )
        
        logger.warning(f"Account {user_id} locked: {reason.value}")
        return lockout
    
    def unlock_account(self, user_id: int, reason: str = "Manual unlock") -> bool:
        """Unlock an account"""
        active_lockouts = self.db.query(AccountLockout).filter(
            and_(
                AccountLockout.user_id == user_id,
                AccountLockout.is_active == True
            )
        ).all()
        
        if not active_lockouts:
            return False
        
        for lockout in active_lockouts:
            lockout.unlock()
        
        self.db.commit()
        
        # Log security event
        self._log_security_event(
            user_id=user_id,
            event_type="account_unlocked",
            event_description=f"Account unlocked: {reason}",
            severity="medium"
        )
        
        logger.info(f"Account {user_id} unlocked: {reason}")
        return True
    
    def is_account_locked(self, user_id: int) -> bool:
        """Check if account is currently locked"""
        active_lockouts = self.db.query(AccountLockout).filter(
            and_(
                AccountLockout.user_id == user_id,
                AccountLockout.is_active == True
            )
        ).all()
        
        return any(lockout.is_locked() for lockout in active_lockouts)
    
    def get_lockout_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get current lockout information"""
        active_lockouts = self.db.query(AccountLockout).filter(
            and_(
                AccountLockout.user_id == user_id,
                AccountLockout.is_active == True
            )
        ).all()
        
        if not active_lockouts:
            return None
        
        # Get the most recent lockout
        lockout = max(active_lockouts, key=lambda x: x.locked_at)
        
        if not lockout.is_locked():
            return None
        
        remaining_time = lockout.get_remaining_lockout_time()
        
        return {
            "is_locked": True,
            "reason": lockout.reason,
            "locked_at": lockout.locked_at,
            "locked_until": lockout.locked_until,
            "is_permanent": lockout.is_permanent,
            "remaining_time_seconds": remaining_time.total_seconds() if remaining_time else None,
            "reason_description": lockout.reason_description,
            "unlock_token": lockout.unlock_token
        }
    
    def get_failed_attempts_count(self, user_id: int, minutes: int = 15) -> int:
        """Get count of failed attempts in specified time window"""
        return self.db.query(FailedLoginAttempt).filter(
            and_(
                FailedLoginAttempt.user_id == user_id,
                FailedLoginAttempt.attempt_time > datetime.utcnow() - timedelta(minutes=minutes)
            )
        ).count()
    
    def get_failed_attempts_count_by_ip(self, ip_address: str, minutes: int = 15) -> int:
        """Get count of failed attempts from IP in specified time window"""
        return self.db.query(FailedLoginAttempt).filter(
            and_(
                FailedLoginAttempt.ip_address == ip_address,
                FailedLoginAttempt.attempt_time > datetime.utcnow() - timedelta(minutes=minutes)
            )
        ).count()
    
    def clear_failed_attempts(self, user_id: int) -> None:
        """Clear failed attempts for user (after successful login)"""
        self.db.query(FailedLoginAttempt).filter(
            FailedLoginAttempt.user_id == user_id
        ).delete()
        self.db.commit()
    
    def _log_security_event(
        self,
        user_id: Optional[int],
        event_type: str,
        event_description: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        severity: str = "medium",
        metadata: Optional[Dict[str, Any]] = None
    ) -> SecurityEvent:
        """Log a security event"""
        event = SecurityEvent(
            user_id=user_id,
            event_type=event_type,
            event_description=event_description,
            ip_address=ip_address,
            user_agent=user_agent,
            severity=severity,
            event_metadata=str(metadata) if metadata else None
        )
        
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        
        return event
    
    def record_suspicious_activity(
        self,
        user_id: Optional[int],
        activity_type: str,
        description: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        risk_score: int = 50
    ) -> SuspiciousActivity:
        """Record suspicious activity"""
        activity = SuspiciousActivity(
            user_id=user_id,
            activity_type=activity_type,
            description=description,
            ip_address=ip_address,
            user_agent=user_agent,
            risk_score=risk_score
        )
        
        self.db.add(activity)
        self.db.commit()
        self.db.refresh(activity)
        
        # Check if risk score exceeds threshold
        if risk_score >= self.suspicious_activity_threshold:
            if user_id:
                self.lock_account(
                    user_id=user_id,
                    reason=LockoutReason.SUSPICIOUS_ACTIVITY,
                    duration_minutes=self.lockout_duration,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    reason_description=f"Suspicious activity detected: {activity_type}"
                )
        
        return activity
    
    def check_password_breach(self, password_hash: str) -> bool:
        """Check if password has been breached (simplified)"""
        # In real implementation, you'd check against breach databases
        # For now, just check against common passwords
        common_hashes = [
            hashlib.sha256(b"password").hexdigest(),
            hashlib.sha256(b"123456").hexdigest(),
            hashlib.sha256(b"qwerty").hexdigest(),
        ]
        
        return password_hash in common_hashes
    
    def record_password_breach(
        self,
        user_id: int,
        breach_source: str,
        breach_description: str,
        password_hash: Optional[str] = None
    ) -> PasswordBreach:
        """Record a password breach"""
        breach = PasswordBreach(
            user_id=user_id,
            breach_source=breach_source,
            breach_description=breach_description,
            password_hash=password_hash
        )
        
        self.db.add(breach)
        self.db.commit()
        self.db.refresh(breach)
        
        # Lock account due to password breach
        self.lock_account(
            user_id=user_id,
            reason=LockoutReason.PASSWORD_BREACH,
            duration_minutes=self.lockout_duration,
            reason_description="Password found in data breach"
        )
        
        return breach
    
    def get_security_events(
        self,
        user_id: Optional[int] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Get security events with filters"""
        query = self.db.query(SecurityEvent)
        
        if user_id:
            query = query.filter(SecurityEvent.user_id == user_id)
        
        if event_type:
            query = query.filter(SecurityEvent.event_type == event_type)
        
        if severity:
            query = query.filter(SecurityEvent.severity == severity)
        
        return query.order_by(desc(SecurityEvent.created_at)).limit(limit).all()
    
    def get_account_security_summary(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive security summary for account"""
        # Get recent failed attempts
        recent_failed_attempts = self.get_failed_attempts_count(user_id, 24 * 60)  # 24 hours
        
        # Get active lockouts
        lockout_info = self.get_lockout_info(user_id)
        
        # Get recent security events
        recent_events = self.get_security_events(user_id, limit=10)
        
        # Get suspicious activities
        suspicious_activities = self.db.query(SuspiciousActivity).filter(
            and_(
                SuspiciousActivity.user_id == user_id,
                SuspiciousActivity.created_at > datetime.utcnow() - timedelta(days=30)
            )
        ).count()
        
        # Get password breaches
        password_breaches = self.db.query(PasswordBreach).filter(
            and_(
                PasswordBreach.user_id == user_id,
                PasswordBreach.is_resolved == False
            )
        ).count()
        
        return {
            "user_id": user_id,
            "is_locked": lockout_info is not None,
            "lockout_info": lockout_info,
            "recent_failed_attempts": recent_failed_attempts,
            "recent_security_events": len(recent_events),
            "suspicious_activities_30d": suspicious_activities,
            "unresolved_password_breaches": password_breaches,
            "security_score": self._calculate_security_score(
                recent_failed_attempts,
                suspicious_activities,
                password_breaches,
                lockout_info is not None
            )
        }
    
    def _calculate_security_score(
        self,
        failed_attempts: int,
        suspicious_activities: int,
        password_breaches: int,
        is_locked: bool
    ) -> int:
        """Calculate security score (0-100, higher is better)"""
        score = 100
        
        # Deduct points for failed attempts
        score -= min(failed_attempts * 5, 30)
        
        # Deduct points for suspicious activities
        score -= min(suspicious_activities * 10, 40)
        
        # Deduct points for password breaches
        score -= min(password_breaches * 20, 40)
        
        # Deduct points for being locked
        if is_locked:
            score -= 30
        
        return max(score, 0)
