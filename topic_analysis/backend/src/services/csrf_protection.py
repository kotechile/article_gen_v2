"""
CSRF protection service for token generation and validation
"""
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import logging
import json

from ..models.csrf_protection import (
    CSRFProtection, 
    CSRFWhitelist, 
    CSRFViolation, 
    CSRFConfiguration,
    CSRFTokenStatus
)
from ..core.config import get_settings
from src.core.supabase_database_service import SupabaseDatabaseService

logger = logging.getLogger(__name__)
settings = get_settings()

class CSRFProtectionService:
    """Service for managing CSRF protection tokens"""
    
    def __init__(self, db: SupabaseDatabaseService):
        self.db = db
        self.token_length = 32
        self.token_ttl = settings.CSRF_TOKEN_TTL_HOURS * 3600  # Convert to seconds
        self.max_violations_per_ip = settings.CSRF_MAX_VIOLATIONS_PER_IP
        self.violation_window = settings.CSRF_VIOLATION_WINDOW_HOURS * 3600
        self.secret_key = settings.CSRF_SECRET_KEY.encode()
    
    def generate_csrf_token(
        self,
        user_id: int,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        is_single_use: bool = True,
        is_permanent: bool = False,
        custom_ttl: Optional[int] = None
    ) -> str:
        """Generate a CSRF protection token"""
        try:
            # Generate random token
            token = secrets.token_urlsafe(self.token_length)
            
            # Create token hash for efficient lookup
            token_hash = self._hash_token(token)
            
            # Calculate expiration time
            if custom_ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=custom_ttl)
            elif is_permanent:
                expires_at = datetime.utcnow() + timedelta(days=365)  # 1 year
            else:
                expires_at = datetime.utcnow() + timedelta(seconds=self.token_ttl)
            
            # Create CSRF protection entry
            csrf_entry = CSRFProtection(
                user_id=user_id,
                token=token,
                token_hash=token_hash,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                status=CSRFTokenStatus.ACTIVE.value,
                expires_at=expires_at,
                is_single_use=is_single_use,
                is_permanent=is_permanent
            )
            
            self.db.add(csrf_entry)
            self.db.commit()
            self.db.refresh(csrf_entry)
            
            logger.info(f"CSRF token generated for user {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Error generating CSRF token: {str(e)}")
            self.db.rollback()
            raise
    
    def validate_csrf_token(
        self,
        token: str,
        user_id: int,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        origin: Optional[str] = None,
        referer: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Validate a CSRF token"""
        try:
            if not token:
                self._log_violation(
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    origin=origin,
                    referer=referer,
                    violation_type="missing_token",
                    severity="high"
                )
                return False, "CSRF token is missing"
            
            # Find token in database
            csrf_entry = self.db.query(CSRFProtection).filter(
                and_(
                    CSRFProtection.token == token,
                    CSRFProtection.user_id == user_id
                )
            ).first()
            
            if not csrf_entry:
                self._log_violation(
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    origin=origin,
                    referer=referer,
                    violation_type="invalid_token",
                    token_provided=token,
                    severity="high"
                )
                return False, "Invalid CSRF token"
            
            # Check if token is expired
            if csrf_entry.is_expired():
                self._log_violation(
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    origin=origin,
                    referer=referer,
                    violation_type="expired_token",
                    token_provided=token,
                    severity="medium"
                )
                return False, "CSRF token has expired"
            
            # Check if token is already used (for single-use tokens)
            if csrf_entry.is_single_use and csrf_entry.status == CSRFTokenStatus.USED.value:
                self._log_violation(
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    origin=origin,
                    referer=referer,
                    violation_type="reused_token",
                    token_provided=token,
                    severity="high"
                )
                return False, "CSRF token has already been used"
            
            # Check if token is revoked
            if csrf_entry.status == CSRFTokenStatus.REVOKED.value:
                self._log_violation(
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    origin=origin,
                    referer=referer,
                    violation_type="revoked_token",
                    token_provided=token,
                    severity="high"
                )
                return False, "CSRF token has been revoked"
            
            # Check origin if provided
            if origin and not self._is_origin_allowed(origin):
                self._log_violation(
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    origin=origin,
                    referer=referer,
                    violation_type="invalid_origin",
                    token_provided=token,
                    severity="high"
                )
                return False, "Invalid origin"
            
            # Mark token as used if it's single-use
            if csrf_entry.is_single_use:
                csrf_entry.mark_as_used()
                self.db.commit()
            
            logger.info(f"CSRF token validated successfully for user {user_id}")
            return True, "CSRF token is valid"
            
        except Exception as e:
            logger.error(f"Error validating CSRF token: {str(e)}")
            return False, "CSRF validation error"
    
    def revoke_user_tokens(
        self,
        user_id: int,
        reason: str = "manual_revoke"
    ) -> int:
        """Revoke all active CSRF tokens for a user"""
        try:
            active_tokens = self.db.query(CSRFProtection).filter(
                and_(
                    CSRFProtection.user_id == user_id,
                    CSRFProtection.status == CSRFTokenStatus.ACTIVE.value
                )
            ).all()
            
            revoked_count = 0
            for token in active_tokens:
                token.revoke()
                revoked_count += 1
            
            self.db.commit()
            
            logger.info(f"Revoked {revoked_count} CSRF tokens for user {user_id}")
            return revoked_count
            
        except Exception as e:
            logger.error(f"Error revoking CSRF tokens: {str(e)}")
            self.db.rollback()
            raise
    
    def revoke_token(
        self,
        token: str,
        user_id: int
    ) -> bool:
        """Revoke a specific CSRF token"""
        try:
            csrf_entry = self.db.query(CSRFProtection).filter(
                and_(
                    CSRFProtection.token == token,
                    CSRFProtection.user_id == user_id
                )
            ).first()
            
            if not csrf_entry:
                return False
            
            csrf_entry.revoke()
            self.db.commit()
            
            logger.info(f"CSRF token {token} revoked for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking CSRF token: {str(e)}")
            self.db.rollback()
            return False
    
    def get_user_active_tokens(self, user_id: int) -> List[CSRFProtection]:
        """Get all active CSRF tokens for a user"""
        return self.db.query(CSRFProtection).filter(
            and_(
                CSRFProtection.user_id == user_id,
                CSRFProtection.status == CSRFTokenStatus.ACTIVE.value,
                CSRFProtection.expires_at > datetime.utcnow()
            )
        ).all()
    
    def cleanup_expired_tokens(self) -> int:
        """Clean up expired CSRF tokens"""
        try:
            now = datetime.utcnow()
            
            # Clean up expired tokens
            expired_tokens = self.db.query(CSRFProtection).filter(
                CSRFProtection.expires_at < now
            ).count()
            
            self.db.query(CSRFProtection).filter(
                CSRFProtection.expires_at < now
            ).delete()
            
            self.db.commit()
            
            logger.info(f"Cleaned up {expired_tokens} expired CSRF tokens")
            return expired_tokens
            
        except Exception as e:
            logger.error(f"Error cleaning up expired CSRF tokens: {str(e)}")
            self.db.rollback()
            raise
    
    def add_whitelist_origin(
        self,
        origin: str,
        description: Optional[str] = None,
        created_by: Optional[int] = None,
        expires_at: Optional[datetime] = None
    ) -> CSRFWhitelist:
        """Add an origin to the CSRF whitelist"""
        try:
            whitelist_entry = CSRFWhitelist(
                origin=origin,
                description=description,
                created_by=created_by,
                expires_at=expires_at
            )
            
            self.db.add(whitelist_entry)
            self.db.commit()
            self.db.refresh(whitelist_entry)
            
            logger.info(f"Added origin {origin} to CSRF whitelist")
            return whitelist_entry
            
        except Exception as e:
            logger.error(f"Error adding whitelist origin: {str(e)}")
            self.db.rollback()
            raise
    
    def is_origin_whitelisted(self, origin: str) -> bool:
        """Check if an origin is whitelisted"""
        whitelist_entry = self.db.query(CSRFWhitelist).filter(
            and_(
                CSRFWhitelist.origin == origin,
                CSRFWhitelist.is_valid()
            )
        ).first()
        
        return whitelist_entry is not None
    
    def get_violation_stats(
        self,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get CSRF violation statistics"""
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            query = self.db.query(CSRFViolation).filter(
                CSRFViolation.created_at >= since
            )
            
            if user_id:
                query = query.filter(CSRFViolation.user_id == user_id)
            
            if ip_address:
                query = query.filter(CSRFViolation.ip_address == ip_address)
            
            violations = query.all()
            
            # Count by type
            violation_types = {}
            for violation in violations:
                violation_type = violation.violation_type
                violation_types[violation_type] = violation_types.get(violation_type, 0) + 1
            
            # Count by severity
            severity_counts = {}
            for violation in violations:
                severity = violation.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                "total_violations": len(violations),
                "violation_types": violation_types,
                "severity_counts": severity_counts,
                "time_period_hours": hours
            }
            
        except Exception as e:
            logger.error(f"Error getting violation stats: {str(e)}")
            return {}
    
    def should_block_ip(self, ip_address: str) -> bool:
        """Check if an IP should be blocked due to too many violations"""
        try:
            since = datetime.utcnow() - timedelta(seconds=self.violation_window)
            
            violation_count = self.db.query(CSRFViolation).filter(
                and_(
                    CSRFViolation.ip_address == ip_address,
                    CSRFViolation.created_at >= since,
                    CSRFViolation.severity.in_(["high", "critical"])
                )
            ).count()
            
            return violation_count >= self.max_violations_per_ip
            
        except Exception as e:
            logger.error(f"Error checking IP block status: {str(e)}")
            return False
    
    def _hash_token(self, token: str) -> str:
        """Create a hash of the CSRF token"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if an origin is allowed (whitelisted or same-origin)"""
        # Check whitelist first
        if self.is_origin_whitelisted(origin):
            return True
        
        # Check if it's a same-origin request (no origin header)
        if not origin:
            return True
        
        # Additional origin validation can be added here
        return False
    
    def _log_violation(
        self,
        user_id: Optional[int],
        ip_address: str,
        user_agent: Optional[str],
        origin: Optional[str],
        referer: Optional[str],
        violation_type: str,
        token_provided: Optional[str] = None,
        expected_token: Optional[str] = None,
        severity: str = "medium"
    ) -> None:
        """Log a CSRF violation"""
        try:
            violation = CSRFViolation(
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                origin=origin,
                referer=referer,
                violation_type=violation_type,
                token_provided=token_provided,
                expected_token=expected_token,
                severity=severity,
                is_blocked=self.should_block_ip(ip_address)
            )
            
            self.db.add(violation)
            self.db.commit()
            
            logger.warning(f"CSRF violation logged: {violation_type} from {ip_address}")
            
        except Exception as e:
            logger.error(f"Error logging CSRF violation: {str(e)}")
            # Don't raise exception for logging failures
