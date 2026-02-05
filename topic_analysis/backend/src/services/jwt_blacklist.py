"""
JWT token blacklist service for token revocation and management
"""
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import logging

from ..models.jwt_blacklist import (
    JWTBlacklist, 
    TokenWhitelist, 
    TokenSession, 
    TokenAuditLog,
    RevocationReason,
    TokenStatus
)
from ..core.config import get_settings
from src.core.supabase_database_service import SupabaseDatabaseService

logger = logging.getLogger(__name__)
settings = get_settings()

class JWTBlacklistService:
    """Service for managing JWT token blacklisting and revocation"""
    
    def __init__(self, db: SupabaseDatabaseService):
        self.db = db
        self.blacklist_ttl = settings.JWT_BLACKLIST_TTL_HOURS * 3600  # Convert to seconds
        self.cleanup_interval = settings.JWT_CLEANUP_INTERVAL_HOURS * 3600
    
    def blacklist_token(
        self,
        token: str,
        user_id: int,
        reason: RevocationReason = RevocationReason.USER_LOGOUT,
        revoked_by: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        is_permanent: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JWTBlacklist:
        """Blacklist a JWT token"""
        try:
            # Decode token to get JTI and expiration
            decoded_token = jwt.decode(
                token, 
                options={"verify_signature": False, "verify_exp": False}
            )
            
            jti = decoded_token.get('jti')
            exp = decoded_token.get('exp')
            token_type = decoded_token.get('type', 'access')
            
            if not jti:
                raise ValueError("Token does not contain JTI")
            
            # Calculate expiration time
            if exp:
                expires_at = datetime.fromtimestamp(exp)
            else:
                expires_at = datetime.utcnow() + timedelta(hours=24)  # Default 24 hours
            
            # Create token hash for efficient lookup
            token_hash = self._hash_token(token)
            
            # Check if token is already blacklisted
            existing = self.db.get_JWTBlacklist_by_id(
                JWTBlacklist.jti == jti
            )
            
            if existing:
                logger.warning(f"Token {jti} is already blacklisted")
                return existing
            
            # Create blacklist entry
            blacklist_entry = JWTBlacklist(
                jti=jti,
                user_id=user_id,
                token_type=token_type,
                token_hash=token_hash,
                expires_at=expires_at,
                reason=reason.value,
                revoked_by=revoked_by,
                ip_address=ip_address,
                user_agent=user_agent,
                is_permanent=is_permanent,
                token_metadata=str(metadata) if metadata else None
            )
            
            self.db.add(blacklist_entry)
            self.db.commit()
            self.db.refresh(blacklist_entry)
            
            # Log audit event
            self._log_token_action(
                jti=jti,
                user_id=user_id,
                action="revoked",
                token_type=token_type,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata={"reason": reason.value, "revoked_by": revoked_by}
            )
            
            # Remove from whitelist if exists
            self._remove_from_whitelist(jti)
            
            logger.info(f"Token {jti} blacklisted for user {user_id}, reason: {reason.value}")
            return blacklist_entry
            
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token for blacklisting: {str(e)}")
            raise ValueError("Invalid token format")
        except Exception as e:
            logger.error(f"Error blacklisting token: {str(e)}")
            self.db.rollback()
            raise
    
    def is_token_blacklisted(self, token: str) -> bool:
        """Check if a token is blacklisted"""
        try:
            # Decode token to get JTI
            decoded_token = jwt.decode(
                token, 
                options={"verify_signature": False, "verify_exp": False}
            )
            
            jti = decoded_token.get('jti')
            if not jti:
                return False
            
            # Check blacklist
            blacklist_entry = self.db.query(JWTBlacklist).filter(
                and_(
                    JWTBlacklist.jti == jti,
                    JWTBlacklist.expires_at > datetime.utcnow()
                )
            ).first()
            
            return blacklist_entry is not None
            
        except jwt.InvalidTokenError:
            return False
        except Exception as e:
            logger.error(f"Error checking token blacklist: {str(e)}")
            return False
    
    def is_jti_blacklisted(self, jti: str) -> bool:
        """Check if a JTI is blacklisted"""
        try:
            blacklist_entry = self.db.query(JWTBlacklist).filter(
                and_(
                    JWTBlacklist.jti == jti,
                    JWTBlacklist.expires_at > datetime.utcnow()
                )
            ).first()
            
            return blacklist_entry is not None
            
        except Exception as e:
            logger.error(f"Error checking JTI blacklist: {str(e)}")
            return False
    
    def blacklist_user_tokens(
        self,
        user_id: int,
        reason: RevocationReason = RevocationReason.ADMIN_REVOKE,
        revoked_by: Optional[int] = None,
        exclude_jti: Optional[str] = None
    ) -> int:
        """Blacklist all active tokens for a user"""
        try:
            # Get all active whitelisted tokens for user
            active_tokens = self.db.query(TokenWhitelist).filter(
                and_(
                    TokenWhitelist.user_id == user_id,
                    TokenWhitelist.is_active == True,
                    TokenWhitelist.expires_at > datetime.utcnow()
                )
            ).all()
            
            blacklisted_count = 0
            
            for token in active_tokens:
                if exclude_jti and token.jti == exclude_jti:
                    continue
                
                # Create blacklist entry
                blacklist_entry = JWTBlacklist(
                    jti=token.jti,
                    user_id=user_id,
                    token_type=token.token_type,
                    token_hash=token.token_hash,
                    expires_at=token.expires_at,
                    reason=reason.value,
                    revoked_by=revoked_by,
                    ip_address=token.ip_address,
                    user_agent=token.user_agent,
                    is_permanent=False
                )
                
                self.db.add(blacklist_entry)
                blacklisted_count += 1
                
                # Log audit event
                self._log_token_action(
                    jti=token.jti,
                    user_id=user_id,
                    action="revoked",
                    token_type=token.token_type,
                    ip_address=token.ip_address,
                    user_agent=token.user_agent,
                    metadata={"reason": reason.value, "revoked_by": revoked_by}
                )
            
            # Remove all tokens from whitelist
            self.db.query(TokenWhitelist).filter(
                and_(
                    TokenWhitelist.user_id == user_id,
                    TokenWhitelist.is_active == True
                )
            ).update({"is_active": False})
            
            self.db.commit()
            
            logger.info(f"Blacklisted {blacklisted_count} tokens for user {user_id}")
            return blacklisted_count
            
        except Exception as e:
            logger.error(f"Error blacklisting user tokens: {str(e)}")
            self.db.rollback()
            raise
    
    def whitelist_token(
        self,
        token: str,
        user_id: int,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> TokenWhitelist:
        """Add token to whitelist (for active token tracking)"""
        try:
            # Decode token to get JTI and expiration
            decoded_token = jwt.decode(
                token, 
                options={"verify_signature": False, "verify_exp": False}
            )
            
            jti = decoded_token.get('jti')
            exp = decoded_token.get('exp')
            token_type = decoded_token.get('type', 'access')
            
            if not jti:
                raise ValueError("Token does not contain JTI")
            
            # Calculate expiration time
            if exp:
                expires_at = datetime.fromtimestamp(exp)
            else:
                expires_at = datetime.utcnow() + timedelta(hours=24)
            
            # Create token hash
            token_hash = self._hash_token(token)
            
            # Check if already whitelisted
            existing = self.db.get_TokenWhitelist_by_id(
                TokenWhitelist.jti == jti
            )
            
            if existing:
                # Update last used time
                existing.last_used = datetime.utcnow()
                self.db.commit()
                return existing
            
            # Create whitelist entry
            whitelist_entry = TokenWhitelist(
                jti=jti,
                user_id=user_id,
                token_type=token_type,
                token_hash=token_hash,
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.db.add(whitelist_entry)
            self.db.commit()
            self.db.refresh(whitelist_entry)
            
            # Log audit event
            self._log_token_action(
                jti=jti,
                user_id=user_id,
                action="issued",
                token_type=token_type,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            return whitelist_entry
            
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token for whitelisting: {str(e)}")
            raise ValueError("Invalid token format")
        except Exception as e:
            logger.error(f"Error whitelisting token: {str(e)}")
            self.db.rollback()
            raise
    
    def get_user_active_tokens(self, user_id: int) -> List[TokenWhitelist]:
        """Get all active tokens for a user"""
        return self.db.query(TokenWhitelist).filter(
            and_(
                TokenWhitelist.user_id == user_id,
                TokenWhitelist.is_active == True,
                TokenWhitelist.expires_at > datetime.utcnow()
            )
        ).all()
    
    def get_user_blacklisted_tokens(self, user_id: int) -> List[JWTBlacklist]:
        """Get all blacklisted tokens for a user"""
        return self.db.query(JWTBlacklist).filter(
            and_(
                JWTBlacklist.user_id == user_id,
                JWTBlacklist.expires_at > datetime.utcnow()
            )
        ).all()
    
    def cleanup_expired_tokens(self) -> Tuple[int, int]:
        """Clean up expired tokens from blacklist and whitelist"""
        try:
            now = datetime.utcnow()
            
            # Clean up expired blacklist entries
            expired_blacklist = self.db.query(JWTBlacklist).filter(
                JWTBlacklist.expires_at < now
            ).count()
            
            self.db.query(JWTBlacklist).filter(
                JWTBlacklist.expires_at < now
            ).delete()
            
            # Clean up expired whitelist entries
            expired_whitelist = self.db.query(TokenWhitelist).filter(
                TokenWhitelist.expires_at < now
            ).count()
            
            self.db.query(TokenWhitelist).filter(
                TokenWhitelist.expires_at < now
            ).delete()
            
            self.db.commit()
            
            logger.info(f"Cleaned up {expired_blacklist} blacklisted and {expired_whitelist} whitelisted tokens")
            return expired_blacklist, expired_whitelist
            
        except Exception as e:
            logger.error(f"Error cleaning up expired tokens: {str(e)}")
            self.db.rollback()
            raise
    
    def get_token_audit_log(
        self,
        user_id: Optional[int] = None,
        jti: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[TokenAuditLog]:
        """Get token audit log with filters"""
        query = self.db.query(TokenAuditLog)
        
        if user_id:
            query = query.filter(TokenAuditLog.user_id == user_id)
        
        if jti:
            query = query.filter(TokenAuditLog.jti == jti)
        
        if action:
            query = query.filter(TokenAuditLog.action == action)
        
        return query.order_by(desc(TokenAuditLog.created_at)).limit(limit).all()
    
    def get_blacklist_stats(self) -> Dict[str, Any]:
        """Get blacklist statistics"""
        try:
            now = datetime.utcnow()
            
            total_blacklisted = self.db.query(JWTBlacklist).count()
            active_blacklisted = self.db.query(JWTBlacklist).filter(
                JWTBlacklist.expires_at > now
            ).count()
            
            total_whitelisted = self.db.query(TokenWhitelist).count()
            active_whitelisted = self.db.query(TokenWhitelist).filter(
                and_(
                    TokenWhitelist.is_active == True,
                    TokenWhitelist.expires_at > now
                )
            ).count()
            
            # Get revocation reasons breakdown
            revocation_reasons = self.db.query(
                JWTBlacklist.reason,
                func.count(JWTBlacklist.id).label('count')
            ).filter(
                JWTBlacklist.expires_at > now
            ).group_by(JWTBlacklist.reason).all()
            
            return {
                "total_blacklisted": total_blacklisted,
                "active_blacklisted": active_blacklisted,
                "total_whitelisted": total_whitelisted,
                "active_whitelisted": active_whitelisted,
                "revocation_reasons": {reason: count for reason, count in revocation_reasons}
            }
            
        except Exception as e:
            logger.error(f"Error getting blacklist stats: {str(e)}")
            return {}
    
    def _hash_token(self, token: str) -> str:
        """Create a hash of the token for efficient lookup"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def _remove_from_whitelist(self, jti: str) -> None:
        """Remove token from whitelist"""
        self.db.query(TokenWhitelist).filter(
            TokenWhitelist.jti == jti
        ).update({"is_active": False})
        self.db.commit()
    
    def _log_token_action(
        self,
        jti: str,
        user_id: int,
        action: str,
        token_type: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log token action for audit trail"""
        try:
            audit_log = TokenAuditLog(
                jti=jti,
                user_id=user_id,
                action=action,
                token_type=token_type,
                ip_address=ip_address,
                user_agent=user_agent,
                log_metadata=str(metadata) if metadata else None
            )
            
            self.db.add(audit_log)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error logging token action: {str(e)}")
            # Don't raise exception for audit logging failures
