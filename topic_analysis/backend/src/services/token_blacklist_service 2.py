"""
JWT Token Blacklisting Service using Redis.

This service provides functionality to blacklist JWT tokens for logout
and security purposes, using Redis for fast token validation.
"""
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
import json

from src.core.redis import get_redis_client, check_redis_connection
from src.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class TokenBlacklistService:
    """Service for managing JWT token blacklisting in Redis."""
    
    def __init__(self):
        self.redis_prefix = "blacklist:token:"
        self.user_sessions_prefix = "user:sessions:"
        self.default_ttl = settings.jwt_expiration_hours * 3600  # Convert to seconds
        
    def _get_token_key(self, token_jti: str) -> str:
        """Get Redis key for token blacklist."""
        return f"{self.redis_prefix}{token_jti}"
    
    def _get_user_sessions_key(self, user_id: str) -> str:
        """Get Redis key for user sessions."""
        return f"{self.user_sessions_prefix}{user_id}"
    
    def _serialize_token_data(self, token_data: Dict[str, Any]) -> str:
        """Serialize token data for Redis storage."""
        return json.dumps(token_data, default=str)
    
    def _deserialize_token_data(self, token_data_str: str) -> Dict[str, Any]:
        """Deserialize token data from Redis."""
        try:
            return json.loads(token_data_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to deserialize token data: {e}")
            return {}
    
    def blacklist_token(
        self, 
        token_jti: str, 
        user_id: str,
        expires_at: datetime,
        token_type: str = "access",
        reason: str = "logout"
    ) -> bool:
        """
        Add a token to the blacklist.
        
        Args:
            token_jti: JWT ID of the token to blacklist
            user_id: ID of the user who owns the token
            expires_at: When the token expires
            token_type: Type of token (access, refresh)
            reason: Reason for blacklisting (logout, security, etc.)
            
        Returns:
            bool: True if token was successfully blacklisted
        """
        try:
            with get_redis_context() as redis_client:
                # Calculate TTL (time to live) for the blacklist entry
                now = datetime.now(timezone.utc)
                ttl_seconds = int((expires_at - now).total_seconds())
                
                # If token is already expired, don't blacklist it
                if ttl_seconds <= 0:
                    logger.info(f"Token {token_jti} is already expired, skipping blacklist")
                    return True
                
                # Prepare token data for storage
                token_data = {
                    "jti": token_jti,
                    "user_id": user_id,
                    "token_type": token_type,
                    "blacklisted_at": now.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "reason": reason,
                    "ttl_seconds": ttl_seconds
                }
                
                # Store in Redis with TTL
                token_key = self._get_token_key(token_jti)
                redis_client.setex(
                    token_key,
                    ttl_seconds,
                    self._serialize_token_data(token_data)
                )
                
                # Add to user's session list for tracking
                user_sessions_key = self._get_user_sessions_key(user_id)
                redis_client.sadd(user_sessions_key, token_jti)
                
                # Set expiration for user sessions set (cleanup after token expires)
                redis_client.expire(user_sessions_key, ttl_seconds)
                
                logger.info(f"Token {token_jti} blacklisted for user {user_id} (TTL: {ttl_seconds}s)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to blacklist token {token_jti}: {e}")
            return False
    
    def is_token_blacklisted(self, token_jti: str) -> bool:
        """
        Check if a token is blacklisted.
        
        Args:
            token_jti: JWT ID of the token to check
            
        Returns:
            bool: True if token is blacklisted
        """
        try:
            with get_redis_context() as redis_client:
                token_key = self._get_token_key(token_jti)
                exists = redis_client.exists(token_key)
                
                if exists:
                    logger.debug(f"Token {token_jti} is blacklisted")
                    return True
                else:
                    logger.debug(f"Token {token_jti} is not blacklisted")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to check token blacklist status for {token_jti}: {e}")
            # In case of Redis error, assume token is not blacklisted to avoid blocking valid users
            return False
    
    def get_blacklisted_token_info(self, token_jti: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a blacklisted token.
        
        Args:
            token_jti: JWT ID of the token
            
        Returns:
            Dict with token information or None if not blacklisted
        """
        try:
            with get_redis_context() as redis_client:
                token_key = self._get_token_key(token_jti)
                token_data_str = redis_client.get(token_key)
                
                if token_data_str:
                    return self._deserialize_token_data(token_data_str)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get blacklisted token info for {token_jti}: {e}")
            return None
    
    def get_user_blacklisted_tokens(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all blacklisted tokens for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of blacklisted token information
        """
        try:
            with get_redis_context() as redis_client:
                user_sessions_key = self._get_user_sessions_key(user_id)
                token_jtis = redis_client.smembers(user_sessions_key)
                
                blacklisted_tokens = []
                for token_jti in token_jtis:
                    token_info = self.get_blacklisted_token_info(token_jti)
                    if token_info:
                        blacklisted_tokens.append(token_info)
                
                return blacklisted_tokens
                
        except Exception as e:
            logger.error(f"Failed to get blacklisted tokens for user {user_id}: {e}")
            return []
    
    def remove_token_from_blacklist(self, token_jti: str) -> bool:
        """
        Remove a token from the blacklist (unblacklist).
        
        Args:
            token_jti: JWT ID of the token to remove
            
        Returns:
            bool: True if token was successfully removed
        """
        try:
            with get_redis_context() as redis_client:
                token_key = self._get_token_key(token_jti)
                
                # Get token info before deletion
                token_info = self.get_blacklisted_token_info(token_jti)
                
                # Remove from blacklist
                deleted = redis_client.delete(token_key)
                
                if deleted and token_info:
                    # Remove from user's session list
                    user_id = token_info.get("user_id")
                    if user_id:
                        user_sessions_key = self._get_user_sessions_key(user_id)
                        redis_client.srem(user_sessions_key, token_jti)
                
                logger.info(f"Token {token_jti} removed from blacklist")
                return bool(deleted)
                
        except Exception as e:
            logger.error(f"Failed to remove token {token_jti} from blacklist: {e}")
            return False
    
    def blacklist_user_tokens(self, user_id: str, reason: str = "security") -> int:
        """
        Blacklist all active tokens for a user.
        
        Args:
            user_id: ID of the user
            reason: Reason for blacklisting all tokens
            
        Returns:
            int: Number of tokens blacklisted
        """
        try:
            with get_redis_context() as redis_client:
                user_sessions_key = self._get_user_sessions_key(user_id)
                token_jtis = redis_client.smembers(user_sessions_key)
                
                blacklisted_count = 0
                for token_jti in token_jtis:
                    token_info = self.get_blacklisted_token_info(token_jti)
                    if token_info and not token_info.get("blacklisted_at"):
                        # Token exists but not blacklisted yet
                        expires_at = datetime.fromisoformat(token_info["expires_at"])
                        if self.blacklist_token(token_jti, user_id, expires_at, reason=reason):
                            blacklisted_count += 1
                
                logger.info(f"Blacklisted {blacklisted_count} tokens for user {user_id}")
                return blacklisted_count
                
        except Exception as e:
            logger.error(f"Failed to blacklist all tokens for user {user_id}: {e}")
            return 0
    
    def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired blacklisted tokens.
        This is typically called by a background task.
        
        Returns:
            int: Number of tokens cleaned up
        """
        try:
            with get_redis_context() as redis_client:
                # Get all blacklist keys
                pattern = f"{self.redis_prefix}*"
                keys = redis_client.keys(pattern)
                
                cleaned_count = 0
                for key in keys:
                    # Check if key has expired (TTL = -1 means expired)
                    ttl = redis_client.ttl(key)
                    if ttl == -1:
                        redis_client.delete(key)
                        cleaned_count += 1
                
                logger.info(f"Cleaned up {cleaned_count} expired blacklisted tokens")
                return cleaned_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired tokens: {e}")
            return 0
    
    def get_blacklist_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the token blacklist.
        
        Returns:
            Dict with blacklist statistics
        """
        try:
            with get_redis_context() as redis_client:
                # Count blacklisted tokens
                pattern = f"{self.redis_prefix}*"
                blacklisted_tokens = len(redis_client.keys(pattern))
                
                # Count user sessions
                user_pattern = f"{self.user_sessions_prefix}*"
                user_sessions = len(redis_client.keys(user_pattern))
                
                return {
                    "blacklisted_tokens": blacklisted_tokens,
                    "user_sessions": user_sessions,
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"Failed to get blacklist stats: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the token blacklist service.
        
        Returns:
            Dict with health status
        """
        try:
            # Test Redis connection
            if not test_redis_connection():
                return {
                    "status": "unhealthy",
                    "error": "Redis connection failed",
                    "timestamp": time.time()
                }
            
            # Test basic operations
            test_jti = "health_check_test"
            test_user_id = "health_check_user"
            test_expires = datetime.utcnow() + timedelta(minutes=1)
            
            # Test blacklisting
            blacklist_success = self.blacklist_token(
                test_jti, 
                test_user_id, 
                test_expires, 
                reason="health_check"
            )
            
            if not blacklist_success:
                return {
                    "status": "unhealthy",
                    "error": "Failed to blacklist test token",
                    "timestamp": time.time()
                }
            
            # Test checking blacklist
            is_blacklisted = self.is_token_blacklisted(test_jti)
            
            if not is_blacklisted:
                return {
                    "status": "unhealthy",
                    "error": "Failed to verify blacklisted token",
                    "timestamp": time.time()
                }
            
            # Clean up test token
            self.remove_token_from_blacklist(test_jti)
            
            return {
                "status": "healthy",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Token blacklist health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

# Global instance
token_blacklist_service = TokenBlacklistService()
