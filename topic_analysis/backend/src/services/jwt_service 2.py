"""
JWT service for token generation and validation with blacklist support.
"""
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from uuid import uuid4
import os
import logging
from src.schemas.auth_schemas import TokenData
from src.services.token_blacklist_service import token_blacklist_service

# Configure logging
logger = logging.getLogger(__name__)

class JWTService:
    """Service for JWT token operations."""
    
    def __init__(self):
        self.secret_key = os.getenv("SECRET_KEY", "your-secret-key-here")
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
        self.refresh_token_expire_days = int(os.getenv("JWT_REFRESH_EXPIRATION_DAYS", "7"))
    
    def create_access_token(self, user_id: str, email: str, role: str) -> str:
        """Create an access token."""
        jti = str(uuid4())
        expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "user_id": user_id,
            "email": email,
            "role": role,
            "jti": jti,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create a refresh token."""
        jti = str(uuid4())
        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "user_id": user_id,
            "jti": jti,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode a JWT token with blacklist check."""
        try:
            # First decode the token to get the JTI
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            jti = payload.get("jti")
            
            # Check if token is blacklisted
            if jti and token_blacklist_service.is_token_blacklisted(jti):
                logger.warning(f"Token with JTI {jti} is blacklisted")
                return None
            
            return TokenData(
                user_id=payload.get("user_id"),
                email=payload.get("email"),
                role=payload.get("role"),
                jti=jti
            )
        except jwt.ExpiredSignatureError:
            logger.debug("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.debug(f"Invalid token: {e}")
            return None
    
    def get_token_expiration(self, token: str) -> Optional[datetime]:
        """Get token expiration time."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
            return datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc)
        except jwt.InvalidTokenError:
            return None
    
    def is_token_expired(self, token: str) -> bool:
        """Check if token is expired."""
        try:
            jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return False
        except jwt.ExpiredSignatureError:
            return True
        except jwt.InvalidTokenError:
            return True
    
    def extract_jti(self, token: str) -> Optional[str]:
        """Extract JTI from token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
            return payload.get("jti")
        except jwt.InvalidTokenError:
            return None
    
    def get_token_payload(self, token: str) -> Optional[Dict[str, Any]]:
        """Get token payload without verification (for debugging)."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
            return payload
        except jwt.InvalidTokenError:
            return None
    
    def is_refresh_token(self, token: str) -> bool:
        """Check if token is a refresh token."""
        payload = self.get_token_payload(token)
        return payload and payload.get("type") == "refresh"
    
    def is_access_token(self, token: str) -> bool:
        """Check if token is an access token."""
        payload = self.get_token_payload(token)
        return payload and payload.get("type") == "access"
    
    def get_user_id_from_token(self, token: str) -> Optional[str]:
        """Extract user ID from token."""
        payload = self.get_token_payload(token)
        return payload.get("user_id") if payload else None
    
    def get_token_type(self, token: str) -> Optional[str]:
        """Get token type (access or refresh)."""
        payload = self.get_token_payload(token)
        return payload.get("type") if payload else None
    
    def blacklist_token(self, token: str, reason: str = "logout") -> bool:
        """
        Blacklist a token.
        
        Args:
            token: JWT token to blacklist
            reason: Reason for blacklisting
            
        Returns:
            bool: True if token was successfully blacklisted
        """
        try:
            # Get token payload to extract JTI and expiration
            payload = self.get_token_payload(token)
            if not payload:
                logger.error("Cannot blacklist invalid token")
                return False
            
            jti = payload.get("jti")
            user_id = payload.get("user_id")
            token_type = payload.get("type", "access")
            exp_timestamp = payload.get("exp")
            
            if not jti or not user_id or not exp_timestamp:
                logger.error("Token missing required fields for blacklisting")
                return False
            
            # Convert expiration timestamp to datetime
            expires_at = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
            
            # Blacklist the token
            return token_blacklist_service.blacklist_token(
                jti,
                user_id,
                expires_at,
                token_type,
                reason
            )
            
        except Exception as e:
            logger.error(f"Failed to blacklist token: {e}")
            return False
    
    def blacklist_user_tokens(self, user_id: str, reason: str = "security") -> int:
        """
        Blacklist all tokens for a user.
        
        Args:
            user_id: ID of the user
            reason: Reason for blacklisting
            
        Returns:
            int: Number of tokens blacklisted
        """
        try:
            return token_blacklist_service.blacklist_user_tokens(user_id, reason)
        except Exception as e:
            logger.error(f"Failed to blacklist user tokens for {user_id}: {e}")
            return 0
    
    def is_token_blacklisted(self, token: str) -> bool:
        """
        Check if a token is blacklisted.
        
        Args:
            token: JWT token to check
            
        Returns:
            bool: True if token is blacklisted
        """
        try:
            jti = self.extract_jti(token)
            if not jti:
                return False
            
            return token_blacklist_service.is_token_blacklisted(jti)
        except Exception as e:
            logger.error(f"Failed to check token blacklist status: {e}")
            return False
    
    def get_blacklist_stats(self) -> Dict[str, Any]:
        """Get token blacklist statistics."""
        try:
            return token_blacklist_service.get_blacklist_stats()
        except Exception as e:
            logger.error(f"Failed to get blacklist stats: {e}")
            return {"error": str(e)}

# Create singleton instance
jwt_service = JWTService()
