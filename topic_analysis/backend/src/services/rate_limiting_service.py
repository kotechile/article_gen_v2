"""
Rate Limiting Service
Manages API rate limiting for external services and user requests
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import structlog
from ..core.redis import cache
from ..core.config import settings

logger = structlog.get_logger()

class RateLimitingService:
    """Service for managing API rate limiting"""
    
    def __init__(self):
        self.redis = cache
        self.default_limits = {
            "user_requests": {"limit": 100, "window": 3600},  # 100 requests per hour
            "api_calls": {"limit": 1000, "window": 3600},      # 1000 API calls per hour
            "file_uploads": {"limit": 10, "window": 3600},    # 10 file uploads per hour
            "trend_analysis": {"limit": 5, "window": 3600},   # 5 trend analyses per hour
            "content_generation": {"limit": 20, "window": 3600}  # 20 content generations per hour
        }
        
        # External API rate limits
        self.external_api_limits = {
            "openai": {"limit": 100, "window": 3600},
            "anthropic": {"limit": 100, "window": 3600},
            "google_ai": {"limit": 100, "window": 3600},
            "dataforseo": {"limit": 1000, "window": 3600},
            "google_trends": {"limit": 100, "window": 3600},
            "linkup": {"limit": 100, "window": 3600},
            "surfer_seo": {"limit": 50, "window": 3600},
            "frase": {"limit": 50, "window": 3600},
            "coschedule": {"limit": 100, "window": 3600}
        }
    
    async def check_rate_limit(
        self,
        user_id: str,
        action: str,
        api_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if user has exceeded rate limit for an action
        
        Args:
            user_id: User ID
            action: Action being performed
            api_provider: Optional API provider for external API limits
            
        Returns:
            Dict containing rate limit status
        """
        try:
            logger.info("Checking rate limit", 
                       user_id=user_id,
                       action=action,
                       api_provider=api_provider)
            
            # Get rate limit configuration
            if api_provider and api_provider in self.external_api_limits:
                limit_config = self.external_api_limits[api_provider]
                limit_key = f"rate_limit:{api_provider}:{user_id}"
            else:
                limit_config = self.default_limits.get(action, self.default_limits["user_requests"])
                limit_key = f"rate_limit:{action}:{user_id}"
            
            # Check current usage
            current_usage = await self._get_current_usage(limit_key, limit_config["window"])
            
            # Check if limit exceeded
            if current_usage >= limit_config["limit"]:
                logger.warning("Rate limit exceeded", 
                             user_id=user_id,
                             action=action,
                             current_usage=current_usage,
                             limit=limit_config["limit"])
                
                return {
                    "allowed": False,
                    "current_usage": current_usage,
                    "limit": limit_config["limit"],
                    "window": limit_config["window"],
                    "reset_time": self._calculate_reset_time(limit_config["window"]),
                    "message": f"Rate limit exceeded for {action}. Try again later."
                }
            
            # Increment usage counter
            await self._increment_usage(limit_key, limit_config["window"])
            
            logger.info("Rate limit check passed", 
                       user_id=user_id,
                       action=action,
                       current_usage=current_usage + 1,
                       limit=limit_config["limit"])
            
            return {
                "allowed": True,
                "current_usage": current_usage + 1,
                "limit": limit_config["limit"],
                "window": limit_config["window"],
                "remaining": limit_config["limit"] - (current_usage + 1)
            }
            
        except Exception as e:
            logger.error("Failed to check rate limit", error=str(e))
            # Allow request on error to avoid blocking legitimate users
            return {
                "allowed": True,
                "current_usage": 0,
                "limit": 1000,
                "window": 3600,
                "remaining": 1000,
                "error": "Rate limit check failed, allowing request"
            }
    
    async def get_rate_limit_status(
        self,
        user_id: str,
        actions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get current rate limit status for user
        
        Args:
            user_id: User ID
            actions: Optional list of actions to check
            
        Returns:
            Dict containing rate limit status for all actions
        """
        try:
            logger.info("Getting rate limit status", 
                       user_id=user_id,
                       actions=actions)
            
            if not actions:
                actions = list(self.default_limits.keys())
            
            status = {}
            
            for action in actions:
                limit_config = self.default_limits.get(action, self.default_limits["user_requests"])
                limit_key = f"rate_limit:{action}:{user_id}"
                
                current_usage = await self._get_current_usage(limit_key, limit_config["window"])
                
                status[action] = {
                    "current_usage": current_usage,
                    "limit": limit_config["limit"],
                    "window": limit_config["window"],
                    "remaining": limit_config["limit"] - current_usage,
                    "percentage_used": (current_usage / limit_config["limit"]) * 100,
                    "reset_time": self._calculate_reset_time(limit_config["window"])
                }
            
            logger.info("Rate limit status retrieved", 
                       user_id=user_id,
                       actions_checked=len(actions))
            
            return {
                "success": True,
                "user_id": user_id,
                "status": status,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get rate limit status", error=str(e))
            raise
    
    async def reset_rate_limit(
        self,
        user_id: str,
        action: str,
        api_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reset rate limit for a user and action
        
        Args:
            user_id: User ID
            action: Action to reset
            api_provider: Optional API provider for external API limits
            
        Returns:
            Dict containing reset results
        """
        try:
            logger.info("Resetting rate limit", 
                       user_id=user_id,
                       action=action,
                       api_provider=api_provider)
            
            # Determine limit key
            if api_provider and api_provider in self.external_api_limits:
                limit_key = f"rate_limit:{api_provider}:{user_id}"
            else:
                limit_key = f"rate_limit:{action}:{user_id}"
            
            # Delete rate limit key
            await self.redis.delete(limit_key)
            
            logger.info("Rate limit reset successfully", 
                       user_id=user_id,
                       action=action,
                       api_provider=api_provider)
            
            return {
                "success": True,
                "user_id": user_id,
                "action": action,
                "api_provider": api_provider,
                "reset_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to reset rate limit", error=str(e))
            raise
    
    async def update_rate_limit_config(
        self,
        action: str,
        limit: int,
        window: int,
        api_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update rate limit configuration
        
        Args:
            action: Action name
            limit: New limit value
            window: New window value in seconds
            api_provider: Optional API provider for external API limits
            
        Returns:
            Dict containing update results
        """
        try:
            logger.info("Updating rate limit config", 
                       action=action,
                       limit=limit,
                       window=window,
                       api_provider=api_provider)
            
            # Update configuration
            if api_provider and api_provider in self.external_api_limits:
                self.external_api_limits[api_provider] = {"limit": limit, "window": window}
            else:
                self.default_limits[action] = {"limit": limit, "window": window}
            
            logger.info("Rate limit config updated successfully", 
                       action=action,
                       limit=limit,
                       window=window,
                       api_provider=api_provider)
            
            return {
                "success": True,
                "action": action,
                "limit": limit,
                "window": window,
                "api_provider": api_provider,
                "updated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to update rate limit config", error=str(e))
            raise
    
    async def get_rate_limit_config(
        self,
        action: Optional[str] = None,
        api_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get rate limit configuration
        
        Args:
            action: Optional action name
            api_provider: Optional API provider
            
        Returns:
            Dict containing rate limit configuration
        """
        try:
            logger.info("Getting rate limit config", 
                       action=action,
                       api_provider=api_provider)
            
            if api_provider and api_provider in self.external_api_limits:
                config = self.external_api_limits[api_provider]
                return {
                    "success": True,
                    "api_provider": api_provider,
                    "config": config,
                    "timestamp": datetime.utcnow().isoformat()
                }
            elif action and action in self.default_limits:
                config = self.default_limits[action]
                return {
                    "success": True,
                    "action": action,
                    "config": config,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": True,
                    "default_limits": self.default_limits,
                    "external_api_limits": self.external_api_limits,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get rate limit config", error=str(e))
            raise
    
    async def _get_current_usage(self, limit_key: str, window: int) -> int:
        """Get current usage count for a rate limit key"""
        try:
            # Get current timestamp
            current_time = int(time.time())
            
            # Get usage count from Redis
            usage_count = await self.redis.get(limit_key)
            
            if usage_count is None:
                return 0
            
            # Parse usage data
            usage_data = eval(usage_count) if isinstance(usage_count, str) else usage_count
            
            # Filter out expired entries
            valid_entries = [
                timestamp for timestamp in usage_data
                if current_time - timestamp < window
            ]
            
            return len(valid_entries)
            
        except Exception as e:
            logger.error("Failed to get current usage", error=str(e))
            return 0
    
    async def _increment_usage(self, limit_key: str, window: int) -> None:
        """Increment usage count for a rate limit key"""
        try:
            # Get current timestamp
            current_time = int(time.time())
            
            # Get existing usage data
            usage_count = await self.redis.get(limit_key)
            
            if usage_count is None:
                usage_data = []
            else:
                usage_data = eval(usage_count) if isinstance(usage_count, str) else usage_count
            
            # Add current timestamp
            usage_data.append(current_time)
            
            # Filter out expired entries
            valid_entries = [
                timestamp for timestamp in usage_data
                if current_time - timestamp < window
            ]
            
            # Store updated usage data
            await self.redis.setex(
                limit_key,
                window,
                str(valid_entries)
            )
            
        except Exception as e:
            logger.error("Failed to increment usage", error=str(e))
    
    def _calculate_reset_time(self, window: int) -> str:
        """Calculate when the rate limit will reset"""
        reset_time = datetime.utcnow() + timedelta(seconds=window)
        return reset_time.isoformat()
    
    async def cleanup_expired_limits(self) -> Dict[str, Any]:
        """Clean up expired rate limit entries"""
        try:
            logger.info("Cleaning up expired rate limits")
            
            # Get all rate limit keys
            pattern = "rate_limit:*"
            keys = await self.redis.keys(pattern)
            
            cleaned_count = 0
            
            for key in keys:
                try:
                    # Check if key has expired
                    ttl = await self.redis.ttl(key)
                    if ttl <= 0:
                        await self.redis.delete(key)
                        cleaned_count += 1
                except Exception as e:
                    logger.warning("Failed to clean up key", key=key, error=str(e))
            
            logger.info("Expired rate limits cleaned up", 
                       keys_checked=len(keys),
                       keys_cleaned=cleaned_count)
            
            return {
                "success": True,
                "keys_checked": len(keys),
                "keys_cleaned": cleaned_count,
                "cleaned_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to cleanup expired limits", error=str(e))
            raise

