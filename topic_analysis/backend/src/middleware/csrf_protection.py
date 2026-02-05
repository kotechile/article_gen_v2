"""
CSRF protection middleware for FastAPI
"""
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import List, Optional, Set
import logging
import time
from urllib.parse import urlparse

from ..services.csrf_protection import CSRFProtectionService
from ..core.database import get_db
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """CSRF protection middleware"""
    
    def __init__(
        self,
        app: ASGIApp,
        exempt_paths: Optional[List[str]] = None,
        exempt_methods: Optional[Set[str]] = None,
        token_header: str = "X-CSRF-Token",
        cookie_name: str = "csrf_token",
        require_https: bool = True
    ):
        super().__init__(app)
        self.exempt_paths = exempt_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/auth/login",
            "/auth/register",
            "/auth/refresh"
        ]
        self.exempt_methods = exempt_methods or {"GET", "HEAD", "OPTIONS"}
        self.token_header = token_header
        self.cookie_name = cookie_name
        self.require_https = require_https
        
        # CSRF protection patterns
        self.safe_methods = {"GET", "HEAD", "OPTIONS", "TRACE"}
        self.unsafe_methods = {"POST", "PUT", "PATCH", "DELETE"}
        
        # Rate limiting for CSRF violations
        self.violation_counts = {}
        self.rate_limit_window = 300  # 5 minutes
        self.max_violations_per_window = 10
    
    async def dispatch(self, request: Request, call_next):
        """Process CSRF protection for each request"""
        start_time = time.time()
        
        try:
            # Skip CSRF protection for exempt paths
            if self._is_exempt_path(request.url.path):
                response = await call_next(request)
                return response
            
            # Skip CSRF protection for safe methods
            if request.method in self.exempt_methods:
                response = await call_next(request)
                return response
            
            # Get database session
            db = next(get_db())
            csrf_service = CSRFProtectionService(db)
            
            try:
                # Extract user information
                user_id = self._get_user_id_from_request(request)
                if not user_id:
                    # No user context, skip CSRF protection
                    response = await call_next(request)
                    return response
                
                # Check if IP is blocked due to violations
                client_ip = self._get_client_ip(request)
                if csrf_service.should_block_ip(client_ip):
                    logger.warning(f"Blocked request from IP {client_ip} due to CSRF violations")
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={"detail": "Access denied due to security violations"}
                    )
                
                # Validate CSRF token
                csrf_token = self._extract_csrf_token(request)
                origin = request.headers.get("origin")
                referer = request.headers.get("referer")
                user_agent = request.headers.get("user-agent")
                session_id = self._get_session_id(request)
                
                is_valid, error_message = csrf_service.validate_csrf_token(
                    token=csrf_token,
                    user_id=user_id,
                    session_id=session_id,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    origin=origin,
                    referer=referer
                )
                
                if not is_valid:
                    # Log the violation
                    self._log_csrf_violation(client_ip, error_message)
                    
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "detail": "CSRF token validation failed",
                            "error": error_message,
                            "code": "csrf_validation_failed"
                        }
                    )
                
                # CSRF validation passed, proceed with request
                response = await call_next(request)
                
                # Add CSRF token to response if needed
                if self._should_add_csrf_token(request, response):
                    new_token = csrf_service.generate_csrf_token(
                        user_id=user_id,
                        session_id=session_id,
                        ip_address=client_ip,
                        user_agent=user_agent
                    )
                    response.set_cookie(
                        key=self.cookie_name,
                        value=new_token,
                        httponly=False,  # Allow JavaScript access
                        secure=self.require_https,
                        samesite="strict"
                    )
                
                # Add security headers
                self._add_csrf_headers(response)
                
                return response
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"CSRF middleware error: {str(e)}")
            # Don't block requests due to middleware errors
            response = await call_next(request)
            return response
    
    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from CSRF protection"""
        for exempt_path in self.exempt_paths:
            if path.startswith(exempt_path):
                return True
        return False
    
    def _get_user_id_from_request(self, request: Request) -> Optional[int]:
        """Extract user ID from request (from JWT token, session, etc.)"""
        # This would typically extract from JWT token or session
        # For now, return None to indicate no user context
        # In a real implementation, this would parse the JWT token
        return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"
    
    def _extract_csrf_token(self, request: Request) -> Optional[str]:
        """Extract CSRF token from request"""
        # Check header first
        token = request.headers.get(self.token_header)
        if token:
            return token
        
        # Check cookie
        token = request.cookies.get(self.cookie_name)
        if token:
            return token
        
        # Check form data for POST requests
        if request.method == "POST":
            try:
                form_data = request.form()
                token = form_data.get("csrf_token")
                if token:
                    return token
            except:
                pass
        
        return None
    
    def _get_session_id(self, request: Request) -> Optional[str]:
        """Get session ID from request"""
        return request.cookies.get("session_id")
    
    def _should_add_csrf_token(self, request: Request, response: Response) -> bool:
        """Determine if CSRF token should be added to response"""
        # Add token for successful responses that might need CSRF protection
        if response.status_code < 400:
            return True
        return False
    
    def _add_csrf_headers(self, response: Response) -> None:
        """Add CSRF-related security headers"""
        response.headers["X-CSRF-Protection"] = "1"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
    
    def _log_csrf_violation(self, ip_address: str, error_message: str) -> None:
        """Log CSRF violation for rate limiting"""
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - self.rate_limit_window
        self.violation_counts = {
            ip: count for ip, count in self.violation_counts.items()
            if count > cutoff_time
        }
        
        # Track violation
        if ip_address not in self.violation_counts:
            self.violation_counts[ip_address] = []
        
        self.violation_counts[ip_address].append(current_time)
        
        # Check if IP should be temporarily blocked
        recent_violations = [
            t for t in self.violation_counts[ip_address]
            if t > cutoff_time
        ]
        
        if len(recent_violations) >= self.max_violations_per_window:
            logger.warning(f"IP {ip_address} has exceeded CSRF violation rate limit")
        
        logger.warning(f"CSRF violation from {ip_address}: {error_message}")

class CSRFProtectionConfig:
    """Configuration for CSRF protection"""
    
    def __init__(self):
        self.enabled = getattr(settings, 'CSRF_PROTECTION_ENABLED', True)
        self.exempt_paths = getattr(settings, 'CSRF_EXEMPT_PATHS', [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/auth/login",
            "/auth/register",
            "/auth/refresh"
        ])
        self.exempt_methods = set(getattr(settings, 'CSRF_EXEMPT_METHODS', [
            "GET", "HEAD", "OPTIONS"
        ]))
        self.token_header = getattr(settings, 'CSRF_TOKEN_HEADER', 'X-CSRF-Token')
        self.cookie_name = getattr(settings, 'CSRF_COOKIE_NAME', 'csrf_token')
        self.require_https = getattr(settings, 'CSRF_REQUIRE_HTTPS', True)
        self.token_ttl_hours = getattr(settings, 'CSRF_TOKEN_TTL_HOURS', 1)
        self.max_violations_per_ip = getattr(settings, 'CSRF_MAX_VIOLATIONS_PER_IP', 5)
        self.violation_window_hours = getattr(settings, 'CSRF_VIOLATION_WINDOW_HOURS', 1)

def create_csrf_middleware() -> CSRFProtectionMiddleware:
    """Create CSRF protection middleware with configuration"""
    config = CSRFProtectionConfig()
    
    return CSRFProtectionMiddleware(
        app=None,  # Will be set by FastAPI
        exempt_paths=config.exempt_paths,
        exempt_methods=config.exempt_methods,
        token_header=config.token_header,
        cookie_name=config.cookie_name,
        require_https=config.require_https
    )
