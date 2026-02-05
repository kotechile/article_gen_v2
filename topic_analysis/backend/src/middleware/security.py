"""
CORS and Security Headers Middleware
Implements comprehensive security headers and CORS configuration
"""

import re
from typing import List, Optional, Dict, Any
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from ..core.config import settings

logger = structlog.get_logger()

class SecurityHeadersMiddleware:
    """Security headers middleware"""
    
    def __init__(self):
        self.allowed_origins = self._get_allowed_origins()
        self.allowed_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
        self.allowed_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-CSRF-Token",
            "X-API-Key"
        ]
        self.exposed_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Request-ID"
        ]
    
    def _get_allowed_origins(self) -> List[str]:
        """Get allowed origins from configuration"""
        origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "https://trendtap.com",
            "https://www.trendtap.com",
            "https://app.trendtap.com"
        ]
        
        # Add development origins if in debug mode
        if settings.DEBUG_MODE:
            origins.extend([
                "http://127.0.0.1:3000",
                "http://127.0.0.1:3001",
                "http://localhost:8080",
                "http://127.0.0.1:8080"
            ])
        
        return origins
    
    async def __call__(self, request: Request, call_next):
        """Process request and add security headers"""
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(request, response)
        
        return response
    
    def _add_security_headers(self, request: Request, response: Response):
        """Add comprehensive security headers"""
        
        # Content Security Policy
        csp_policy = self._get_csp_policy()
        response.headers["Content-Security-Policy"] = csp_policy
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = self._get_permissions_policy()
        
        # HSTS (only for HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Cross-Origin policies
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        
        # Additional security headers
        response.headers["X-DNS-Prefetch-Control"] = "off"
        response.headers["X-Download-Options"] = "noopen"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        
        # Remove server information
        if "Server" in response.headers:
            del response.headers["Server"]
        
        # Add custom security headers
        response.headers["X-API-Version"] = settings.APP_VERSION
        response.headers["X-Request-ID"] = self._generate_request_id()
    
    def _get_csp_policy(self) -> str:
        """Get Content Security Policy"""
        if settings.DEBUG_MODE:
            # More permissive CSP for development
            return (
                "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: blob: https:; "
                "connect-src 'self' https: wss:; "
                "frame-src 'none'; "
                "object-src 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
        else:
            # Strict CSP for production
            return (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https:; "
                "frame-src 'none'; "
                "object-src 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
    
    def _get_permissions_policy(self) -> str:
        """Get Permissions Policy"""
        return (
            "accelerometer=(), "
            "ambient-light-sensor=(), "
            "autoplay=(), "
            "battery=(), "
            "camera=(), "
            "display-capture=(), "
            "document-domain=(), "
            "encrypted-media=(), "
            "fullscreen=(self), "
            "geolocation=(), "
            "gyroscope=(), "
            "magnetometer=(), "
            "microphone=(), "
            "midi=(), "
            "payment=(), "
            "picture-in-picture=(), "
            "publickey-credentials-get=(), "
            "screen-wake-lock=(), "
            "sync-xhr=(), "
            "usb=(), "
            "web-share=(), "
            "xr-spatial-tracking=()"
        )
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import uuid
        return str(uuid.uuid4())

class CORSConfig:
    """CORS configuration manager"""
    
    def __init__(self):
        self.allowed_origins = self._get_allowed_origins()
        self.allowed_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
        self.allowed_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-CSRF-Token",
            "X-API-Key",
            "X-Request-ID"
        ]
        self.exposed_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Request-ID",
            "X-API-Version"
        ]
        self.allow_credentials = True
        self.max_age = 3600
    
    def _get_allowed_origins(self) -> List[str]:
        """Get allowed origins from configuration"""
        origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "https://trendtap.com",
            "https://www.trendtap.com",
            "https://app.trendtap.com"
        ]
        
        # Add development origins if in debug mode
        if settings.DEBUG_MODE:
            origins.extend([
                "http://127.0.0.1:3000",
                "http://127.0.0.1:3001",
                "http://localhost:8080",
                "http://127.0.0.1:8080"
            ])
        
        return origins
    
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        if not origin:
            return False
        
        # Check exact match
        if origin in self.allowed_origins:
            return True
        
        # Check wildcard patterns
        for allowed_origin in self.allowed_origins:
            if "*" in allowed_origin:
                pattern = allowed_origin.replace("*", ".*")
                if re.match(pattern, origin):
                    return True
        
        return False
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "allow_origins": self.allowed_origins,
            "allow_methods": self.allowed_methods,
            "allow_headers": self.allowed_headers,
            "expose_headers": self.exposed_headers,
            "allow_credentials": self.allow_credentials,
            "max_age": self.max_age
        }

class CSRFProtection:
    """CSRF protection middleware"""
    
    def __init__(self):
        self.exempt_paths = [
            "/api/auth/login",
            "/api/auth/register",
            "/api/auth/forgot-password",
            "/api/health"
        ]
        self.exempt_methods = ["GET", "HEAD", "OPTIONS"]
    
    def is_exempt(self, request: Request) -> bool:
        """Check if request is exempt from CSRF protection"""
        # Check path
        if request.url.path in self.exempt_paths:
            return True
        
        # Check method
        if request.method in self.exempt_methods:
            return True
        
        # Check if it's an API endpoint (exempt for now)
        if request.url.path.startswith("/api/"):
            return True
        
        return False
    
    async def __call__(self, request: Request, call_next):
        """CSRF protection middleware"""
        if self.is_exempt(request):
            return await call_next(request)
        
        # Check CSRF token
        csrf_token = request.headers.get("X-CSRF-Token")
        if not csrf_token:
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF token missing"}
            )
        
        # Validate CSRF token (simplified)
        if not self._validate_csrf_token(csrf_token, request):
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid CSRF token"}
            )
        
        return await call_next(request)
    
    def _validate_csrf_token(self, token: str, request: Request) -> bool:
        """Validate CSRF token (simplified implementation)"""
        # In a real implementation, you would validate the token
        # against a stored value or session
        return len(token) > 0

class SecurityMiddleware:
    """Main security middleware coordinator"""
    
    def __init__(self):
        self.security_headers = SecurityHeadersMiddleware()
        self.cors_config = CORSConfig()
        self.csrf_protection = CSRFProtection()
    
    async def __call__(self, request: Request, call_next):
        """Main security middleware"""
        try:
            # Add security headers
            response = await call_next(request)
            self.security_headers._add_security_headers(request, response)
            
            return response
            
        except Exception as e:
            logger.error("Security middleware error", error=str(e))
            return await call_next(request)

# Global instances
security_middleware = SecurityMiddleware()
cors_config = CORSConfig()
csrf_protection = CSRFProtection()

# CORS middleware configuration
def get_cors_middleware():
    """Get configured CORS middleware"""
    return CORSMiddleware(
        allow_origins=cors_config.allowed_origins,
        allow_credentials=cors_config.allow_credentials,
        allow_methods=cors_config.allowed_methods,
        allow_headers=cors_config.allowed_headers,
        expose_headers=cors_config.exposed_headers,
        max_age=cors_config.max_age
    )

# Utility functions
def is_origin_allowed(origin: str) -> bool:
    """Check if origin is allowed"""
    return cors_config.is_origin_allowed(origin)

def get_security_headers() -> Dict[str, str]:
    """Get security headers configuration"""
    return {
        "Content-Security-Policy": security_middleware.security_headers._get_csp_policy(),
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": security_middleware.security_headers._get_permissions_policy()
    }

def validate_csrf_token(token: str, request: Request) -> bool:
    """Validate CSRF token"""
    return csrf_protection._validate_csrf_token(token, request)

def is_csrf_exempt(request: Request) -> bool:
    """Check if request is exempt from CSRF protection"""
    return csrf_protection.is_exempt(request)