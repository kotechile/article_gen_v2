"""
Enhanced CORS and security headers middleware for FastAPI application.
"""
import logging
import re
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urlparse

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.core.config import settings

logger = logging.getLogger(__name__)

class EnhancedCORSMiddleware(BaseHTTPMiddleware):
    """Enhanced CORS middleware with advanced security features."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.allowed_origins = self._parse_origins(settings.get_cors_origins())
        self.allowed_methods = [
            "GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"
        ]
        self.allowed_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Request-ID",
            "X-CSRF-Token",
            "X-API-Key",
            "X-Client-Version",
            "X-Client-Platform"
        ]
        self.expose_headers = [
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Response-Time"
        ]
        self.max_age = 3600  # 1 hour
        self.allow_credentials = True
        self.allow_wildcard_subdomains = settings.is_development()
        
        # Security features
        self.enable_origin_validation = True
        self.enable_method_validation = True
        self.enable_header_validation = True
        self.block_suspicious_origins = True
        
        # Suspicious patterns
        self.suspicious_patterns = [
            r"\.(tk|ml|ga|cf)$",  # Free domains often used for attacks
            r"localhost:\d+",     # Localhost with port (development only)
            r"127\.0\.0\.1:\d+",  # Local IP with port
            r"0\.0\.0\.0:\d+",    # All interfaces
        ]
    
    def _parse_origins(self, origins: List[str]) -> Set[str]:
        """Parse and normalize CORS origins."""
        parsed_origins = set()
        
        for origin in origins:
            if origin == "*":
                parsed_origins.add("*")
            elif origin:
                # Normalize origin (remove trailing slash, ensure protocol)
                normalized = origin.rstrip("/")
                if not normalized.startswith(("http://", "https://")):
                    normalized = f"https://{normalized}"
                parsed_origins.add(normalized)
        
        return parsed_origins
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed with enhanced validation."""
        if not origin:
            return False
        
        # Block suspicious origins in production
        if self.block_suspicious_origins and not settings.is_development():
            for pattern in self.suspicious_patterns:
                if re.search(pattern, origin, re.IGNORECASE):
                    logger.warning(f"Blocked suspicious origin: {origin}")
                    return False
        
        # Allow wildcard in development
        if "*" in self.allowed_origins and settings.is_development():
            return True
        
        # Allow exact matches
        if origin in self.allowed_origins:
            return True
        
        # Allow subdomain wildcards
        if self.allow_wildcard_subdomains:
            for allowed_origin in self.allowed_origins:
                if allowed_origin.startswith("*."):
                    domain = allowed_origin[2:]  # Remove "*."
                    if origin.endswith(domain):
                        return True
        
        # Allow localhost with any port in development
        if settings.is_development() and origin.startswith(("http://localhost:", "https://localhost:")):
            return True
        
        return False
    
    def _is_method_allowed(self, method: str) -> bool:
        """Check if HTTP method is allowed."""
        if not self.enable_method_validation:
            return True
        
        return method.upper() in self.allowed_methods
    
    def _is_header_allowed(self, header: str) -> bool:
        """Check if header is allowed."""
        if not self.enable_header_validation:
            return True
        
        # Convert to lowercase for case-insensitive comparison
        header_lower = header.lower()
        
        # Check against allowed headers (case-insensitive)
        for allowed_header in self.allowed_headers:
            if header_lower == allowed_header.lower():
                return True
        
        # Allow standard headers
        standard_headers = [
            "accept", "accept-language", "content-language", "content-type",
            "authorization", "x-requested-with", "x-request-id"
        ]
        
        return header_lower in standard_headers
    
    def _validate_request_headers(self, request: Request) -> bool:
        """Validate request headers for security."""
        # Check for suspicious headers
        suspicious_headers = [
            "x-forwarded-host",
            "x-originating-ip",
            "x-remote-ip",
            "x-remote-addr"
        ]
        
        for header in suspicious_headers:
            if header in request.headers:
                logger.warning(f"Suspicious header detected: {header}")
                return False
        
        return True
    
    def _get_cors_headers(self, origin: str) -> Dict[str, str]:
        """Get CORS headers for the given origin."""
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": ", ".join(self.allowed_methods),
            "Access-Control-Allow-Headers": ", ".join(self.allowed_headers),
            "Access-Control-Expose-Headers": ", ".join(self.expose_headers),
            "Access-Control-Max-Age": str(self.max_age),
            "Access-Control-Allow-Credentials": "true" if self.allow_credentials else "false",
            "Vary": "Origin"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Handle CORS preflight and add CORS headers."""
        origin = request.headers.get("Origin")
        
        # Validate request headers
        if not self._validate_request_headers(request):
            return Response(
                status_code=400,
                content="Invalid request headers",
                headers={"Content-Type": "text/plain"}
            )
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            if not origin:
                return Response(status_code=400)
            
            if not self._is_origin_allowed(origin):
                logger.warning(f"CORS preflight rejected for origin: {origin}")
                return Response(
                    status_code=403,
                    content="CORS preflight request not allowed",
                    headers={"Content-Type": "text/plain"}
                )
            
            # Validate requested method
            requested_method = request.headers.get("Access-Control-Request-Method")
            if requested_method and not self._is_method_allowed(requested_method):
                logger.warning(f"CORS preflight rejected for method: {requested_method}")
                return Response(
                    status_code=403,
                    content="Requested method not allowed",
                    headers={"Content-Type": "text/plain"}
                )
            
            # Validate requested headers
            requested_headers = request.headers.get("Access-Control-Request-Headers", "")
            if requested_headers:
                headers_list = [h.strip() for h in requested_headers.split(",")]
                for header in headers_list:
                    if not self._is_header_allowed(header):
                        logger.warning(f"CORS preflight rejected for header: {header}")
                        return Response(
                            status_code=403,
                            content="Requested header not allowed",
                            headers={"Content-Type": "text/plain"}
                        )
            
            # Return preflight response
            cors_headers = self._get_cors_headers(origin)
            return Response(
                status_code=200,
                headers=cors_headers
            )
        
        # Process the request
        response = await call_next(request)
        
        # Add CORS headers to response
        if origin and self._is_origin_allowed(origin):
            cors_headers = self._get_cors_headers(origin)
            for header, value in cors_headers.items():
                response.headers[header] = value
        elif origin:
            logger.warning(f"CORS request rejected for origin: {origin}")
            # Don't add CORS headers for rejected origins
        
        return response

class EnhancedSecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enhanced security headers middleware with comprehensive protection."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.enable_csp = True
        self.enable_hsts = not settings.is_development()
        self.enable_csrf_protection = True
        self.security_headers = self._get_security_headers()
        
    def _get_security_headers(self) -> Dict[str, str]:
        """Get comprehensive security headers."""
        headers = {
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            
            # XSS protection
            "X-XSS-Protection": "1; mode=block",
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "speaker=(), "
                "vibrate=(), "
                "fullscreen=(self), "
                "payment=()"
            ),
            
            # Content Security Policy
            "Content-Security-Policy": self._get_csp_header(),
            
            # Cross-Origin policies
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin",
            
            # Remove server information
            "Server": "TrendAnalysis-API",
            
            # Cache control for sensitive endpoints
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        
        # Add HSTS header in production
        if self.enable_hsts:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        return headers
    
    def _get_csp_header(self) -> str:
        """Get Content Security Policy header."""
        if not self.enable_csp:
            return ""
        
        # Base CSP policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Allow inline scripts for development
            "style-src 'self' 'unsafe-inline'",  # Allow inline styles
            "img-src 'self' data: https: blob:",
            "font-src 'self' data: https:",
            "connect-src 'self' https: wss:",
            "media-src 'self' https: blob:",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
            "frame-src 'none'",
            "manifest-src 'self'",
            "worker-src 'self' blob:",
            "child-src 'self' blob:",
            "upgrade-insecure-requests",
            "block-all-mixed-content"
        ]
        
        # Add nonce support for development
        if settings.is_development():
            csp_directives.append("script-src 'self' 'unsafe-inline' 'unsafe-eval' 'nonce-{nonce}'")
        
        return "; ".join(csp_directives)
    
    def _add_csrf_headers(self, response: Response) -> None:
        """Add CSRF protection headers."""
        if not self.enable_csrf_protection:
            return
        
        # Add CSRF token header
        response.headers["X-CSRF-Token"] = "required"
        
        # Add SameSite cookie attribute (handled by FastAPI)
        response.headers["Set-Cookie"] = "csrf-token=required; SameSite=Strict; Secure; HttpOnly"
    
    def _add_performance_headers(self, response: Response) -> None:
        """Add performance-related headers."""
        # Add response time header
        if hasattr(response, 'headers'):
            response.headers["X-Response-Time"] = "0ms"  # Will be updated by timing middleware
        
        # Add cache headers for static content
        if response.headers.get("Content-Type", "").startswith(("text/css", "application/javascript", "image/")):
            response.headers["Cache-Control"] = "public, max-age=31536000"  # 1 year
            response.headers["Expires"] = "Thu, 31 Dec 2025 23:59:59 GMT"
    
    def _add_cors_security_headers(self, response: Response) -> None:
        """Add CORS-related security headers."""
        # Add CORS preflight headers
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "3600"
        
        # Add CORS security headers
        response.headers["Access-Control-Allow-Origin"] = "null"  # Will be overridden by CORS middleware
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add CSRF protection headers
        self._add_csrf_headers(response)
        
        # Add performance headers
        self._add_performance_headers(response)
        
        # Add CORS security headers
        self._add_cors_security_headers(response)
        
        # Log security header addition
        logger.debug(f"Added security headers to response for {request.url.path}")
        
        return response

class SecurityMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting security-related metrics."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "cors_preflight_requests": 0,
            "cors_rejected_requests": 0,
            "suspicious_origins_blocked": 0,
            "suspicious_headers_blocked": 0,
            "security_headers_added": 0,
            "requests_by_origin": {},
            "requests_by_method": {},
            "requests_by_status": {}
        }
    
    async def dispatch(self, request: Request, call_next):
        """Collect security metrics."""
        # Track request
        self.metrics["total_requests"] += 1
        
        origin = request.headers.get("Origin", "unknown")
        method = request.method
        
        # Track by origin
        self.metrics["requests_by_origin"][origin] = self.metrics["requests_by_origin"].get(origin, 0) + 1
        
        # Track by method
        self.metrics["requests_by_method"][method] = self.metrics["requests_by_method"].get(method, 0) + 1
        
        # Track CORS preflight requests
        if method == "OPTIONS":
            self.metrics["cors_preflight_requests"] += 1
        
        # Process request
        response = await call_next(request)
        
        # Track response status
        status_code = response.status_code
        self.metrics["requests_by_status"][status_code] = self.metrics["requests_by_status"].get(status_code, 0) + 1
        
        # Track CORS rejections
        if status_code == 403 and "CORS" in str(response.body):
            self.metrics["cors_rejected_requests"] += 1
        
        # Track security headers
        security_header_count = sum(1 for header in response.headers.keys() 
                                  if header.startswith(("X-", "Strict-", "Content-Security", "Cross-Origin")))
        self.metrics["security_headers_added"] += security_header_count
        
        return response
    
    def get_security_metrics(self) -> Dict[str, any]:
        """Get current security metrics."""
        return dict(self.metrics)

# Create global security metrics instance
security_metrics = SecurityMetricsMiddleware(None)
