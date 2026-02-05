"""
Request/Response Logging Middleware
Comprehensive logging of HTTP requests and responses
"""

import time
import json
import uuid
from typing import Dict, Any, Optional
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
import structlog

from ..core.config import settings

logger = structlog.get_logger()

class RequestLoggingMiddleware:
    """Request and response logging middleware"""
    
    def __init__(self):
        self.sensitive_headers = {
            "authorization",
            "x-api-key",
            "x-csrf-token",
            "cookie",
            "set-cookie"
        }
        self.sensitive_body_fields = {
            "password",
            "token",
            "secret",
            "key",
            "authorization"
        }
        self.max_body_length = 10000  # Max body length to log
        self.excluded_paths = {
            "/health",
            "/metrics",
            "/favicon.ico"
        }
    
    async def __call__(self, request: Request, call_next):
        """Process request and log details"""
        # Skip logging for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            await self._log_response(request, response, request_id, process_time)
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            await self._log_error(request, e, request_id, process_time)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request"""
        try:
            # Get client information
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "")
            
            # Get request body (if applicable)
            body = await self._get_request_body(request)
            
            # Prepare request data
            request_data = {
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "headers": self._sanitize_headers(dict(request.headers)),
                "body": self._sanitize_body(body),
                "timestamp": time.time()
            }
            
            # Log request
            logger.info(
                "HTTP Request",
                **request_data
            )
            
        except Exception as e:
            logger.error("Error logging request", error=str(e), request_id=request_id)
    
    async def _log_response(self, request: Request, response: Response, request_id: str, process_time: float):
        """Log outgoing response"""
        try:
            # Get response headers
            response_headers = dict(response.headers)
            
            # Get response body (if applicable)
            response_body = await self._get_response_body(response)
            
            # Prepare response data
            response_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "headers": self._sanitize_headers(response_headers),
                "body": self._sanitize_body(response_body),
                "process_time": round(process_time, 4),
                "timestamp": time.time()
            }
            
            # Log response
            logger.info(
                "HTTP Response",
                **response_data
            )
            
        except Exception as e:
            logger.error("Error logging response", error=str(e), request_id=request_id)
    
    async def _log_error(self, request: Request, error: Exception, request_id: str, process_time: float):
        """Log error details"""
        try:
            error_data = {
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "process_time": round(process_time, 4),
                "timestamp": time.time()
            }
            
            logger.error(
                "HTTP Error",
                **error_data
            )
            
        except Exception as e:
            logger.error("Error logging error", error=str(e), request_id=request_id)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return request.client.host if request.client else "unknown"
    
    async def _get_request_body(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get request body"""
        try:
            # Only read body for certain content types
            content_type = request.headers.get("content-type", "")
            if not any(ct in content_type for ct in ["application/json", "application/x-www-form-urlencoded"]):
                return None
            
            # Read body
            body = await request.body()
            if not body:
                return None
            
            # Parse JSON if possible
            if "application/json" in content_type:
                try:
                    return json.loads(body.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return {"raw": body.decode()[:self.max_body_length]}
            
            # Parse form data
            elif "application/x-www-form-urlencoded" in content_type:
                from urllib.parse import parse_qs
                return {k: v[0] if len(v) == 1 else v for k, v in parse_qs(body.decode()).items()}
            
            return None
            
        except Exception as e:
            logger.warning("Error reading request body", error=str(e))
            return None
    
    async def _get_response_body(self, response: Response) -> Optional[Dict[str, Any]]:
        """Get response body"""
        try:
            # Only read body for certain content types
            content_type = response.headers.get("content-type", "")
            if not any(ct in content_type for ct in ["application/json", "text/"]):
                return None
            
            # For streaming responses, we can't easily get the body
            if isinstance(response, StreamingResponse):
                return {"type": "streaming"}
            
            # Get body from response
            if hasattr(response, "body"):
                body = response.body
                if not body:
                    return None
                
                # Parse JSON if possible
                if "application/json" in content_type:
                    try:
                        return json.loads(body.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        return {"raw": body.decode()[:self.max_body_length]}
                
                # Return text content
                return {"text": body.decode()[:self.max_body_length]}
            
            return None
            
        except Exception as e:
            logger.warning("Error reading response body", error=str(e))
            return None
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize sensitive headers"""
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        return sanitized
    
    def _sanitize_body(self, body: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sanitize sensitive body fields"""
        if not body:
            return body
        
        if not isinstance(body, dict):
            return body
        
        sanitized = {}
        for key, value in body.items():
            if key.lower() in self.sensitive_body_fields:
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_body(value)
            else:
                sanitized[key] = value
        
        return sanitized

class PerformanceLoggingMiddleware:
    """Performance monitoring middleware"""
    
    def __init__(self):
        self.slow_request_threshold = 2.0  # seconds
        self.very_slow_request_threshold = 5.0  # seconds
    
    async def __call__(self, request: Request, call_next):
        """Monitor request performance"""
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # Log performance metrics
        await self._log_performance(request, response, process_time)
        
        return response
    
    async def _log_performance(self, request: Request, response: Response, process_time: float):
        """Log performance metrics"""
        try:
            performance_data = {
                "request_id": getattr(request.state, "request_id", "unknown"),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time": round(process_time, 4),
                "timestamp": time.time()
            }
            
            # Log based on performance level
            if process_time >= self.very_slow_request_threshold:
                logger.warning(
                    "Very slow request",
                    **performance_data
                )
            elif process_time >= self.slow_request_threshold:
                logger.info(
                    "Slow request",
                    **performance_data
                )
            else:
                logger.debug(
                    "Request performance",
                    **performance_data
                )
                
        except Exception as e:
            logger.error("Error logging performance", error=str(e))

class AuditLoggingMiddleware:
    """Audit logging for security events"""
    
    def __init__(self):
        self.audit_events = {
            "authentication": ["login", "logout", "register", "password_reset"],
            "authorization": ["access_denied", "permission_denied"],
            "data_access": ["create", "read", "update", "delete"],
            "system": ["error", "warning", "info"]
        }
    
    async def __call__(self, request: Request, call_next):
        """Audit request for security events"""
        # Check for security-relevant paths
        if self._is_security_relevant(request):
            await self._log_audit_event(request, "access_attempt")
        
        response = await call_next(request)
        
        # Check for security events in response
        if response.status_code in [401, 403, 429]:
            await self._log_audit_event(request, "security_event", {
                "status_code": response.status_code,
                "reason": "authentication_or_authorization_failure"
            })
        
        return response
    
    def _is_security_relevant(self, request: Request) -> bool:
        """Check if request is security-relevant"""
        security_paths = [
            "/api/auth/",
            "/api/admin/",
            "/api/users/",
            "/api/export/"
        ]
        
        return any(request.url.path.startswith(path) for path in security_paths)
    
    async def _log_audit_event(self, request: Request, event_type: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log audit event"""
        try:
            audit_data = {
                "request_id": getattr(request.state, "request_id", "unknown"),
                "event_type": event_type,
                "method": request.method,
                "path": request.url.path,
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                "timestamp": time.time()
            }
            
            if extra_data:
                audit_data.update(extra_data)
            
            logger.info(
                "Audit Event",
                **audit_data
            )
            
        except Exception as e:
            logger.error("Error logging audit event", error=str(e))
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"

# Global instances
request_logging = RequestLoggingMiddleware()
performance_logging = PerformanceLoggingMiddleware()
audit_logging = AuditLoggingMiddleware()

# Combined logging middleware
async def logging_middleware(request: Request, call_next):
    """Combined logging middleware"""
    # Request logging
    await request_logging(request, call_next)
    
    # Performance logging
    await performance_logging(request, call_next)
    
    # Audit logging
    await audit_logging(request, call_next)
    
    return await call_next(request)

# Utility functions
def log_user_action(user_id: str, action: str, resource: str, details: Optional[Dict[str, Any]] = None):
    """Log user action for audit purposes"""
    try:
        log_data = {
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "timestamp": time.time()
        }
        
        if details:
            log_data.update(details)
        
        logger.info("User Action", **log_data)
        
    except Exception as e:
        logger.error("Error logging user action", error=str(e))

def log_security_event(event_type: str, details: Dict[str, Any]):
    """Log security event"""
    try:
        log_data = {
            "event_type": event_type,
            "timestamp": time.time()
        }
        log_data.update(details)
        
        logger.warning("Security Event", **log_data)
        
    except Exception as e:
        logger.error("Error logging security event", error=str(e))

def log_system_event(event_type: str, details: Dict[str, Any]):
    """Log system event"""
    try:
        log_data = {
            "event_type": event_type,
            "timestamp": time.time()
        }
        log_data.update(details)
        
        logger.info("System Event", **log_data)
        
    except Exception as e:
        logger.error("Error logging system event", error=str(e))