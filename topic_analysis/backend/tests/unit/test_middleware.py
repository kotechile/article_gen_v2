"""
Backend Unit Tests for Middleware
T110: Backend unit tests for all services
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add backend src to path
backend_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(backend_src))

from middleware.auth import AuthMiddleware, create_access_token, require_role, require_permission
from middleware.rate_limiting import RateLimitingMiddleware
from middleware.security import SecurityMiddleware
from middleware.logging import RequestLoggingMiddleware
from middleware.error_handling import ErrorHandlingMiddleware


class TestAuthMiddleware:
    """Test Authentication Middleware"""
    
    @pytest.fixture
    def app(self):
        app = FastAPI()
        app.add_middleware(AuthMiddleware)
        return app
    
    @pytest.fixture
    def client(self, app):
        return TestClient(app)
    
    def test_create_access_token(self):
        """Test access token creation"""
        data = {"sub": "user123", "role": "user"}
        token = create_access_token(data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_require_role_decorator(self):
        """Test require_role decorator"""
        # Mock request with user role
        request = Mock()
        request.state.user_role = "admin"
        
        # Test admin role requirement
        admin_checker = require_role("admin")
        result = admin_checker(request)
        assert result is True
        
        # Test user role requirement (should fail)
        user_checker = require_role("user")
        with pytest.raises(Exception):  # Should raise HTTPException
            user_checker(request)
    
    def test_require_permission_decorator(self):
        """Test require_permission decorator"""
        # Mock request with user permissions
        request = Mock()
        request.state.user_permissions = ["read", "write"]
        
        # Test read permission (should pass)
        read_checker = require_permission("read")
        result = read_checker(request)
        assert result is True
        
        # Test admin permission (should fail)
        admin_checker = require_permission("admin")
        with pytest.raises(Exception):  # Should raise HTTPException
            admin_checker(request)


class TestRateLimitingMiddleware:
    """Test Rate Limiting Middleware"""
    
    @pytest.fixture
    def app(self):
        app = FastAPI()
        app.add_middleware(RateLimitingMiddleware, default_limit=10, default_window=60)
        return app
    
    @pytest.fixture
    def client(self, app):
        return TestClient(app)
    
    def test_rate_limiting_initialization(self):
        """Test rate limiting middleware initialization"""
        middleware = RateLimitingMiddleware(Mock(), default_limit=10, default_window=60)
        
        assert middleware.default_limit == 10
        assert middleware.default_window == 60
        assert middleware.in_memory_limits is not None
    
    def test_get_client_id(self):
        """Test client ID extraction"""
        middleware = RateLimitingMiddleware(Mock())
        
        # Test with X-Forwarded-For header
        request = Mock()
        request.headers = {"X-Forwarded-For": "192.168.1.1"}
        request.client = Mock()
        request.client.host = "127.0.0.1"
        
        client_id = middleware._get_client_id(request)
        assert client_id == "192.168.1.1"
        
        # Test without X-Forwarded-For header
        request.headers = {}
        client_id = middleware._get_client_id(request)
        assert client_id == "127.0.0.1"


class TestSecurityMiddleware:
    """Test Security Middleware"""
    
    @pytest.fixture
    def app(self):
        app = FastAPI()
        app.add_middleware(SecurityMiddleware)
        return app
    
    @pytest.fixture
    def client(self, app):
        return TestClient(app)
    
    def test_security_middleware_initialization(self):
        """Test security middleware initialization"""
        middleware = SecurityMiddleware(Mock())
        
        assert middleware is not None
        assert hasattr(middleware, 'cors_config')
        assert hasattr(middleware, 'csrf_protection')
    
    def test_is_origin_allowed(self):
        """Test origin validation"""
        middleware = SecurityMiddleware(Mock())
        
        # Test allowed origin
        result = middleware.is_origin_allowed("http://localhost:3000")
        assert result is True
        
        # Test disallowed origin
        result = middleware.is_origin_allowed("http://malicious.com")
        assert result is False
    
    def test_get_security_headers(self):
        """Test security headers generation"""
        middleware = SecurityMiddleware(Mock())
        
        headers = middleware.get_security_headers()
        
        assert isinstance(headers, dict)
        assert "Content-Security-Policy" in headers
        assert "X-Frame-Options" in headers
        assert "X-Content-Type-Options" in headers
        assert "Referrer-Policy" in headers


class TestLoggingMiddleware:
    """Test Logging Middleware"""
    
    @pytest.fixture
    def app(self):
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
        return app
    
    @pytest.fixture
    def client(self, app):
        return TestClient(app)
    
    def test_logging_middleware_initialization(self):
        """Test logging middleware initialization"""
        middleware = RequestLoggingMiddleware(Mock())
        
        assert middleware is not None
        assert hasattr(middleware, 'performance_logging')
        assert hasattr(middleware, 'audit_logging')
    
    def test_log_user_action(self):
        """Test user action logging"""
        from middleware.logging import log_user_action
        
        # Should not raise an exception
        log_user_action("user123", "test_action", "test_resource", {"test": "data"})
    
    def test_log_security_event(self):
        """Test security event logging"""
        from middleware.logging import log_security_event
        
        # Should not raise an exception
        log_security_event("test_event", {"test": "data"})
    
    def test_log_system_event(self):
        """Test system event logging"""
        from middleware.logging import log_system_event
        
        # Should not raise an exception
        log_system_event("test_event", {"test": "data"})


class TestErrorHandlingMiddleware:
    """Test Error Handling Middleware"""
    
    @pytest.fixture
    def app(self):
        app = FastAPI()
        app.add_middleware(ErrorHandlingMiddleware)
        return app
    
    @pytest.fixture
    def client(self, app):
        return TestClient(app)
    
    def test_error_handling_middleware_initialization(self):
        """Test error handling middleware initialization"""
        middleware = ErrorHandlingMiddleware(Mock())
        
        assert middleware is not None
        assert hasattr(middleware, 'circuit_breaker')
        assert hasattr(middleware, 'retry_handler')
    
    def test_handle_validation_error(self):
        """Test validation error handling"""
        from middleware.error_handling import handle_validation_error
        
        errors = [{"field": "test", "message": "error"}]
        result = handle_validation_error(errors)
        
        assert isinstance(result, dict)
        assert "error" in result
        assert "details" in result
    
    def test_handle_authentication_error(self):
        """Test authentication error handling"""
        from middleware.error_handling import handle_authentication_error
        
        result = handle_authentication_error("Invalid token")
        
        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] == "Authentication failed"
    
    def test_handle_authorization_error(self):
        """Test authorization error handling"""
        from middleware.error_handling import handle_authorization_error
        
        result = handle_authorization_error("Insufficient permissions")
        
        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] == "Authorization failed"
    
    def test_handle_not_found_error(self):
        """Test not found error handling"""
        from middleware.error_handling import handle_not_found_error
        
        result = handle_not_found_error("Resource not found")
        
        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] == "Resource not found"


class TestMiddlewareIntegration:
    """Test Middleware Integration"""
    
    def test_all_middleware_initialization(self):
        """Test that all middleware can be initialized"""
        app = FastAPI()
        
        # Add all middleware
        app.add_middleware(AuthMiddleware)
        app.add_middleware(RateLimitingMiddleware, default_limit=10, default_window=60)
        app.add_middleware(SecurityMiddleware)
        app.add_middleware(RequestLoggingMiddleware)
        app.add_middleware(ErrorHandlingMiddleware)
        
        # Should not raise any exceptions
        assert len(app.user_middleware) == 5
    
    def test_middleware_order(self):
        """Test middleware execution order"""
        app = FastAPI()
        
        # Add middleware in specific order
        app.add_middleware(ErrorHandlingMiddleware)  # Should be first
        app.add_middleware(RequestLoggingMiddleware)
        app.add_middleware(SecurityMiddleware)
        app.add_middleware(RateLimitingMiddleware)
        app.add_middleware(AuthMiddleware)  # Should be last
        
        # Check middleware order
        middleware_classes = [mw.cls for mw in app.user_middleware]
        
        # Error handling should be first
        assert middleware_classes[0] == ErrorHandlingMiddleware
        
        # Auth should be last
        assert middleware_classes[-1] == AuthMiddleware


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
