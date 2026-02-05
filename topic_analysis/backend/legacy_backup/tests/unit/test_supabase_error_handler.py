"""
Unit tests for Supabase error handler
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException, status

from src.services.supabase_error_handler import SupabaseErrorHandler

class TestSupabaseErrorHandler:
    """Test cases for SupabaseErrorHandler"""
    
    @pytest.fixture
    def error_handler(self):
        """SupabaseErrorHandler instance"""
        return SupabaseErrorHandler()
    
    def test_handle_supabase_error_api_response_exception(self, error_handler):
        """Test handling APIResponseException"""
        from postgrest.exceptions import APIResponseException
        
        error = APIResponseException(
            message="Row level security policy violation",
            details="The user does not have permission to access this resource",
            hint="Check your RLS policies",
            code="42501",
            status=403
        )
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_supabase_error(error, "test operation")
        
        assert exc_info.value.status_code == 403
        assert "test operation failed" in str(exc_info.value.detail)
        assert "Row level security policy violation" in str(exc_info.value.detail)
        assert "Code: 42501" in str(exc_info.value.detail)
    
    def test_handle_supabase_error_auth_api_error(self, error_handler):
        """Test handling AuthApiError"""
        from gotrue.exceptions import AuthApiError
        
        error = AuthApiError(
            message="Invalid JWT token",
            status=401
        )
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_supabase_error(error, "authentication")
        
        assert exc_info.value.status_code == 401
        assert "Authentication failed" in str(exc_info.value.detail)
        assert "Invalid JWT token" in str(exc_info.value.detail)
    
    def test_handle_supabase_error_generic_exception(self, error_handler):
        """Test handling generic exception"""
        error = Exception("Unknown error occurred")
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_supabase_error(error, "test operation")
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "test operation encountered an unexpected error" in str(exc_info.value.detail)
        assert "Unknown error occurred" in str(exc_info.value.detail)
    
    def test_handle_generic_error(self, error_handler):
        """Test handling generic error"""
        error = ValueError("Invalid input")
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_generic_error(error, "validation")
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "validation encountered an unexpected error" in str(exc_info.value.detail)
        assert "Invalid input" in str(exc_info.value.detail)
    
    def test_handle_generic_error_with_kwargs(self, error_handler):
        """Test handling generic error with additional context"""
        error = ConnectionError("Connection timeout")
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_generic_error(
                error, 
                "database operation",
                table_name="users",
                operation_type="insert"
            )
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "database operation encountered an unexpected error" in str(exc_info.value.detail)
        assert "Connection timeout" in str(exc_info.value.detail)
    
    def test_handle_connection_error(self, error_handler):
        """Test handling connection error"""
        error = ConnectionError("Supabase service unavailable")
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_connection_error(error, "initial connection")
        
        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Database connection unavailable" in str(exc_info.value.detail)
        assert "Supabase service unavailable" in str(exc_info.value.detail)
    
    def test_handle_connection_error_with_context(self, error_handler):
        """Test handling connection error with context"""
        error = TimeoutError("Connection timeout after 30 seconds")
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_connection_error(error, "reconnection attempt")
        
        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Database connection unavailable" in str(exc_info.value.detail)
        assert "Connection timeout after 30 seconds" in str(exc_info.value.detail)
    
    def test_handle_supabase_error_with_none_status(self, error_handler):
        """Test handling APIResponseException with None status"""
        from postgrest.exceptions import APIResponseException
        
        error = APIResponseException(
            message="Database error",
            details="Internal server error",
            hint="Contact support",
            code="50000",
            status=None
        )
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_supabase_error(error, "test operation")
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "test operation failed" in str(exc_info.value.detail)
    
    def test_handle_supabase_error_with_none_auth_status(self, error_handler):
        """Test handling AuthApiError with None status"""
        from gotrue.exceptions import AuthApiError
        
        error = AuthApiError(
            message="Token expired",
            status=None
        )
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_supabase_error(error, "token refresh")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authentication failed" in str(exc_info.value.detail)
    
    def test_handle_supabase_error_with_empty_message(self, error_handler):
        """Test handling error with empty message"""
        from postgrest.exceptions import APIResponseException
        
        error = APIResponseException(
            message="",
            details="",
            hint="",
            code="",
            status=400
        )
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_supabase_error(error, "test operation")
        
        assert exc_info.value.status_code == 400
        assert "test operation failed" in str(exc_info.value.detail)
    
    def test_handle_generic_error_with_empty_message(self, error_handler):
        """Test handling generic error with empty message"""
        error = Exception("")
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_generic_error(error, "test operation")
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "test operation encountered an unexpected error" in str(exc_info.value.detail)
    
    def test_handle_connection_error_with_empty_message(self, error_handler):
        """Test handling connection error with empty message"""
        error = ConnectionError("")
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_connection_error(error, "test connection")
        
        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Database connection unavailable" in str(exc_info.value.detail)
    
    def test_error_handler_initialization(self, error_handler):
        """Test SupabaseErrorHandler initialization"""
        assert error_handler is not None
        assert hasattr(error_handler, 'handle_supabase_error')
        assert hasattr(error_handler, 'handle_generic_error')
        assert hasattr(error_handler, 'handle_connection_error')
    
    def test_handle_supabase_error_with_context_none(self, error_handler):
        """Test handling Supabase error with None context"""
        from postgrest.exceptions import APIResponseException
        
        error = APIResponseException(
            message="Test error",
            details="Test details",
            hint="Test hint",
            code="TEST001",
            status=400
        )
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_supabase_error(error, None)
        
        assert exc_info.value.status_code == 400
        assert "Supabase operation failed" in str(exc_info.value.detail)
    
    def test_handle_generic_error_with_context_none(self, error_handler):
        """Test handling generic error with None context"""
        error = Exception("Test error")
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_generic_error(error, None)
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Generic operation encountered an unexpected error" in str(exc_info.value.detail)
    
    def test_handle_connection_error_with_context_none(self, error_handler):
        """Test handling connection error with None context"""
        error = ConnectionError("Test connection error")
        
        with pytest.raises(HTTPException) as exc_info:
            error_handler.handle_connection_error(error, None)
        
        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Database connection unavailable" in str(exc_info.value.detail)

