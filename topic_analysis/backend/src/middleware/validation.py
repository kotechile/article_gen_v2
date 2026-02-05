"""
Validation Middleware

Provides input validation for all API endpoints.
"""

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import logging
import re
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)

class ValidationMiddleware:
    """Middleware for validating API inputs"""
    
    @staticmethod
    def validate_file_id(file_id: str) -> bool:
        """Validate file ID format"""
        try:
            UUID(file_id)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """Validate user ID format"""
        if not user_id or not isinstance(user_id, str):
            return False
        
        # Basic format validation (adjust based on your user ID format)
        return len(user_id) > 0 and len(user_id) <= 255
    
    @staticmethod
    def validate_analysis_id(analysis_id: str) -> bool:
        """Validate analysis ID format"""
        try:
            UUID(analysis_id)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_report_id(report_id: str) -> bool:
        """Validate report ID format"""
        try:
            UUID(report_id)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_pagination_params(limit: int, offset: int) -> bool:
        """Validate pagination parameters"""
        if limit < 1 or limit > 1000:
            return False
        if offset < 0:
            return False
        return True
    
    @staticmethod
    def validate_export_format(format: str) -> bool:
        """Validate export format"""
        valid_formats = ["json", "csv", "pdf"]
        return format.lower() in valid_formats
    
    @staticmethod
    def validate_file_upload_request(request_data: Dict[str, Any]) -> List[str]:
        """Validate file upload request data"""
        errors = []
        
        # Check required fields
        required_fields = ["filename", "file_size", "content_type"]
        for field in required_fields:
            if field not in request_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate file size
        if "file_size" in request_data:
            file_size = request_data["file_size"]
            if not isinstance(file_size, int) or file_size <= 0:
                errors.append("File size must be a positive integer")
            elif file_size > 10 * 1024 * 1024:  # 10MB limit
                errors.append("File size exceeds maximum allowed size (10MB)")
        
        # Validate content type
        if "content_type" in request_data:
            content_type = request_data["content_type"]
            valid_types = ["text/tab-separated-values", "text/csv", "application/octet-stream"]
            if content_type not in valid_types:
                errors.append(f"Invalid content type: {content_type}")
        
        # Validate filename
        if "filename" in request_data:
            filename = request_data["filename"]
            if not isinstance(filename, str) or not filename.strip():
                errors.append("Filename must be a non-empty string")
            elif len(filename) > 255:
                errors.append("Filename must be 255 characters or less")
        
        return errors
    
    @staticmethod
    def validate_analysis_request(request_data: Dict[str, Any]) -> List[str]:
        """Validate analysis request data"""
        errors = []
        
        # Check required fields
        required_fields = ["file_id"]
        for field in required_fields:
            if field not in request_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate file_id
        if "file_id" in request_data:
            file_id = request_data["file_id"]
            if not ValidationMiddleware.validate_file_id(file_id):
                errors.append("Invalid file_id format")
        
        return errors
    
    @staticmethod
    def validate_report_request(request_data: Dict[str, Any]) -> List[str]:
        """Validate report request data"""
        errors = []
        
        # Check required fields
        required_fields = ["report_id"]
        for field in required_fields:
            if field not in request_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate report_id
        if "report_id" in request_data:
            report_id = request_data["report_id"]
            if not ValidationMiddleware.validate_report_id(report_id):
                errors.append("Invalid report_id format")
        
        return errors

def validate_request_data(request_data: Dict[str, Any], validation_type: str) -> List[str]:
    """
    Validate request data based on type
    
    Args:
        request_data: Request data to validate
        validation_type: Type of validation to perform
        
    Returns:
        List of validation errors
    """
    try:
        if validation_type == "file_upload":
            return ValidationMiddleware.validate_file_upload_request(request_data)
        elif validation_type == "analysis":
            return ValidationMiddleware.validate_analysis_request(request_data)
        elif validation_type == "report":
            return ValidationMiddleware.validate_report_request(request_data)
        else:
            return [f"Unknown validation type: {validation_type}"]
    
    except Exception as e:
        logger.error(f"Error validating request data: {str(e)}")
        return [f"Validation error: {str(e)}"]

def create_validation_error_response(errors: List[str]) -> JSONResponse:
    """
    Create a standardized validation error response
    
    Args:
        errors: List of validation errors
        
    Returns:
        JSONResponse with validation errors
    """
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation failed",
            "message": "Request data validation failed",
            "details": errors,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

def create_error_response(
    status_code: int, 
    message: str, 
    details: Optional[str] = None
) -> JSONResponse:
    """
    Create a standardized error response
    
    Args:
        status_code: HTTP status code
        message: Error message
        details: Additional error details
        
    Returns:
        JSONResponse with error information
    """
    error_response = {
        "error": "Request failed",
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if details:
        error_response["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )
