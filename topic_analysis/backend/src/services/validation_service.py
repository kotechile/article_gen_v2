"""
Validation Service
Provides comprehensive data validation for all inputs and operations
"""

import re
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import structlog
from ..core.config import settings

logger = structlog.get_logger()

class ValidationService:
    """Service for comprehensive data validation"""
    
    def __init__(self):
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.password_pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')
        self.url_pattern = re.compile(r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$')
        
        # File validation settings
        self.max_file_size = settings.max_file_size if hasattr(settings, 'max_file_size') else 10 * 1024 * 1024  # 10MB
        self.allowed_file_types = settings.allowed_file_types if hasattr(settings, 'allowed_file_types') else ['.csv', '.xlsx', '.json']
        
        # Content validation settings
        self.max_title_length = 200
        self.max_description_length = 2000
        self.max_keyword_length = 100
        self.max_topic_length = 500
    
    async def validate_user_registration(
        self,
        email: str,
        password: str,
        full_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate user registration data
        
        Args:
            email: User email
            password: User password
            full_name: Optional full name
            
        Returns:
            Dict containing validation results
        """
        try:
            logger.info("Validating user registration", email=email)
            
            errors = []
            warnings = []
            
            # Validate email
            email_validation = await self._validate_email(email)
            if not email_validation["valid"]:
                errors.extend(email_validation["errors"])
            if email_validation.get("warnings"):
                warnings.extend(email_validation["warnings"])
            
            # Validate password
            password_validation = await self._validate_password(password)
            if not password_validation["valid"]:
                errors.extend(password_validation["errors"])
            if password_validation.get("warnings"):
                warnings.extend(password_validation["warnings"])
            
            # Validate full name if provided
            if full_name:
                name_validation = await self._validate_full_name(full_name)
                if not name_validation["valid"]:
                    errors.extend(name_validation["errors"])
                if name_validation.get("warnings"):
                    warnings.extend(name_validation["warnings"])
            
            is_valid = len(errors) == 0
            
            logger.info("User registration validation completed", 
                       email=email,
                       valid=is_valid,
                       error_count=len(errors),
                       warning_count=len(warnings))
            
            return {
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "validated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to validate user registration", error=str(e))
            raise
    
    async def validate_content_idea(
        self,
        title: str,
        description: str,
        content_type: str,
        keywords: List[str],
        target_audience: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate content idea data
        
        Args:
            title: Content title
            description: Content description
            content_type: Type of content
            keywords: List of keywords
            target_audience: Optional target audience
            
        Returns:
            Dict containing validation results
        """
        try:
            logger.info("Validating content idea", title=title[:50])
            
            errors = []
            warnings = []
            
            # Validate title
            title_validation = await self._validate_title(title)
            if not title_validation["valid"]:
                errors.extend(title_validation["errors"])
            if title_validation.get("warnings"):
                warnings.extend(title_validation["warnings"])
            
            # Validate description
            description_validation = await self._validate_description(description)
            if not description_validation["valid"]:
                errors.extend(description_validation["errors"])
            if description_validation.get("warnings"):
                warnings.extend(description_validation["warnings"])
            
            # Validate content type
            content_type_validation = await self._validate_content_type(content_type)
            if not content_type_validation["valid"]:
                errors.extend(content_type_validation["errors"])
            
            # Validate keywords
            keywords_validation = await self._validate_keywords(keywords)
            if not keywords_validation["valid"]:
                errors.extend(keywords_validation["errors"])
            if keywords_validation.get("warnings"):
                warnings.extend(keywords_validation["warnings"])
            
            # Validate target audience if provided
            if target_audience:
                audience_validation = await self._validate_target_audience(target_audience)
                if not audience_validation["valid"]:
                    errors.extend(audience_validation["errors"])
                if audience_validation.get("warnings"):
                    warnings.extend(audience_validation["warnings"])
            
            is_valid = len(errors) == 0
            
            logger.info("Content idea validation completed", 
                       title=title[:50],
                       valid=is_valid,
                       error_count=len(errors),
                       warning_count=len(warnings))
            
            return {
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "validated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to validate content idea", error=str(e))
            raise
    
    async def validate_trend_analysis_request(
        self,
        topics: List[str],
        analysis_type: str,
        time_range: Optional[str] = None,
        geographic_scope: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate trend analysis request
        
        Args:
            topics: List of topics to analyze
            analysis_type: Type of analysis
            time_range: Optional time range
            geographic_scope: Optional geographic scope
            
        Returns:
            Dict containing validation results
        """
        try:
            logger.info("Validating trend analysis request", 
                       topics_count=len(topics),
                       analysis_type=analysis_type)
            
            errors = []
            warnings = []
            
            # Validate topics
            topics_validation = await self._validate_topics(topics)
            if not topics_validation["valid"]:
                errors.extend(topics_validation["errors"])
            if topics_validation.get("warnings"):
                warnings.extend(topics_validation["warnings"])
            
            # Validate analysis type
            analysis_type_validation = await self._validate_analysis_type(analysis_type)
            if not analysis_type_validation["valid"]:
                errors.extend(analysis_type_validation["errors"])
            
            # Validate time range if provided
            if time_range:
                time_range_validation = await self._validate_time_range(time_range)
                if not time_range_validation["valid"]:
                    errors.extend(time_range_validation["errors"])
                if time_range_validation.get("warnings"):
                    warnings.extend(time_range_validation["warnings"])
            
            # Validate geographic scope if provided
            if geographic_scope:
                geo_validation = await self._validate_geographic_scope(geographic_scope)
                if not geo_validation["valid"]:
                    errors.extend(geo_validation["errors"])
                if geo_validation.get("warnings"):
                    warnings.extend(geo_validation["warnings"])
            
            is_valid = len(errors) == 0
            
            logger.info("Trend analysis request validation completed", 
                       topics_count=len(topics),
                       valid=is_valid,
                       error_count=len(errors),
                       warning_count=len(warnings))
            
            return {
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "validated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to validate trend analysis request", error=str(e))
            raise
    
    async def validate_file_upload(
        self,
        filename: str,
        file_size: int,
        file_type: str,
        content: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Validate file upload
        
        Args:
            filename: File name
            file_size: File size in bytes
            file_type: File MIME type
            content: Optional file content
            
        Returns:
            Dict containing validation results
        """
        try:
            logger.info("Validating file upload", 
                       filename=filename,
                       file_size=file_size,
                       file_type=file_type)
            
            errors = []
            warnings = []
            
            # Validate file size
            if file_size > self.max_file_size:
                errors.append(f"File size {file_size} exceeds maximum allowed size {self.max_file_size}")
            elif file_size > self.max_file_size * 0.8:
                warnings.append(f"File size is close to maximum limit")
            
            # Validate file type
            file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
            if f'.{file_extension}' not in self.allowed_file_types:
                errors.append(f"File type .{file_extension} is not allowed. Allowed types: {self.allowed_file_types}")
            
            # Validate file name
            if not filename or len(filename) > 255:
                errors.append("Invalid filename")
            
            # Validate file content if provided
            if content is not None:
                content_validation = await self._validate_file_content(content, file_type)
                if not content_validation["valid"]:
                    errors.extend(content_validation["errors"])
                if content_validation.get("warnings"):
                    warnings.extend(content_validation["warnings"])
            
            is_valid = len(errors) == 0
            
            logger.info("File upload validation completed", 
                       filename=filename,
                       valid=is_valid,
                       error_count=len(errors),
                       warning_count=len(warnings))
            
            return {
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "validated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to validate file upload", error=str(e))
            raise
    
    async def validate_api_request(
        self,
        endpoint: str,
        method: str,
        headers: Dict[str, str],
        query_params: Dict[str, Any],
        body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate API request
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            headers: Request headers
            query_params: Query parameters
            body: Optional request body
            
        Returns:
            Dict containing validation results
        """
        try:
            logger.info("Validating API request", 
                       endpoint=endpoint,
                       method=method)
            
            errors = []
            warnings = []
            
            # Validate endpoint
            if not endpoint or not endpoint.startswith('/'):
                errors.append("Invalid endpoint format")
            
            # Validate method
            valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
            if method.upper() not in valid_methods:
                errors.append(f"Invalid HTTP method. Allowed: {valid_methods}")
            
            # Validate headers
            headers_validation = await self._validate_headers(headers)
            if not headers_validation["valid"]:
                errors.extend(headers_validation["errors"])
            if headers_validation.get("warnings"):
                warnings.extend(headers_validation["warnings"])
            
            # Validate query parameters
            query_validation = await self._validate_query_params(query_params)
            if not query_validation["valid"]:
                errors.extend(query_validation["errors"])
            if query_validation.get("warnings"):
                warnings.extend(query_validation["warnings"])
            
            # Validate body if provided
            if body is not None:
                body_validation = await self._validate_request_body(body)
                if not body_validation["valid"]:
                    errors.extend(body_validation["errors"])
                if body_validation.get("warnings"):
                    warnings.extend(body_validation["warnings"])
            
            is_valid = len(errors) == 0
            
            logger.info("API request validation completed", 
                       endpoint=endpoint,
                       valid=is_valid,
                       error_count=len(errors),
                       warning_count=len(warnings))
            
            return {
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "validated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to validate API request", error=str(e))
            raise
    
    async def _validate_email(self, email: str) -> Dict[str, Any]:
        """Validate email address"""
        errors = []
        warnings = []
        
        if not email:
            errors.append("Email is required")
            return {"valid": False, "errors": errors}
        
        if not self.email_pattern.match(email):
            errors.append("Invalid email format")
        
        if len(email) > 254:
            errors.append("Email is too long")
        
        if len(email) > 100:
            warnings.append("Email is very long")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password"""
        errors = []
        warnings = []
        
        if not password:
            errors.append("Password is required")
            return {"valid": False, "errors": errors}
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        if len(password) > 128:
            errors.append("Password is too long")
        
        if not self.password_pattern.match(password):
            errors.append("Password must contain at least one lowercase letter, one uppercase letter, one digit, and one special character")
        
        if len(password) < 12:
            warnings.append("Consider using a longer password for better security")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_full_name(self, full_name: str) -> Dict[str, Any]:
        """Validate full name"""
        errors = []
        warnings = []
        
        if not full_name:
            errors.append("Full name is required")
            return {"valid": False, "errors": errors}
        
        if len(full_name) < 2:
            errors.append("Full name must be at least 2 characters long")
        
        if len(full_name) > 100:
            errors.append("Full name is too long")
        
        if not re.match(r'^[a-zA-Z\s\-\.]+$', full_name):
            errors.append("Full name contains invalid characters")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_title(self, title: str) -> Dict[str, Any]:
        """Validate content title"""
        errors = []
        warnings = []
        
        if not title:
            errors.append("Title is required")
            return {"valid": False, "errors": errors}
        
        if len(title) > self.max_title_length:
            errors.append(f"Title is too long (max {self.max_title_length} characters)")
        
        if len(title) < 10:
            warnings.append("Title is very short")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_description(self, description: str) -> Dict[str, Any]:
        """Validate content description"""
        errors = []
        warnings = []
        
        if not description:
            errors.append("Description is required")
            return {"valid": False, "errors": errors}
        
        if len(description) > self.max_description_length:
            errors.append(f"Description is too long (max {self.max_description_length} characters)")
        
        if len(description) < 50:
            warnings.append("Description is very short")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_content_type(self, content_type: str) -> Dict[str, Any]:
        """Validate content type"""
        errors = []
        
        valid_types = ['article', 'guide', 'review', 'tutorial', 'listicle', 'video', 'podcast']
        
        if not content_type:
            errors.append("Content type is required")
            return {"valid": False, "errors": errors}
        
        if content_type.lower() not in valid_types:
            errors.append(f"Invalid content type. Allowed: {valid_types}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _validate_keywords(self, keywords: List[str]) -> Dict[str, Any]:
        """Validate keywords list"""
        errors = []
        warnings = []
        
        if not keywords:
            errors.append("Keywords are required")
            return {"valid": False, "errors": errors}
        
        if len(keywords) > 50:
            errors.append("Too many keywords (max 50)")
        
        if len(keywords) < 3:
            warnings.append("Consider adding more keywords")
        
        for keyword in keywords:
            if len(keyword) > self.max_keyword_length:
                errors.append(f"Keyword '{keyword}' is too long (max {self.max_keyword_length} characters)")
            
            if not keyword.strip():
                errors.append("Empty keyword found")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_target_audience(self, target_audience: str) -> Dict[str, Any]:
        """Validate target audience"""
        errors = []
        warnings = []
        
        if not target_audience:
            errors.append("Target audience is required")
            return {"valid": False, "errors": errors}
        
        if len(target_audience) > 500:
            errors.append("Target audience description is too long")
        
        if len(target_audience) < 20:
            warnings.append("Target audience description is very short")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_topics(self, topics: List[str]) -> Dict[str, Any]:
        """Validate topics list"""
        errors = []
        warnings = []
        
        if not topics:
            errors.append("Topics are required")
            return {"valid": False, "errors": errors}
        
        if len(topics) > 10:
            errors.append("Too many topics (max 10)")
        
        if len(topics) < 1:
            errors.append("At least one topic is required")
        
        for topic in topics:
            if len(topic) > self.max_topic_length:
                errors.append(f"Topic '{topic}' is too long (max {self.max_topic_length} characters)")
            
            if not topic.strip():
                errors.append("Empty topic found")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_analysis_type(self, analysis_type: str) -> Dict[str, Any]:
        """Validate analysis type"""
        errors = []
        
        valid_types = ['trend', 'keyword', 'competitor', 'content', 'affiliate']
        
        if not analysis_type:
            errors.append("Analysis type is required")
            return {"valid": False, "errors": errors}
        
        if analysis_type.lower() not in valid_types:
            errors.append(f"Invalid analysis type. Allowed: {valid_types}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _validate_time_range(self, time_range: str) -> Dict[str, Any]:
        """Validate time range"""
        errors = []
        warnings = []
        
        valid_ranges = ['1d', '7d', '30d', '90d', '1y', '5y']
        
        if not time_range:
            errors.append("Time range is required")
            return {"valid": False, "errors": errors}
        
        if time_range not in valid_ranges:
            errors.append(f"Invalid time range. Allowed: {valid_ranges}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_geographic_scope(self, geographic_scope: str) -> Dict[str, Any]:
        """Validate geographic scope"""
        errors = []
        warnings = []
        
        if not geographic_scope:
            errors.append("Geographic scope is required")
            return {"valid": False, "errors": errors}
        
        if len(geographic_scope) > 100:
            errors.append("Geographic scope is too long")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_file_content(self, content: bytes, file_type: str) -> Dict[str, Any]:
        """Validate file content"""
        errors = []
        warnings = []
        
        if not content:
            errors.append("File content is empty")
            return {"valid": False, "errors": errors}
        
        # Check for suspicious content
        if b'<script' in content.lower():
            errors.append("File contains potentially malicious script content")
        
        if b'<?php' in content.lower():
            errors.append("File contains PHP code")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_headers(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Validate request headers"""
        errors = []
        warnings = []
        
        # Check for required headers
        if 'content-type' not in headers:
            warnings.append("Content-Type header is missing")
        
        # Check for suspicious headers
        for key, value in headers.items():
            if len(key) > 100:
                errors.append(f"Header key '{key}' is too long")
            
            if len(value) > 1000:
                errors.append(f"Header value for '{key}' is too long")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_query_params(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate query parameters"""
        errors = []
        warnings = []
        
        for key, value in query_params.items():
            if len(key) > 100:
                errors.append(f"Query parameter key '{key}' is too long")
            
            if isinstance(value, str) and len(value) > 1000:
                errors.append(f"Query parameter value for '{key}' is too long")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_request_body(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request body"""
        errors = []
        warnings = []
        
        if not body:
            errors.append("Request body is required")
            return {"valid": False, "errors": errors}
        
        # Check body size
        body_str = json.dumps(body)
        if len(body_str) > 10000:
            errors.append("Request body is too large")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

