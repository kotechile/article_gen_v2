"""
Data Validation and Schema Enforcement Middleware

This module provides middleware for data validation and schema enforcement
with comprehensive validation rules and error handling.
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List, Union
import re
from datetime import datetime
from pydantic import BaseModel, ValidationError, validator

from ..core.logging import db_operation_logger, get_logger

logger = get_logger(__name__)

class ValidationMiddleware:
    """
    Middleware for data validation and schema enforcement.
    """
    
    def __init__(self):
        """Initialize the validation middleware."""
        self.logger = db_operation_logger
        self._validation_rules: Dict[str, Dict[str, Any]] = {}
        self._schema_cache: Dict[str, BaseModel] = {}
    
    def register_validation_rule(self, table_name: str, field_name: str, rule: Dict[str, Any]) -> None:
        """
        Register validation rule for a table field.
        
        Args:
            table_name: Table name
            field_name: Field name
            rule: Validation rule
        """
        if table_name not in self._validation_rules:
            self._validation_rules[table_name] = {}
        
        self._validation_rules[table_name][field_name] = rule
        
        logger.info(
            "Validation rule registered",
            table_name=table_name,
            field_name=field_name,
            rule_type=rule.get("type")
        )
    
    def validate_data(self, table_name: str, data: Dict[str, Any], 
                     operation_type: str = "create") -> Dict[str, Any]:
        """
        Validate data against schema rules.
        
        Args:
            table_name: Table name
            data: Data to validate
            operation_type: Type of operation (create, update, delete)
            
        Returns:
            Validated data
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if table_name not in self._validation_rules:
                return data
            
            rules = self._validation_rules[table_name]
            validated_data = {}
            
            for field_name, value in data.items():
                if field_name in rules:
                    rule = rules[field_name]
                    validated_value = self._validate_field(field_name, value, rule, operation_type)
                    validated_data[field_name] = validated_value
                else:
                    validated_data[field_name] = value
            
            # Log successful validation
            self.logger.log_operation_success(
                operation_id="validation",
                execution_time_ms=0,
                table_name=table_name,
                operation_type=operation_type
            )
            
            return validated_data
            
        except Exception as e:
            # Log validation error
            self.logger.log_operation_error(
                operation_id="validation",
                error_message=str(e),
                error_type="validation_error",
                table_name=table_name,
                operation_type=operation_type
            )
            raise ValidationError(str(e))
    
    def _validate_field(self, field_name: str, value: Any, rule: Dict[str, Any], 
                       operation_type: str) -> Any:
        """
        Validate a single field.
        
        Args:
            field_name: Field name
            value: Field value
            rule: Validation rule
            operation_type: Operation type
            
        Returns:
            Validated value
            
        Raises:
            ValidationError: If validation fails
        """
        # Check if field is required
        if rule.get("required", False) and (value is None or value == ""):
            if operation_type == "create":
                raise ValidationError(f"Field '{field_name}' is required")
        
        # Skip validation for None values unless required
        if value is None and not rule.get("required", False):
            return value
        
        # Type validation
        field_type = rule.get("type")
        if field_type:
            value = self._validate_type(field_name, value, field_type)
        
        # Length validation
        if "max_length" in rule and len(str(value)) > rule["max_length"]:
            raise ValidationError(f"Field '{field_name}' exceeds maximum length of {rule['max_length']}")
        
        if "min_length" in rule and len(str(value)) < rule["min_length"]:
            raise ValidationError(f"Field '{field_name}' is below minimum length of {rule['min_length']}")
        
        # Range validation
        if "max_value" in rule and value > rule["max_value"]:
            raise ValidationError(f"Field '{field_name}' exceeds maximum value of {rule['max_value']}")
        
        if "min_value" in rule and value < rule["min_value"]:
            raise ValidationError(f"Field '{field_name}' is below minimum value of {rule['min_value']}")
        
        # Pattern validation
        if "pattern" in rule and not re.match(rule["pattern"], str(value)):
            raise ValidationError(f"Field '{field_name}' does not match required pattern")
        
        # Enum validation
        if "enum" in rule and value not in rule["enum"]:
            raise ValidationError(f"Field '{field_name}' must be one of: {rule['enum']}")
        
        # Custom validation
        if "custom_validator" in rule:
            validator_func = rule["custom_validator"]
            if not validator_func(value):
                raise ValidationError(f"Field '{field_name}' failed custom validation")
        
        return value
    
    def _validate_type(self, field_name: str, value: Any, field_type: str) -> Any:
        """
        Validate field type.
        
        Args:
            field_name: Field name
            value: Field value
            field_type: Expected type
            
        Returns:
            Converted value
            
        Raises:
            ValidationError: If type conversion fails
        """
        try:
            if field_type == "string":
                return str(value)
            elif field_type == "integer":
                return int(value)
            elif field_type == "float":
                return float(value)
            elif field_type == "boolean":
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() in ["true", "1", "yes", "on"]
                else:
                    return bool(value)
            elif field_type == "email":
                if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str(value)):
                    raise ValidationError(f"Field '{field_name}' must be a valid email address")
                return str(value)
            elif field_type == "url":
                if not re.match(r'^https?://', str(value)):
                    raise ValidationError(f"Field '{field_name}' must be a valid URL")
                return str(value)
            elif field_type == "uuid":
                if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', str(value)):
                    raise ValidationError(f"Field '{field_name}' must be a valid UUID")
                return str(value)
            elif field_type == "date":
                if isinstance(value, str):
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                return value
            else:
                return value
                
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Field '{field_name}' must be of type {field_type}: {e}")
    
    def validate_request_data(self, request: Request, table_name: str, 
                            operation_type: str = "create") -> Dict[str, Any]:
        """
        Validate request data.
        
        Args:
            request: FastAPI request object
            table_name: Table name
            operation_type: Operation type
            
        Returns:
            Validated data
            
        Raises:
            HTTPException: If validation fails
        """
        try:
            # Get request data
            if hasattr(request, 'json'):
                data = request.json()
            else:
                data = {}
            
            # Validate data
            validated_data = self.validate_data(table_name, data, operation_type)
            
            return validated_data
            
        except ValidationError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Validation Error",
                    "message": str(e),
                    "table_name": table_name,
                    "operation_type": operation_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    def setup_default_validation_rules(self) -> None:
        """
        Setup default validation rules for common tables.
        """
        # Users table validation
        self.register_validation_rule("users", "email", {
            "type": "email",
            "required": True,
            "max_length": 255
        })
        
        self.register_validation_rule("users", "name", {
            "type": "string",
            "required": True,
            "min_length": 1,
            "max_length": 100
        })
        
        self.register_validation_rule("users", "status", {
            "type": "string",
            "enum": ["active", "inactive", "pending", "suspended"],
            "default": "active"
        })
        
        # Trend analysis table validation
        self.register_validation_rule("trend_analysis", "keyword", {
            "type": "string",
            "required": True,
            "min_length": 1,
            "max_length": 200
        })
        
        self.register_validation_rule("trend_analysis", "user_id", {
            "type": "uuid",
            "required": True
        })
        
        self.register_validation_rule("trend_analysis", "status", {
            "type": "string",
            "enum": ["pending", "processing", "completed", "failed"],
            "default": "pending"
        })
        
        # Affiliate programs table validation
        self.register_validation_rule("affiliate_programs", "name", {
            "type": "string",
            "required": True,
            "min_length": 1,
            "max_length": 200
        })
        
        self.register_validation_rule("affiliate_programs", "url", {
            "type": "url",
            "required": True
        })
        
        self.register_validation_rule("affiliate_programs", "commission_rate", {
            "type": "float",
            "min_value": 0.0,
            "max_value": 100.0
        })
        
        logger.info("Default validation rules setup completed")
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """
        Get validation metrics for monitoring.
        
        Returns:
            Dict containing validation metrics
        """
        return {
            "registered_tables": len(self._validation_rules),
            "total_rules": sum(len(rules) for rules in self._validation_rules.values()),
            "tables_with_rules": list(self._validation_rules.keys()),
            "schema_cache_size": len(self._schema_cache)
        }

# Global middleware instance
validation_middleware = ValidationMiddleware()

# Setup default validation rules
validation_middleware.setup_default_validation_rules()

async def validation_middleware_handler(request: Request, call_next):
    """
    FastAPI middleware for data validation.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware/handler
        
    Returns:
        Response object
    """
    try:
        # Extract table name from request path
        path_parts = request.url.path.split('/')
        if len(path_parts) > 2 and path_parts[1] == "database":
            table_name = path_parts[2] if len(path_parts) > 2 else "unknown"
            
            # Determine operation type
            operation_type = "read"
            if request.method == "POST":
                operation_type = "create"
            elif request.method == "PUT":
                operation_type = "update"
            elif request.method == "DELETE":
                operation_type = "delete"
            
            # Validate request data if it's a write operation
            if operation_type in ["create", "update"]:
                try:
                    validated_data = validation_middleware.validate_request_data(
                        request, table_name, operation_type
                    )
                    # Store validated data in request state
                    request.state.validated_data = validated_data
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Validation Error",
                            "message": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
        
        # Process request
        response = await call_next(request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Validation middleware error",
            error=str(e),
            request_path=request.url.path
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "Validation middleware failed",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
