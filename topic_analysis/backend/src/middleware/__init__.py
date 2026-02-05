"""
Middleware package for FastAPI application.
"""

from .auth_middleware import (
    AuthenticationMiddleware,
    get_current_user,
    get_optional_user,
    require_permission
)

from .auth import (
    require_admin,
)

from .authorization import (
    AuthorizationMiddleware,
    Permission,
    ResourceType,
    require_permissions,
    require_admin_permission,
    require_resource_ownership,
)

from .rate_limiting import (
    rate_limit_middleware,
)

from .error_handling import (
    ErrorHandlingMiddleware,
)

from .logging import (
    RequestLoggingMiddleware,
    PerformanceLoggingMiddleware,
    AuditLoggingMiddleware,
    logging_middleware,
)

from .security import (
    SecurityHeadersMiddleware,
    SecurityMiddleware,
    CORSMiddleware,
)

__all__ = [
    # Authentication middleware
    "AuthenticationMiddleware",
    "get_current_user",
    "get_optional_user",
    "require_admin",
    
    # Authorization middleware
    "AuthorizationMiddleware",
    "Permission",
    "ResourceType",
    "require_permission",
    "require_permissions",
    "require_admin_permission",
    "require_resource_ownership",
    
    # Rate limiting
    "rate_limit_middleware",
    
    # Error handling middleware
    "ErrorHandlingMiddleware",
    
    # Logging
    "RequestLoggingMiddleware",
    "PerformanceLoggingMiddleware",
    "AuditLoggingMiddleware",
    "logging_middleware",
    
    # Security middleware
    "SecurityHeadersMiddleware",
    "SecurityMiddleware",
    "CORSMiddleware",
]
