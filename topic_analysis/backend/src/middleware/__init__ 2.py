"""
Middleware package for FastAPI application.
"""

from .auth_middleware import (
    get_current_user,
    get_optional_user,
    require_permission
)

from .authorization import (
    Permission,
    ResourceType,
    require_permission,
    require_permissions,
    require_admin_permission,
    require_resource_ownership
)

__all__ = [
    # Authentication
    "get_current_user",
    "get_optional_user",
    "require_permission",
    
    # Authorization
    "Permission",
    "ResourceType",
    "require_permission",
    "require_permissions",
    "require_admin_permission",
    "require_resource_ownership",
]
