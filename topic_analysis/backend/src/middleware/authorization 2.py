"""
Authorization middleware for role-based access control.
"""
import logging
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
from functools import wraps

from fastapi import Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..schemas.auth_schemas import TokenData
from ..schemas.user_schemas import UserRole
from .auth_middleware import get_current_user

logger = logging.getLogger(__name__)

class Permission(str, Enum):
    """Permission enumeration for fine-grained access control."""
    # User permissions
    READ_OWN_PROFILE = "read:own_profile"
    UPDATE_OWN_PROFILE = "update:own_profile"
    CHANGE_OWN_PASSWORD = "change:own_password"
    DEACTIVATE_OWN_ACCOUNT = "deactivate:own_account"
    MANAGE_OWN_SESSIONS = "manage:own_sessions"
    
    # Admin permissions
    READ_ALL_USERS = "read:all_users"
    UPDATE_ALL_USERS = "update:all_users"
    DELETE_USERS = "delete:users"
    MANAGE_USER_ROLES = "manage:user_roles"
    MANAGE_USER_SESSIONS = "manage:user_sessions"
    VIEW_AUDIT_LOGS = "view:audit_logs"
    MANAGE_SYSTEM_SETTINGS = "manage:system_settings"
    
    # Content permissions
    READ_CONTENT = "read:content"
    CREATE_CONTENT = "create:content"
    UPDATE_CONTENT = "update:content"
    DELETE_CONTENT = "delete:content"
    
    # Analytics permissions
    VIEW_ANALYTICS = "view:analytics"
    MANAGE_ANALYTICS = "manage:analytics"

class ResourceType(str, Enum):
    """Resource types for authorization."""
    USER = "user"
    CONTENT = "content"
    ANALYTICS = "analytics"
    SYSTEM = "system"
    AUDIT = "audit"

# Role-based permission mapping
ROLE_PERMISSIONS: Dict[UserRole, List[Permission]] = {
    UserRole.USER: [
        # Own profile management
        Permission.READ_OWN_PROFILE,
        Permission.UPDATE_OWN_PROFILE,
        Permission.CHANGE_OWN_PASSWORD,
        Permission.DEACTIVATE_OWN_ACCOUNT,
        Permission.MANAGE_OWN_SESSIONS,
        
        # Content permissions
        Permission.READ_CONTENT,
        Permission.CREATE_CONTENT,
        Permission.UPDATE_CONTENT,
        
        # Analytics permissions
        Permission.VIEW_ANALYTICS,
    ],
    UserRole.ADMIN: [
        # All user permissions
        Permission.READ_OWN_PROFILE,
        Permission.UPDATE_OWN_PROFILE,
        Permission.CHANGE_OWN_PASSWORD,
        Permission.DEACTIVATE_OWN_ACCOUNT,
        Permission.MANAGE_OWN_SESSIONS,
        
        # Admin-only permissions
        Permission.READ_ALL_USERS,
        Permission.UPDATE_ALL_USERS,
        Permission.DELETE_USERS,
        Permission.MANAGE_USER_ROLES,
        Permission.MANAGE_USER_SESSIONS,
        Permission.VIEW_AUDIT_LOGS,
        Permission.MANAGE_SYSTEM_SETTINGS,
        
        # Content permissions
        Permission.READ_CONTENT,
        Permission.CREATE_CONTENT,
        Permission.UPDATE_CONTENT,
        Permission.DELETE_CONTENT,
        
        # Analytics permissions
        Permission.VIEW_ANALYTICS,
        Permission.MANAGE_ANALYTICS,
    ],
}

class AuthorizationMiddleware(BaseHTTPMiddleware):
    """Middleware for role-based access control."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.protected_paths = {
            "/api/v1/users/": ["user", "admin"],
            "/api/v1/admin/": ["admin"],
            "/api/v1/content/": ["user", "admin"],
            "/api/v1/analytics/": ["user", "admin"],
        }
        self.permission_required_paths = {
            "/api/v1/admin/users": [Permission.READ_ALL_USERS],
            "/api/v1/admin/users/{user_id}": [Permission.READ_ALL_USERS],
            "/api/v1/admin/users/{user_id}/update": [Permission.UPDATE_ALL_USERS],
            "/api/v1/admin/users/{user_id}/delete": [Permission.DELETE_USERS],
            "/api/v1/admin/users/{user_id}/role": [Permission.MANAGE_USER_ROLES],
            "/api/v1/admin/users/{user_id}/sessions": [Permission.MANAGE_USER_SESSIONS],
            "/api/v1/admin/audit-logs": [Permission.VIEW_AUDIT_LOGS],
            "/api/v1/admin/settings": [Permission.MANAGE_SYSTEM_SETTINGS],
        }
    
    def _get_required_roles(self, path: str) -> List[str]:
        """Get required roles for a path."""
        for protected_path, roles in self.protected_paths.items():
            if path.startswith(protected_path):
                return roles
        return []
    
    def _get_required_permissions(self, path: str) -> List[Permission]:
        """Get required permissions for a path."""
        # Check exact matches first
        if path in self.permission_required_paths:
            return self.permission_required_paths[path]
        
        # Check pattern matches
        for pattern, permissions in self.permission_required_paths.items():
            if self._path_matches_pattern(path, pattern):
                return permissions
        
        return []
    
    def _path_matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches a pattern with parameters."""
        path_parts = path.split('/')
        pattern_parts = pattern.split('/')
        
        if len(path_parts) != len(pattern_parts):
            return False
        
        for path_part, pattern_part in zip(path_parts, pattern_parts):
            if pattern_part.startswith('{') and pattern_part.endswith('}'):
                # This is a parameter, match any value
                continue
            elif path_part != pattern_part:
                return False
        
        return True
    
    def _has_permission(self, user_role: UserRole, permission: Permission) -> bool:
        """Check if user role has a specific permission."""
        user_permissions = ROLE_PERMISSIONS.get(user_role, [])
        return permission in user_permissions
    
    def _has_any_permission(self, user_role: UserRole, permissions: List[Permission]) -> bool:
        """Check if user role has any of the required permissions."""
        if not permissions:
            return True
        
        user_permissions = ROLE_PERMISSIONS.get(user_role, [])
        return any(permission in user_permissions for permission in permissions)
    
    def _get_client_info(self, request: Request) -> Dict[str, Any]:
        """Extract client information from request."""
        return {
            "ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "forwarded_for": request.headers.get("X-Forwarded-For"),
            "real_ip": request.headers.get("X-Real-IP")
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process authorization for requests."""
        # Skip authorization for public paths
        if request.url.path in ["/", "/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Skip authorization for auth endpoints (handled by authentication middleware)
        if request.url.path.startswith("/api/v1/auth/"):
            return await call_next(request)
        
        # Check if user is authenticated (set by authentication middleware)
        if not hasattr(request.state, 'authenticated') or not request.state.authenticated:
            return await call_next(request)
        
        user_role = request.state.user_role
        user_id = request.state.user_id
        path = request.url.path
        
        # Check role-based access
        required_roles = self._get_required_roles(path)
        if required_roles and user_role not in required_roles:
            logger.warning(f"User {user_id} with role {user_role} denied access to {path}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "INSUFFICIENT_ROLE",
                    "message": f"Access denied. Required roles: {', '.join(required_roles)}",
                    "status_code": 403
                }
            )
        
        # Check permission-based access
        required_permissions = self._get_required_permissions(path)
        if required_permissions and not self._has_any_permission(user_role, required_permissions):
            logger.warning(f"User {user_id} with role {user_role} denied access to {path} - missing permissions: {required_permissions}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "INSUFFICIENT_PERMISSIONS",
                    "message": f"Access denied. Required permissions: {', '.join([p.value for p in required_permissions])}",
                    "status_code": 403
                }
            )
        
        # Log successful authorization
        client_info = self._get_client_info(request)
        logger.info(
            f"User {user_id} with role {user_role} authorized for {path}",
            extra={
                "user_id": user_id,
                "user_role": user_role,
                "path": path,
                "method": request.method,
                "client_info": client_info
            }
        )
        
        # Process request
        return await call_next(request)

def require_permission(permission: Permission):
    """FastAPI dependency factory to require a specific permission."""
    def permission_checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        user_role = UserRole(current_user.role)
        
        if not _has_permission(user_role, permission):
            logger.warning(f"User {current_user.user_id} with role {user_role} denied access - missing permission: {permission}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required permission: {permission.value}"
            )
        
        return current_user
    
    return permission_checker

def require_permissions(permissions: List[Permission], require_all: bool = False):
    """FastAPI dependency factory to require multiple permissions."""
    def permission_checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        user_role = UserRole(current_user.role)
        
        if require_all:
            # User must have ALL permissions
            missing_permissions = [p for p in permissions if not _has_permission(user_role, p)]
            if missing_permissions:
                logger.warning(f"User {current_user.user_id} with role {user_role} denied access - missing permissions: {missing_permissions}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required permissions: {', '.join([p.value for p in missing_permissions])}"
                )
        else:
            # User must have ANY permission
            if not _has_any_permission(user_role, permissions):
                logger.warning(f"User {current_user.user_id} with role {user_role} denied access - missing any of permissions: {permissions}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required permissions: {', '.join([p.value for p in permissions])}"
                )
        
        return current_user
    
    return permission_checker

def require_admin_permission(permission: Permission):
    """FastAPI dependency to require admin role and specific permission."""
    def admin_permission_checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        user_role = UserRole(current_user.role)
        
        if user_role != UserRole.ADMIN:
            logger.warning(f"Non-admin user {current_user.user_id} attempted to access admin resource")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        if not _has_permission(user_role, permission):
            logger.warning(f"Admin user {current_user.user_id} denied access - missing permission: {permission}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required permission: {permission.value}"
            )
        
        return current_user
    
    return admin_permission_checker

def require_resource_ownership(resource_type: ResourceType, resource_id_param: str = "user_id"):
    """FastAPI dependency to require resource ownership."""
    def ownership_checker(
        request: Request,
        current_user: TokenData = Depends(get_current_user)
    ) -> TokenData:
        # Get resource ID from path parameters
        resource_id = request.path_params.get(resource_id_param)
        
        if not resource_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Resource ID not found in path parameter: {resource_id_param}"
            )
        
        # Check if user is accessing their own resource
        if resource_type == ResourceType.USER and resource_id == current_user.user_id:
            return current_user
        
        # Check if user has admin permissions for this resource type
        user_role = UserRole(current_user.role)
        if user_role == UserRole.ADMIN:
            return current_user
        
        # User is trying to access someone else's resource without admin privileges
        logger.warning(f"User {current_user.user_id} attempted to access {resource_type.value} {resource_id} without permission")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied. You can only access your own {resource_type.value} resources"
        )
    
    return ownership_checker

def _has_permission(user_role: UserRole, permission: Permission) -> bool:
    """Check if user role has a specific permission."""
    user_permissions = ROLE_PERMISSIONS.get(user_role, [])
    return permission in user_permissions

def _has_any_permission(user_role: UserRole, permissions: List[Permission]) -> bool:
    """Check if user role has any of the required permissions."""
    if not permissions:
        return True
    
    user_permissions = ROLE_PERMISSIONS.get(user_role, [])
    return any(permission in user_permissions for permission in permissions)

class AuthorizationMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting authorization metrics."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = {
            "total_authz_requests": 0,
            "successful_authz_requests": 0,
            "failed_authz_requests": 0,
            "role_denied_requests": 0,
            "permission_denied_requests": 0,
            "authz_requests_by_role": {},
            "authz_requests_by_path": {},
            "authz_requests_by_permission": {}
        }
    
    async def dispatch(self, request: Request, call_next):
        """Collect authorization metrics."""
        # Only track protected paths
        if not any(request.url.path.startswith(path) for path in ["/api/v1/users/", "/api/v1/admin/", "/api/v1/content/", "/api/v1/analytics/"]):
            return await call_next(request)
        
        # Track request
        self.metrics["total_authz_requests"] += 1
        self.metrics["authz_requests_by_path"][request.url.path] = self.metrics["authz_requests_by_path"].get(request.url.path, 0) + 1
        
        # Process request
        response = await call_next(request)
        
        # Track response based on status code
        if response.status_code == 403:
            self.metrics["failed_authz_requests"] += 1
            # Try to determine the specific failure reason
            if hasattr(response, 'body'):
                content = response.body.decode() if hasattr(response.body, 'decode') else str(response.body)
                if "INSUFFICIENT_ROLE" in content:
                    self.metrics["role_denied_requests"] += 1
                elif "INSUFFICIENT_PERMISSIONS" in content:
                    self.metrics["permission_denied_requests"] += 1
        elif 200 <= response.status_code < 300:
            self.metrics["successful_authz_requests"] += 1
            # Track role if available
            if hasattr(request.state, 'user_role'):
                role = request.state.user_role
                self.metrics["authz_requests_by_role"][role] = self.metrics["authz_requests_by_role"].get(role, 0) + 1
        
        return response
    
    def get_authz_metrics(self) -> Dict[str, Any]:
        """Get current authorization metrics."""
        return dict(self.metrics)

# Create global authz metrics instance
authz_metrics = AuthorizationMetricsMiddleware(None)
