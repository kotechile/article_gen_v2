"""
API module initialization
"""

from fastapi import APIRouter
from .affiliate_routes import router as affiliate_router
from .trend_routes import router as trend_router
from .keyword_routes import router as keyword_router
from .content_routes import router as content_router
from .software_routes import router as software_router
from .export_routes import router as export_router
from .calendar_routes import router as calendar_router
from .user_routes import router as user_router

# Create main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(affiliate_router)
api_router.include_router(trend_router)
api_router.include_router(keyword_router)
api_router.include_router(content_router)
api_router.include_router(software_router)
api_router.include_router(export_router)
api_router.include_router(calendar_router)
api_router.include_router(user_router)

__all__ = ["api_router"]