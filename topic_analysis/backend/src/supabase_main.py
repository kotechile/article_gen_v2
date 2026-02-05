"""
TrendTap Backend API - Supabase Version
AI Research Workspace using Supabase SDK instead of SQLAlchemy
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import structlog

# Import Supabase-based API routers
from .api.supabase_health_routes import router as supabase_health_router
from .api.supabase_api_routes import router as supabase_api_router
from .api.settings_routes import router as settings_router
from .core.supabase_database_service import get_supabase_db

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Create FastAPI application
app = FastAPI(
    title="TrendTap API - Supabase",
    description="AI Research Workspace using Supabase SDK",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://trendtap.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "trendtap.com", "*.trendtap.com"]
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TrendTap API - Supabase Version",
        "version": "1.0.0",
        "status": "running",
        "database": "supabase",
        "docs": "/docs"
    }

@app.get("/supabase-health")
async def supabase_health():
    """Check Supabase database health"""
    try:
        db = get_supabase_db()
        health_status = db.health_check()
        return health_status
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Supabase connection failed: {str(e)}",
            "timestamp": "2024-01-01T00:00:00Z"
        }

# Include Supabase-based API routers
app.include_router(supabase_health_router)
app.include_router(supabase_api_router)
app.include_router(settings_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "supabase_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
