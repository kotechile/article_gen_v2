"""
TrendTap Backend API
AI Research Workspace for affiliate research, trend analysis, and content generation
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import structlog

# Import API routers
from .api import health_routes, keyword_routes, keyword_enhancer_routes, affiliate_research_routes, content_ideas_routes, research_topics_routes, llm_routes, settings_routes, enhanced_topic_routes
from .routers import dataforseo_router

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
    title="TrendTap API",
    description="AI Research Workspace for affiliate research, trend analysis, and content generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import Request

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP Error: {exc.status_code} - {exc.detail} - Path: {request.url.path}")
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation Error: {exc.errors()} - Body: {exc.body} - Path: {request.url.path}")
    return await request_validation_exception_handler(request, exc)

@app.on_event("startup")
async def startup_event():
    print("Startup: Listing all registered routes")
    for route in app.routes:
        print(f"Route: {route.path} [{route.methods}]")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TrendTap API",
        "version": "1.0.1",
        "build_id": "debug-2026-01-28-v2", # User can verify this
        "status": "running",
        "docs": "/docs"
    }

# Include API routers
app.include_router(health_routes.router)
app.include_router(dataforseo_router.router)
app.include_router(keyword_routes.router)
app.include_router(keyword_enhancer_routes.router)
app.include_router(affiliate_research_routes.router)
app.include_router(content_ideas_routes.router)
app.include_router(research_topics_routes.router)
app.include_router(llm_routes.router)
app.include_router(llm_routes.router)
app.include_router(settings_routes.router)
app.include_router(enhanced_topic_routes.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )