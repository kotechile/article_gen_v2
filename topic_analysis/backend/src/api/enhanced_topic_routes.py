"""
Enhanced topics API routes
FastAPI routes for Google Autocomplete integration with topic decomposition
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ..services.enhanced_topic_decomposition_service import EnhancedTopicDecompositionService
from ..services.enhanced_idea_generator import enhanced_idea_generator
from ..integrations.google_autocomplete import GoogleAutocompleteService
from ..models.enhanced_subtopic import EnhancedSubtopic, SubtopicSource
from ..models.autocomplete_result import AutocompleteResult
from ..monitoring.performance_metrics import PerformanceMonitor, record_autocomplete_performance, record_decomposition_performance, performance_tracker
from ..core.api_key_manager import api_key_manager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/enhanced-topics", tags=["Enhanced Topics"])

def get_default_llm_provider():
    """Get the default LLM provider based on available API keys - prioritize DeepSeek"""
    # Check which providers have API keys available, prioritize DeepSeek
    if api_key_manager.get_deepseek_key():
        return "deepseek"
    elif api_key_manager.get_openai_key():
        return "openai"
    elif api_key_manager.get_anthropic_key():
        return "anthropic"
    elif api_key_manager.get_google_ai_key():
        return "google_ai"
    else:
        return "openai"  # fallback

# Global service instances (in production, these would be dependency injected)
google_autocomplete_service = GoogleAutocompleteService()
# Initialize with both autocomplete and LLM for hybrid approach
enhanced_topic_service = EnhancedTopicDecompositionService(
    google_autocomplete_service=google_autocomplete_service
)

# Request/Response Models
class EnhancedTopicDecompositionRequest(BaseModel):
    """Request model for enhanced topic decomposition"""
    search_query: str = Field(..., min_length=1, max_length=200, description="The topic to decompose")
    user_id: str = Field(..., min_length=1, max_length=100, description="User identifier")
    max_subtopics: int = Field(default=6, ge=1, le=10, description="Maximum number of subtopics")
    use_autocomplete: bool = Field(default=True, description="Whether to use Google Autocomplete")
    use_llm: bool = Field(default=True, description="Whether to use LLM processing")
    
    @validator('search_query')
    def validate_search_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Search query cannot be empty')
        return v.strip()
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        return v.strip()

class EnhancedTopicDecompositionResponse(BaseModel):
    """Response model for enhanced topic decomposition"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable message")
    original_query: str = Field(..., description="The original search query")
    subtopics: list = Field(..., description="List of enhanced subtopics")
    autocomplete_data: Optional[dict] = Field(None, description="Autocomplete data")
    processing_time: float = Field(..., description="Total processing time in seconds")
    enhancement_methods: list = Field(..., description="Methods used for enhancement")

class AutocompleteRequest(BaseModel):
    """Request model for autocomplete suggestions"""
    query: str = Field(..., min_length=1, max_length=200, description="Search query")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class AutocompleteResponse(BaseModel):
    """Response model for autocomplete suggestions"""
    success: bool = Field(..., description="Whether the request was successful")
    query: str = Field(..., description="The processed query")
    suggestions: list = Field(..., description="Autocomplete suggestions")
    total_suggestions: int = Field(..., description="Total number of suggestions")
    processing_time: float = Field(..., description="Processing time in seconds")

class MethodComparisonRequest(BaseModel):
    """Request model for method comparison"""
    search_query: str = Field(..., min_length=1, max_length=200, description="The topic to compare methods for")
    user_id: str = Field(..., min_length=1, max_length=100, description="User identifier")
    max_subtopics: int = Field(default=6, ge=1, le=10, description="Maximum number of subtopics per method")
    
    @validator('search_query')
    def validate_search_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Search query cannot be empty')
        return v.strip()
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        return v.strip()

class MethodComparisonResponse(BaseModel):
    """Response model for method comparison"""
    success: bool = Field(..., description="Whether the comparison was successful")
    original_query: str = Field(..., description="The original query")
    comparison: dict = Field(..., description="Method comparison results")
    recommendation: str = Field(..., description="Recommendation based on comparison")
    processing_time: float = Field(..., description="Total processing time in seconds")

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")

# Rate limiting (simple in-memory implementation)
class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited"""
        import time
        current_time = time.time()
        
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Remove old requests outside the window
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if current_time - req_time < self.window_seconds
        ]
        
        # Check if limit exceeded
        if len(self.requests[user_id]) >= self.max_requests:
            return True
        
        # Add current request
        self.requests[user_id].append(current_time)
        return False

rate_limiter = RateLimiter()

# Simple request/response models for frontend compatibility
class SimpleTopicAnalysisRequest(BaseModel):
    """Simple request model for topic analysis (frontend compatibility)"""
    topic: str = Field(..., min_length=1, max_length=200, description="The topic to analyze")
    
    @validator('topic')
    def validate_topic(cls, v):
        if not v or not v.strip():
            raise ValueError('Topic cannot be empty')
        return v.strip()

    success: bool = Field(default=True, description="Whether the operation was successful")
    message: str = Field(default="Analysis completed", description="Human-readable message")

class SimpleTopicAnalysisResponse(BaseModel):
    """Simple response model for topic analysis"""
    subtopics: list = Field(..., description="List of subtopics")
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable message")

class IdeaBurstRequest(BaseModel):
    """Request model for Idea Burst generation"""
    user_id: str = Field(..., description="User identifier")
    topic_id: str = Field(..., description="Topic identifier")
    subtopic: str = Field(..., description="Subtopic name")
    keywords: list = Field(default=[], description="List of keywords")
    affiliate_offers: list = Field(default=[], description="List of affiliate offers")

class IdeaBurstResponse(BaseModel):
    """Response model for Idea Burst generation"""
    success: bool
    blog_ideas: list
    software_ideas: list
    message: str

# API Routes
@router.post("/analyze-topic", response_model=SimpleTopicAnalysisResponse)
async def analyze_topic_simple(request: SimpleTopicAnalysisRequest):
    """
    Simple topic analysis endpoint for frontend compatibility
    
    This endpoint provides a simple interface that the frontend expects
    """
    try:
        # Convert to enhanced decomposition request
        enhanced_request = EnhancedTopicDecompositionRequest(
            search_query=request.topic,
            user_id="frontend-user",  # Default user ID for frontend
            max_subtopics=6,
            use_autocomplete=True,
            use_llm=True
        )
        
        # Call the enhanced decomposition
        result = await decompose_topic_enhanced(enhanced_request)
        
        # Convert to simple response format
        return SimpleTopicAnalysisResponse(
            subtopics=result.subtopics,
            success=result.success,
            message=result.message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in simple topic analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/decompose", response_model=EnhancedTopicDecompositionResponse)
async def decompose_topic_enhanced(request: EnhancedTopicDecompositionRequest):
    """
    Enhanced topic decomposition with Google Autocomplete
    
    Decompose a topic into subtopics using Google Autocomplete + LLM hybrid approach
    """
    try:
        # Rate limiting check
        if rate_limiter.is_rate_limited(request.user_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Validate request
        if not request.search_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )
        
        # Decompose topic with performance monitoring
        async with PerformanceMonitor("enhanced_decomposition", {
            "query": request.search_query,
            "user_id": request.user_id,
            "max_subtopics": request.max_subtopics,
            "use_autocomplete": request.use_autocomplete,
            "use_llm": request.use_llm
        }):
            result = await enhanced_topic_service.decompose_topic_enhanced(
                query=request.search_query,
                user_id=request.user_id,
                max_subtopics=request.max_subtopics,
                use_autocomplete=request.use_autocomplete,
                use_llm=request.use_llm
            )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
        
        return EnhancedTopicDecompositionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enhanced topic decomposition: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/autocomplete/{query}", response_model=AutocompleteResponse)
async def get_autocomplete_suggestions(query: str):
    """
    Get Google Autocomplete suggestions
    
    Direct access to Google Autocomplete suggestions for a query
    """
    try:
        # Validate query
        if not query or len(query) > 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid query. Must be 1-200 characters."
            )
        
        # Get suggestions with performance monitoring
        async with PerformanceMonitor("autocomplete", {"query": query}):
            result = await google_autocomplete_service.get_suggestions(query)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error_message or "Failed to get autocomplete suggestions"
            )
        
        return AutocompleteResponse(
            success=result.success,
            query=result.query,
            suggestions=result.suggestions,
            total_suggestions=result.total_suggestions,
            processing_time=result.processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting autocomplete suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/compare-methods", response_model=MethodComparisonResponse)
async def compare_decomposition_methods(request: MethodComparisonRequest):
    """
    Compare different decomposition methods
    
    Run side-by-side comparison of LLM-only, autocomplete-only, and hybrid approaches
    """
    try:
        # Rate limiting check
        if rate_limiter.is_rate_limited(request.user_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Validate request
        if not request.search_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )
        
        # Compare methods
        result = await enhanced_topic_service.compare_methods(
            query=request.search_query,
            user_id=request.user_id,
            max_subtopics=request.max_subtopics
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("recommendation", "Error comparing methods")
            )
        
        return MethodComparisonResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in method comparison: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )



@router.post("/idea-burst", response_model=IdeaBurstResponse)
async def generate_idea_burst(request: IdeaBurstRequest):
    """
    Generate Idea Burst content (Step 5)
    
    Generates blog titles and software ideas for a specific subtopic
    """
    try:
        # Rate limiting check
        if rate_limiter.is_rate_limited(request.user_id):
             raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
            
        result = await enhanced_idea_generator.generate_content_for_subtopic(
            topic_id=request.topic_id,
            subtopic=request.subtopic,
            keywords=request.keywords,
            affiliate_offers=request.affiliate_offers,
            user_id=request.user_id
        )
        
        success = len(result["blog_ideas"]) > 0 or len(result["software_ideas"]) > 0
        message = "Content generated successfully" if success else "Failed to generate content"
        
        return IdeaBurstResponse(
            success=success,
            blog_ideas=result["blog_ideas"],
            software_ideas=result["software_ideas"],
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in idea burst generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check service health
        autocomplete_healthy = google_autocomplete_service.is_healthy()
        
        return {
            "status": "healthy" if autocomplete_healthy else "degraded",
            "services": {
                "autocomplete": "healthy" if autocomplete_healthy else "unhealthy",
                "enhanced_topic_service": "healthy"
            },
            "cache_stats": enhanced_topic_service.get_cache_stats()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.post("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    try:
        enhanced_topic_service.clear_cache()
        google_autocomplete_service.clear_cache()
        
        return {
            "success": True,
            "message": "All caches cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        return {
            "enhanced_topic_service": enhanced_topic_service.get_cache_stats(),
            "google_autocomplete_service": google_autocomplete_service.get_cache_stats()
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}"
        )

@router.get("/performance/metrics")
async def get_performance_metrics():
    """Get performance metrics"""
    try:
        stats = await performance_tracker.get_all_stats()
        health_score = await performance_tracker.get_health_score()
        recent_metrics = await performance_tracker.get_recent_metrics(50)
        
        return {
            "health_score": health_score,
            "operation_stats": {
                operation: {
                    "total_requests": stats.total_requests,
                    "successful_requests": stats.successful_requests,
                    "failed_requests": stats.failed_requests,
                    "success_rate": stats.successful_requests / stats.total_requests if stats.total_requests > 0 else 0,
                    "avg_duration": stats.avg_duration,
                    "min_duration": stats.min_duration if stats.min_duration != float('inf') else 0,
                    "max_duration": stats.max_duration
                }
                for operation, stats in stats.items()
            },
            "recent_metrics": [
                {
                    "operation": metric.operation,
                    "duration": metric.duration,
                    "success": metric.success,
                    "timestamp": metric.timestamp,
                    "metadata": metric.metadata
                }
                for metric in recent_metrics
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )

@router.post("/performance/clear")
async def clear_performance_metrics():
    """Clear performance metrics"""
    try:
        await performance_tracker.clear_metrics()
        return {
            "success": True,
            "message": "Performance metrics cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing performance metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear performance metrics: {str(e)}"
        )

# Error handlers are handled at the app level in main.py