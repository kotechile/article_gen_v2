"""
Keyword Management API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from ..core.database import get_db
from ..services.keyword_service import KeywordService
from ..models.user import User
from ..models.keyword_data import KeywordData, KeywordSource
from ..core.supabase_auth import get_current_user
from ..schemas.keyword_schemas import (
    KeywordUploadRequest,
    KeywordUploadResponse,
    KeywordCrawlRequest,
    KeywordCrawlResponse,
    KeywordDataResponse,
    KeywordDataListResponse,
    KeywordAnalysisResponse,
    KeywordClusterResponse
)
from pydantic import BaseModel
from src.core.supabase_database_service import SupabaseDatabaseService

logger = structlog.get_logger()
router = APIRouter(prefix="/api/keywords", tags=["keyword-management"])

# Pydantic models for keyword generation
class KeywordGenerationRequest(BaseModel):
    subtopics: List[str]
    topicId: str
    topicTitle: str

class KeywordGenerationResponse(BaseModel):
    success: bool
    keywords: List[str]
    message: Optional[str] = None

def get_keyword_service(db: SupabaseDatabaseService = Depends(get_db)) -> KeywordService:
    """Get keyword service dependency"""
    return KeywordService(db)


@router.post("/upload", response_model=KeywordUploadResponse)
async def upload_keywords(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """Upload keywords from CSV file"""
    try:
        logger.info("Uploading keywords", user_id=current_user.id, filename=file.filename)
        
        # Check user limits
        if not await keyword_service.check_user_limits(current_user.id, "keyword_data"):
            raise HTTPException(
                status_code=403,
                detail="Keyword data limit reached for your subscription tier"
            )
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read file content
        content = await file.read()
        
        # Process keywords
        result = await keyword_service.process_uploaded_keywords(
            user_id=current_user.id,
            file_content=content,
            filename=file.filename
        )
        
        logger.info("Keywords uploaded successfully", user_id=current_user.id, keywords_count=result["keywords_processed"])
        
        return KeywordUploadResponse(
            success=True,
            keywords_processed=result["keywords_processed"],
            keywords_skipped=result["keywords_skipped"],
            processing_time=result["processing_time"],
            keyword_data_id=result["keyword_data_id"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload keywords", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/crawl", response_model=KeywordCrawlResponse)
async def crawl_keywords(
    request: KeywordCrawlRequest,
    current_user: User = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """Crawl keywords using DataForSEO"""
    try:
        logger.info("Starting keyword crawl", user_id=current_user.id, seed_keyword=request.seed_keyword)
        
        # Check user limits
        if not await keyword_service.check_user_limits(current_user.id, "keyword_data"):
            raise HTTPException(
                status_code=403,
                detail="Keyword data limit reached for your subscription tier"
            )
        
        # Start crawl
        result = await keyword_service.start_keyword_crawl(
            user_id=current_user.id,
            seed_keyword=request.seed_keyword,
            depth=request.depth,
            geo=request.geo,
            language=request.language
        )
        
        logger.info("Keyword crawl started", user_id=current_user.id, crawl_id=result["crawl_id"])
        
        return KeywordCrawlResponse(
            success=True,
            crawl_id=result["crawl_id"],
            keywords_crawled=result["keywords_crawled"],
            estimated_completion_time=result["estimated_completion_time"]
        )
        
    except ValueError as e:
        logger.error("Invalid request for keyword crawl", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to start keyword crawl", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/data/{keyword_data_id}", response_model=KeywordDataResponse)
async def get_keyword_data(
    keyword_data_id: str,
    current_user: User = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """Get keyword data by ID"""
    try:
        keyword_data = await keyword_service.get_keyword_data(keyword_data_id)
        
        if not keyword_data:
            raise HTTPException(status_code=404, detail="Keyword data not found")
        
        # Check if user owns this data
        if keyword_data.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return KeywordDataResponse(
            id=keyword_data.id,
            source=keyword_data.source.value,
            keywords_count=len(keyword_data.keywords),
            priority_score=keyword_data.priority_score,
            created_at=keyword_data.created_at.isoformat(),
            keywords=keyword_data.keywords,
            clusters=keyword_data.clusters,
            gap_analysis=keyword_data.gap_analysis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get keyword data", keyword_data_id=keyword_data_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/data", response_model=KeywordDataListResponse)
async def list_keyword_data(
    skip: int = 0,
    limit: int = 20,
    source: Optional[KeywordSource] = None,
    current_user: User = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """List user's keyword data"""
    try:
        keyword_data_list = await keyword_service.list_user_keyword_data(
            user_id=current_user.id,
            skip=skip,
            limit=limit,
            source=source
        )
        
        return KeywordDataListResponse(
            keyword_data=[
                KeywordDataResponse(
                    id=kd.id,
                    source=kd.source.value,
                    keywords_count=len(kd.keywords),
                    priority_score=kd.priority_score,
                    created_at=kd.created_at.isoformat(),
                    keywords=kd.keywords[:5],  # Show only first 5 keywords
                    clusters=kd.clusters,
                    gap_analysis=kd.gap_analysis
                )
                for kd in keyword_data_list
            ],
            total=len(keyword_data_list)
        )
        
    except Exception as e:
        logger.error("Failed to list keyword data", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/data/{keyword_data_id}")
async def delete_keyword_data(
    keyword_data_id: str,
    current_user: User = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """Delete keyword data"""
    try:
        keyword_data = await keyword_service.get_keyword_data(keyword_data_id)
        
        if not keyword_data:
            raise HTTPException(status_code=404, detail="Keyword data not found")
        
        # Check if user owns this data
        if keyword_data.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await keyword_service.delete_keyword_data(keyword_data_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete keyword data")
        
        return {"message": "Keyword data deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete keyword data", keyword_data_id=keyword_data_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/data/{keyword_data_id}/analysis", response_model=KeywordAnalysisResponse)
async def get_keyword_analysis(
    keyword_data_id: str,
    current_user: User = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """Get keyword analysis"""
    try:
        keyword_data = await keyword_service.get_keyword_data(keyword_data_id)
        
        if not keyword_data:
            raise HTTPException(status_code=404, detail="Keyword data not found")
        
        # Check if user owns this data
        if keyword_data.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        analysis = await keyword_service.get_keyword_analysis(keyword_data_id)
        
        return KeywordAnalysisResponse(
            keyword_data_id=keyword_data_id,
            total_keywords=analysis["total_keywords"],
            high_priority_keywords=analysis["high_priority_keywords"],
            medium_priority_keywords=analysis["medium_priority_keywords"],
            low_priority_keywords=analysis["low_priority_keywords"],
            clusters=analysis["clusters"],
            gap_analysis=analysis["gap_analysis"],
            competition_analysis=analysis["competition_analysis"],
            search_volume_analysis=analysis["search_volume_analysis"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get keyword analysis", keyword_data_id=keyword_data_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/data/{keyword_data_id}/clusters", response_model=List[KeywordClusterResponse])
async def get_keyword_clusters(
    keyword_data_id: str,
    current_user: User = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """Get keyword clusters"""
    try:
        keyword_data = await keyword_service.get_keyword_data(keyword_data_id)
        
        if not keyword_data:
            raise HTTPException(status_code=404, detail="Keyword data not found")
        
        # Check if user owns this data
        if keyword_data.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        clusters = await keyword_service.get_keyword_clusters(keyword_data_id)
        
        return [
            KeywordClusterResponse(
                id=cluster["id"],
                name=cluster["name"],
                keywords=cluster["keywords"],
                priority_score=cluster["priority_score"],
                search_volume=cluster["search_volume"],
                competition_level=cluster["competition_level"],
                opportunity_score=cluster["opportunity_score"]
            )
            for cluster in clusters
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get keyword clusters", keyword_data_id=keyword_data_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/data/{keyword_data_id}/enrich")
async def enrich_keywords(
    keyword_data_id: str,
    current_user: User = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """Enrich keywords with additional data"""
    try:
        keyword_data = await keyword_service.get_keyword_data(keyword_data_id)
        
        if not keyword_data:
            raise HTTPException(status_code=404, detail="Keyword data not found")
        
        # Check if user owns this data
        if keyword_data.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await keyword_service.enrich_keywords(keyword_data_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to enrich keywords")
        
        return {"message": "Keywords enrichment initiated", "keyword_data_id": keyword_data_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to enrich keywords", keyword_data_id=keyword_data_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/data/{keyword_data_id}/cluster")
async def cluster_keywords(
    keyword_data_id: str,
    current_user: User = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """Cluster keywords"""
    try:
        keyword_data = await keyword_service.get_keyword_data(keyword_data_id)
        
        if not keyword_data:
            raise HTTPException(status_code=404, detail="Keyword data not found")
        
        # Check if user owns this data
        if keyword_data.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await keyword_service.cluster_keywords(keyword_data_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to cluster keywords")
        
        return {"message": "Keywords clustering initiated", "keyword_data_id": keyword_data_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cluster keywords", keyword_data_id=keyword_data_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/data/{keyword_data_id}/export")
async def export_keywords(
    keyword_data_id: str,
    format: str = "csv",
    current_user: User = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """Export keywords to file"""
    try:
        keyword_data = await keyword_service.get_keyword_data(keyword_data_id)
        
        if not keyword_data:
            raise HTTPException(status_code=404, detail="Keyword data not found")
        
        # Check if user owns this data
        if keyword_data.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if format not in ["csv", "json", "xlsx"]:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        export_data = await keyword_service.export_keywords(keyword_data_id, format)
        
        return {
            "success": True,
            "format": format,
            "download_url": export_data["download_url"],
            "expires_at": export_data["expires_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export keywords", keyword_data_id=keyword_data_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/suggestions")
async def get_keyword_suggestions(
    query: str,
    geo: str = "US",
    language: str = "en",
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """Get keyword suggestions"""
    try:
        suggestions = await keyword_service.get_keyword_suggestions(
            query=query,
            geo=geo,
            language=language,
            limit=limit
        )
        
        return {
            "query": query,
            "geo": geo,
            "language": language,
            "suggestions": suggestions
        }
        
    except Exception as e:
        logger.error("Failed to get keyword suggestions", query=query, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/data/{keyword_data_id}/analytics", response_model=Dict[str, Any])
async def get_keyword_analytics(
    keyword_data_id: str,
    current_user: User = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """Get keyword analytics"""
    try:
        keyword_data = await keyword_service.get_keyword_data(keyword_data_id)
        
        if not keyword_data:
            raise HTTPException(status_code=404, detail="Keyword data not found")
        
        # Check if user owns this data
        if keyword_data.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        analytics = await keyword_service.get_keyword_analytics(keyword_data_id)
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get keyword analytics", keyword_data_id=keyword_data_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/generate", response_model=KeywordGenerationResponse)
async def generate_keywords(
    request: KeywordGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    keyword_service: KeywordService = Depends(get_keyword_service)
):
    """Generate keywords using LLM based on subtopics"""
    try:
        user_id = current_user.get("id") or current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        
        logger.info("Generating keywords with LLM", 
                   user_id=user_id, 
                   topic_id=request.topicId,
                   subtopics_count=len(request.subtopics))
        
        # Check user limits
        if not await keyword_service.check_user_limits(user_id, "keyword_data"):
            raise HTTPException(
                status_code=403,
                detail="Keyword data limit reached for your subscription tier"
            )
        
        # Generate keywords using LLM
        keywords = await keyword_service.generate_keywords_with_llm(
            subtopics=request.subtopics,
            topic_title=request.topicTitle,
            user_id=user_id
        )
        
        logger.info("Keywords generated successfully", 
                   user_id=user_id, 
                   keywords_count=len(keywords))
        
        return KeywordGenerationResponse(
            success=True,
            keywords=keywords,
            message=f"Generated {len(keywords)} keywords successfully"
        )
        
    except ValueError as e:
        logger.error("Invalid request for keyword generation", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to generate keywords", 
                    user_id=user_id if 'user_id' in locals() else 'unknown', 
                    error=str(e))
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to generate keywords: {str(e)}")
