"""
Content Generation API Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import structlog

from ..dependencies import get_db, get_current_user
from ..services.content_generation_service import ContentGenerationService
from ..services.keyword_clustering_service import KeywordClusteringService
from ..services.external_tool_integration_service import ExternalToolIntegrationService
from ..schemas.content_generation import (
    ContentGenerationRequest,
    ContentGenerationResponse,
    ContentIdeaCreate,
    ContentIdeaUpdate,
    ContentIdeaResponse,
    KeywordClusteringRequest,
    KeywordClusteringResponse,
    ExternalToolRequest,
    ExternalToolResponse
)
from ..models.user import User

logger = structlog.get_logger()
router = APIRouter()

@router.post("/content/generate", response_model=ContentGenerationResponse)
async def generate_content_ideas_endpoint(
    request: ContentGenerationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate content ideas based on trend analysis and keyword clusters
    """
    try:
        service = ContentGenerationService(db)
        result = await service.generate_content_ideas(
            user_id=str(current_user.id),
            workflow_session_id=request.workflow_session_id,
            trend_analysis_id=request.trend_analysis_id,
            topic_decomposition_id=request.topic_decomposition_id,
            keyword_clusters=request.keyword_clusters,
            content_types=request.content_types,
            target_audience=request.target_audience,
            max_ideas=request.max_ideas
        )
        return ContentGenerationResponse(**result)
    except ValueError as e:
        logger.error("Validation error generating content ideas", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to generate content ideas", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate content ideas")

@router.get("/content/ideas", response_model=List[ContentIdeaResponse])
async def get_content_ideas_endpoint(
    workflow_session_id: Optional[str] = None,
    status: Optional[str] = None,
    content_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get content ideas with optional filtering
    """
    try:
        service = ContentGenerationService(db)
        ideas = await service.get_content_ideas(
            user_id=str(current_user.id),
            workflow_session_id=workflow_session_id,
            status=status,
            content_type=content_type,
            limit=limit,
            offset=offset
        )
        return [ContentIdeaResponse(**idea) for idea in ideas]
    except Exception as e:
        logger.error("Failed to get content ideas", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get content ideas")

@router.get("/content/ideas/{idea_id}", response_model=ContentIdeaResponse)
async def get_content_idea_endpoint(
    idea_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get specific content idea by ID
    """
    try:
        service = ContentGenerationService(db)
        idea = await service.get_content_idea(idea_id, str(current_user.id))
        if not idea:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Content idea not found")
        return ContentIdeaResponse(**idea)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get content idea", error=str(e), idea_id=idea_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get content idea")

@router.put("/content/ideas/{idea_id}", response_model=ContentIdeaResponse)
async def update_content_idea_endpoint(
    idea_id: str,
    request: ContentIdeaUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update content idea
    """
    try:
        service = ContentGenerationService(db)
        updates = request.dict(exclude_unset=True)
        idea = await service.update_content_idea(idea_id, str(current_user.id), updates)
        if not idea:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Content idea not found")
        return ContentIdeaResponse(**idea.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update content idea", error=str(e), idea_id=idea_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update content idea")

@router.delete("/content/ideas/{idea_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_content_idea_endpoint(
    idea_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete content idea
    """
    try:
        service = ContentGenerationService(db)
        deleted = await service.delete_content_idea(idea_id, str(current_user.id))
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Content idea not found")
        return {"message": "Content idea deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete content idea", error=str(e), idea_id=idea_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete content idea")

@router.post("/keywords/cluster", response_model=KeywordClusteringResponse)
async def cluster_keywords_endpoint(
    request: KeywordClusteringRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Cluster keywords using specified method
    """
    try:
        service = KeywordClusteringService(db)
        result = await service.cluster_keywords(
            user_id=str(current_user.id),
            workflow_session_id=request.workflow_session_id,
            external_tool_result_id=request.external_tool_result_id,
            keywords_data=request.keywords_data,
            cluster_method=request.cluster_method,
            n_clusters=request.n_clusters,
            min_cluster_size=request.min_cluster_size,
            max_clusters=request.max_clusters
        )
        return KeywordClusteringResponse(**result)
    except ValueError as e:
        logger.error("Validation error clustering keywords", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to cluster keywords", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to cluster keywords")

@router.get("/keywords/clusters", response_model=List[Dict[str, Any]])
async def get_keyword_clusters_endpoint(
    workflow_session_id: Optional[str] = None,
    is_active: Optional[bool] = None,
    is_processed: Optional[bool] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get keyword clusters with optional filtering
    """
    try:
        service = KeywordClusteringService(db)
        clusters = await service.get_clusters(
            user_id=str(current_user.id),
            workflow_session_id=workflow_session_id,
            is_active=is_active,
            is_processed=is_processed,
            limit=limit,
            offset=offset
        )
        return clusters
    except Exception as e:
        logger.error("Failed to get keyword clusters", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get keyword clusters")

@router.put("/keywords/clusters/{cluster_id}", response_model=Dict[str, Any])
async def update_keyword_cluster_endpoint(
    cluster_id: str,
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update keyword cluster
    """
    try:
        service = KeywordClusteringService(db)
        cluster = await service.update_cluster(cluster_id, str(current_user.id), request)
        if not cluster:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Keyword cluster not found")
        return cluster.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update keyword cluster", error=str(e), cluster_id=cluster_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update keyword cluster")

@router.delete("/keywords/clusters/{cluster_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_keyword_cluster_endpoint(
    cluster_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete keyword cluster
    """
    try:
        service = KeywordClusteringService(db)
        deleted = await service.delete_cluster(cluster_id, str(current_user.id))
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Keyword cluster not found")
        return {"message": "Keyword cluster deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete keyword cluster", error=str(e), cluster_id=cluster_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete keyword cluster")

@router.post("/external-tools/process", response_model=ExternalToolResponse)
async def process_external_tool_endpoint(
    request: ExternalToolRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Process external tool request (Semrush, Ahrefs, Ubersuggest)
    """
    try:
        service = ExternalToolIntegrationService(db)
        result = await service.process_external_tool_request(
            user_id=str(current_user.id),
            workflow_session_id=request.workflow_session_id,
            tool_name=request.tool_name,
            query_type=request.query_type,
            query_parameters=request.query_parameters,
            seed_keywords=request.seed_keywords,
            trend_analysis_id=request.trend_analysis_id
        )
        return ExternalToolResponse(**result)
    except ValueError as e:
        logger.error("Validation error processing external tool", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to process external tool", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process external tool")

@router.get("/external-tools/results", response_model=List[ExternalToolResponse])
async def get_external_tool_results_endpoint(
    workflow_session_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get external tool results with optional filtering
    """
    try:
        service = ExternalToolIntegrationService(db)
        results = await service.get_external_tool_results(
            user_id=str(current_user.id),
            workflow_session_id=workflow_session_id,
            tool_name=tool_name,
            status=status,
            limit=limit,
            offset=offset
        )
        return [ExternalToolResponse(**result) for result in results]
    except Exception as e:
        logger.error("Failed to get external tool results", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get external tool results")

@router.post("/external-tools/results/{result_id}/retry", response_model=ExternalToolResponse)
async def retry_external_tool_endpoint(
    result_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retry a failed external tool request
    """
    try:
        service = ExternalToolIntegrationService(db)
        result = await service.retry_failed_request(result_id, str(current_user.id))
        return ExternalToolResponse(**result)
    except ValueError as e:
        logger.error("Validation error retrying external tool", error=str(e), result_id=result_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to retry external tool", error=str(e), result_id=result_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retry external tool")

@router.get("/content/ideas/{idea_id}/metrics", response_model=Dict[str, Any])
async def get_content_idea_metrics_endpoint(
    idea_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get content idea metrics and analytics
    """
    try:
        service = ContentGenerationService(db)
        idea = await service.get_content_idea(idea_id, str(current_user.id))
        if not idea:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Content idea not found")
        
        # Calculate metrics
        metrics = {
            "keyword_count": idea.get("keyword_count", 0),
            "affiliate_offer_count": idea.get("affiliate_offer_count", 0),
            "quality_score": idea.get("quality_score", 0),
            "monetization_potential": idea.get("monetization_potential", "unknown"),
            "seo_potential": idea.get("seo_potential", "unknown"),
            "content_metrics": idea.get("content_metrics", {})
        }
        
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get content idea metrics", error=str(e), idea_id=idea_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get content idea metrics")

@router.get("/keywords/clusters/{cluster_id}/summary", response_model=Dict[str, Any])
async def get_cluster_summary_endpoint(
    cluster_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get keyword cluster summary and analytics
    """
    try:
        service = KeywordClusteringService(db)
        cluster = await service.get_cluster(cluster_id, str(current_user.id))
        if not cluster:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Keyword cluster not found")
        
        return cluster.get_cluster_summary()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get cluster summary", error=str(e), cluster_id=cluster_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get cluster summary")
