"""
Trend Analysis API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from ..core.database import get_db
from ..services.trend_service import TrendService
from ..models.user import User
from ..models.trend_analysis import TrendAnalysis, AnalysisStatus
from src.core.supabase_database_service import SupabaseDatabaseService
from ..schemas.trend_schemas import (
    TrendAnalysisRequest,
    TrendAnalysisResponse,
    TrendAnalysisListResponse,
    TrendForecastResponse,
    TrendAnalysisUpdate
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/trends", tags=["trend-analysis"])

def get_trend_service(db: SupabaseDatabaseService = Depends(get_db)) -> TrendService:
    """Get trend service dependency"""
    return TrendService(db)

def get_current_user(db: SupabaseDatabaseService = Depends(get_db)) -> User:
    """Get current authenticated user (placeholder - implement auth middleware)"""
    # This is a placeholder - in real implementation, this would extract user from JWT token
    user = db.query(User).first()
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

@router.post("/analyze", response_model=TrendAnalysisResponse)
async def start_trend_analysis(
    request: TrendAnalysisRequest,
    current_user: User = Depends(get_current_user),
    trend_service: TrendService = Depends(get_trend_service)
):
    """Start trend analysis for a keyword"""
    try:
        logger.info("Starting trend analysis", user_id=current_user.id, keyword=request.keyword)
        
        # Check user limits
        if not await trend_service.check_user_limits(current_user.id, "trend_analyses"):
            raise HTTPException(
                status_code=403,
                detail="Trend analysis limit reached for your subscription tier"
            )
        
        # Start analysis
        analysis = await trend_service.start_trend_analysis(
            user_id=current_user.id,
            keyword=request.keyword,
            geo=request.geo,
            time_range=request.time_range,
            affiliate_research_id=request.affiliate_research_id
        )
        
        logger.info("Trend analysis started", analysis_id=analysis.id, user_id=current_user.id)
        
        return TrendAnalysisResponse(
            id=analysis.id,
            keyword=analysis.keyword,
            geo=analysis.geo,
            status=analysis.status.value,
            opportunity_score=analysis.opportunity_score,
            created_at=analysis.created_at.isoformat(),
            estimated_completion_time=analysis.estimated_completion_time
        )
        
    except ValueError as e:
        logger.error("Invalid request for trend analysis", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to start trend analysis", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/analysis/{analysis_id}", response_model=TrendAnalysisResponse)
async def get_trend_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    trend_service: TrendService = Depends(get_trend_service)
):
    """Get trend analysis by ID"""
    try:
        analysis = await trend_service.get_trend_analysis(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Check if user owns this analysis
        if analysis.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return TrendAnalysisResponse(
            id=analysis.id,
            keyword=analysis.keyword,
            geo=analysis.geo,
            status=analysis.status.value,
            opportunity_score=analysis.opportunity_score,
            created_at=analysis.created_at.isoformat(),
            estimated_completion_time=analysis.estimated_completion_time,
            historical_data=analysis.historical_data,
            forecast_data=analysis.forecast_data,
            trend_insights=analysis.trend_insights
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get trend analysis", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/analysis", response_model=TrendAnalysisListResponse)
async def list_trend_analyses(
    skip: int = 0,
    limit: int = 20,
    status: Optional[AnalysisStatus] = None,
    current_user: User = Depends(get_current_user),
    trend_service: TrendService = Depends(get_trend_service)
):
    """List user's trend analyses"""
    try:
        analyses = await trend_service.list_user_analyses(
            user_id=current_user.id,
            skip=skip,
            limit=limit,
            status=status
        )
        
        return TrendAnalysisListResponse(
            analyses=[
                TrendAnalysisResponse(
                    id=analysis.id,
                    keyword=analysis.keyword,
                    geo=analysis.geo,
                    status=analysis.status.value,
                    opportunity_score=analysis.opportunity_score,
                    created_at=analysis.created_at.isoformat(),
                    estimated_completion_time=analysis.estimated_completion_time
                )
                for analysis in analyses
            ],
            total=len(analyses)
        )
        
    except Exception as e:
        logger.error("Failed to list trend analyses", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/analysis/{analysis_id}", response_model=TrendAnalysisResponse)
async def update_trend_analysis(
    analysis_id: str,
    request: TrendAnalysisUpdate,
    current_user: User = Depends(get_current_user),
    trend_service: TrendService = Depends(get_trend_service)
):
    """Update trend analysis"""
    try:
        analysis = await trend_service.get_trend_analysis(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Check if user owns this analysis
        if analysis.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update analysis
        updated_analysis = await trend_service.update_trend_analysis(
            analysis_id=analysis_id,
            notes=request.notes
        )
        
        return TrendAnalysisResponse(
            id=updated_analysis.id,
            keyword=updated_analysis.keyword,
            geo=updated_analysis.geo,
            status=updated_analysis.status.value,
            opportunity_score=updated_analysis.opportunity_score,
            created_at=updated_analysis.created_at.isoformat(),
            estimated_completion_time=updated_analysis.estimated_completion_time,
            historical_data=updated_analysis.historical_data,
            forecast_data=updated_analysis.forecast_data,
            trend_insights=updated_analysis.trend_insights
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update trend analysis", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/analysis/{analysis_id}")
async def delete_trend_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    trend_service: TrendService = Depends(get_trend_service)
):
    """Delete trend analysis"""
    try:
        analysis = await trend_service.get_trend_analysis(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Check if user owns this analysis
        if analysis.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await trend_service.delete_trend_analysis(analysis_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete analysis")
        
        return {"message": "Analysis deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete trend analysis", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/analysis/{analysis_id}/forecast", response_model=TrendForecastResponse)
async def get_trend_forecast(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    trend_service: TrendService = Depends(get_trend_service)
):
    """Get trend forecast for analysis"""
    try:
        analysis = await trend_service.get_trend_analysis(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Check if user owns this analysis
        if analysis.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        forecast = await trend_service.get_trend_forecast(analysis_id)
        
        return TrendForecastResponse(
            analysis_id=analysis_id,
            keyword=analysis.keyword,
            forecast_data=forecast.get("forecast_data", {}),
            confidence_score=forecast.get("confidence_score", 0.0),
            trend_direction=forecast.get("trend_direction", "stable"),
            seasonal_patterns=forecast.get("seasonal_patterns", []),
            news_impact=forecast.get("news_impact", {}),
            llm_insights=forecast.get("llm_insights", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get trend forecast", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/analysis/{analysis_id}/insights", response_model=Dict[str, Any])
async def get_trend_insights(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    trend_service: TrendService = Depends(get_trend_service)
):
    """Get trend insights for analysis"""
    try:
        analysis = await trend_service.get_trend_analysis(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Check if user owns this analysis
        if analysis.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        insights = await trend_service.get_trend_insights(analysis_id)
        
        return insights
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get trend insights", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/analysis/{analysis_id}/refresh")
async def refresh_trend_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    trend_service: TrendService = Depends(get_trend_service)
):
    """Refresh trend analysis with latest data"""
    try:
        analysis = await trend_service.get_trend_analysis(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Check if user owns this analysis
        if analysis.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await trend_service.refresh_trend_analysis(analysis_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to refresh analysis")
        
        return {"message": "Analysis refresh initiated", "analysis_id": analysis_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to refresh trend analysis", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/keywords/suggestions")
async def get_keyword_suggestions(
    query: str,
    geo: str = "US",
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    trend_service: TrendService = Depends(get_trend_service)
):
    """Get keyword suggestions for trend analysis"""
    try:
        suggestions = await trend_service.get_keyword_suggestions(
            query=query,
            geo=geo,
            limit=limit
        )
        
        return {
            "query": query,
            "geo": geo,
            "suggestions": suggestions
        }
        
    except Exception as e:
        logger.error("Failed to get keyword suggestions", query=query, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/regions", response_model=List[Dict[str, Any]])
async def get_available_regions():
    """Get available regions for trend analysis"""
    try:
        regions = [
            {"code": "US", "name": "United States", "language": "en"},
            {"code": "GB", "name": "United Kingdom", "language": "en"},
            {"code": "CA", "name": "Canada", "language": "en"},
            {"code": "AU", "name": "Australia", "language": "en"},
            {"code": "DE", "name": "Germany", "language": "de"},
            {"code": "FR", "name": "France", "language": "fr"},
            {"code": "ES", "name": "Spain", "language": "es"},
            {"code": "IT", "name": "Italy", "language": "it"},
            {"code": "BR", "name": "Brazil", "language": "pt"},
            {"code": "MX", "name": "Mexico", "language": "es"},
            {"code": "JP", "name": "Japan", "language": "ja"},
            {"code": "KR", "name": "South Korea", "language": "ko"},
            {"code": "IN", "name": "India", "language": "en"},
            {"code": "CN", "name": "China", "language": "zh"},
            {"code": "RU", "name": "Russia", "language": "ru"}
        ]
        
        return regions
        
    except Exception as e:
        logger.error("Failed to get available regions", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/analysis/{analysis_id}/analytics", response_model=Dict[str, Any])
async def get_analysis_analytics(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    trend_service: TrendService = Depends(get_trend_service)
):
    """Get analysis analytics"""
    try:
        analysis = await trend_service.get_trend_analysis(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Check if user owns this analysis
        if analysis.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        analytics = await trend_service.get_analysis_analytics(analysis_id)
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get analysis analytics", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
