"""
Analytics API Routes
Provides endpoints for analytics and reporting
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from ..services.analytics_service import AnalyticsService
from ..core.supabase_database import get_supabase_db
from ..schemas.analytics_schemas import (
    UserAnalyticsRequest,
    PlatformAnalyticsRequest,
    ContentPerformanceRequest,
    TrendAnalyticsRequest,
    AnalyticsExportRequest
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

@router.post("/user", response_model=Dict[str, Any])
async def get_user_analytics(
    request: UserAnalyticsRequest,
    db: Any = Depends(get_supabase_db)
):
    """
    Get comprehensive analytics for a user
    """
    try:
        logger.info("Getting user analytics", 
                   user_id=request.user_id,
                   start_date=request.start_date.isoformat() if request.start_date else None,
                   end_date=request.end_date.isoformat() if request.end_date else None)
        
        analytics_service = AnalyticsService()
        
        result = await analytics_service.get_user_analytics(
            user_id=request.user_id,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get user analytics"
            )
        
        logger.info("User analytics retrieved successfully", 
                   user_id=request.user_id)
        
        return {
            "success": True,
            "user_id": result["user_id"],
            "period": result["period"],
            "analytics": result["analytics"],
            "insights": result["insights"],
            "recommendations": result["recommendations"],
            "generated_at": result["generated_at"]
        }
        
    except Exception as e:
        logger.error("Failed to get user analytics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/platform", response_model=Dict[str, Any])
async def get_platform_analytics(
    request: PlatformAnalyticsRequest,
    db: Any = Depends(get_supabase_db)
):
    """
    Get platform-wide analytics (admin only)
    """
    try:
        logger.info("Getting platform analytics", 
                   start_date=request.start_date.isoformat() if request.start_date else None,
                   end_date=request.end_date.isoformat() if request.end_date else None)
        
        analytics_service = AnalyticsService()
        
        result = await analytics_service.get_platform_analytics(
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get platform analytics"
            )
        
        logger.info("Platform analytics retrieved successfully")
        
        return {
            "success": True,
            "period": result["period"],
            "platform_data": result["platform_data"],
            "insights": result["insights"],
            "generated_at": result["generated_at"]
        }
        
    except Exception as e:
        logger.error("Failed to get platform analytics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/content-performance", response_model=Dict[str, Any])
async def get_content_performance_analytics(
    request: ContentPerformanceRequest,
    db: Any = Depends(get_supabase_db)
):
    """
    Get content performance analytics
    """
    try:
        logger.info("Getting content performance analytics", 
                   user_id=request.user_id,
                   content_count=len(request.content_ids) if request.content_ids else None,
                   start_date=request.start_date.isoformat() if request.start_date else None,
                   end_date=request.end_date.isoformat() if request.end_date else None)
        
        analytics_service = AnalyticsService()
        
        result = await analytics_service.get_content_performance_analytics(
            user_id=request.user_id,
            content_ids=request.content_ids,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get content performance analytics"
            )
        
        logger.info("Content performance analytics retrieved successfully", 
                   user_id=request.user_id)
        
        return {
            "success": True,
            "user_id": result["user_id"],
            "period": result["period"],
            "performance_data": result["performance_data"],
            "metrics": result["metrics"],
            "insights": result["insights"],
            "generated_at": result["generated_at"]
        }
        
    except Exception as e:
        logger.error("Failed to get content performance analytics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/trends", response_model=Dict[str, Any])
async def get_trend_analytics(
    request: TrendAnalyticsRequest,
    db: Any = Depends(get_supabase_db)
):
    """
    Get trend analysis analytics
    """
    try:
        logger.info("Getting trend analytics", 
                   user_id=request.user_id,
                   start_date=request.start_date.isoformat() if request.start_date else None,
                   end_date=request.end_date.isoformat() if request.end_date else None)
        
        analytics_service = AnalyticsService()
        
        result = await analytics_service.get_trend_analytics(
            user_id=request.user_id,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get trend analytics"
            )
        
        logger.info("Trend analytics retrieved successfully", 
                   user_id=request.user_id)
        
        return {
            "success": True,
            "user_id": result["user_id"],
            "period": result["period"],
            "trend_data": result["trend_data"],
            "insights": result["insights"],
            "predictions": result["predictions"],
            "generated_at": result["generated_at"]
        }
        
    except Exception as e:
        logger.error("Failed to get trend analytics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/export", response_model=Dict[str, Any])
async def export_analytics_report(
    request: AnalyticsExportRequest,
    db: Any = Depends(get_supabase_db)
):
    """
    Export analytics report in specified format
    """
    try:
        logger.info("Exporting analytics report", 
                   user_id=request.user_id,
                   report_type=request.report_type,
                   format=request.format,
                   start_date=request.start_date.isoformat() if request.start_date else None,
                   end_date=request.end_date.isoformat() if request.end_date else None)
        
        analytics_service = AnalyticsService()
        
        result = await analytics_service.export_analytics_report(
            user_id=request.user_id,
            report_type=request.report_type,
            format=request.format,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to export analytics report"
            )
        
        logger.info("Analytics report exported successfully", 
                   report_type=request.report_type,
                   format=request.format)
        
        return {
            "success": True,
            "report_type": result["report_type"],
            "format": result["format"],
            "export_result": result["export_result"],
            "generated_at": result["generated_at"]
        }
        
    except Exception as e:
        logger.error("Failed to export analytics report", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

