"""
Trend Analysis API Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import structlog

from ..dependencies import get_db, get_current_user
from ..services.trend_analysis_service import TrendAnalysisService
from ..services.csv_upload_service import CSVUploadService
from ..schemas.trend_analysis import (
    TrendAnalysisCreate, 
    TrendAnalysisUpdate, 
    TrendAnalysisResponse,
    TrendAnalysisStart,
    CSVUploadRequest,
    CSVUploadResponse
)
from ..models.user import User
from ..models.trend_analysis import TrendAnalysisSource

logger = structlog.get_logger()
router = APIRouter()

@router.post("/trend-analysis", response_model=TrendAnalysisResponse, status_code=status.HTTP_201_CREATED)
async def create_trend_analysis_endpoint(
    request: TrendAnalysisCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new trend analysis
    """
    try:
        service = TrendAnalysisService(db)
        analysis_data = await service.create_trend_analysis(
            user_id=str(current_user.id),
            workflow_session_id=request.workflow_session_id,
            analysis_name=request.analysis_name,
            keywords=request.keywords,
            timeframe=request.timeframe,
            geo=request.geo,
            category=request.category,
            description=request.description,
            topic_decomposition_id=request.topic_decomposition_id
        )
        return TrendAnalysisResponse(**analysis_data)
    except ValueError as e:
        logger.error("Validation error creating trend analysis", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to create trend analysis", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create trend analysis")

@router.post("/trend-analysis/{analysis_id}/start", response_model=TrendAnalysisResponse)
async def start_trend_analysis_endpoint(
    analysis_id: str,
    request: TrendAnalysisStart,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Start trend analysis processing
    """
    try:
        service = TrendAnalysisService(db)
        analysis_data = await service.start_trend_analysis(
            analysis_id=analysis_id,
            user_id=str(current_user.id),
            source=TrendAnalysisSource(request.source)
        )
        return TrendAnalysisResponse(**analysis_data)
    except ValueError as e:
        logger.error("Validation error starting trend analysis", error=str(e), analysis_id=analysis_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to start trend analysis", error=str(e), analysis_id=analysis_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to start trend analysis")

@router.get("/trend-analysis/{analysis_id}", response_model=TrendAnalysisResponse)
async def get_trend_analysis_endpoint(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get trend analysis by ID
    """
    try:
        service = TrendAnalysisService(db)
        analysis_data = await service.get_trend_analysis(
            analysis_id=analysis_id,
            user_id=str(current_user.id)
        )
        if not analysis_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trend analysis not found")
        return TrendAnalysisResponse(**analysis_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get trend analysis", error=str(e), analysis_id=analysis_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get trend analysis")

@router.get("/trend-analysis", response_model=List[TrendAnalysisResponse])
async def list_trend_analyses_endpoint(
    workflow_session_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List trend analyses for the current user
    """
    try:
        service = TrendAnalysisService(db)
        analyses = await service.list_trend_analyses(
            user_id=str(current_user.id),
            workflow_session_id=workflow_session_id
        )
        return [TrendAnalysisResponse(**analysis) for analysis in analyses]
    except Exception as e:
        logger.error("Failed to list trend analyses", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list trend analyses")

@router.put("/trend-analysis/{analysis_id}", response_model=TrendAnalysisResponse)
async def update_trend_analysis_endpoint(
    analysis_id: str,
    request: TrendAnalysisUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update trend analysis
    """
    try:
        service = TrendAnalysisService(db)
        updates = request.dict(exclude_unset=True)
        analysis_data = await service.update_trend_analysis(
            analysis_id=analysis_id,
            user_id=str(current_user.id),
            updates=updates
        )
        if not analysis_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trend analysis not found")
        return TrendAnalysisResponse(**analysis_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update trend analysis", error=str(e), analysis_id=analysis_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update trend analysis")

@router.delete("/trend-analysis/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_trend_analysis_endpoint(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete trend analysis
    """
    try:
        service = TrendAnalysisService(db)
        deleted = await service.delete_trend_analysis(
            analysis_id=analysis_id,
            user_id=str(current_user.id)
        )
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trend analysis not found")
        return {"message": "Trend analysis deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete trend analysis", error=str(e), analysis_id=analysis_id, user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete trend analysis")

@router.post("/trend-analysis/csv-upload", response_model=CSVUploadResponse)
async def upload_csv_for_trend_analysis_endpoint(
    file: UploadFile = File(...),
    tool_type: str = Form(default="google_trends"),
    workflow_session_id: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload CSV file for trend analysis
    """
    try:
        service = CSVUploadService(db)
        upload_result = await service.process_csv_upload(
            file=file,
            user_id=str(current_user.id),
            tool_type=tool_type,
            workflow_session_id=workflow_session_id
        )
        return CSVUploadResponse(**upload_result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload CSV for trend analysis", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload CSV file")

@router.post("/trend-analysis/csv-upload/validate", response_model=Dict[str, Any])
async def validate_csv_structure_endpoint(
    file: UploadFile = File(...),
    tool_type: str = Form(default="google_trends"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Validate CSV structure before upload
    """
    try:
        service = CSVUploadService(db)
        validation_result = await service.validate_csv_structure(
            file=file,
            tool_type=tool_type
        )
        return validation_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to validate CSV structure", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to validate CSV structure")

@router.get("/trend-analysis/csv-upload/cached", response_model=Dict[str, Any])
async def get_cached_csv_data_endpoint(
    workflow_session_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get cached CSV upload data
    """
    try:
        service = CSVUploadService(db)
        cached_data = await service.get_cached_trend_data(
            user_id=str(current_user.id),
            workflow_session_id=workflow_session_id
        )
        if not cached_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No cached CSV data found")
        return cached_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get cached CSV data", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get cached CSV data")