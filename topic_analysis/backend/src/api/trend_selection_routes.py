"""
Trend Selection API Routes
Handles trend selection, CSV upload, and trend management
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import structlog
import pandas as pd
import io
from pydantic import BaseModel, Field

from ..services.trend_selection_service import TrendSelectionService
from ..schemas.trend_schemas import TrendSelectionRequest, TrendSelectionResponse

logger = structlog.get_logger()
router = APIRouter(prefix="/api/trends", tags=["trend-selection"])

class TrendCSVUploadResponse(BaseModel):
    success: bool
    message: str
    trends_imported: int
    data: Optional[Dict[str, Any]] = None

class TrendSelectionUpdateRequest(BaseModel):
    selected_trend_ids: List[str] = Field(..., description="List of selected trend IDs")
    search_term: str = Field(..., description="Original search term")
    user_id: Optional[str] = Field(None, description="User ID for tracking")

def get_trend_selection_service() -> TrendSelectionService:
    """Get trend selection service dependency"""
    return TrendSelectionService()

@router.post("/upload-csv", response_model=TrendCSVUploadResponse)
async def upload_trend_csv(
    file: UploadFile = File(...),
    search_term: str = Form(...),
    user_id: Optional[str] = Form(None),
    trend_service: TrendSelectionService = Depends(get_trend_selection_service)
):
    """
    Upload CSV file with trend data for analysis
    Expected CSV format: topic,search_volume,trend_direction,competition,opportunity_score
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['topic', 'search_volume', 'trend_direction', 'competition', 'opportunity_score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Process CSV data
        result = await trend_service.process_trend_csv(
            df=df,
            search_term=search_term,
            user_id=user_id
        )
        
        return TrendCSVUploadResponse(
            success=True,
            message=f"Successfully imported {result['trends_imported']} trends",
            trends_imported=result['trends_imported'],
            data=result
        )
        
    except Exception as e:
        logger.error("CSV upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"CSV upload failed: {str(e)}")

@router.post("/select-trends", response_model=TrendSelectionResponse)
async def select_trends(
    request: TrendSelectionUpdateRequest,
    trend_service: TrendSelectionService = Depends(get_trend_selection_service)
):
    """
    Select specific trends for content/software generation
    """
    try:
        result = await trend_service.select_trends_for_generation(
            selected_trend_ids=request.selected_trend_ids,
            search_term=request.search_term,
            user_id=request.user_id
        )
        
        return TrendSelectionResponse(
            success=True,
            message="Trends selected successfully",
            data=result
        )
        
    except Exception as e:
        logger.error("Trend selection failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Trend selection failed: {str(e)}")

@router.get("/available-trends/{search_term}")
async def get_available_trends(
    search_term: str,
    user_id: Optional[str] = None,
    trend_service: TrendSelectionService = Depends(get_trend_selection_service)
):
    """
    Get all available trends for a search term (from analysis + CSV uploads)
    """
    try:
        trends = await trend_service.get_available_trends(
            search_term=search_term,
            user_id=user_id
        )
        
        return {
            "success": True,
            "data": trends
        }
        
    except Exception as e:
        logger.error("Failed to get available trends", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")

@router.get("/selected-trends/{search_term}")
async def get_selected_trends(
    search_term: str,
    user_id: Optional[str] = None,
    trend_service: TrendSelectionService = Depends(get_trend_selection_service)
):
    """
    Get currently selected trends for a search term
    """
    try:
        trends = await trend_service.get_selected_trends(
            search_term=search_term,
            user_id=user_id
        )
        
        return {
            "success": True,
            "data": trends
        }
        
    except Exception as e:
        logger.error("Failed to get selected trends", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get selected trends: {str(e)}")

