"""
Keyword Enhancer API Routes
Handles seed keyword generation, external tool integration, and keyword analysis
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import structlog
import pandas as pd
import io
from pydantic import BaseModel, Field

from ..services.keyword_enhancer_service import KeywordEnhancerService

logger = structlog.get_logger()
router = APIRouter(prefix="/api/keywords", tags=["keyword-enhancer"])

class SeedKeywordRequest(BaseModel):
    search_term: str = Field(..., description="Main search term")
    selected_trends: List[str] = Field(..., description="Selected trend topics")
    content_ideas: List[str] = Field(..., description="Content ideas to enhance")
    user_id: Optional[str] = Field(None, description="User ID for tracking")

class ExternalToolUploadRequest(BaseModel):
    tool_name: str = Field(..., description="External tool name (ahrefs, semrush, etc.)")
    search_term: str = Field(..., description="Original search term")
    user_id: Optional[str] = Field(None, description="User ID for tracking")

class KeywordAnalysisResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any]

def get_keyword_enhancer_service() -> KeywordEnhancerService:
    """Get keyword enhancer service dependency"""
    return KeywordEnhancerService()

@router.post("/generate-seed-keywords", response_model=KeywordAnalysisResponse)
async def generate_seed_keywords(
    request: SeedKeywordRequest,
    keyword_service: KeywordEnhancerService = Depends(get_keyword_enhancer_service)
):
    """
    Generate seed keywords based on search term, trends, and content ideas
    """
    try:
        logger.info("Generating seed keywords",
                   search_term=request.search_term,
                   trends_count=len(request.selected_trends),
                   content_ideas_count=len(request.content_ideas))
        
        result = await keyword_service.generate_seed_keywords(
            search_term=request.search_term,
            selected_trends=request.selected_trends,
            content_ideas=request.content_ideas,
            user_id=request.user_id
        )
        
        return KeywordAnalysisResponse(
            success=True,
            message="Seed keywords generated successfully",
            data=result
        )
        
    except Exception as e:
        logger.error("Seed keyword generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Seed keyword generation failed: {str(e)}")

@router.post("/upload-external-results", response_model=KeywordAnalysisResponse)
async def upload_external_keyword_results(
    tool_name: str,
    search_term: str,
    user_id: Optional[str] = None,
    keyword_service: KeywordEnhancerService = Depends(get_keyword_enhancer_service)
):
    """
    Upload keyword analysis results from external tools (Ahrefs, Semrush, etc.)
    Expected CSV format varies by tool
    """
    try:
        # For now, return a mock response since file upload is disabled
        return KeywordAnalysisResponse(
            success=True,
            message=f"Mock processing of {tool_name} keyword data",
            data={
                "tool_name": tool_name,
                "search_term": search_term,
                "processed_keywords": 0,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        )
        
    except Exception as e:
        logger.error("External keyword upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"External keyword upload failed: {str(e)}")

@router.get("/suggested-format/{tool_name}")
async def get_suggested_csv_format(tool_name: str):
    """
    Get suggested CSV format for external tool uploads
    """
    formats = {
        "ahrefs": {
            "required_columns": [
                "keyword", "search_volume", "difficulty", "cpc", "competition",
                "trend", "click_potential", "parent_keyword"
            ],
            "description": "Ahrefs Keywords Explorer export format",
            "sample_row": {
                "keyword": "eco friendly homes",
                "search_volume": 12000,
                "difficulty": 45,
                "cpc": 2.50,
                "competition": 0.65,
                "trend": "rising",
                "click_potential": 0.8,
                "parent_keyword": "green homes"
            }
        },
        "semrush": {
            "required_columns": [
                "keyword", "search_volume", "difficulty", "cpc", "competition",
                "trend", "related_keywords", "questions"
            ],
            "description": "Semrush Keyword Magic Tool export format",
            "sample_row": {
                "keyword": "sustainable housing",
                "search_volume": 8500,
                "difficulty": 38,
                "cpc": 1.95,
                "competition": 0.52,
                "trend": "stable",
                "related_keywords": "green building, eco homes",
                "questions": "What is sustainable housing?"
            }
        },
        "ubersuggest": {
            "required_columns": [
                "keyword", "search_volume", "difficulty", "cpc", "competition",
                "trend", "related_keywords"
            ],
            "description": "Ubersuggest keyword research export format",
            "sample_row": {
                "keyword": "green building materials",
                "search_volume": 6500,
                "difficulty": 42,
                "cpc": 2.10,
                "competition": 0.58,
                "trend": "rising",
                "related_keywords": "eco materials, sustainable construction"
            }
        }
    }
    
    if tool_name.lower() not in formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported tool: {tool_name}. Supported tools: {list(formats.keys())}"
        )
    
    return {
        "success": True,
        "tool_name": tool_name,
        "format": formats[tool_name.lower()]
    }

@router.post("/analyze-keyword-clusters", response_model=KeywordAnalysisResponse)
async def analyze_keyword_clusters(
    search_term: str,
    user_id: Optional[str] = None,
    keyword_service: KeywordEnhancerService = Depends(get_keyword_enhancer_service)
):
    """
    Analyze keyword clusters and generate content strategy
    """
    try:
        result = await keyword_service.analyze_keyword_clusters(
            search_term=search_term,
            user_id=user_id
        )
        
        return KeywordAnalysisResponse(
            success=True,
            message="Keyword cluster analysis completed",
            data=result
        )
        
    except Exception as e:
        logger.error("Keyword cluster analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Keyword cluster analysis failed: {str(e)}")

@router.get("/keyword-analysis/{search_term}")
async def get_keyword_analysis(
    search_term: str,
    user_id: Optional[str] = None,
    keyword_service: KeywordEnhancerService = Depends(get_keyword_enhancer_service)
):
    """
    Get complete keyword analysis for a search term
    """
    try:
        result = await keyword_service.get_complete_keyword_analysis(
            search_term=search_term,
            user_id=user_id
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error("Failed to get keyword analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get keyword analysis: {str(e)}")

@router.post("/export-keywords")
async def export_keywords_for_external_tools(
    search_term: str,
    tool_name: str,
    user_id: Optional[str] = None,
    keyword_service: KeywordEnhancerService = Depends(get_keyword_enhancer_service)
):
    """
    Export keywords in format suitable for external tools
    """
    try:
        csv_content = await keyword_service.export_keywords_for_tool(
            search_term=search_term,
            tool_name=tool_name,
            user_id=user_id
        )
        
        return {
            "success": True,
            "csv_content": csv_content,
            "filename": f"{search_term}_{tool_name}_keywords.csv"
        }
        
    except Exception as e:
        logger.error("Keyword export failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Keyword export failed: {str(e)}")

