"""
Enhanced Trends Analysis API Routes
Integrates with affiliate research for comprehensive trend analysis
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import structlog
import asyncio
from datetime import datetime

from ..core.supabase_database import get_supabase_db
from ..core.supabase_database_service import SupabaseDatabaseService
from ..schemas.trends_schemas import (
    TrendAnalysisRequest,
    TrendAnalysisResponse,
    TrendData,
    AffiliateResearchResponse,
    ContentIdeaRequest,
    ContentIdeaResponse
)
from ..services.trend_service import TrendService
from ..services.affiliate_research_service import AffiliateResearchService
from ..services.content_service import ContentService
from src.core.supabase_database_service import SupabaseDatabaseService

router = APIRouter()
logger = structlog.get_logger()

@router.get("/affiliate-researches")
async def get_affiliate_researches(
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """Get all affiliate research records for trend analysis"""
    try:
        # Get recent affiliate research records
        researches = await db.get_recent_affiliate_researches(limit=20)
        
        formatted_researches = []
        for research in researches:
            formatted_researches.append({
                "id": research.get("id"),
                "main_topic": research.get("main_topic"),
                "subtopics": research.get("subtopics", []),
                "programs_count": len(research.get("programs", [])),
                "created_at": research.get("created_at"),
                "status": research.get("status", "completed")
            })
        
        return {
            "success": True,
            "data": formatted_researches,
            "message": f"Found {len(formatted_researches)} research records"
        }
    except Exception as e:
        logger.error("Failed to get affiliate researches", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get affiliate researches: {str(e)}")

@router.get("/affiliate-research/{research_id}")
async def get_affiliate_research(
    research_id: str,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """Get specific affiliate research with programs"""
    try:
        research = await db.get_affiliate_research_by_id(research_id)
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        return {
            "success": True,
            "data": research,
            "message": "Research retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get affiliate research", research_id=research_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get research: {str(e)}")

@router.post("/trend-analysis", response_model=TrendAnalysisResponse)
async def analyze_trends(
    request: TrendAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """Analyze trends for multiple topics/subtopics"""
    try:
        trend_service = TrendService()
        affiliate_service = AffiliateResearchService()
        
        # Get affiliate research data if provided
        affiliate_programs = []
        if request.affiliate_research_id:
            research = await db.get_affiliate_research_by_id(request.affiliate_research_id)
            if research:
                affiliate_programs = research.get("programs", [])
        
        # Analyze trends for each topic/subtopic
        all_topics = request.topics + request.subtopics
        trend_results = []
        
        for i, topic in enumerate(all_topics):
            try:
                # Analyze trend for this topic
                trend_analysis = await trend_service.analyze_trend(
                    keyword=topic,
                    geo=request.geo,
                    time_range=request.time_range,
                    include_forecast=True
                )
                
                # Calculate opportunity score based on affiliate programs
                opportunity_score = await _calculate_opportunity_score(
                    topic, affiliate_programs, trend_analysis
                )
                
                trend_data = TrendData(
                    id=f"trend-{i}",
                    topic=topic if topic in request.topics else "",
                    subtopic=topic if topic in request.subtopics else "",
                    trend_score=trend_analysis.get("trend_score", 0),
                    opportunity_score=opportunity_score,
                    search_volume=trend_analysis.get("search_volume", 0),
                    competition=trend_analysis.get("competition_level", "medium"),
                    trend_direction=trend_analysis.get("trend_direction", "stable"),
                    forecast_data=trend_analysis.get("forecast", {}),
                    selected=False
                )
                
                trend_results.append(trend_data)
                
            except Exception as e:
                logger.warning("Failed to analyze trend for topic", topic=topic, error=str(e))
                # Add fallback trend data
                trend_data = TrendData(
                    id=f"trend-{i}",
                    topic=topic if topic in request.topics else "",
                    subtopic=topic if topic in request.subtopics else "",
                    trend_score=50,
                    opportunity_score=50,
                    search_volume=1000,
                    competition="medium",
                    trend_direction="stable",
                    forecast_data={},
                    selected=False
                )
                trend_results.append(trend_data)
        
        # Save trend analysis results
        analysis_id = await _save_trend_analysis(
            request, trend_results, affiliate_programs, db
        )
        
        return TrendAnalysisResponse(
            success=True,
            data={
                "analysis_id": analysis_id,
                "trends": trend_results,
                "affiliate_programs": affiliate_programs,
                "analysis_summary": {
                    "total_topics": len(all_topics),
                    "high_opportunity_trends": len([t for t in trend_results if t.opportunity_score >= 80]),
                    "trending_up": len([t for t in trend_results if t.trend_direction == "up"]),
                    "low_competition": len([t for t in trend_results if t.competition == "low"])
                }
            },
            message=f"Analyzed trends for {len(all_topics)} topics"
        )
        
    except Exception as e:
        logger.error("Trend analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@router.post("/content-ideas", response_model=ContentIdeaResponse)
async def generate_content_ideas(
    request: ContentIdeaRequest,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """Generate content ideas based on selected trends and affiliate programs"""
    try:
        content_service = ContentService()
        
        # Get selected trends data
        selected_trends = request.selected_trends
        affiliate_programs = request.affiliate_programs
        
        # Generate content ideas using LLM
        content_ideas = await content_service.generate_content_ideas_from_trends(
            trends=selected_trends,
            affiliate_programs=affiliate_programs,
            content_types=request.content_types,
            max_ideas=request.max_ideas
        )
        
        return ContentIdeaResponse(
            success=True,
            data={
                "content_ideas": content_ideas,
                "trends_used": len(selected_trends),
                "affiliate_programs_used": len(affiliate_programs),
                "generated_at": datetime.utcnow().isoformat()
            },
            message=f"Generated {len(content_ideas)} content ideas"
        )
        
    except Exception as e:
        logger.error("Content idea generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Content idea generation failed: {str(e)}")

async def _calculate_opportunity_score(
    topic: str, 
    affiliate_programs: List[Dict[str, Any]], 
    trend_analysis: Dict[str, Any]
) -> int:
    """Calculate opportunity score based on affiliate programs and trend data"""
    base_score = trend_analysis.get("trend_score", 50)
    
    # Boost score if there are relevant affiliate programs
    relevant_programs = [
        p for p in affiliate_programs 
        if topic.lower() in p.get("name", "").lower() or 
           topic.lower() in p.get("description", "").lower()
    ]
    
    if relevant_programs:
        # Higher commission rates = higher opportunity
        avg_commission = sum([
            float(p.get("commission_rate", "0").replace("%", "")) 
            for p in relevant_programs 
            if p.get("commission_rate", "0").replace("%", "").isdigit()
        ]) / len(relevant_programs) if relevant_programs else 0
        
        commission_boost = min(avg_commission * 2, 30)  # Max 30 point boost
        base_score += commission_boost
    
    # Boost for high search volume
    search_volume = trend_analysis.get("search_volume", 0)
    if search_volume > 10000:
        base_score += 10
    elif search_volume > 5000:
        base_score += 5
    
    # Boost for low competition
    competition = trend_analysis.get("competition_level", "medium")
    if competition == "low":
        base_score += 15
    elif competition == "medium":
        base_score += 5
    
    return min(int(base_score), 100)

async def _save_trend_analysis(
    request: TrendAnalysisRequest,
    trend_results: List[TrendData],
    affiliate_programs: List[Dict[str, Any]],
    db: SupabaseDatabaseService
) -> str:
    """Save trend analysis results to database"""
    try:
        analysis_data = {
            "affiliate_research_id": request.affiliate_research_id,
            "topics": request.topics,
            "subtopics": request.subtopics,
            "geo": request.geo,
            "time_range": request.time_range,
            "trend_results": [trend.dict() for trend in trend_results],
            "affiliate_programs": affiliate_programs,
            "created_at": datetime.utcnow().isoformat(),
            "status": "completed"
        }
        
        result = await db.create_trend_analysis(analysis_data)
        return result.get("id", "temp-analysis-id")
        
    except Exception as e:
        logger.warning("Failed to save trend analysis", error=str(e))
        return "temp-analysis-id"

