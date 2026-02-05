"""
Analysis API Endpoints

Handles keyword analysis, content idea generation, and results retrieval.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any, List, Optional
import logging
import uuid
from datetime import datetime

from ..models.keyword import Keyword
from ..models.analysis_report import KeywordAnalysisReport
from ..models.seo_content_idea import SEOContentIdea
from ..services.keyword_analyzer import KeywordAnalyzer
from ..services.content_idea_generator import ContentIdeaGenerator
from ..services.database import DatabaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])

# Initialize services
keyword_analyzer = KeywordAnalyzer()
content_idea_generator = ContentIdeaGenerator()
database = DatabaseService()

# In-memory storage for analysis status (in production, use Redis or database)
analysis_status = {}

@router.post("/{file_id}/start")
async def start_analysis(
    file_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = None  # This would come from authentication middleware
) -> Dict[str, Any]:
    """
    Start keyword analysis for an uploaded file
    
    Args:
        file_id: Unique file identifier
        background_tasks: FastAPI background tasks
        user_id: Authenticated user ID
        
    Returns:
        Analysis start response
    """
    try:
        # Check if file exists and is processed
        ahrefs_file = await database.get_ahrefs_file(file_id)
        if not ahrefs_file:
            raise HTTPException(status_code=404, detail="File not found")
        
        if ahrefs_file.status != "completed":
            raise HTTPException(status_code=400, detail="File not ready for analysis")
        
        # Check if analysis already exists
        existing_report = await database.get_analysis_report_by_file_id(file_id)
        if existing_report:
            raise HTTPException(status_code=400, detail="Analysis already exists for this file")
        
        # Create analysis report
        analysis_id = str(uuid.uuid4())
        analysis_report = KeywordAnalysisReport(
            id=analysis_id,
            file_id=file_id,
            user_id=user_id,
            status="started",
            created_at=datetime.utcnow()
        )
        
        # Save analysis report
        await database.save_analysis_report(analysis_report)
        
        # Initialize analysis status
        analysis_status[analysis_id] = {
            "status": "started",
            "progress": 0,
            "message": "Analysis started",
            "created_at": datetime.utcnow()
        }
        
        # Start background analysis
        background_tasks.add_task(perform_analysis, analysis_id, file_id)
        
        logger.info(f"Analysis started: {analysis_id} for file {file_id}")
        
        return {
            "analysis_id": analysis_id,
            "file_id": file_id,
            "status": "started",
            "message": "Analysis started successfully",
            "created_at": analysis_report.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis start failed: {str(e)}")

@router.get("/{file_id}/status")
async def get_analysis_status(file_id: str) -> Dict[str, Any]:
    """
    Get analysis status for a file
    
    Args:
        file_id: Unique file identifier
        
    Returns:
        Analysis status information
    """
    try:
        # Get analysis report
        analysis_report = await database.get_analysis_report_by_file_id(file_id)
        if not analysis_report:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Get status from tracking
        status_info = analysis_status.get(analysis_report.id, {
            "status": analysis_report.status,
            "progress": 0,
            "message": "Status unknown"
        })
        
        return {
            "analysis_id": analysis_report.id,
            "file_id": file_id,
            "status": status_info["status"],
            "progress": status_info["progress"],
            "message": status_info["message"],
            "created_at": analysis_report.created_at.isoformat(),
            "updated_at": status_info.get("created_at", analysis_report.created_at).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.get("/{file_id}/results")
async def get_analysis_results(file_id: str) -> Dict[str, Any]:
    """
    Get analysis results for a file
    
    Args:
        file_id: Unique file identifier
        
    Returns:
        Complete analysis results
    """
    try:
        # Get analysis report
        analysis_report = await database.get_analysis_report_by_file_id(file_id)
        if not analysis_report:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        if analysis_report.status != "completed":
            raise HTTPException(status_code=400, detail="Analysis not completed")
        
        # Get keywords
        keywords = await database.get_keywords_by_file_id(file_id)
        
        # Get content opportunities
        content_opportunities = await database.get_content_opportunities_by_analysis_id(analysis_report.id)
        
        # Get SEO content ideas
        seo_content_ideas = await database.get_seo_content_ideas_by_analysis_id(analysis_report.id)
        
        # Calculate summary statistics
        total_keywords = len(keywords)
        total_volume = sum(k.volume for k in keywords)
        avg_difficulty = sum(k.difficulty for k in keywords) / total_keywords if total_keywords > 0 else 0
        avg_cpc = sum(k.cpc for k in keywords) / total_keywords if total_keywords > 0 else 0
        
        # Intent distribution
        intent_counts = {}
        for keyword in keywords:
            for intent in keyword.intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "analysis_id": analysis_report.id,
            "file_id": file_id,
            "status": analysis_report.status,
            "summary": {
                "total_keywords": total_keywords,
                "total_volume": total_volume,
                "average_difficulty": round(avg_difficulty, 2),
                "average_cpc": round(avg_cpc, 2),
                "intent_distribution": intent_counts
            },
            "keywords": [
                {
                    "keyword": k.keyword,
                    "volume": k.volume,
                    "difficulty": k.difficulty,
                    "cpc": k.cpc,
                    "intents": k.intents,
                    "opportunity_score": k.opportunity_score
                }
                for k in keywords
            ],
            "content_opportunities": [
                {
                    "keyword": co.keyword,
                    "opportunity_score": co.opportunity_score,
                    "content_suggestions": co.content_suggestions,
                    "priority": co.priority
                }
                for co in content_opportunities
            ],
            "seo_content_ideas": [
                {
                    "title": idea.title,
                    "content_type": idea.content_type,
                    "primary_keywords": idea.primary_keywords,
                    "secondary_keywords": idea.secondary_keywords,
                    "seo_optimization_score": idea.seo_optimization_score,
                    "traffic_potential_score": idea.traffic_potential_score,
                    "total_search_volume": idea.total_search_volume,
                    "average_difficulty": idea.average_difficulty,
                    "average_cpc": idea.average_cpc,
                    "optimization_tips": idea.optimization_tips,
                    "content_outline": idea.content_outline
                }
                for idea in seo_content_ideas
            ],
            "created_at": analysis_report.created_at.isoformat(),
            "completed_at": analysis_report.completed_at.isoformat() if analysis_report.completed_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")

async def perform_analysis(analysis_id: str, file_id: str) -> None:
    """
    Background task to perform keyword analysis
    
    Args:
        analysis_id: Unique analysis identifier
        file_id: Unique file identifier
    """
    try:
        logger.info(f"Starting analysis: {analysis_id}")
        
        # Update status
        analysis_status[analysis_id].update({
            "status": "analyzing",
            "progress": 20,
            "message": "Loading keywords"
        })
        
        # Get keywords from database
        keywords = await database.get_keywords_by_file_id(file_id)
        
        # Update status
        analysis_status[analysis_id].update({
            "status": "analyzing",
            "progress": 40,
            "message": f"Analyzing {len(keywords)} keywords"
        })
        
        # Perform keyword analysis
        analyzed_keywords = keyword_analyzer.analyze_keywords(keywords)
        
        # Update status
        analysis_status[analysis_id].update({
            "status": "analyzing",
            "progress": 60,
            "message": "Generating content opportunities"
        })
        
        # Generate content opportunities
        content_opportunities = keyword_analyzer.generate_content_opportunities(analyzed_keywords)
        
        # Update status
        analysis_status[analysis_id].update({
            "status": "analyzing",
            "progress": 80,
            "message": "Generating SEO content ideas"
        })
        
        # Generate SEO content ideas
        seo_content_ideas = content_idea_generator.generate_content_ideas(
            analyzed_keywords, 
            content_opportunities
        )
        
        # Save results to database
        await database.save_analyzed_keywords(analysis_id, analyzed_keywords)
        await database.save_content_opportunities(analysis_id, content_opportunities)
        await database.save_seo_content_ideas(analysis_id, seo_content_ideas)
        
        # Update analysis report
        analysis_report = await database.get_analysis_report(analysis_id)
        if analysis_report:
            analysis_report.status = "completed"
            analysis_report.completed_at = datetime.utcnow()
            analysis_report.keywords_analyzed = len(analyzed_keywords)
            analysis_report.content_opportunities_count = len(content_opportunities)
            analysis_report.seo_content_ideas_count = len(seo_content_ideas)
            await database.update_analysis_report(analysis_report)
        
        # Update status
        analysis_status[analysis_id].update({
            "status": "completed",
            "progress": 100,
            "message": f"Analysis completed: {len(seo_content_ideas)} content ideas generated"
        })
        
        logger.info(f"Analysis completed: {analysis_id}")
        
    except Exception as e:
        logger.error(f"Error performing analysis {analysis_id}: {str(e)}")
        
        # Update status with error
        analysis_status[analysis_id].update({
            "status": "error",
            "progress": 0,
            "message": f"Analysis failed: {str(e)}"
        })
        
        # Update analysis report
        analysis_report = await database.get_analysis_report(analysis_id)
        if analysis_report:
            analysis_report.status = "error"
            analysis_report.error_message = str(e)
            await database.update_analysis_report(analysis_report)

