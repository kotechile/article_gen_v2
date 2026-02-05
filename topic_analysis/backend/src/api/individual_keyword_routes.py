"""
Individual Keyword Optimization API Routes
Handles individual keyword upload, optimization, and management
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import List, Optional, Dict, Any
import structlog
from datetime import datetime
import uuid

from ..services.individual_keyword_optimizer import IndividualKeywordOptimizer
from ..core.supabase_database import get_supabase_db
from ..schemas.individual_keyword_schemas import (
    IndividualKeywordUploadRequest,
    IndividualKeywordOptimizationRequest,
    IndividualKeywordResponse,
    OptimizationSessionResponse,
    ContentIdeaEnhancementResponse
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/individual-keywords", tags=["individual-keywords"])

def get_individual_keyword_optimizer() -> IndividualKeywordOptimizer:
    return IndividualKeywordOptimizer()

@router.post("/upload-ahrefs-csv")
async def upload_ahrefs_csv(
    file: UploadFile = File(...),
    content_idea_id: str = Form(...),
    user_id: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    optimizer: IndividualKeywordOptimizer = Depends(get_individual_keyword_optimizer)
):
    """
    Upload Ahrefs CSV file and process individual keywords
    """
    try:
        logger.info("Processing Ahrefs CSV upload", 
                   filename=file.filename, 
                   content_idea_id=content_idea_id)
        
        # Read file content
        content = await file.read()
        csv_text = content.decode('utf-8')
        
        # Parse CSV and extract keywords
        keywords = await _parse_ahrefs_csv(csv_text)
        
        if not keywords:
            raise HTTPException(status_code=400, detail="No valid keywords found in CSV file")
        
        # Create optimization session
        session_id = str(uuid.uuid4())
        
        # Start background optimization
        background_tasks.add_task(
            _process_keyword_optimization,
            keywords,
            content_idea_id,
            user_id,
            session_id,
            optimizer
        )
        
        return {
            "success": True,
            "message": f"Successfully processed {len(keywords)} keywords",
            "session_id": session_id,
            "keywords_count": len(keywords),
            "status": "processing"
        }
        
    except Exception as e:
        logger.error("Ahrefs CSV upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/optimize-keywords")
async def optimize_keywords(
    request: IndividualKeywordOptimizationRequest,
    optimizer: IndividualKeywordOptimizer = Depends(get_individual_keyword_optimizer)
):
    """
    Optimize selected keywords using LLM analysis
    """
    try:
        logger.info("Starting keyword optimization", 
                   keyword_count=len(request.keywords),
                   content_idea_id=request.content_idea_id)
        
        result = await optimizer.optimize_keywords(
            keywords=request.keywords,
            content_idea_id=request.content_idea_id,
            user_id=request.user_id,
            optimization_session_id=request.session_id
        )
        
        return result
        
    except Exception as e:
        logger.error("Keyword optimization failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.get("/optimization-session/{session_id}")
async def get_optimization_session(
    session_id: str,
    user_id: str,
    db = Depends(get_supabase_db)
):
    """
    Get optimization session details
    """
    try:
        result = db.table('keyword_optimization_sessions').select('*').eq('id', session_id).eq('user_id', user_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Optimization session not found")
        
        session = result.data[0]
        
        # Get optimized keywords for this session
        keywords_result = db.table('individual_keywords').select('*').eq('content_idea_id', session['content_idea_id']).execute()
        
        return {
            "success": True,
            "session": session,
            "optimized_keywords": keywords_result.data
        }
        
    except Exception as e:
        logger.error("Failed to get optimization session", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@router.get("/content-idea/{content_idea_id}/keywords")
async def get_content_idea_keywords(
    content_idea_id: str,
    user_id: str,
    keyword_type: Optional[str] = None,
    optimized_only: bool = False,
    db = Depends(get_supabase_db)
):
    """
    Get keywords for a specific content idea
    """
    try:
        query = db.table('individual_keywords').select('*').eq('content_idea_id', content_idea_id).eq('user_id', user_id)
        
        if keyword_type:
            query = query.eq('keyword_type', keyword_type)
        
        if optimized_only:
            query = query.eq('is_optimized', True)
        
        result = query.execute()
        
        return {
            "success": True,
            "keywords": result.data,
            "total_count": len(result.data)
        }
        
    except Exception as e:
        logger.error("Failed to get content idea keywords", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get keywords: {str(e)}")

@router.get("/content-idea/{content_idea_id}/enhancement")
async def get_content_idea_enhancement(
    content_idea_id: str,
    user_id: str,
    db = Depends(get_supabase_db)
):
    """
    Get enhanced content idea with keyword metrics
    """
    try:
        # Get content idea
        content_result = db.table('content_ideas').select('*').eq('id', content_idea_id).eq('user_id', user_id).execute()
        
        if not content_result.data:
            raise HTTPException(status_code=404, detail="Content idea not found")
        
        content_idea = content_result.data[0]
        
        # Get keyword metrics
        keywords_result = db.table('individual_keywords').select('*').eq('content_idea_id', content_idea_id).eq('user_id', user_id).execute()
        
        # Get optimization sessions
        sessions_result = db.table('keyword_optimization_sessions').select('*').eq('content_idea_id', content_idea_id).eq('user_id', user_id).order('created_at', desc=True).execute()
        
        return {
            "success": True,
            "content_idea": content_idea,
            "keywords": keywords_result.data,
            "optimization_sessions": sessions_result.data,
            "enhancement_summary": {
                "total_keywords": len(keywords_result.data),
                "optimized_keywords": len([k for k in keywords_result.data if k.get('is_optimized')]),
                "high_priority_keywords": len([k for k in keywords_result.data if k.get('priority_score', 0) >= 80]),
                "affiliate_potential_keywords": len([k for k in keywords_result.data if k.get('affiliate_potential_score', 0) >= 60]),
                "is_enhanced": content_idea.get('is_enhanced', False)
            }
        }
        
    except Exception as e:
        logger.error("Failed to get content idea enhancement", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get enhancement: {str(e)}")

@router.put("/keyword/{keyword_id}/optimize")
async def optimize_single_keyword(
    keyword_id: str,
    user_id: str,
    optimizer: IndividualKeywordOptimizer = Depends(get_individual_keyword_optimizer),
    db = Depends(get_supabase_db)
):
    """
    Optimize a single keyword
    """
    try:
        # Get keyword data
        keyword_result = db.table('individual_keywords').select('*').eq('id', keyword_id).eq('user_id', user_id).execute()
        
        if not keyword_result.data:
            raise HTTPException(status_code=404, detail="Keyword not found")
        
        keyword_data = keyword_result.data[0]
        
        # Optimize the keyword
        optimized_keyword = await optimizer._optimize_single_keyword(
            keyword_data, 
            keyword_data['content_idea_id'], 
            user_id
        )
        
        # Update keyword in database
        update_data = {
            'is_optimized': True,
            'optimized_at': datetime.utcnow().isoformat(),
            'llm_optimized_title': optimized_keyword.get('llm_optimized_title'),
            'llm_optimized_description': optimized_keyword.get('llm_optimized_description'),
            'llm_keyword_variations': optimized_keyword.get('llm_keyword_variations', []),
            'llm_content_angle': optimized_keyword.get('llm_content_angle'),
            'llm_target_audience': optimized_keyword.get('llm_target_audience'),
            'content_suggestions': optimized_keyword.get('content_suggestions', []),
            'heading_suggestions': optimized_keyword.get('heading_suggestions', []),
            'internal_link_suggestions': optimized_keyword.get('internal_link_suggestions', []),
            'related_questions': optimized_keyword.get('related_questions', []),
            'affiliate_potential_score': optimized_keyword.get('affiliate_potential_score', 0),
            'suggested_affiliate_networks': optimized_keyword.get('suggested_affiliate_networks', []),
            'monetization_opportunities': optimized_keyword.get('monetization_opportunities', []),
            'relevance_score': optimized_keyword.get('relevance_score', 0),
            'optimization_score': optimized_keyword.get('optimization_score', 0),
            'priority_score': optimized_keyword.get('priority_score', 0)
        }
        
        db.table('individual_keywords').update(update_data).eq('id', keyword_id).execute()
        
        return {
            "success": True,
            "message": "Keyword optimized successfully",
            "optimized_keyword": optimized_keyword
        }
        
    except Exception as e:
        logger.error("Single keyword optimization failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.delete("/keyword/{keyword_id}")
async def delete_keyword(
    keyword_id: str,
    user_id: str,
    db = Depends(get_supabase_db)
):
    """
    Delete a keyword
    """
    try:
        result = db.table('individual_keywords').delete().eq('id', keyword_id).eq('user_id', user_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Keyword not found")
        
        return {
            "success": True,
            "message": "Keyword deleted successfully"
        }
        
    except Exception as e:
        logger.error("Failed to delete keyword", error=str(e))
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@router.get("/analytics/{content_idea_id}")
async def get_keyword_analytics(
    content_idea_id: str,
    user_id: str,
    db = Depends(get_supabase_db)
):
    """
    Get keyword analytics for a content idea
    """
    try:
        # Get all keywords for the content idea
        keywords_result = db.table('individual_keywords').select('*').eq('content_idea_id', content_idea_id).eq('user_id', user_id).execute()
        
        keywords = keywords_result.data
        
        if not keywords:
            return {
                "success": True,
                "analytics": {
                    "total_keywords": 0,
                    "metrics": {},
                    "insights": []
                }
            }
        
        # Calculate analytics
        total_keywords = len(keywords)
        optimized_keywords = len([k for k in keywords if k.get('is_optimized')])
        high_priority_keywords = len([k for k in keywords if k.get('priority_score', 0) >= 80])
        
        # Calculate average metrics
        avg_search_volume = sum(k.get('search_volume', 0) for k in keywords) / total_keywords
        avg_difficulty = sum(k.get('keyword_difficulty', 0) for k in keywords) / total_keywords
        avg_opportunity_score = sum(k.get('opportunity_score', 0) for k in keywords) / total_keywords
        avg_affiliate_potential = sum(k.get('affiliate_potential_score', 0) for k in keywords) / total_keywords
        
        # Group by keyword type
        keyword_types = {}
        for k in keywords:
            kw_type = k.get('keyword_type', 'primary')
            if kw_type not in keyword_types:
                keyword_types[kw_type] = 0
            keyword_types[kw_type] += 1
        
        # Get top performing keywords
        top_keywords = sorted(keywords, key=lambda k: k.get('priority_score', 0), reverse=True)[:10]
        
        analytics = {
            "total_keywords": total_keywords,
            "optimized_keywords": optimized_keywords,
            "high_priority_keywords": high_priority_keywords,
            "optimization_rate": (optimized_keywords / total_keywords) * 100 if total_keywords > 0 else 0,
            "metrics": {
                "avg_search_volume": round(avg_search_volume, 2),
                "avg_difficulty": round(avg_difficulty, 2),
                "avg_opportunity_score": round(avg_opportunity_score, 2),
                "avg_affiliate_potential": round(avg_affiliate_potential, 2)
            },
            "keyword_types": keyword_types,
            "top_keywords": top_keywords,
            "insights": [
                f"Optimized {optimized_keywords} out of {total_keywords} keywords",
                f"{high_priority_keywords} high-priority keywords identified",
                f"Average opportunity score: {round(avg_opportunity_score, 1)}%",
                f"Average affiliate potential: {round(avg_affiliate_potential, 1)}%"
            ]
        }
        
        return {
            "success": True,
            "analytics": analytics
        }
        
    except Exception as e:
        logger.error("Failed to get keyword analytics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

async def _parse_ahrefs_csv(csv_text: str) -> List[Dict[str, Any]]:
    """
    Parse Ahrefs CSV and extract keyword data
    """
    lines = csv_text.split('\n')
    if len(lines) < 2:
        raise ValueError("CSV file must have at least a header row and one data row")
    
    headers = lines[0].split('\t')
    keywords = []
    
    # Map Ahrefs columns
    column_mapping = {
        'keyword': ['keyword', 'query', 'search term'],
        'search_volume': ['search volume', 'volume'],
        'keyword_difficulty': ['keyword difficulty', 'kd', 'difficulty'],
        'cpc': ['cpc', 'cost per click'],
        'traffic_potential': ['traffic potential', 'traffic'],
        'clicks': ['clicks'],
        'impressions': ['impressions'],
        'ctr': ['ctr', 'click-through rate'],
        'position': ['position', 'avg position']
    }
    
    def find_column_index(target_columns):
        for target in target_columns:
            for i, header in enumerate(headers):
                if target.lower() in header.lower():
                    return i
        return -1
    
    keyword_index = find_column_index(column_mapping['keyword'])
    if keyword_index == -1:
        raise ValueError("Keyword column not found in CSV")
    
    for i, line in enumerate(lines[1:], 1):
        if not line.strip():
            continue
            
        values = line.split('\t')
        if len(values) < len(headers):
            continue
        
        keyword_data = {
            'keyword': values[keyword_index].strip(),
            'search_volume': int(values[find_column_index(column_mapping['search_volume'])] or 0),
            'keyword_difficulty': int(values[find_column_index(column_mapping['keyword_difficulty'])] or 0),
            'cpc': float(values[find_column_index(column_mapping['cpc'])] or 0),
            'traffic_potential': int(values[find_column_index(column_mapping['traffic_potential'])] or 0),
            'clicks': int(values[find_column_index(column_mapping['clicks'])] or 0),
            'impressions': int(values[find_column_index(column_mapping['impressions'])] or 0),
            'ctr': float(values[find_column_index(column_mapping['ctr'])] or 0),
            'position': float(values[find_column_index(column_mapping['position'])] or 0),
            'keyword_type': 'primary',  # Default, will be determined during optimization
            'search_intent': 'informational',  # Default, will be analyzed
            'competition_level': 'medium',  # Default, will be calculated
            'trend_score': 0,  # Will be calculated
            'opportunity_score': 0,  # Will be calculated
            'row_number': i
        }
        
        if keyword_data['keyword']:
            keywords.append(keyword_data)
    
    return keywords

async def _process_keyword_optimization(
    keywords: List[Dict[str, Any]],
    content_idea_id: str,
    user_id: str,
    session_id: str,
    optimizer: IndividualKeywordOptimizer
):
    """
    Background task to process keyword optimization
    """
    try:
        logger.info("Starting background keyword optimization", session_id=session_id)
        
        result = await optimizer.optimize_keywords(
            keywords=keywords,
            content_idea_id=content_idea_id,
            user_id=user_id,
            optimization_session_id=session_id
        )
        
        logger.info("Background keyword optimization completed", session_id=session_id)
        
    except Exception as e:
        logger.error("Background keyword optimization failed", 
                   session_id=session_id, 
                   error=str(e))

