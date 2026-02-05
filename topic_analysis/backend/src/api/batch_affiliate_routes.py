"""
Batch affiliate research API routes for efficient processing
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
from ..core.supabase_database import get_supabase_db
from ..services.affiliate_research_service import AffiliateResearchService
import asyncio
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class BatchAffiliateRequest(BaseModel):
    main_topic: str
    subtopics: List[str]
    user_id: str = None

class BatchAffiliateResponse(BaseModel):
    main_topic: str
    subtopics: List[str]
    programs: List[Dict[str, Any]]
    success: bool
    message: str
    processing_time: float

@router.post("/batch-affiliate-research", response_model=BatchAffiliateResponse)
async def batch_affiliate_research(
    request: BatchAffiliateRequest,
    db = Depends(get_supabase_db)
):
    """
    Efficiently process affiliate research for main topic + all subtopics in one call
    """
    import time
    start_time = time.time()
    
    try:
        affiliate_service = AffiliateResearchService()
        
        # Combine main topic with subtopics for comprehensive search
        all_topics = [request.main_topic] + request.subtopics[:3]  # Limit to 3 subtopics for performance
        combined_search = f"{request.main_topic} {' '.join(request.subtopics[:3])}"
        
        logger.info("Starting batch affiliate research", 
                   main_topic=request.main_topic, 
                   subtopics_count=len(request.subtopics),
                   combined_search=combined_search)
        
        # Single comprehensive search instead of multiple individual searches
        result = await affiliate_service.search_affiliate_programs(
            search_term=combined_search,
            niche=request.main_topic,
            user_id=request.user_id
        )
        
        processing_time = time.time() - start_time
        
        logger.info("Batch affiliate research completed", 
                   programs_found=len(result.get('programs', [])),
                   processing_time=processing_time)
        
        return BatchAffiliateResponse(
            main_topic=request.main_topic,
            subtopics=request.subtopics,
            programs=result.get('programs', []),
            success=True,
            message=f"Found {len(result.get('programs', []))} affiliate programs in {processing_time:.2f}s",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error("Batch affiliate research failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch research failed: {str(e)}")

@router.post("/optimized-affiliate-research", response_model=Dict[str, Any])
async def optimized_affiliate_research(
    request: Dict[str, Any],
    db = Depends(get_supabase_db)
):
    """
    Optimized affiliate research that processes main topic + subtopics efficiently
    """
    import time
    start_time = time.time()
    
    try:
        affiliate_service = AffiliateResearchService()
        
        # Extract parameters
        main_topic = request.get('main_topic', request.get('search_term', ''))
        subtopics = request.get('subtopics', [])
        user_id = request.get('user_id')
        
        # Create comprehensive search strategy
        if subtopics:
            # Use the most relevant subtopics for better results
            relevant_subtopics = subtopics[:2]  # Use top 2 subtopics
            combined_search = f"{main_topic} {' '.join(relevant_subtopics)}"
        else:
            combined_search = main_topic
        
        logger.info("Optimized affiliate research", 
                   main_topic=main_topic, 
                   subtopics=subtopics,
                   combined_search=combined_search)
        
        # Single comprehensive search
        result = await affiliate_service.search_affiliate_programs(
            search_term=combined_search,
            niche=main_topic,
            user_id=user_id
        )
        
        processing_time = time.time() - start_time
        
        # Add performance metrics
        result['processing_time'] = processing_time
        result['search_strategy'] = 'optimized_batch'
        result['subtopics_processed'] = len(subtopics)
        
        logger.info("Optimized research completed", 
                   programs_found=len(result.get('programs', [])),
                   processing_time=processing_time)
        
        return result
        
    except Exception as e:
        logger.error("Optimized affiliate research failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Optimized research failed: {str(e)}")

