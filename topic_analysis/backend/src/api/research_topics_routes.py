"""
Research Topics API routes for dataflow persistence
This module provides REST API endpoints for managing research topics and their related data
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Request, Body
from fastapi.responses import JSONResponse

from ..models.research_topic import (
    ResearchTopicCreate, ResearchTopicUpdate, ResearchTopicResponse,
    ResearchTopicListResponse, ResearchTopicComplete, ResearchTopicStats,
    ResearchTopicStatus
)
from ..models.topic_decomposition import TopicDecompositionCreate, TopicDecompositionResponse
from ..services.research_topic_service import ResearchTopicService
from ..services.topic_decomposition_service import TopicDecompositionService
from ..services.trend_analysis_service import TrendAnalysisService

from ..services.content_idea_service import ContentIdeaService
from ..services.subtopics_service import SubtopicsService
from ..models.subtopic import SubtopicCreate, SubtopicUpdate, SubtopicResponse, SubtopicListResponse
from ..middleware.auth_middleware import get_current_user
from ..services.enhanced_affiliate_research_service import EnhancedAffiliateResearchService
from ..services.trend_service import TrendService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/research-topics", tags=["research-topics"])

# Initialize services
research_topic_service = ResearchTopicService()
topic_decomposition_service = TopicDecompositionService()
trend_analysis_service = TrendAnalysisService()

content_idea_service = ContentIdeaService()
subtopics_service = SubtopicsService()
enhanced_affiliate_service = EnhancedAffiliateResearchService()
trend_service = TrendService()

# Lazy loading for enhanced service to avoid circular imports or heavy initialization
_enhanced_decomposition_service = None

def get_enhanced_decomposition_service():
    global _enhanced_decomposition_service
    if _enhanced_decomposition_service is None:
        from ..services.enhanced_topic_decomposition_service import EnhancedTopicDecompositionService
        _enhanced_decomposition_service = EnhancedTopicDecompositionService()
    return _enhanced_decomposition_service

async def get_current_user_id(request: Request) -> UUID:
    """Get current user ID from authentication context"""
    try:
        user_info = await get_current_user(request)
        logger.info(f"DEBUG: user_id string is '{user_info.get('user_id')}'")
        return UUID(user_info["user_id"])
    except Exception as e:
        logger.error(f"Error getting current user ID: {e}")
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )

# Common dependency for all routes
async def get_user_id(request: Request) -> UUID:
    """Get user ID from request"""
    return await get_current_user_id(request)

@router.post("/", response_model=ResearchTopicResponse, status_code=201)
async def create_research_topic(
    topic_data: ResearchTopicCreate,
    user_id: UUID = Depends(get_user_id)
):
    """Create a new research topic"""
    try:
        topic = await research_topic_service.create(topic_data, user_id)
        if not topic:
            raise HTTPException(status_code=500, detail="Failed to create research topic")
        
        return topic
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating research topic: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{topic_id}", response_model=ResearchTopicResponse)
async def get_research_topic(
    topic_id: UUID = Path(..., description="Research topic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """Get a research topic by ID"""
    try:
        topic = await research_topic_service.get_by_id(topic_id, user_id)
        if not topic:
            raise HTTPException(status_code=404, detail="Research topic not found")
        
        return topic
        
    except Exception as e:
        logger.error(f"Error getting research topic: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/", response_model=ResearchTopicListResponse)
async def list_research_topics(
    status: Optional[ResearchTopicStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    order_by: str = Query("created_at", description="Field to sort by"),
    order_direction: str = Query("desc", description="Sort direction (asc/desc)"),
    user_id: UUID = Depends(get_user_id)
):
    """List research topics with pagination and filtering"""
    try:
        logger.info(f"Listing topics for user_id: {user_id}")
        topics = await research_topic_service.get_all(
            user_id=user_id,
            status=status,
            page=page,
            size=size,
            order_by=order_by,
            order_direction=order_direction
        )
        logger.info(f"Retrieved {len(topics.items)} topics from service, serializing response...")
        return topics
        
    except Exception as e:
        logger.error(f"Error listing research topics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/{topic_id}", response_model=ResearchTopicResponse)
async def update_research_topic(
    topic_id: UUID = Path(..., description="Research topic ID"),
    topic_data: ResearchTopicUpdate = ...,
    user_id: UUID = Depends(get_user_id)
):
    """Update a research topic"""
    try:
        topic = await research_topic_service.update(topic_id, topic_data, user_id)
        if not topic:
            raise HTTPException(status_code=404, detail="Research topic not found")
        
        return topic
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating research topic: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{topic_id}", status_code=204)
async def delete_research_topic(
    topic_id: UUID = Path(..., description="Research topic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """Delete a research topic"""
    try:
        success = await research_topic_service.delete(topic_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Research topic not found")
        
        return JSONResponse(content=None, status_code=204)
        
    except Exception as e:
        logger.error(f"Error deleting research topic: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



@router.get("/{topic_id}/stats", response_model=ResearchTopicStats)
async def get_research_topic_stats(
    topic_id: UUID = Path(..., description="Research topic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """Get statistics for a research topic"""
    try:
        # Verify topic exists
        topic = await research_topic_service.get_by_id(topic_id, user_id)
        if not topic:
            raise HTTPException(status_code=404, detail="Research topic not found")
        
        stats = await research_topic_service.get_stats(user_id)
        return stats
        
    except Exception as e:
        logger.error(f"Error getting research topic stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{topic_id}/subtopics", response_model=SubtopicResponse, status_code=201)
async def create_subtopic(
    subtopic_data: SubtopicCreate,
    topic_id: UUID = Path(..., description="Research topic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """Create a new subtopic"""
    try:
        result = await subtopics_service.create(
            research_topic_id=topic_id,
            name=subtopic_data.name,
            user_id=user_id,
            trend_data=subtopic_data.dict(exclude={"name"})
        )
        return result
        
    except Exception as e:
        logger.error(f"Error creating subtopic: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{topic_id}/subtopics", response_model=SubtopicListResponse)
async def get_topic_subtopics(
    topic_id: UUID = Path(..., description="Research topic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """Get enriched subtopics from the new normalized table, with fallback to legacy JSON storage"""
    try:
        # Try retrieving from new normalized table first
        try:
            subtopics = await subtopics_service.get_by_research_topic(topic_id, user_id)
            if subtopics:
                # Map project_id to research_topic_id for Pydantic model compatibility
                # And ensure other required fields (user_id, updated_at) are present
                for sub in subtopics:
                    if "project_id" in sub and "research_topic_id" not in sub:
                         sub["research_topic_id"] = sub["project_id"]
                    
                    if "user_id" not in sub:
                         sub["user_id"] = str(user_id)
                         
                    if "updated_at" not in sub:
                        sub["updated_at"] = sub.get("created_at", datetime.utcnow().isoformat())
            
            return {
                "items": subtopics or [],
                "total": len(subtopics) if subtopics else 0
            }
        except Exception as e:
            logger.error(f"Failed to get subtopics: {e}")
            return {
                "items": [],
                "total": 0
            }
        
    except Exception as e:
        logger.error(f"Error getting topic subtopics (all methods failed): {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{topic_id}/subtopics/generate", response_model=SubtopicListResponse)
async def generate_subtopics(
    topic_id: UUID = Path(..., description="Research topic ID"),
    payload: Any = Body(None),
    user_id: UUID = Depends(get_user_id)
):
    """Generate subtopics using LLM + Search and save to normalized table"""
    try:
        # 1. Get the research topic
        topic = await research_topic_service.get_by_id(topic_id, user_id)
        if not topic:
            raise HTTPException(status_code=404, detail="Research topic not found")
            
        logger.info(f"Generating subtopics for: {topic.title} (User: {user_id})")

        # --- FEEDBACK MECHANISM: Create Status Subtopic ---
        status_subtopic_name = " ⏳ Generating Topic Ideas (Please Wait)..."
        status_sub = await subtopics_service.create(
            research_topic_id=topic_id,
            name=status_subtopic_name,
            user_id=user_id,
            trend_data={"trend_direction": "stable"} # Dummy data
        )
        status_id = status_sub["id"] if status_sub else None
        
        # 2. Call EnhancedDecompositionService
        service = get_enhanced_decomposition_service()
        decomposition_result = await service.decompose_topic_enhanced(
            query=topic.title,
            user_id=str(user_id),
            max_subtopics=8,  # Generate a good amount
            use_autocomplete=False, # Disable autocomplete to ensure pure LLM subtopics
            use_llm=True
        )
        
        if not decomposition_result["success"]:
            raise HTTPException(status_code=500, detail=f"Failed to generate subtopics: {decomposition_result['message']}")
            
        # 3. Save to normalized table
        generated_subtopics = decomposition_result["subtopics"]
        saved_subtopics = []
        
        # Collect seed keywords for mapping later
        seed_to_subtopic_map = {} # seed_keyword -> subtopic_id
        all_seed_keywords = []
        keywords_batch = []

        async def process_subtopic(item):
            # Map enhanced subtopic to creation schema
            sub_name = item["title"]
            
            # DEBUG: Log what metrics we are receiving from the service
            logger.info(f"DEBUG Subtopic Item '{sub_name}': Vol={item.get('search_volume')}, CPC={item.get('cpc')}, KD={item.get('keyword_difficulty')}")
            
            # Initialize trend data structure with explicit None values
            # Initialize trend data structure using Pre-Verified Data if available
            trend_data = {
                "trend_direction": None, 
                "trend_score": None,     
                "search_volume": int(item.get("search_volume") or 0),   # Explicit cast, handle None
                "cpc": float(item.get("cpc") or 0.0),                   # Explicit cast, handle None
                "seo_difficulty": int(item.get("keyword_difficulty") or 50), # Use max KD
                "keywords": item.get("seed_keywords", []),
                "rationale": item.get("rationale"),
                "target_audience": item.get("target_audience"),
                "trend_analysis": item.get("trend_analysis"),   # Rich Trend Data
                "monetization": item.get("monetization_data"),  # Rich Offers Data
                "interest_over_time": (item.get("trend_analysis") or {}).get("historical_data", [])
            }
            
            # Create the record immediately WITHOUT fetching trends
            new_subtopic = await subtopics_service.create(
                research_topic_id=topic_id,
                name=sub_name,
                user_id=user_id,
                trend_data=trend_data
            )
            
            if new_subtopic:
                if "project_id" in new_subtopic and "research_topic_id" not in new_subtopic:
                    new_subtopic["research_topic_id"] = new_subtopic["project_id"]
                
                # Retrieve ID for mapping
                sub_id_str = new_subtopic.get("id")
                
                if sub_id_str and "seed_keywords" in item:
                     # Check if seed_keywords contains objects (our new format) or strings
                     seeds = item.get("seed_keywords", [])
                     if seeds and isinstance(seeds[0], dict):
                         # Pre-verified data with metrics!
                         for kw_obj in seeds:
                             kw_dict = {
                                "keyword": kw_obj.get("keyword"),
                                "search_volume": kw_obj.get("search_volume", 0),
                                "cpc": kw_obj.get("cpc", 0.0),
                                "keyword_difficulty": kw_obj.get("keyword_difficulty", 0),
                                "seed_keyword": kw_obj.get("seed", new_subtopic.get("name")), # Fix 23502 Violation
                                "research_topic_id": str(topic_id),
                                "subtopic_id": sub_id_str,
                                "user_id": str(user_id),
                                "source": "cluster_decomposition",
                                "status": "active",
                                "profitability_score": kw_obj.get("profitability_score"),
                                "main_intent": kw_obj.get("main_intent")
                             }
                             keywords_batch.append(kw_dict)
                     else:
                         # Fallback for legacy string lists (shouldn't happen with new Service)
                         for seed in seeds:
                             if isinstance(seed, str):
                                 all_seed_keywords.append(seed)
                                 seed_to_subtopic_map[seed.strip().lower()] = sub_id_str
            
            return new_subtopic

        # Execute all subtopic processing in parallel
        results = await asyncio.gather(*(process_subtopic(item) for item in generated_subtopics))
        
        # Filter out None results (failed creations)
        saved_subtopics = [r for r in results if r is not None]
        
        # Save Keywords Batch (New Efficient Method)
        if keywords_batch:
            try:
                from ..services.keywords_service import KeywordsService
                keywords_persistence = KeywordsService()
                await keywords_persistence.create_batch(keywords_batch)
                logger.info(f"Persisted {len(keywords_batch)} detailed keywords to database directly.")
            except Exception as e:
                logger.error(f"Failed to persist batch keywords: {e}")

        test_count = len(saved_subtopics)
        logger.info(f"Saved {test_count} generated subtopics to table")
        
        # DEBUG: Validate manually to catch 500 errors
        try:
            # CLEANUP: Remove status message
            if 'status_id' in locals() and status_id:
                try:
                    await subtopics_service.delete(status_id, user_id)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup status subtopic: {cleanup_error}")

            # We must import this if not already imported, but it is imported as SubtopicListResponse
            # Re-validate to see error
            from src.models.subtopic import SubtopicListResponse
            return SubtopicListResponse(items=saved_subtopics, total=len(saved_subtopics))
        except Exception as val_error:
            logger.error(f"CRITICAL: Pydantic Validation Error: {val_error}")
            # Identify the first item to see what's wrong
            if saved_subtopics:
                logger.error(f"First item keys: {saved_subtopics[0].keys()}")
            raise HTTPException(status_code=500, detail=f"Data validation error: {str(val_error)}")

    except ValueError as e:
        if 'status_id' in locals() and status_id:
            try: await subtopics_service.delete(status_id, user_id)
            except: pass
            
        # Catch strict quality policy errors
        error_msg = str(e)
        logger.warning(f"Generation failed due to quality policy: {error_msg}")
        status_code = 503 if "service" in error_msg.lower() or "unavailable" in error_msg.lower() else 422
        raise HTTPException(status_code=status_code, detail=error_msg)
    except HTTPException as e:
        if 'status_id' in locals() and status_id:
            try: await subtopics_service.delete(status_id, user_id)
            except: pass
        raise e
    except Exception as e:
        import traceback
        error_msg = str(e)
        logger.error(f"Generate subtopics failed: {error_msg}")
        logger.error(traceback.format_exc())
        
        if 'status_id' in locals() and status_id:
            try: await subtopics_service.delete(status_id, user_id)
            except: pass
        
        # In strict quality mode, any failure to generate is effectively a service failure
        raise HTTPException(status_code=500, detail=f"Service unavailable: {error_msg}")

@router.post("/{topic_id}/enrich", response_model=SubtopicListResponse)
async def enrich_research_topic(
    topic_id: UUID = Path(..., description="Research topic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """
    Enrich subtopics with real-world data:
    1. Trend Analysis (Volume, Growth)
    2. Affiliate Opportunities (Offer Count, EPC)
    """
    try:
        # 1. Get all subtopics for this topic
        subtopics = await subtopics_service.get_by_research_topic(topic_id, user_id)
        if not subtopics:
            raise HTTPException(status_code=404, detail="No subtopics found to enrich")
            
        enriched_count = 0
        
        # Parallelize the enrichment process
        from .enrichment_helper import _process_single_subtopic_enrichment
        
        enrichment_tasks = []
        for sub in subtopics:
            sub_id = sub["id"]
            sub_name = sub["name"]
            enrichment_tasks.append(_process_single_subtopic_enrichment(sub_id, sub_name, sub, user_id))
        
        # Execute in parallel
        # return_exceptions=True so one failure doesn't stop others
        results = await asyncio.gather(*enrichment_tasks, return_exceptions=True)
        
        for res in results:
            if res is True:
                enriched_count += 1

        # Return updated list
        updated_subtopics = await subtopics_service.get_by_research_topic(topic_id, user_id)
        
        # Ensure compatibility fields are present (similar to get_topic_subtopics)
        for sub in updated_subtopics:
            if "project_id" in sub and "research_topic_id" not in sub:
                sub["research_topic_id"] = sub["project_id"]
            if "user_id" not in sub:
                 sub["user_id"] = str(user_id)
            if "updated_at" not in sub:
                sub["updated_at"] = sub.get("created_at", datetime.utcnow().isoformat())

        return {
            "items": updated_subtopics,
            "total": len(updated_subtopics)
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logger.error(f"Error enriching subtopics: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/{topic_id}/subtopics/{subtopic_id}/enrich", response_model=SubtopicResponse)
async def enrich_single_subtopic(
    topic_id: UUID = Path(..., description="Research topic ID"),
    subtopic_id: UUID = Path(..., description="Subtopic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """
    Enrich a specific subtopic with real-world data (DataForSEO)
    """
    try:
        # 1. Get the subtopic
        sub = await subtopics_service.get_by_id(subtopic_id, user_id)
        if not sub:
            raise HTTPException(status_code=404, detail="Subtopic not found")
            
        sub_name = sub["name"]
        
        # Use the shared helper which enforces proper timeouts (60s) and comprehensive scope
        from .enrichment_helper import _process_single_subtopic_enrichment
        
        logger.info(f"Enriching subtopic {sub_name} ({subtopic_id})...")
        success = await _process_single_subtopic_enrichment(str(subtopic_id), sub_name, sub, user_id)
        
        if not success:
             logger.warning(f"Enrichment helper returned False for {sub_name}")
             # We don't raise error, just return current state, or maybe 500?
             # User reported "No data" so let's continue to return whatever we have
        
        # Reload to get the persisted data
        updated_sub = await subtopics_service.get_by_id(subtopic_id, user_id)
        if not updated_sub:
             raise HTTPException(status_code=500, detail="Failed to retrieve updated subtopic")

        # Map compatibility fields
        if "project_id" in updated_sub and "research_topic_id" not in updated_sub:
            updated_sub["research_topic_id"] = updated_sub["project_id"]
        
        return updated_sub

    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        error_msg = f"Error enriching subtopic {subtopic_id}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        with open("backend_error.log", "w") as f:
            f.write(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{topic_id}/subtopics/{subtopic_id}", response_model=SubtopicResponse)
async def update_subtopic(
    subtopic_data: SubtopicUpdate,
    topic_id: UUID = Path(..., description="Research topic ID"),
    subtopic_id: UUID = Path(..., description="Subtopic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """Update a subtopic"""
    try:
        result = await subtopics_service.update(
            subtopic_id=subtopic_id,
            update_data=subtopic_data.dict(exclude_unset=True),
            user_id=user_id
        )
        if not result:
            raise HTTPException(status_code=404, detail="Subtopic not found")
        return result
        
    except Exception as e:
        logger.error(f"Error updating subtopic: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{topic_id}/subtopics/{subtopic_id}", status_code=204)
async def delete_subtopic(
    topic_id: UUID = Path(..., description="Research topic ID"),
    subtopic_id: UUID = Path(..., description="Subtopic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """Delete a subtopic"""
    try:
        success = await subtopics_service.delete(subtopic_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Subtopic not found")
        return JSONResponse(status_code=204, content=None)
        
    except Exception as e:
        logger.error(f"Error deleting subtopic: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats/overview", response_model=ResearchTopicStats)
async def get_overview_stats(
    user_id: UUID = Depends(get_user_id)
):
    """Get overview statistics for all research topics"""
    try:
        stats = await research_topic_service.get_stats(user_id)
        return stats
        
    except Exception as e:
        logger.error(f"Error getting overview stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Phase 3 Routes: Keyword & Content Topic Management ---

@router.get("/{topic_id}/keywords", response_model=List[Dict[str, Any]])
async def get_topic_keywords(
    topic_id: UUID = Path(..., description="Research topic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """Get all keywords for a research topic (across all subtopics)"""
    try:
        from ..services.keywords_service import KeywordsService
        keywords_service = KeywordsService()
        
        keywords = await keywords_service.get_by_topic(topic_id, user_id)
        return keywords
        
    except Exception as e:
        logger.error(f"Error getting topic keywords: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{topic_id}/subtopics/{subtopic_id}/keywords", response_model=List[Dict[str, Any]])
async def get_subtopic_keywords(
    topic_id: UUID = Path(..., description="Research topic ID"),
    subtopic_id: UUID = Path(..., description="Subtopic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """Get expanded keywords for a subtopic"""
    try:
        from ..services.keywords_service import KeywordsService
        keywords_service = KeywordsService()
        
        # Logging for debug
        logger.info(f"Fetching keywords for subtopic {subtopic_id}")
        
        keywords = await keywords_service.get_by_subtopic(subtopic_id, user_id)
        return keywords
        
    except Exception as e:
        import traceback
        error_msg = f"Error getting keywords: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        with open("backend_error.log", "a") as f:
            f.write(error_msg)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{topic_id}/subtopics/{subtopic_id}/keywords/expand", response_model=Dict[str, Any])
async def expand_subtopic_keywords(
    topic_id: UUID = Path(..., description="Research topic ID"),
    subtopic_id: UUID = Path(..., description="Subtopic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """Manually trigger keyword expansion for a subtopic"""
    try:
        from ..services.keyword_expansion_service import KeywordExpansionService
        from ..services.keywords_service import KeywordsService
        
        # 1. Get subtopic to find seed keywords
        subtopic = await subtopics_service.get_by_id(subtopic_id, user_id)
        if not subtopic:
            error_msg = f"Subtopic not found in Expand. ID: {subtopic_id}, User: {user_id}"
            logger.error(error_msg)
            with open("backend_error.log", "a") as f:
                f.write(error_msg + "\n")
            raise HTTPException(status_code=404, detail="Subtopic not found")
        
        seed_keywords = subtopic.get("keywords", [])
        if not seed_keywords:
            # Fallback to name if no keywords
            seed_keywords = [subtopic["name"]]
            
        # 2. Expand
        expansion_service = KeywordExpansionService()
        expanded_models = await expansion_service.expand_seed_keywords(seed_keywords)
        
        logger.info(f"Expanded {len(expanded_models)} raw keywords for {subtopic['name']}")
        
        # 3. Filter
        profitable_kws = expansion_service.apply_profitability_filter(expanded_models)
        
        logger.info(f"Filtered down to {len(profitable_kws)} profitable keywords")

        # 4. Save
        keywords_service = KeywordsService()
        saved_count = 0
        
        if profitable_kws:
            batch = []
            for kw in profitable_kws:
                kw_dict = kw.dict()
                kw_dict["research_topic_id"] = str(topic_id)
                kw_dict["subtopic_id"] = str(subtopic_id)
                kw_dict["user_id"] = str(user_id)
                batch.append(kw_dict)
            
            await keywords_service.create_batch(batch)
            saved_count = len(batch)
            
        # 5. Update Subtopic Metrics (Volume, CPC, KD)
        # We look for the keyword that matches the subtopic name (or the seed itself)
        # to populate the main row metrics.
        subtopic_name_lower = subtopic["name"].lower()
        matching_metrics = None
        
        # Priority 1: Exact match with Subtopic Name
        if expanded_models:
            for kw in expanded_models:
                if kw.keyword.lower().strip() == subtopic_name_lower:
                    matching_metrics = kw
                    break
            
            # Priority 2: Match with ANY of the seed keywords
            if not matching_metrics:
                for kw in expanded_models:
                     if kw.keyword.lower().strip() in [s.lower().strip() for s in seed_keywords]:
                         matching_metrics = kw
                         break
            
            # Priority 3: Just take the one with highest volume (proxy)
            if not matching_metrics:
                 # Sort by volume desc
                 sorted_by_vol = sorted(expanded_models, key=lambda k: k.search_volume or 0, reverse=True)
                 if sorted_by_vol:
                     matching_metrics = sorted_by_vol[0]
        
        if matching_metrics:
            logger.info(f"Updating subtopic {subtopic['name']} metrics from keyword '{matching_metrics.keyword}'")
            update_data = {
                "search_volume": matching_metrics.search_volume,
                "cpc": matching_metrics.cpc,
                "seo_difficulty": matching_metrics.difficulty,
                # Update viability score if not already set or boost it
                "viability_score": matching_metrics.profitability_score
            }
            await subtopics_service.update(subtopic_id, update_data, user_id)
            
        return {
            "success": True, 
            "keywords_found": len(profitable_kws),
            "keywords_saved": saved_count
        }

    except Exception as e:
        import traceback
        error_msg = f"Error in expand_subtopic_keywords: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        with open("backend_error.log", "w") as f:
            f.write(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{topic_id}/keywords/expand_all", response_model=Dict[str, Any])
async def expand_all_topic_keywords(
    topic_id: UUID = Path(..., description="Research topic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """
    Expand keywords for ALL subtopics in a research topic in one batch.
    This optimizes DataForSEO API usage by aggregating seed keywords.
    """
    try:
        from ..services.keyword_expansion_service import KeywordExpansionService
        from ..services.keywords_service import KeywordsService
        
        # 1. Fetch all subtopics
        logging.info("Step 1: Fetching subtopics")
        with open("backend_error.log", "a") as f:
            f.write(f"Step 1: Fetching subtopics for topic {topic_id}\n")
            
        subtopics = await subtopics_service.get_by_research_topic(topic_id, user_id)
        if not subtopics:
             raise HTTPException(status_code=404, detail="No subtopics found for this topic")
             
        # 2. Aggregate seeds and map them to subtopics
        # Map: seed_keyword -> Set[subtopic_id]
        with open("backend_error.log", "a") as f:
            f.write(f"Step 2: Aggregating seeds from {len(subtopics)} subtopics\n")
            
        seed_map = {}
        all_unique_seeds = set()
        
        for sub in subtopics:
            seeds = sub.get("keywords", [])
            # If no keywords, use name as fallback
            if not seeds:
                seeds = [sub["name"]]
                
            sub_id = sub["id"]
            for seed in seeds:
                seed_str = ""
                if isinstance(seed, dict):
                    # Handle robust keyword object
                    seed_str = seed.get("keyword") or seed.get("seed_keyword") or ""
                elif isinstance(seed, str):
                    seed_str = seed
                
                if seed_str:
                    normalized_seed = seed_str.strip().lower()
                    if normalized_seed:
                        if normalized_seed not in seed_map:
                            seed_map[normalized_seed] = set()
                        seed_map[normalized_seed].add(sub_id)
                        all_unique_seeds.add(normalized_seed)
        
        if not all_unique_seeds:
             return {"success": True, "keywords_found": 0, "keywords_saved": 0, "message": "No seed keywords found"}
             
        # 3. Batch Expand
        with open("backend_error.log", "a") as f:
            f.write(f"Step 3: Expanding {len(all_unique_seeds)} unique seeds\n")
            
        expansion_service = KeywordExpansionService()
        # Convert set to list
        expanded_models = await expansion_service.expand_seed_keywords(list(all_unique_seeds))
        
        with open("backend_error.log", "a") as f:
            f.write(f"Step 4: Got {len(expanded_models)} expanded models. Filtering...\n")
            
        # 4. Filter
        profitable_kws = expansion_service.apply_profitability_filter(expanded_models)
        
        with open("backend_error.log", "a") as f:
            f.write(f"Step 5: {len(profitable_kws)} profitable keywords found. Saving...\n")
            
        # 5. Distribute & Save
        keywords_service = KeywordsService()
        batch_to_save = []
        
        for kw in profitable_kws:
            # DataForSEO returns the seed that generated this idea
            # We map it back to the subtopics that requested it
            generated_from = kw.seed_keyword.strip().lower() if kw.seed_keyword else ""
            
            target_subtopics = seed_map.get(generated_from, set())
            
            # If we can't map it back, maybe we should skip or assign to topic?
            # Assigning to topic without subtopic is safer fallback
            if not target_subtopics:
                # Fallback: create one generic record
                kw_dict = kw.dict()
                kw_dict["research_topic_id"] = str(topic_id)
                kw_dict["subtopic_id"] = None
                kw_dict["user_id"] = str(user_id)
                batch_to_save.append(kw_dict)
            else:
                # Create a record for EACH subtopic that has this seed
                for sub_id in target_subtopics:
                    kw_dict = kw.dict()
                    kw_dict["research_topic_id"] = str(topic_id)
                    kw_dict["subtopic_id"] = str(sub_id)
                    kw_dict["user_id"] = str(user_id)
                    batch_to_save.append(kw_dict)
        
        saved_count = 0
        if batch_to_save:
            # We might have duplicates if multiple subtopics target same seed AND yield same result?
            # KeywordsService.create_batch usually handles basic insertion, but ideally we dedup?
            # But here they differ by subtopic_id, so they are unique records.
            # If a keyword "best coffee" comes from seed "coffee", and we have 2 subtopics "coffee types" and "coffee brewing"
            # both using "coffee" seed... we probably want "best coffee" in both panels.
            with open("backend_error.log", "a") as f:
                f.write(f"Step 6: Saving {len(batch_to_save)} records to DB\n")
                
            await keywords_service.create_batch(batch_to_save)
            saved_count = len(batch_to_save)

        # 6. Update Subtopic Metrics (Volume, CPC, KD)
        # Iterate over all subtopics and find their best match in the expanded models
        updated_subtopics_count = 0
        for sub in subtopics:
            subtopic_id = sub["id"]
            subtopic_name_lower = sub["name"].lower()
            seed_keywords = sub.get("keywords", [])
            matching_metrics = None
            
            # Priority 1: Exact match with Subtopic Name
            if expanded_models:
                for kw in expanded_models:
                     if kw.keyword.lower().strip() == subtopic_name_lower:
                        matching_metrics = kw
                        break
            
            # Prepare safe string list of seeds
            relevant_seeds_strs = []
            for s in seed_keywords:
                if isinstance(s, str):
                    relevant_seeds_strs.append(s.strip().lower())
                elif isinstance(s, dict):
                    val = s.get("keyword") or s.get("seed_keyword")
                    if val:
                        relevant_seeds_strs.append(val.strip().lower())

            # Priority 2: Match with ANY of the seed keywords
            if not matching_metrics and relevant_seeds_strs:
                for kw in expanded_models:
                     if kw.keyword.lower().strip() in relevant_seeds_strs:
                         matching_metrics = kw
                         break
            
            # Priority 3: Assign highest volume keyword found for this subtopic (via seed_map)
            if not matching_metrics and relevant_seeds_strs:
                relevant_models = []
                for kw in expanded_models:
                    # Check if this keyword came from a seed relevant to this subtopic
                    if kw.seed_keyword and kw.seed_keyword.lower().strip() in relevant_seeds_strs:
                        relevant_models.append(kw)
                
                # Sort by volume desc
                sorted_relevant = sorted(relevant_models, key=lambda k: k.search_volume or 0, reverse=True)
                if sorted_relevant:
                    matching_metrics = sorted_relevant[0]

            if matching_metrics:
                try:
                    update_data = {
                        "search_volume": matching_metrics.search_volume,
                        "cpc": matching_metrics.cpc,
                        "seo_difficulty": matching_metrics.difficulty,
                        "viability_score": matching_metrics.profitability_score
                    }
                    await subtopics_service.update(subtopic_id, update_data, user_id)
                    updated_subtopics_count += 1
                except Exception as update_error:
                    logger.warning(f"Failed to update subtopic {sub['name']} metrics: {update_error}")
            
        with open("backend_error.log", "a") as f:
            f.write(f"Step 7: Updated metrics for {updated_subtopics_count} subtopics. Complete.\n")
            
        return {
            "success": True,
            "keywords_found": len(profitable_kws),
            "keywords_saved": saved_count,
            "subtopics_processed": len(subtopics),
            "subtopics_updated": updated_subtopics_count
        }

    except Exception as e:
        import traceback
        error_msg = f"Error in expand_all_topic_keywords: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        with open("backend_error.log", "w") as f:
            f.write(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{topic_id}/keywords/cluster", response_model=Dict[str, Any])
async def cluster_topic_keywords(
    topic_id: UUID = Path(..., description="Research topic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """Cluster all keywords in a topic and generate Content Topics"""
    try:
        from ..services.keyword_expansion_service import KeywordExpansionService
        from ..services.keywords_service import KeywordsService
        from ..services.content_topics_service import ContentTopicsService
        from ..models.keyword import KeywordBase
        
        expansion_service = KeywordExpansionService()
        keywords_service = KeywordsService()
        content_topics_service = ContentTopicsService()
        
        # 1. Fetch all keywords for topic
        raw_keywords = await keywords_service.get_by_topic(topic_id, user_id)
        if not raw_keywords:
             raise HTTPException(status_code=404, detail="No keywords found for this topic to cluster")
             
        # Convert DB dicts back to Pydantic models for service logic
        # Note: KeywordBase doesn't have ID, but we need it? 
        # Actually generate_content_topics doesn't strictly need IDs if we just want to create new topics
        # But for linking supporting keywords, we might want IDs.
        # The service `generate_content_topics` implementation I wrote:
        # "supporting_indices = [] # We don't have IDs yet..."
        # It takes `clustered_keywords: Dict[str, List[KeywordBase]]`.
        
        keyword_models = []
        for k in raw_keywords:
            try:
                # We need to adapt DB fields to Model fields if there are differences
                # DB might store snake_case, Model expects same.
                # Just unpack.
                # Remove fields not in KeywordBase if any extra
                valid_fields = k.copy()
                # Filter valid fields for KeywordBase? Or let Pydantic ignore extras? Pydantic ignores by default usually.
                model = KeywordBase(**valid_fields)
                # We monkey-patch the ID or store it separately if needed?
                # The service uses `kw.keyword` string mostly.
                keyword_models.append(model)
            except Exception as parse_e:
                logger.warning(f"Skipping invalid keyword record: {parse_e}")
                continue
                
        # 2. Cluster
        clusters = expansion_service.cluster_keywords_by_intent(keyword_models)
        
        # 3. Generate Topics
        content_topic_models = expansion_service.generate_content_topics(
            clustered_keywords=clusters,
            user_id=user_id,
            research_topic_id=topic_id
        )
        
        # 4. Save Content Topics
        saved_topics = []
        if content_topic_models:
            batch = []
            for topic in content_topic_models:
                t_dict = topic.dict()
                t_dict["user_id"] = str(user_id)
                t_dict["research_topic_id"] = str(topic_id)
                batch.append(t_dict)
                
            saved_topics = await content_topics_service.create_batch(batch)
            
        return {
            "success": True,
            "clusters_identified": len(clusters),
            "content_topics_generated": len(saved_topics)
        }
        
    except Exception as e:
        logger.error(f"Error clustering keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{topic_id}/content-topics", response_model=List[Dict[str, Any]])
async def get_content_topics(
    topic_id: UUID = Path(..., description="Research topic ID"),
    user_id: UUID = Depends(get_user_id)
):
    """Get generated content topics for a research topic"""
    try:
        from ..services.content_topics_service import ContentTopicsService
        content_topics_service = ContentTopicsService()
        
        topics = await content_topics_service.get_by_topic(topic_id, user_id)
        return topics
        
    except Exception as e:
        logger.error(f"Error getting content topics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
