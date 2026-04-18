from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from ..services.research.deep_research_service import DeepResearchService
from ..core.supabase_singleton import get_supabase_client
from ..core.config import get_settings

router = APIRouter(prefix="/api/research", tags=["Deep Research"])
logger = logging.getLogger(__name__)

class DeepResearchRequest(BaseModel):
    title_ids: List[str]
    collection_name: str = "deep_research_collection"
    user_id: str
    
class DeepResearchResponse(BaseModel):
    message: str
    job_ids: Dict[str, str] # title_id -> doc_id (or job_status_id in future)

def get_deep_research_service():
    return DeepResearchService()

@router.post("/deep-gap-fill", response_model=DeepResearchResponse)
async def trigger_deep_gap_fill(
    request: DeepResearchRequest,
    background_tasks: BackgroundTasks,
    service: DeepResearchService = Depends(get_deep_research_service)
):
    """
    Trigger Deep Research for a list of title IDs.
    Running in background to avoid timeouts.
    """
    logger.info(f"Received Deep Gap Fill request for {len(request.title_ids)} titles.")
    
    # We'll return immediately and run in background
    # Realistically, we might want to store a job ID in DB, but for now we'll just fire and forget
    # or rely on the service to update Title status. 
    
    # Since background_tasks.add_task doesn't support async generators easily without wrapper,
    # we'll define a wrapper function.
    
    async def process_titles(titles: List[str], user_id: str, collection: str):
        supabase = get_supabase_client()
        for title_id in titles:
            try:
                # Fetch outline first
                # Assuming title has 'content_outline' field
                response = supabase.table('Titles').select('content_outline, title').eq('id', title_id).execute()
                if not response.data:
                    logger.error(f"Title {title_id} not found.")
                    continue
                
                title_data = response.data[0]
                outline = title_data.get('content_outline') or title_data.get('title') # Fallback to title
                
                # Perform Research
                result = await service.perform_deep_research(
                    title_id=title_id,
                    outline=outline,
                    user_id=user_id,
                    collection_name=collection
                )
                
                if result['success']:
                    dossier = result.get("research_dossier") or {}
                    dossier_quality_score = int(dossier.get("dossier_quality_score", 0) or 0)
                    dossier_status = "ready" if dossier_quality_score >= 30 else "needs_review"
                    update_payload = {
                        'status': 'Research Complete',
                        'research_dossier': dossier,
                        'dossier_status': dossier_status,
                        'dossier_last_updated_at': result.get('research_dossier', {}).get('generated_at'),
                        'dossier_quality_score': dossier_quality_score,
                    }
                    try:
                        supabase.table('Titles').update(update_payload).eq('id', title_id).execute()
                    except Exception as update_error:
                        # Backward-compatible fallback before dossier migration is applied.
                        if 'research_dossier' in str(update_error) or 'dossier_' in str(update_error):
                            supabase.table('Titles').update({'status': 'Research Complete'}).eq('id', title_id).execute()
                        else:
                            raise
                else:
                    supabase.table('Titles').update({'dossier_status': 'failed'}).eq('id', title_id).execute()
                    
            except Exception as e:
                logger.error(f"Error processing title {title_id}: {e}")

    background_tasks.add_task(process_titles, request.title_ids, request.user_id, request.collection_name)
    
    return DeepResearchResponse(
        message=f"Deep research started for {len(request.title_ids)} titles.",
        job_ids={} # Async process
    )
