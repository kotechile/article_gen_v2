from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from uuid import UUID

from ..core.supabase_client import get_supabase_client
from ..middleware.auth_middleware import get_current_user

router = APIRouter(prefix="/api/settings", tags=["settings"])
logger = logging.getLogger(__name__)

class ResearchSettings(BaseModel):
    min_volume: int = 50
    max_difficulty: int = 50
    min_cpc: float = 0.5
    strict_mode: bool = True

class SettingsResponse(BaseModel):
    success: bool
    data: Optional[ResearchSettings] = None
    message: Optional[str] = None

@router.get("/research", response_model=SettingsResponse)
async def get_research_settings(user_info: Dict[str, Any] = Depends(get_current_user)):
    """Get research settings for the current user"""
    try:
        supabase = get_supabase_client()
        # For now, we use a global settings row (id=1) 
        # In a multi-tenant app, we would filter by user_id or have a per-user settings table
        response = supabase.table('application_settings').select('research_settings').eq('id', 1).execute()
        
        settings_dict = {}
        if response.data and len(response.data) > 0:
            settings_dict = response.data[0].get('research_settings') or {}
        
        # Merge with defaults
        settings = ResearchSettings(**settings_dict) if settings_dict else ResearchSettings()
        
        return SettingsResponse(
            success=True,
            data=settings
        )
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        return SettingsResponse(
            success=False, 
            message="Failed to retrieve settings",
            data=ResearchSettings()
        )

@router.post("/research", response_model=SettingsResponse)
async def update_research_settings(
    settings: ResearchSettings, 
    user_info: Dict[str, Any] = Depends(get_current_user)
):
    """Update research settings in application_settings table"""
    try:
        supabase = get_supabase_client()
        
        # Update the research_settings field in the global settings row
        update_data = {
            "research_settings": settings.dict()
        }
        
        response = supabase.table('application_settings').update(update_data).eq('id', 1).execute()
        
        if not response.data:
            # If row 1 doesn't exist, try to insert it (though it should exist)
            supabase.table('application_settings').insert({"id": 1, **update_data}).execute()
        
        logger.info(f"Updated research settings: {settings}")
        
        return SettingsResponse(
            success=True,
            data=settings,
            message="Settings saved successfully"
        )
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")
