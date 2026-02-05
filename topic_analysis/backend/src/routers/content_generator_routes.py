from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import os

# Import migrated services
from ..services.content_generator.article_structure_generator import ArticleStructureGenerator
from ..services.content_generator.content_generator import ContentGenerator
from ..services.content_generator.llm_client import create_llm_client

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/content-generator", tags=["Content Generator"])

# Request models
class StructureRequest(BaseModel):
    topic: str
    keywords: Optional[str] = None
    tone: Optional[str] = "journalistic"
    target_word_count: Optional[int] = 2000
    brief: Optional[str] = None
    draft_title: Optional[str] = None

class SectionRequest(BaseModel):
    section_outline: Dict[str, Any]
    research_data: Dict[str, Any]
    claims: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    previous_sections: Optional[List[Dict[str, Any]]] = None

# Dependency to get LLM client
def get_llm_client():
    # Use environment variables for configuration
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4-turbo")
    
    api_key_env = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(api_key_env)
    
    if not api_key:
        logger.warning(f"No API key found for provider {provider} (checked {api_key_env}). using mock/fallback or failing.")
        # passing empty string might cause failure in client if not handled
    
    return create_llm_client(
        provider=provider,
        model=model,
        api_key=api_key or "dummy_key"
    )

@router.get("/health")
async def health_check():
    """Check Content Generator health."""
    return {"status": "healthy", "service": "content-generator"}

@router.post("/structure")
async def generate_structure(request: StructureRequest):
    """Generate article structure based on topic and parameters."""
    try:
        llm_client = get_llm_client()
        structure_generator = ArticleStructureGenerator(llm_client=llm_client)
        
        # Prepare research data
        research_data = {
            "brief": request.brief or request.topic,
            "keywords": request.keywords,
            "tone": request.tone,
            "target_word_count": request.target_word_count,
            "draft_title": request.draft_title
        }
        
        # Claims and evidence would typically come from research phase
        # For now we assume empty lists if not provided, or this endpoint 
        # is used after research.
        # Ideally, the user passes this in?
        # The request model doesn't have claims/evidence yet.
        # Assuming minimal input for structure generation initially.
        claims = [] 
        evidence = [] 
        
        structure = structure_generator.generate_structure(research_data, claims, evidence)
        
        return structure
    except Exception as e:
        logger.error(f"Error generating structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/section")
async def generate_section(request: SectionRequest):
    """Generate content for a specific section."""
    try:
        llm_client = get_llm_client()
        content_generator = ContentGenerator(llm_client=llm_client)
        
        # Convert previous sections dicts back to objects if needed, 
        # but content_generator might accept dicts or need objects.
        # Checking content_generator.py:
        # def generate_section_content(..., previous_sections: List[SectionContent] = None)
        # It expects SectionContent objects. We need to handle this deserialization.
        
        # For now, let's keep it simple and see if we can pass dicts if methods allow,
        # or we might need to reconstruct the objects.
        # content_generator.generate_section_content calls methods that access attributes like section.title
        # So passing dicts directly will fail.
        
        # Note: migrating complex objects via API is tricky.
        # Ideally, we should persist state in DB and just pass IDs.
        # But keeping it stateless for now requires reconstructing objects.
        
        # This implementation is a placeholder for the verified connection.
        # Real implementation requires thorough object mapping.
        
        raise HTTPException(status_code=501, detail="Section generation endpoint not fully implemented yet for stateless usage.")
        
    except Exception as e:
        logger.error(f"Error generating section: {e}")
        raise HTTPException(status_code=500, detail=str(e))
