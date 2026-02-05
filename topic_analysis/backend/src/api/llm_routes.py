from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from ..services.llm.llm_service import llm_service
from ..services.llm.llm_provider import LLMResponse

router = APIRouter(prefix="/api/llm", tags=["llm"])

class GenerateRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

@router.post("/generate", response_model=LLMResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text using an LLM.
    If provider is not specified, uses the default provider from the database.
    """
    try:
        # Filter out None values to let provider defaults or DB defaults take over
        kwargs = {k: v for k, v in request.dict().items() 
                  if k not in ["prompt", "provider"] and v is not None}
        
        response = await llm_service.generate_text(
            prompt=request.prompt,
            provider=request.provider,
            **kwargs
        )
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
