"""
Simple Content Generation API endpoints for frontend integration
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from ..services.llm_service import LLMService
from ..core.database import get_db
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/content", tags=["simple-content"])

class SimpleContentRequest(BaseModel):
    topic: str
    program_ids: List[str]
    content_type: str = "article_ideas"
    count: int = 5

class ContentIdea(BaseModel):
    headline: str
    description: str
    target_keywords: List[str]
    estimated_traffic: int
    difficulty: str
    content_type: str

class SimpleContentResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any]

@router.post("/generate", response_model=SimpleContentResponse)
async def generate_simple_content(
    request: SimpleContentRequest
):
    """Generate simple content ideas for frontend"""
    try:
        # For now, use mock data to avoid database issues
        # TODO: Integrate with LLM service when database is stable
        
        # Use mock data for now (will integrate LLM later)
        content_data = {
                "ideas": [
                    {
                        "headline": f"Ultimate Guide to {request.topic}",
                        "description": f"Comprehensive guide covering all aspects of {request.topic} with expert insights and recommendations",
                        "target_keywords": [f"{request.topic} guide", f"{request.topic} tutorial", f"how to {request.topic}"],
                        "estimated_traffic": 5000,
                        "difficulty": "Medium",
                        "content_type": "article"
                    },
                    {
                        "headline": f"Best {request.topic} Tools and Resources",
                        "description": f"Curated list of the best tools, resources, and products for {request.topic} enthusiasts",
                        "target_keywords": [f"{request.topic} tools", f"{request.topic} resources", f"{request.topic} products"],
                        "estimated_traffic": 3000,
                        "difficulty": "Easy",
                        "content_type": "article"
                    },
                    {
                        "headline": f"{request.topic} Mistakes to Avoid",
                        "description": f"Common mistakes people make with {request.topic} and how to avoid them",
                        "target_keywords": [f"{request.topic} mistakes", f"{request.topic} tips", f"{request.topic} advice"],
                        "estimated_traffic": 4000,
                        "difficulty": "Easy",
                        "content_type": "article"
                    },
                    {
                        "headline": f"Advanced {request.topic} Techniques",
                        "description": f"Expert-level strategies and techniques for {request.topic} professionals",
                        "target_keywords": [f"advanced {request.topic}", f"{request.topic} techniques", f"{request.topic} strategies"],
                        "estimated_traffic": 2000,
                        "difficulty": "Hard",
                        "content_type": "article"
                    },
                    {
                        "headline": f"{request.topic} Case Studies and Success Stories",
                        "description": f"Real-world examples and success stories from {request.topic} practitioners",
                        "target_keywords": [f"{request.topic} case studies", f"{request.topic} success stories", f"{request.topic} examples"],
                        "estimated_traffic": 2500,
                        "difficulty": "Medium",
                        "content_type": "article"
                    }
                ]
            }
        
        # Convert to response models
        ideas = content_data.get("ideas", [])[:request.count]
        
        return SimpleContentResponse(
            success=True,
            message=f"Generated {len(ideas)} content ideas",
            data={
                "ideas": ideas,
                "topic": request.topic,
                "program_ids": request.program_ids,
                "content_type": request.content_type,
                "generated_at": "2024-01-01T00:00:00Z"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content generation failed: {str(e)}")
