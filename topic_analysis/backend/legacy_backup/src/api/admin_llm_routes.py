from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime

from ..core.database import get_db
from ..core.supabase_database import get_supabase_db
from ..services.llm_service import LLMService
from ..models.llm_config import LLMProvider, LLMConfiguration
from ..core.security import require_admin

router = APIRouter(prefix="/api/admin/llm", tags=["admin-llm"])

# Request/Response Models
class LLMProviderCreate(BaseModel):
    name: str = Field(..., description="Provider name (e.g., 'OpenAI GPT-4')")
    provider_type: str = Field(..., description="Provider type: openai, anthropic, google, local, custom")
    model_name: str = Field(..., description="Model name (e.g., 'gpt-4', 'claude-3-sonnet')")
    api_key_env_var: Optional[str] = Field(None, description="Environment variable name for API key")
    base_url: Optional[str] = Field(None, description="Custom API endpoint URL")
    api_version: Optional[str] = Field(None, description="API version")
    max_tokens: int = Field(2000, description="Maximum tokens per request")
    temperature: float = Field(0.7, description="Temperature (0.0-2.0)")
    top_p: float = Field(1.0, description="Top-p sampling")
    frequency_penalty: float = Field(0.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, description="Presence penalty")
    cost_per_1k_tokens: float = Field(0.0, description="Cost per 1000 tokens")
    max_requests_per_minute: int = Field(60, description="Rate limit per minute")
    priority: int = Field(0, description="Priority (higher = more preferred)")
    custom_config: Optional[Dict[str, Any]] = Field(None, description="Custom configuration")

class LLMProviderUpdate(BaseModel):
    name: Optional[str] = None
    model_name: Optional[str] = None
    api_key_env_var: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    cost_per_1k_tokens: Optional[float] = None
    max_requests_per_minute: Optional[int] = None
    is_active: Optional[bool] = None
    priority: Optional[int] = None
    custom_config: Optional[Dict[str, Any]] = None

class LLMProviderResponse(BaseModel):
    id: str
    name: str
    provider_type: str
    model_name: str
    is_active: bool
    is_default: bool
    priority: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens_used: int
    total_cost: float
    last_used: Optional[datetime]
    created_at: datetime

class LLMTestRequest(BaseModel):
    test_topic: str = Field("best wireless headphones", description="Topic to test with")

class LLMTestResponse(BaseModel):
    success: bool
    response_time_ms: Optional[int] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class LLMUsageStatsResponse(BaseModel):
    period_days: int
    providers: List[Dict[str, Any]]

# API Endpoints

@router.get("/providers", response_model=List[LLMProviderResponse])
async def get_llm_providers(
    current_user = Depends(require_admin)
):
    """Get all LLM providers"""
    try:
        db = get_supabase_db()
        providers_data = db.get_llm_providers()
        
        return [
            LLMProviderResponse(
                id=str(provider.get('id', '')),
                name=provider.get('name', ''),
                provider_type=provider.get('provider_type', ''),
                model_name=provider.get('model_name', ''),
                is_active=provider.get('is_active', True),
                is_default=provider.get('is_default', False),
                priority=provider.get('priority', 0),
                total_requests=provider.get('total_requests', 0),
                successful_requests=provider.get('successful_requests', 0),
                failed_requests=provider.get('failed_requests', 0),
                total_tokens_used=provider.get('total_tokens_used', 0),
                total_cost=provider.get('total_cost', 0.0),
                last_used=provider.get('last_used'),
                created_at=provider.get('created_at')
            )
            for provider in providers_data
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch providers: {str(e)}")

@router.post("/providers", response_model=LLMProviderResponse)
async def create_llm_provider(
    provider_data: LLMProviderCreate,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Create a new LLM provider"""
    llm_service = LLMService(db)
    
    # Convert to dict and add required fields
    provider_dict = provider_data.dict()
    provider_dict.update({
        "is_active": True,
        "is_default": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })
    
    provider = llm_service.create_provider(provider_dict)
    
    return LLMProviderResponse(
        id=str(provider.id),
        name=provider.name,
        provider_type=provider.provider_type,
        model_name=provider.model_name,
        is_active=provider.is_active,
        is_default=provider.is_default,
        priority=provider.priority,
        total_requests=provider.total_requests,
        successful_requests=provider.successful_requests,
        failed_requests=provider.failed_requests,
        total_tokens_used=provider.total_tokens_used,
        total_cost=provider.total_cost,
        last_used=provider.last_used,
        created_at=provider.created_at
    )

@router.put("/providers/{provider_id}", response_model=LLMProviderResponse)
async def update_llm_provider(
    provider_id: str,
    updates: LLMProviderUpdate,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Update an LLM provider"""
    llm_service = LLMService(db)
    
    # Convert updates to dict, removing None values
    update_dict = {k: v for k, v in updates.dict().items() if v is not None}
    update_dict["updated_at"] = datetime.utcnow()
    
    success = llm_service.update_provider(provider_id, update_dict)
    if not success:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    provider = llm_service.get_provider_by_id(provider_id)
    return LLMProviderResponse(
        id=str(provider.id),
        name=provider.name,
        provider_type=provider.provider_type,
        model_name=provider.model_name,
        is_active=provider.is_active,
        is_default=provider.is_default,
        priority=provider.priority,
        total_requests=provider.total_requests,
        successful_requests=provider.successful_requests,
        failed_requests=provider.failed_requests,
        total_tokens_used=provider.total_tokens_used,
        total_cost=provider.total_cost,
        last_used=provider.last_used,
        created_at=provider.created_at
    )

@router.delete("/providers/{provider_id}")
async def delete_llm_provider(
    provider_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Delete an LLM provider (soft delete)"""
    llm_service = LLMService(db)
    
    success = llm_service.update_provider(provider_id, {"is_active": False})
    if not success:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    return {"message": "Provider deactivated successfully"}

@router.post("/providers/{provider_id}/set-default")
async def set_default_provider(
    provider_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Set a provider as the default"""
    llm_service = LLMService(db)
    
    success = llm_service.set_default_provider(provider_id)
    if not success:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    return {"message": "Default provider updated successfully"}

@router.post("/providers/{provider_id}/test", response_model=LLMTestResponse)
async def test_llm_provider(
    provider_id: str,
    test_request: LLMTestRequest,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Test an LLM provider with a sample topic"""
    llm_service = LLMService(db)
    
    result = await llm_service.test_provider(provider_id, test_request.test_topic)
    
    return LLMTestResponse(**result)

@router.get("/usage-stats", response_model=LLMUsageStatsResponse)
async def get_usage_stats(
    days: int = Query(30, description="Number of days to include in stats"),
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Get LLM usage statistics"""
    llm_service = LLMService(db)
    stats = llm_service.get_usage_stats(days)
    
    return LLMUsageStatsResponse(**stats)

@router.get("/config")
async def get_llm_config(
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Get global LLM configuration"""
    config = db.query(LLMConfiguration).first()
    if not config:
        # Return default configuration
        return {
            "enable_llm_analysis": True,
            "enable_auto_fallback": True,
            "enable_cost_tracking": True,
            "global_rate_limit_per_minute": 100,
            "user_rate_limit_per_minute": 10,
            "daily_cost_limit": 50.0,
            "monthly_cost_limit": 1000.0
        }
    
    return {
        "enable_llm_analysis": config.enable_llm_analysis,
        "enable_auto_fallback": config.enable_auto_fallback,
        "enable_cost_tracking": config.enable_cost_tracking,
        "global_rate_limit_per_minute": config.global_rate_limit_per_minute,
        "user_rate_limit_per_minute": config.user_rate_limit_per_minute,
        "daily_cost_limit": config.daily_cost_limit,
        "monthly_cost_limit": config.monthly_cost_limit,
        "cost_alert_threshold": config.cost_alert_threshold
    }

@router.put("/config")
async def update_llm_config(
    config_data: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Update global LLM configuration"""
    config = db.query(LLMConfiguration).first()
    
    if not config:
        config = LLMConfiguration(**config_data)
        db.add(config)
    else:
        for key, value in config_data.items():
            setattr(config, key, value)
        config.updated_at = datetime.utcnow()
        config.updated_by = current_user.id
    
    db.commit()
    
    return {"message": "Configuration updated successfully"}

@router.get("/providers/{provider_id}/analytics")
async def get_provider_analytics(
    provider_id: str,
    days: int = Query(30, description="Number of days to include"),
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Get detailed analytics for a specific provider"""
    from datetime import timedelta
    
    since_date = datetime.utcnow() - timedelta(days=days)
    
    # Get usage logs for the provider
    usage_logs = db.query(LLMUsageLog).filter(
        LLMUsageLog.provider_id == provider_id,
        LLMUsageLog.request_timestamp >= since_date
    ).all()
    
    # Calculate analytics
    total_requests = len(usage_logs)
    successful_requests = sum(1 for log in usage_logs if log.success)
    failed_requests = total_requests - successful_requests
    total_tokens = sum(log.total_tokens for log in usage_logs)
    total_cost = sum(log.cost for log in usage_logs)
    avg_response_time = sum(log.response_time_ms for log in usage_logs) / total_requests if total_requests > 0 else 0
    
    # Popular topics
    from sqlalchemy import func
    popular_topics = db.query(
        LLMUsageLog.topic,
        func.count(LLMUsageLog.id).label('count')
    ).filter(
        LLMUsageLog.provider_id == provider_id,
        LLMUsageLog.request_timestamp >= since_date
    ).group_by(LLMUsageLog.topic).order_by(
        func.count(LLMUsageLog.id).desc()
    ).limit(10).all()
    
    return {
        "provider_id": provider_id,
        "period_days": days,
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "avg_response_time_ms": avg_response_time,
        "popular_topics": [
            {"topic": topic, "count": count}
            for topic, count in popular_topics
        ]
    }
