"""
Software Solution API schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class SoftwareType(str, Enum):
    """Software type enumeration"""
    CALCULATOR = "calculator"
    ANALYZER = "analyzer"
    CONVERTER = "converter"
    GENERATOR = "generator"
    OPTIMIZER = "optimizer"
    VALIDATOR = "validator"
    COMPARATOR = "comparator"
    SIMULATOR = "simulator"

class SoftwareStatus(str, Enum):
    """Software status enumeration"""
    IDEA = "idea"
    IN_DEVELOPMENT = "in_development"
    BETA = "beta"
    RELEASED = "released"
    MAINTENANCE = "maintenance"
    ARCHIVED = "archived"

class DevelopmentComplexity(str, Enum):
    """Development complexity enumeration"""
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"
    EXPERT = "expert"

class SoftwareSolutionRequest(BaseModel):
    """Request schema for software solution generation"""
    title: str = Field(..., min_length=1, max_length=200, description="Software title")
    description: str = Field(..., min_length=10, max_length=1000, description="Software description")
    software_type: SoftwareType = Field(..., description="Type of software")
    target_audience: Optional[str] = Field(None, description="Target audience")
    keywords: List[str] = Field(..., min_items=1, description="Target keywords")
    features: Optional[List[str]] = Field(None, description="Desired features")
    monetization_strategy: Optional[str] = Field(None, description="Monetization strategy")
    technology_stack: Optional[List[str]] = Field(None, description="Preferred technology stack")
    budget_range: Optional[str] = Field(None, description="Budget range")
    timeline: Optional[str] = Field(None, description="Development timeline")
    additional_requirements: Optional[str] = Field(None, description="Additional requirements")
    
    @validator('keywords')
    def validate_keywords(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one keyword is required')
        return v

class SoftwareGenerationRequest(BaseModel):
    """Request schema for software generation (alias for SoftwareSolutionRequest)"""
    title: str = Field(..., min_length=1, max_length=200, description="Software title")
    description: str = Field(..., min_length=10, max_length=1000, description="Software description")
    software_type: SoftwareType = Field(..., description="Type of software")
    target_audience: Optional[str] = Field(None, description="Target audience")
    keywords: List[str] = Field(..., min_items=1, description="Target keywords")
    features: Optional[List[str]] = Field(None, description="Desired features")
    monetization_strategy: Optional[str] = Field(None, description="Monetization strategy")
    technology_stack: Optional[List[str]] = Field(None, description="Preferred technology stack")
    budget_range: Optional[str] = Field(None, description="Budget range")
    timeline: Optional[str] = Field(None, description="Development timeline")
    additional_requirements: Optional[str] = Field(None, description="Additional requirements")
    
    @validator('keywords')
    def validate_keywords(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one keyword is required')
        return v

class SoftwareSolutionResponse(BaseModel):
    """Response schema for software solution"""
    id: int = Field(..., description="Solution ID")
    user_id: int = Field(..., description="User ID")
    title: str = Field(..., description="Software title")
    description: str = Field(..., description="Software description")
    software_type: SoftwareType = Field(..., description="Software type")
    target_audience: Optional[str] = Field(None, description="Target audience")
    keywords: List[str] = Field(..., description="Target keywords")
    features: List[str] = Field(..., description="Planned features")
    monetization_strategy: Optional[str] = Field(None, description="Monetization strategy")
    technology_stack: List[str] = Field(..., description="Recommended technology stack")
    budget_range: Optional[str] = Field(None, description="Budget range")
    timeline: Optional[str] = Field(None, description="Development timeline")
    status: SoftwareStatus = Field(..., description="Development status")
    
    # Analysis results
    market_analysis: Optional[Dict[str, Any]] = Field(None, description="Market analysis")
    competitor_analysis: Optional[Dict[str, Any]] = Field(None, description="Competitor analysis")
    development_complexity: Optional[DevelopmentComplexity] = Field(None, description="Development complexity")
    complexity_score: Optional[int] = Field(None, ge=1, le=10, description="Complexity score (1-10)")
    estimated_development_time: Optional[int] = Field(None, ge=1, description="Estimated development time in weeks")
    estimated_cost: Optional[Dict[str, Any]] = Field(None, description="Estimated development cost")
    revenue_potential: Optional[Dict[str, Any]] = Field(None, description="Revenue potential analysis")
    
    # Technical specifications
    technical_requirements: Optional[List[str]] = Field(None, description="Technical requirements")
    api_requirements: Optional[List[str]] = Field(None, description="API requirements")
    database_requirements: Optional[Dict[str, Any]] = Field(None, description="Database requirements")
    hosting_requirements: Optional[Dict[str, Any]] = Field(None, description="Hosting requirements")
    security_requirements: Optional[List[str]] = Field(None, description="Security requirements")
    
    # Development plan
    development_phases: Optional[List[Dict[str, Any]]] = Field(None, description="Development phases")
    milestones: Optional[List[Dict[str, Any]]] = Field(None, description="Development milestones")
    risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Risk assessment")
    success_metrics: Optional[List[str]] = Field(None, description="Success metrics")
    
    # Additional data
    additional_requirements: Optional[str] = Field(None, description="Additional requirements")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    analyzed_at: Optional[datetime] = Field(None, description="Analysis completion date")

class SoftwareGenerationResponse(BaseModel):
    """Response schema for software generation results"""
    solution_id: int = Field(..., description="Software solution ID")
    generated_solution: SoftwareSolutionResponse = Field(..., description="Generated solution")
    generation_metadata: Dict[str, Any] = Field(..., description="Generation metadata")
    processing_time: float = Field(..., description="Processing time in seconds")
    tokens_used: Optional[int] = Field(None, description="Tokens used for generation")
    model_used: Optional[str] = Field(None, description="AI model used")
    created_at: datetime = Field(..., description="Generation date")

class SoftwareSolutionListResponse(BaseModel):
    """Response schema for software solution list"""
    solutions: List[SoftwareSolutionResponse] = Field(..., description="List of solutions")
    total: int = Field(..., description="Total number of solutions")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class SoftwareSolutionsResponse(BaseModel):
    """Response schema for software solutions (alias for SoftwareSolutionResponse)"""
    id: int = Field(..., description="Solution ID")
    user_id: int = Field(..., description="User ID")
    title: str = Field(..., description="Software title")
    description: str = Field(..., description="Software description")
    software_type: SoftwareType = Field(..., description="Software type")
    target_audience: Optional[str] = Field(None, description="Target audience")
    keywords: List[str] = Field(..., description="Target keywords")
    features: List[str] = Field(..., description="Planned features")
    monetization_strategy: Optional[str] = Field(None, description="Monetization strategy")
    technology_stack: List[str] = Field(..., description="Recommended technology stack")
    budget_range: Optional[str] = Field(None, description="Budget range")
    timeline: Optional[str] = Field(None, description="Development timeline")
    status: SoftwareStatus = Field(..., description="Development status")
    
    # Analysis results
    market_analysis: Optional[Dict[str, Any]] = Field(None, description="Market analysis")
    competitor_analysis: Optional[Dict[str, Any]] = Field(None, description="Competitor analysis")
    development_complexity: Optional[DevelopmentComplexity] = Field(None, description="Development complexity")
    complexity_score: Optional[int] = Field(None, ge=1, le=10, description="Complexity score (1-10)")
    estimated_development_time: Optional[int] = Field(None, ge=1, description="Estimated development time in weeks")
    estimated_cost: Optional[Dict[str, Any]] = Field(None, description="Estimated development cost")
    revenue_potential: Optional[Dict[str, Any]] = Field(None, description="Revenue potential analysis")
    
    # Technical specifications
    technical_requirements: Optional[List[str]] = Field(None, description="Technical requirements")
    api_requirements: Optional[List[str]] = Field(None, description="API requirements")
    database_requirements: Optional[Dict[str, Any]] = Field(None, description="Database requirements")
    hosting_requirements: Optional[Dict[str, Any]] = Field(None, description="Hosting requirements")
    security_requirements: Optional[List[str]] = Field(None, description="Security requirements")
    
    # Development plan
    development_phases: Optional[List[Dict[str, Any]]] = Field(None, description="Development phases")
    milestones: Optional[List[Dict[str, Any]]] = Field(None, description="Development milestones")
    risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Risk assessment")
    success_metrics: Optional[List[str]] = Field(None, description="Success metrics")
    
    # Additional data
    additional_requirements: Optional[str] = Field(None, description="Additional requirements")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    analyzed_at: Optional[datetime] = Field(None, description="Analysis completion date")

class SoftwareSolutionsListResponse(BaseModel):
    """Response schema for software solutions list (alias for SoftwareSolutionListResponse)"""
    solutions: List[SoftwareSolutionsResponse] = Field(..., description="List of solutions")
    total: int = Field(..., description="Total number of solutions")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class SoftwareSolutionUpdateRequest(BaseModel):
    """Request schema for updating software solution"""
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="Updated title")
    description: Optional[str] = Field(None, min_length=10, max_length=1000, description="Updated description")
    software_type: Optional[SoftwareType] = Field(None, description="Updated software type")
    target_audience: Optional[str] = Field(None, description="Updated target audience")
    keywords: Optional[List[str]] = Field(None, description="Updated keywords")
    features: Optional[List[str]] = Field(None, description="Updated features")
    monetization_strategy: Optional[str] = Field(None, description="Updated monetization strategy")
    technology_stack: Optional[List[str]] = Field(None, description="Updated technology stack")
    budget_range: Optional[str] = Field(None, description="Updated budget range")
    timeline: Optional[str] = Field(None, description="Updated timeline")
    status: Optional[SoftwareStatus] = Field(None, description="Updated status")
    additional_requirements: Optional[str] = Field(None, description="Updated requirements")
    notes: Optional[str] = Field(None, description="Updated notes")

class SoftwareUpdateRequest(BaseModel):
    """Request schema for updating software (alias for SoftwareSolutionUpdateRequest)"""
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="Updated title")
    description: Optional[str] = Field(None, min_length=10, max_length=1000, description="Updated description")
    software_type: Optional[SoftwareType] = Field(None, description="Updated software type")
    target_audience: Optional[str] = Field(None, description="Updated target audience")
    keywords: Optional[List[str]] = Field(None, description="Updated keywords")
    features: Optional[List[str]] = Field(None, description="Updated features")
    monetization_strategy: Optional[str] = Field(None, description="Updated monetization strategy")
    technology_stack: Optional[List[str]] = Field(None, description="Updated technology stack")
    budget_range: Optional[str] = Field(None, description="Updated budget range")
    timeline: Optional[str] = Field(None, description="Updated timeline")
    status: Optional[SoftwareStatus] = Field(None, description="Updated status")
    additional_requirements: Optional[str] = Field(None, description="Updated requirements")
    notes: Optional[str] = Field(None, description="Updated notes")

class SoftwareAnalysisRequest(BaseModel):
    """Request schema for software analysis"""
    solution_id: int = Field(..., description="Solution ID to analyze")
    analysis_type: Optional[str] = Field("comprehensive", description="Type of analysis")
    include_market_research: Optional[bool] = Field(True, description="Include market research")
    include_competitor_analysis: Optional[bool] = Field(True, description="Include competitor analysis")
    include_technical_analysis: Optional[bool] = Field(True, description="Include technical analysis")
    include_financial_analysis: Optional[bool] = Field(True, description="Include financial analysis")
    include_risk_analysis: Optional[bool] = Field(True, description="Include risk analysis")

class SoftwareAnalysisResponse(BaseModel):
    """Response schema for software analysis results"""
    solution_id: int = Field(..., description="Solution ID")
    analysis_type: str = Field(..., description="Analysis type")
    market_analysis: Optional[Dict[str, Any]] = Field(None, description="Market analysis")
    competitor_analysis: Optional[Dict[str, Any]] = Field(None, description="Competitor analysis")
    technical_analysis: Optional[Dict[str, Any]] = Field(None, description="Technical analysis")
    financial_analysis: Optional[Dict[str, Any]] = Field(None, description="Financial analysis")
    risk_analysis: Optional[Dict[str, Any]] = Field(None, description="Risk analysis")
    recommendations: List[str] = Field(..., description="Analysis recommendations")
    insights: List[str] = Field(..., description="Key insights")
    next_steps: List[str] = Field(..., description="Recommended next steps")
    created_at: datetime = Field(..., description="Analysis date")

class SoftwareTemplateResponse(BaseModel):
    """Response schema for software template"""
    id: int = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    software_type: SoftwareType = Field(..., description="Software type")
    description: str = Field(..., description="Template description")
    template: Dict[str, Any] = Field(..., description="Template structure")
    variables: List[str] = Field(..., description="Template variables")
    is_public: bool = Field(..., description="Is template public")
    usage_count: int = Field(..., description="Usage count")
    created_at: datetime = Field(..., description="Template creation date")
    updated_at: datetime = Field(..., description="Last update date")

class SoftwareTemplateListResponse(BaseModel):
    """Response schema for software template list"""
    templates: List[SoftwareTemplateResponse] = Field(..., description="List of templates")
    total: int = Field(..., description="Total number of templates")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class SoftwareStatsResponse(BaseModel):
    """Response schema for software statistics"""
    total_solutions: int = Field(..., description="Total solutions")
    solutions_by_type: List[Dict[str, Any]] = Field(..., description="Solutions by type")
    solutions_by_status: List[Dict[str, Any]] = Field(..., description="Solutions by status")
    average_complexity: float = Field(..., description="Average complexity score")
    average_development_time: float = Field(..., description="Average development time")
    total_estimated_cost: float = Field(..., description="Total estimated cost")
    total_revenue_potential: float = Field(..., description="Total revenue potential")
    top_technologies: List[Dict[str, Any]] = Field(..., description="Top technologies used")
    last_updated: datetime = Field(..., description="Last database update")
