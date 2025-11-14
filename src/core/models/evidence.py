"""
Evidence-related data models and schemas.

This module defines the data structures for evidence collection,
ranking, and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class EvidenceSource(str, Enum):
    """Evidence source types."""
    RAG = "rag"
    WEB_SEARCH = "web_search"
    ACADEMIC = "academic"
    NEWS = "news"
    BLOG = "blog"
    REPORT = "report"
    OTHER = "other"


class EvidenceQuality(str, Enum):
    """Evidence quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ClaimType(str, Enum):
    """Claim types."""
    FACTUAL = "factual"
    OPINION = "opinion"
    STATISTICAL = "statistical"
    ANECDOTAL = "anecdotal"
    EXPERT = "expert"
    RESEARCH = "research"


class EvidenceRanking(BaseModel):
    """Evidence ranking model."""
    
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to claim")
    credibility_score: float = Field(..., ge=0.0, le=1.0, description="Source credibility")
    recency_score: float = Field(..., ge=0.0, le=1.0, description="Information recency")
    authority_score: float = Field(..., ge=0.0, le=1.0, description="Source authority")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall ranking score")
    
    ranking_factors: Dict[str, Any] = Field(default_factory=dict, description="Detailed ranking factors")
    
    @validator('overall_score')
    def validate_overall_score(cls, v, values):
        """Validate overall score is within bounds."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Overall score must be between 0.0 and 1.0')
        return v


class Evidence(BaseModel):
    """Evidence model."""
    
    id: str = Field(..., description="Unique evidence ID")
    content: str = Field(..., min_length=10, description="Evidence content")
    source: EvidenceSource = Field(..., description="Evidence source")
    source_url: Optional[str] = Field(None, description="Source URL")
    source_title: Optional[str] = Field(None, description="Source title")
    author: Optional[str] = Field(None, description="Author name")
    publication: Optional[str] = Field(None, description="Publication name")
    publication_date: Optional[datetime] = Field(None, description="Publication date")
    
    # Quality Metrics
    quality: EvidenceQuality = Field(..., description="Evidence quality")
    ranking: EvidenceRanking = Field(..., description="Evidence ranking")
    
    # Metadata
    word_count: int = Field(..., description="Word count")
    character_count: int = Field(..., description="Character count")
    language: str = Field(default="en", description="Content language")
    
    # Processing Info
    collected_at: datetime = Field(default_factory=datetime.utcnow, description="Collection timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    
    # Validation
    fact_checked: bool = Field(default=False, description="Whether evidence has been fact-checked")
    verified: bool = Field(default=False, description="Whether evidence has been verified")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def mark_processed(self):
        """Mark evidence as processed."""
        self.processed_at = datetime.utcnow()
    
    def mark_fact_checked(self):
        """Mark evidence as fact-checked."""
        self.fact_checked = True
    
    def mark_verified(self):
        """Mark evidence as verified."""
        self.verified = True


class Claim(BaseModel):
    """Claim model."""
    
    id: str = Field(..., description="Unique claim ID")
    content: str = Field(..., min_length=10, description="Claim content")
    claim_type: ClaimType = Field(..., description="Type of claim")
    
    # Evidence
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting evidence")
    evidence_count: int = Field(default=0, description="Number of evidence pieces")
    
    # Validation
    verified: bool = Field(default=False, description="Whether claim is verified")
    fact_checked: bool = Field(default=False, description="Whether claim is fact-checked")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in claim")
    
    # Processing
    extracted_at: datetime = Field(default_factory=datetime.utcnow, description="Extraction timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_evidence(self, evidence: Evidence):
        """Add evidence to claim."""
        self.evidence.append(evidence)
        self.evidence_count = len(self.evidence)
    
    def remove_evidence(self, evidence_id: str):
        """Remove evidence from claim."""
        self.evidence = [e for e in self.evidence if e.id != evidence_id]
        self.evidence_count = len(self.evidence)
    
    def get_high_quality_evidence(self) -> List[Evidence]:
        """Get high quality evidence."""
        return [e for e in self.evidence if e.quality in [EvidenceQuality.HIGH, EvidenceQuality.VERY_HIGH]]
    
    def get_verified_evidence(self) -> List[Evidence]:
        """Get verified evidence."""
        return [e for e in self.evidence if e.verified]
    
    def calculate_confidence_score(self) -> float:
        """Calculate confidence score based on evidence."""
        if not self.evidence:
            return 0.0
        
        # Weight by evidence quality and ranking
        total_score = 0.0
        total_weight = 0.0
        
        for evidence in self.evidence:
            weight = 1.0
            if evidence.quality == EvidenceQuality.VERY_HIGH:
                weight = 2.0
            elif evidence.quality == EvidenceQuality.HIGH:
                weight = 1.5
            elif evidence.quality == EvidenceQuality.MEDIUM:
                weight = 1.0
            else:  # LOW
                weight = 0.5
            
            if evidence.verified:
                weight *= 1.2
            
            total_score += evidence.ranking.overall_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def mark_processed(self):
        """Mark claim as processed."""
        self.processed_at = datetime.utcnow()
        self.confidence_score = self.calculate_confidence_score()
    
    def mark_verified(self):
        """Mark claim as verified."""
        self.verified = True
    
    def mark_fact_checked(self):
        """Mark claim as fact-checked."""
        self.fact_checked = True
