"""
SoftwareSolutions model for TrendTap
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Enum, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base


class SoftwareType(PyEnum):
    """Software type enumeration"""
    CALCULATOR = "calculator"
    ANALYZER = "analyzer"
    GENERATOR = "generator"
    CONVERTER = "converter"
    ESTIMATOR = "estimator"


class DevelopmentStatus(PyEnum):
    """Development status enumeration"""
    IDEA = "idea"
    PLANNED = "planned"
    IN_DEVELOPMENT = "in_development"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class SoftwareSolutions(Base):
    """Software solutions model"""
    __tablename__ = "software_solutions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    keyword_data_id = Column(Integer, ForeignKey("keyword_data.id"), nullable=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    software_type = Column(Enum(SoftwareType), nullable=False)
    
    # Complexity and priority
    complexity_score = Column(Integer, nullable=False)  # 1-10 scale
    priority_score = Column(Float, nullable=True)  # 0-1 scale
    
    # Technical requirements
    technical_requirements = Column(JSON, nullable=True)  # Technical specs
    estimated_development_time = Column(String(100), nullable=True)  # e.g., "3-4 weeks"
    development_phases = Column(JSON, nullable=True)  # Development phases
    
    # Target keywords and SEO
    target_keywords = Column(JSON, nullable=True)  # Target keywords array
    seo_optimization = Column(JSON, nullable=True)  # SEO optimization data
    
    # Monetization
    monetization_strategy = Column(JSON, nullable=True)  # Monetization strategy
    estimated_revenue_potential = Column(Float, nullable=True)  # Revenue potential score
    
    # Development status
    development_status = Column(Enum(DevelopmentStatus), default=DevelopmentStatus.IDEA, nullable=False)
    planned_start_date = Column(DateTime(timezone=True), nullable=True)
    planned_completion_date = Column(DateTime(timezone=True), nullable=True)
    actual_start_date = Column(DateTime(timezone=True), nullable=True)
    actual_completion_date = Column(DateTime(timezone=True), nullable=True)
    
    # Development progress
    progress_percentage = Column(Integer, default=0, nullable=False)  # 0-100
    current_phase = Column(String(100), nullable=True)
    development_notes = Column(Text, nullable=True)
    
    # Generation metadata
    generation_model = Column(String(50), nullable=True)
    generation_prompt = Column(Text, nullable=True)
    generation_parameters = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="software_solutions")
    keyword_data = relationship("KeywordData", back_populates="software_solutions")
    content_calendar = relationship("ContentCalendar", back_populates="software_solution")
    
    def __repr__(self):
        return f"<SoftwareSolutions(id={self.id}, name='{self.name}', type='{self.software_type.value}')>"
    
    def to_dict(self):
        """Convert software solution to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "keyword_data_id": self.keyword_data_id,
            "name": self.name,
            "description": self.description,
            "software_type": self.software_type.value,
            "complexity_score": self.complexity_score,
            "priority_score": self.priority_score,
            "technical_requirements": self.technical_requirements,
            "estimated_development_time": self.estimated_development_time,
            "development_phases": self.development_phases,
            "target_keywords": self.target_keywords,
            "seo_optimization": self.seo_optimization,
            "monetization_strategy": self.monetization_strategy,
            "estimated_revenue_potential": self.estimated_revenue_potential,
            "development_status": self.development_status.value,
            "planned_start_date": self.planned_start_date.isoformat() if self.planned_start_date else None,
            "planned_completion_date": self.planned_completion_date.isoformat() if self.planned_completion_date else None,
            "actual_start_date": self.actual_start_date.isoformat() if self.actual_start_date else None,
            "actual_completion_date": self.actual_completion_date.isoformat() if self.actual_completion_date else None,
            "progress_percentage": self.progress_percentage,
            "current_phase": self.current_phase,
            "development_notes": self.development_notes,
            "generation_model": self.generation_model,
            "generation_prompt": self.generation_prompt,
            "generation_parameters": self.generation_parameters,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def is_completed(self) -> bool:
        """Check if software is completed"""
        return self.development_status == DevelopmentStatus.COMPLETED
    
    def is_in_development(self) -> bool:
        """Check if software is in development"""
        return self.development_status == DevelopmentStatus.IN_DEVELOPMENT
    
    def is_planned(self) -> bool:
        """Check if software is planned"""
        return self.development_status == DevelopmentStatus.PLANNED
    
    def get_complexity_category(self) -> str:
        """Get complexity category"""
        if self.complexity_score <= 3:
            return "simple"
        elif self.complexity_score <= 6:
            return "moderate"
        elif self.complexity_score <= 8:
            return "complex"
        else:
            return "very_complex"
    
    def get_priority_category(self) -> str:
        """Get priority category"""
        if not self.priority_score:
            return "unknown"
        
        if self.priority_score >= 0.8:
            return "high"
        elif self.priority_score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def get_target_keywords_list(self) -> list:
        """Get target keywords as list"""
        if not self.target_keywords:
            return []
        
        return self.target_keywords
    
    def get_technical_requirements(self) -> dict:
        """Get technical requirements"""
        if not self.technical_requirements:
            return {}
        
        return self.technical_requirements
    
    def get_development_phases(self) -> list:
        """Get development phases"""
        if not self.development_phases:
            return []
        
        return self.development_phases
    
    def get_monetization_strategy(self) -> dict:
        """Get monetization strategy"""
        if not self.monetization_strategy:
            return {}
        
        return self.monetization_strategy
    
    def get_seo_optimization(self) -> dict:
        """Get SEO optimization data"""
        if not self.seo_optimization:
            return {}
        
        return self.seo_optimization
    
    def get_meta_description(self) -> str:
        """Get meta description from SEO optimization"""
        seo = self.get_seo_optimization()
        return seo.get("meta_description", "")
    
    def get_estimated_development_weeks(self) -> int:
        """Get estimated development time in weeks"""
        if not self.estimated_development_time:
            return 0
        
        # Parse development time string (e.g., "3-4 weeks" -> 4)
        import re
        match = re.search(r'(\d+)-?(\d+)?\s*weeks?', self.estimated_development_time.lower())
        if match:
            if match.group(2):
                return int(match.group(2))  # Use upper bound
            else:
                return int(match.group(1))
        
        return 0
    
    def get_development_phase_count(self) -> int:
        """Get number of development phases"""
        phases = self.get_development_phases()
        return len(phases)
    
    def get_current_phase_index(self) -> int:
        """Get current phase index"""
        if not self.current_phase:
            return 0
        
        phases = self.get_development_phases()
        for i, phase in enumerate(phases):
            if phase.get("phase") == self.current_phase:
                return i
        
        return 0
    
    def get_phase_progress(self) -> float:
        """Get progress within current phase"""
        if not self.current_phase:
            return 0.0
        
        phases = self.get_development_phases()
        current_phase_index = self.get_current_phase_index()
        
        if current_phase_index >= len(phases):
            return 1.0
        
        # Calculate progress based on current phase and overall progress
        phase_progress = self.progress_percentage / 100.0
        phase_count = len(phases)
        
        if phase_count == 0:
            return 0.0
        
        # Distribute progress across phases
        phase_progress = (current_phase_index + phase_progress) / phase_count
        return min(phase_progress, 1.0)
    
    def update_development_status(self, status: DevelopmentStatus, notes: str = None):
        """Update development status"""
        self.development_status = status
        
        if status == DevelopmentStatus.IN_DEVELOPMENT and not self.actual_start_date:
            self.actual_start_date = func.now()
        elif status == DevelopmentStatus.COMPLETED and not self.actual_completion_date:
            self.actual_completion_date = func.now()
            self.progress_percentage = 100
        
        if notes:
            self.development_notes = notes
    
    def update_progress(self, percentage: int, current_phase: str = None):
        """Update development progress"""
        self.progress_percentage = max(0, min(percentage, 100))
        
        if current_phase:
            self.current_phase = current_phase
    
    def add_development_note(self, note: str):
        """Add development note"""
        if not self.development_notes:
            self.development_notes = note
        else:
            self.development_notes += f"\n\n{note}"
    
    def calculate_revenue_potential(self) -> float:
        """Calculate revenue potential score"""
        if not self.monetization_strategy:
            return 0.0
        
        # Calculate based on monetization strategy
        primary_strategy = self.monetization_strategy.get("primary", "")
        affiliate_products = self.monetization_strategy.get("affiliate_products", [])
        
        score = 0.0
        
        # Base score for different strategies
        strategy_scores = {
            "affiliate_marketing": 0.7,
            "freemium_model": 0.8,
            "subscription_model": 0.9,
            "one_time_purchase": 0.6
        }
        
        score += strategy_scores.get(primary_strategy, 0.5)
        
        # Bonus for multiple affiliate products
        if len(affiliate_products) > 3:
            score += 0.2
        elif len(affiliate_products) > 1:
            score += 0.1
        
        # Normalize to 0-1 scale
        return min(score, 1.0)
    
    def get_development_summary(self) -> dict:
        """Get development summary"""
        return {
            "id": self.id,
            "name": self.name,
            "software_type": self.software_type.value,
            "complexity_score": self.complexity_score,
            "complexity_category": self.get_complexity_category(),
            "priority_score": self.priority_score,
            "priority_category": self.get_priority_category(),
            "development_status": self.development_status.value,
            "progress_percentage": self.progress_percentage,
            "current_phase": self.current_phase,
            "estimated_development_time": self.estimated_development_time,
            "estimated_development_weeks": self.get_estimated_development_weeks(),
            "development_phase_count": self.get_development_phase_count(),
            "phase_progress": self.get_phase_progress(),
            "target_keywords_count": len(self.get_target_keywords_list()),
            "revenue_potential": self.estimated_revenue_potential
        }
