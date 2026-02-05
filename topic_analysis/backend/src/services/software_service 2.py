"""
SoftwareService for software solution generation and management
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from ..core.database import get_db
from ..core.redis import cache
from ..core.config import get_settings
from ..models.software_solutions import SoftwareSolutions, SoftwareType, DevelopmentStatus
from ..models.trend_analysis import TrendAnalysis
from ..models.keyword_data import KeywordData

logger = structlog.get_logger()
settings = get_settings()

class SoftwareService:
    """Service for software solution generation and management"""
    
    def __init__(self):
        self.openai_api_key = settings.openai_api_key
        self.anthropic_api_key = settings.anthropic_api_key
        self.google_ai_api_key = settings.google_ai_api_key
        
        # Model configuration
        self.llm_model = "gpt-4" if self.openai_api_key else "claude-3-sonnet"
        self.max_solutions = 10
        self.software_types = [SoftwareType.CALCULATOR, SoftwareType.ANALYZER, SoftwareType.GENERATOR, SoftwareType.CONVERTER, SoftwareType.ESTIMATOR]
        
        # Complexity scoring factors
        self.complexity_factors = {
            "frontend": {"simple": 1, "moderate": 3, "complex": 6, "very_complex": 8},
            "backend": {"simple": 1, "moderate": 3, "complex": 6, "very_complex": 8},
            "database": {"simple": 1, "moderate": 2, "complex": 4, "very_complex": 6},
            "apis": {"simple": 1, "moderate": 2, "complex": 4, "very_complex": 6},
            "ml_services": {"simple": 0, "moderate": 3, "complex": 6, "very_complex": 9},
            "features": {"simple": 1, "moderate": 2, "complex": 3, "very_complex": 4}
        }
    
    async def generate_solutions(self, user_id: int, trend_analysis_id: int, keyword_data_id: int,
                               software_types: List[str], max_solutions: int = 5) -> Dict[str, Any]:
        """Generate software solutions"""
        try:
            # Validate inputs
            if max_solutions > self.max_solutions:
                max_solutions = self.max_solutions
            
            # Get trend analysis and keyword data
            db = next(get_db())
            trend_analysis = db.get_TrendAnalysis_by_id(TrendAnalysis.id == trend_analysis_id)
            keyword_data = db.get_KeywordData_by_id(KeywordData.id == keyword_data_id)
            
            if not trend_analysis or not keyword_data:
                raise ValueError("Trend analysis or keyword data not found")
            
            # Create software solutions record
            software_solutions = SoftwareSolutions(
                user_id=user_id,
                keyword_data_id=keyword_data_id,
                software_solutions=[]
            )
            db.add(software_solutions)
            db.commit()
            db.refresh(software_solutions)
            
            # Start background generation
            asyncio.create_task(self._generate_software_solutions(
                software_solutions.id, trend_analysis, keyword_data, software_types, max_solutions
            ))
            
            logger.info("Software solutions generation initiated", software_solutions_id=software_solutions.id)
            return software_solutions.to_dict()
            
        except Exception as e:
            logger.error("Failed to generate software solutions", error=str(e))
            raise
    
    async def get_software_solution(self, software_solution_id: int) -> Dict[str, Any]:
        """Get software solution by ID"""
        try:
            db = next(get_db())
            software_solution = db.get_SoftwareSolutions_by_id(SoftwareSolutions.id == software_solution_id)
            
            if not software_solution:
                raise ValueError("Software solution not found")
            
            return software_solution.to_dict()
            
        except Exception as e:
            logger.error("Failed to get software solution", software_solution_id=software_solution_id, error=str(e))
            raise
    
    async def update_software_status(self, software_solution_id: int, status: str, 
                                   planned_start_date: Optional[datetime] = None, notes: str = None) -> bool:
        """Update software solution status"""
        try:
            db = next(get_db())
            software_solution = db.get_SoftwareSolutions_by_id(SoftwareSolutions.id == software_solution_id)
            
            if not software_solution:
                raise ValueError("Software solution not found")
            
            # Update status
            software_solution.update_development_status(DevelopmentStatus(status), notes)
            
            if planned_start_date:
                software_solution.planned_start_date = planned_start_date
            
            db.commit()
            
            logger.info("Software status updated", software_solution_id=software_solution_id, status=status)
            return True
            
        except Exception as e:
            logger.error("Failed to update software status", software_solution_id=software_solution_id, error=str(e))
            raise
    
    async def _generate_software_solutions(self, software_solutions_id: int, trend_analysis: TrendAnalysis,
                                         keyword_data: KeywordData, software_types: List[str], max_solutions: int):
        """Generate software solutions in background"""
        try:
            db = next(get_db())
            software_solutions = db.get_SoftwareSolutions_by_id(SoftwareSolutions.id == software_solutions_id)
            
            if not software_solutions:
                return
            
            # Get high-opportunity topics
            high_opportunity_topics = trend_analysis.get_high_opportunity_topics(70.0)
            
            # Get top keywords
            top_keywords = keyword_data.get_top_keywords(20)
            
            # Generate solutions for each software type
            generated_solutions = []
            for software_type in software_types:
                if len(generated_solutions) >= max_solutions:
                    break
                
                solutions = await self._generate_solutions_for_type(
                    software_type, high_opportunity_topics, top_keywords, trend_analysis, keyword_data
                )
                generated_solutions.extend(solutions)
            
            # Limit to max_solutions
            generated_solutions = generated_solutions[:max_solutions]
            
            # Update software solutions
            software_solutions.software_solutions = generated_solutions
            db.commit()
            
            logger.info("Software solutions generation completed", 
                       software_solutions_id=software_solutions_id, 
                       solutions_count=len(generated_solutions))
            
        except Exception as e:
            logger.error("Software solutions generation failed", software_solutions_id=software_solutions_id, error=str(e))
    
    async def _generate_solutions_for_type(self, software_type: str, topics: List[str], 
                                         keywords: List[Dict[str, Any]], trend_analysis: TrendAnalysis,
                                         keyword_data: KeywordData) -> List[Dict[str, Any]]:
        """Generate solutions for specific software type"""
        solutions = []
        
        for topic in topics:
            solution = await self._generate_single_solution(
                software_type, topic, keywords, trend_analysis, keyword_data
            )
            if solution:
                solutions.append(solution)
        
        return solutions
    
    async def _generate_single_solution(self, software_type: str, topic: str,
                                      keywords: List[Dict[str, Any]], trend_analysis: TrendAnalysis,
                                      keyword_data: KeywordData) -> Optional[Dict[str, Any]]:
        """Generate single software solution"""
        try:
            # Get relevant keywords for topic
            topic_keywords = [k for k in keywords if topic.lower() in k.get("keyword", "").lower()]
            if not topic_keywords:
                topic_keywords = keywords[:5]  # Use top keywords as fallback
            
            # Generate solution details
            name = self._generate_solution_name(software_type, topic, topic_keywords)
            description = self._generate_solution_description(software_type, topic, topic_keywords)
            
            # Generate technical requirements
            technical_requirements = self._generate_technical_requirements(software_type, topic, topic_keywords)
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(technical_requirements)
            
            # Generate development phases
            development_phases = self._generate_development_phases(software_type, complexity_score)
            
            # Calculate priority score
            priority_score = self._calculate_priority_score(topic, trend_analysis, keyword_data)
            
            # Generate monetization strategy
            monetization_strategy = self._generate_monetization_strategy(software_type, topic, topic_keywords)
            
            # Generate SEO optimization
            seo_optimization = self._generate_seo_optimization(topic, topic_keywords)
            
            # Calculate estimated development time
            estimated_development_time = self._calculate_development_time(complexity_score, development_phases)
            
            return {
                "id": f"solution_{hash(topic + software_type)}",
                "name": name,
                "description": description,
                "software_type": software_type,
                "complexity_score": complexity_score,
                "priority_score": priority_score,
                "target_keywords": [k["keyword"] for k in topic_keywords[:5]],
                "technical_requirements": technical_requirements,
                "estimated_development_time": estimated_development_time,
                "development_phases": development_phases,
                "monetization_strategy": monetization_strategy,
                "seo_optimization": seo_optimization,
                "status": "IDEA"
            }
            
        except Exception as e:
            logger.error("Failed to generate single solution", error=str(e))
            return None
    
    def _generate_solution_name(self, software_type: str, topic: str, keywords: List[Dict[str, Any]]) -> str:
        """Generate solution name"""
        top_keyword = keywords[0]["keyword"] if keywords else topic
        
        name_templates = {
            "calculator": f"{topic.title()} Calculator",
            "analyzer": f"{topic.title()} Analyzer",
            "generator": f"{topic.title()} Generator",
            "converter": f"{topic.title()} Converter",
            "estimator": f"{topic.title()} Estimator"
        }
        
        return name_templates.get(software_type, f"{topic.title()} Tool")
    
    def _generate_solution_description(self, software_type: str, topic: str, keywords: List[Dict[str, Any]]) -> str:
        """Generate solution description"""
        top_keyword = keywords[0]["keyword"] if keywords else topic
        
        descriptions = {
            "calculator": f"Calculate {topic} values with precision. Perfect for {top_keyword} professionals and enthusiasts.",
            "analyzer": f"Analyze {topic} data and get detailed insights. Ideal for {top_keyword} analysis and optimization.",
            "generator": f"Generate {topic} content and solutions automatically. Streamline your {top_keyword} workflow.",
            "converter": f"Convert between different {topic} formats and units. Essential for {top_keyword} compatibility.",
            "estimator": f"Estimate {topic} values and projections. Make informed decisions about {top_keyword}."
        }
        
        return descriptions.get(software_type, f"Professional {topic} tool for {top_keyword}.")
    
    def _generate_technical_requirements(self, software_type: str, topic: str, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate technical requirements"""
        # Base requirements
        requirements = {
            "frontend": "React with TypeScript",
            "backend": "Node.js with Express",
            "database": "PostgreSQL",
            "features": []
        }
        
        # Add type-specific requirements
        if software_type == "calculator":
            requirements["features"] = [
                "Interactive calculation interface",
                "Real-time result updates",
                "Input validation",
                "Result export functionality",
                "Calculation history"
            ]
        elif software_type == "analyzer":
            requirements["features"] = [
                "Data upload and processing",
                "Interactive charts and graphs",
                "Export analysis reports",
                "Data visualization",
                "Statistical analysis"
            ]
            requirements["apis"] = ["Data processing API", "Chart generation API"]
        elif software_type == "generator":
            requirements["features"] = [
                "Template-based generation",
                "Customizable parameters",
                "Batch processing",
                "Output format options",
                "Preview functionality"
            ]
        elif software_type == "converter":
            requirements["features"] = [
                "Multiple format support",
                "Batch conversion",
                "Format validation",
                "Conversion history",
                "API integration"
            ]
        elif software_type == "estimator":
            requirements["features"] = [
                "Parameter input forms",
                "Estimation algorithms",
                "Confidence intervals",
                "Scenario comparison",
                "Report generation"
            ]
        
        # Add complexity-based requirements
        if "analyzer" in software_type.lower():
            requirements["ml_services"] = ["Data analysis API", "Pattern recognition API"]
            requirements["backend"] = "Python with FastAPI"
        
        return requirements
    
    def _calculate_complexity_score(self, technical_requirements: Dict[str, Any]) -> int:
        """Calculate complexity score (1-10)"""
        score = 0
        
        # Frontend complexity
        frontend = technical_requirements.get("frontend", "simple")
        if "React" in frontend:
            score += 3
        elif "Vue" in frontend or "Angular" in frontend:
            score += 2
        else:
            score += 1
        
        # Backend complexity
        backend = technical_requirements.get("backend", "simple")
        if "Python" in backend or "Node.js" in backend:
            score += 2
        else:
            score += 1
        
        # Database complexity
        database = technical_requirements.get("database", "simple")
        if "PostgreSQL" in database or "MongoDB" in database:
            score += 2
        else:
            score += 1
        
        # API complexity
        apis = technical_requirements.get("apis", [])
        if apis:
            score += min(len(apis), 3)
        
        # ML services complexity
        ml_services = technical_requirements.get("ml_services", [])
        if ml_services:
            score += min(len(ml_services) * 2, 4)
        
        # Features complexity
        features = technical_requirements.get("features", [])
        score += min(len(features) // 2, 2)
        
        return min(max(score, 1), 10)
    
    def _generate_development_phases(self, software_type: str, complexity_score: int) -> List[Dict[str, Any]]:
        """Generate development phases"""
        phases = []
        
        # Planning phase
        phases.append({
            "phase": "Planning & Design",
            "duration": "1-2 weeks",
            "tasks": [
                "UI/UX design",
                "Database schema design",
                "API specification",
                "Technical architecture"
            ]
        })
        
        # Core development phase
        core_duration = "2-4 weeks" if complexity_score <= 5 else "4-6 weeks"
        phases.append({
            "phase": "Core Development",
            "duration": core_duration,
            "tasks": [
                "Frontend implementation",
                "Backend API development",
                "Database implementation",
                "Core feature development"
            ]
        })
        
        # Advanced features phase (if complex)
        if complexity_score > 6:
            phases.append({
                "phase": "Advanced Features",
                "duration": "2-3 weeks",
                "tasks": [
                    "Advanced functionality",
                    "API integrations",
                    "Performance optimization",
                    "Security implementation"
                ]
            })
        
        # Testing phase
        phases.append({
            "phase": "Testing & Optimization",
            "duration": "1-2 weeks",
            "tasks": [
                "Unit testing",
                "Integration testing",
                "User acceptance testing",
                "Performance optimization"
            ]
        })
        
        return phases
    
    def _calculate_priority_score(self, topic: str, trend_analysis: TrendAnalysis, keyword_data: KeywordData) -> float:
        """Calculate priority score for software solution"""
        # Get opportunity score from trend analysis
        opportunity_score = trend_analysis.get_opportunity_score(topic) / 100.0
        
        # Get keyword priority scores
        topic_keywords = [k for k in keyword_data.keywords if topic.lower() in k.get("keyword", "").lower()]
        if topic_keywords:
            avg_keyword_priority = sum(k.get("priority_score", 0) for k in topic_keywords) / len(topic_keywords)
        else:
            avg_keyword_priority = 0.5
        
        # Weighted average
        priority_score = (opportunity_score * 0.6 + avg_keyword_priority * 0.4)
        
        return min(max(priority_score, 0), 1)
    
    def _generate_monetization_strategy(self, software_type: str, topic: str, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate monetization strategy"""
        # Base strategy
        strategy = {
            "primary": "Freemium model",
            "free_features": [
                "Basic functionality",
                "Limited calculations",
                "Standard templates"
            ],
            "premium_features": [
                "Advanced features",
                "Unlimited usage",
                "Export functionality",
                "API access"
            ],
            "affiliate_products": []
        }
        
        # Add topic-specific affiliate products
        if "coffee" in topic.lower():
            strategy["affiliate_products"].extend([
                "Coffee equipment",
                "Coffee beans",
                "Coffee books and guides"
            ])
        elif "software" in topic.lower():
            strategy["affiliate_products"].extend([
                "Development tools",
                "Software licenses",
                "Programming courses"
            ])
        else:
            strategy["affiliate_products"].append(f"{topic} related products")
        
        # Adjust strategy based on software type
        if software_type == "calculator":
            strategy["primary"] = "Freemium model"
        elif software_type == "analyzer":
            strategy["primary"] = "Subscription model"
        elif software_type == "generator":
            strategy["primary"] = "Pay-per-use model"
        
        return strategy
    
    def _generate_seo_optimization(self, topic: str, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate SEO optimization data"""
        # Get target keywords
        target_keywords = [k["keyword"] for k in keywords[:5]]
        
        # Generate meta description
        meta_description = f"Free {topic} tool. Calculate, analyze, and optimize {target_keywords[0] if target_keywords else 'your data'} with our professional tool."
        
        # Generate content strategy
        content_strategy = [
            f"{topic} tool landing page",
            f"How to use {topic} tool guide",
            f"{topic} tool features and benefits",
            f"{topic} tool tutorials and examples"
        ]
        
        return {
            "target_keywords": target_keywords,
            "meta_description": meta_description,
            "content_strategy": content_strategy
        }
    
    def _calculate_development_time(self, complexity_score: int, development_phases: List[Dict[str, Any]]) -> str:
        """Calculate estimated development time"""
        total_weeks = 0
        
        for phase in development_phases:
            duration = phase.get("duration", "1 week")
            if "-" in duration:
                # Range like "2-4 weeks"
                weeks = int(duration.split("-")[1].split()[0])
            else:
                # Single value like "2 weeks"
                weeks = int(duration.split()[0])
            total_weeks += weeks
        
        if total_weeks <= 4:
            return f"{total_weeks} weeks"
        elif total_weeks <= 8:
            return f"{total_weeks//4}-{total_weeks//4 + 1} months"
        else:
            return f"{total_weeks//4} months"
    
    async def get_user_software_solutions(self, user_id: int, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get user's software solutions"""
        try:
            db = next(get_db())
            software_solutions = db.query(SoftwareSolutions).filter(
                SoftwareSolutions.user_id == user_id
            ).order_by(SoftwareSolutions.created_at.desc()).offset(offset).limit(limit).all()
            
            return [ss.to_dict() for ss in software_solutions]
            
        except Exception as e:
            logger.error("Failed to get user software solutions", user_id=user_id, error=str(e))
            raise
    
    async def delete_software_solution(self, software_solution_id: int, user_id: int) -> bool:
        """Delete software solution"""
        try:
            db = next(get_db())
            software_solution = db.get_SoftwareSolutions_by_id(
                SoftwareSolutions.id == software_solution_id,
                SoftwareSolutions.user_id == user_id
            )
            
            if not software_solution:
                raise ValueError("Software solution not found")
            
            db.delete(software_solution)
            db.commit()
            
            logger.info("Software solution deleted", software_solution_id=software_solution_id, user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete software solution", software_solution_id=software_solution_id, error=str(e))
            raise
