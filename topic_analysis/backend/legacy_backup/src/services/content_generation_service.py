"""
Content Generation Service
Handles content idea generation with enhanced keywords and affiliate integration
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from ..models.content_idea import ContentIdea, ContentType, ContentStatus, ContentPriority
from ..models.keyword_cluster import KeywordCluster
from ..models.external_tool_result import ExternalToolResult
from ..core.llm_config import LLMConfigManager
from ..core.redis import cache

logger = structlog.get_logger()

class ContentGenerationService:
    """Service for generating content ideas with enhanced keywords and affiliate integration"""
    
    def __init__(self, db: Session):
        self.db = db
        self.llm_config_manager = LLMConfigManager()
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    async def generate_content_ideas(
        self,
        user_id: str,
        workflow_session_id: str,
        trend_analysis_id: Optional[str] = None,
        topic_decomposition_id: Optional[str] = None,
        keyword_clusters: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        target_audience: Optional[str] = None,
        max_ideas: int = 10
    ) -> Dict[str, Any]:
        """Generate content ideas based on trend analysis and keyword clusters"""
        try:
            logger.info("Generating content ideas", 
                       user_id=user_id, 
                       workflow_session_id=workflow_session_id,
                       max_ideas=max_ideas)
            
            # Get trend analysis data
            trend_data = await self._get_trend_analysis_data(trend_analysis_id, user_id)
            
            # Get keyword clusters
            clusters_data = await self._get_keyword_clusters_data(keyword_clusters, user_id)
            
            # Get topic decomposition data
            topic_data = await self._get_topic_decomposition_data(topic_decomposition_id, user_id)
            
            # Generate content ideas using LLM
            content_ideas = await self._generate_ideas_with_llm(
                trend_data=trend_data,
                clusters_data=clusters_data,
                topic_data=topic_data,
                content_types=content_types or [ContentType.BLOG_POST.value],
                target_audience=target_audience,
                max_ideas=max_ideas
                )
            
            # Save content ideas to database
            saved_ideas = []
            for idea_data in content_ideas:
                content_idea = await self._save_content_idea(
                    user_id=user_id,
                    workflow_session_id=workflow_session_id,
                    trend_analysis_id=trend_analysis_id,
                    topic_decomposition_id=topic_decomposition_id,
                    idea_data=idea_data
                )
                saved_ideas.append(content_idea)
            
            # Update keyword clusters as used for content
            if keyword_clusters:
                await self._mark_clusters_as_used(keyword_clusters, user_id)
            
            logger.info("Content ideas generated successfully", 
                       ideas_count=len(saved_ideas),
                       user_id=user_id)
            
            return {
                "content_ideas": [idea.to_dict() for idea in saved_ideas],
                "generation_metadata": {
                    "total_ideas": len(saved_ideas),
                    "trend_analysis_id": trend_analysis_id,
                    "topic_decomposition_id": topic_decomposition_id,
                    "keyword_clusters_used": keyword_clusters or [],
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error("Failed to generate content ideas", 
                        error=str(e), 
                        user_id=user_id)
            raise
    
    async def _get_trend_analysis_data(self, trend_analysis_id: Optional[str], user_id: str) -> Dict[str, Any]:
        """Get trend analysis data"""
        if not trend_analysis_id:
            return {}
        
        try:
            # This would query the trend analysis from database
            # For now, return mock data
            return {
                "trends": [
                    {
                        "keyword": "electric cars",
                        "search_volume": 50000,
                        "trend_score": 85,
                        "growth_rate": 25.5
                    }
                ],
                "insights": {
                    "top_trending": ["electric cars", "sustainable transportation"],
                    "growth_opportunities": ["EV charging", "battery technology"]
                }
            }
        except Exception as e:
            logger.warning("Failed to get trend analysis data", error=str(e))
            return {}
    
    async def _get_keyword_clusters_data(self, cluster_ids: Optional[List[str]], user_id: str) -> List[Dict[str, Any]]:
        """Get keyword clusters data"""
        if not cluster_ids:
            return []
        
        try:
            clusters = self.db.query(KeywordCluster).filter(
                KeywordCluster.id.in_(cluster_ids),
                KeywordCluster.user_id == user_id,
                KeywordCluster.is_active == True
            ).all()
            
            return [cluster.to_dict() for cluster in clusters]
        except Exception as e:
            logger.warning("Failed to get keyword clusters data", error=str(e))
            return []
    
    async def _get_topic_decomposition_data(self, topic_decomposition_id: Optional[str], user_id: str) -> Dict[str, Any]:
        """Get topic decomposition data"""
        if not topic_decomposition_id:
            return {}
        
        try:
            # This would query the topic decomposition from database
            # For now, return mock data
            return {
                "subtopics": [
                    {
                        "name": "Electric Vehicle Technology",
                        "description": "Latest developments in EV technology",
                        "relevance_score": 0.95,
                        "category": "automotive"
                    }
                ]
            }
        except Exception as e:
            logger.warning("Failed to get topic decomposition data", error=str(e))
            return {}
    
    async def _generate_ideas_with_llm(
        self,
        trend_data: Dict[str, Any],
        clusters_data: List[Dict[str, Any]],
        topic_data: Dict[str, Any],
        content_types: List[str],
        target_audience: Optional[str],
        max_ideas: int
    ) -> List[Dict[str, Any]]:
        """Generate content ideas using LLM"""
        try:
            llm_provider = self.llm_config_manager.get_active_provider()
            if not llm_provider:
                raise ValueError("No active LLM provider configured")
            
            # Prepare context for LLM
            context = self._prepare_llm_context(
                trend_data, clusters_data, topic_data, 
                content_types, target_audience
            )
            
            # Generate ideas
            prompt = self._create_content_generation_prompt(context, max_ideas)
            
            llm_response = await llm_provider.generate_text(
                prompt, 
                temperature=0.7, 
                max_tokens=2000
            )
            
            # Parse LLM response
            ideas = self._parse_llm_response(llm_response)
            
            return ideas[:max_ideas]  # Limit to requested number
            
        except Exception as e:
            logger.error("Failed to generate ideas with LLM", error=str(e))
            # Return fallback ideas
            return self._generate_fallback_ideas(trend_data, clusters_data, max_ideas)
    
    def _prepare_llm_context(
        self,
        trend_data: Dict[str, Any],
        clusters_data: List[Dict[str, Any]],
        topic_data: Dict[str, Any],
        content_types: List[str],
        target_audience: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare context for LLM generation"""
        context = {
            "trends": trend_data.get("trends", []),
            "insights": trend_data.get("insights", {}),
            "keyword_clusters": clusters_data,
            "subtopics": topic_data.get("subtopics", []),
            "content_types": content_types,
            "target_audience": target_audience or "general audience"
        }
        
        # Extract key information
        context["primary_keywords"] = []
        context["search_volumes"] = []
        context["trending_topics"] = []
        
        for trend in context["trends"]:
            if isinstance(trend, dict):
                context["primary_keywords"].append(trend.get("keyword", ""))
                context["search_volumes"].append(trend.get("search_volume", 0))
                if trend.get("trend_score", 0) > 70:
                    context["trending_topics"].append(trend.get("keyword", ""))
        
        for cluster in context["keyword_clusters"]:
            if isinstance(cluster, dict):
                context["primary_keywords"].extend(cluster.get("secondary_keywords", []))
        
        return context
    
    def _create_content_generation_prompt(self, context: Dict[str, Any], max_ideas: int) -> str:
        """Create prompt for content generation"""
        prompt = f"""
You are an expert content strategist. Generate {max_ideas} high-quality content ideas based on the following data:

TREND DATA:
- Primary Keywords: {', '.join(context['primary_keywords'][:10])}
- Trending Topics: {', '.join(context['trending_topics'][:5])}
- Search Volumes: {context['search_volumes'][:5]}

KEYWORD CLUSTERS:
{self._format_clusters_for_prompt(context['keyword_clusters'])}

TOPIC DECOMPOSITION:
{self._format_subtopics_for_prompt(context['subtopics'])}

CONTENT TYPES: {', '.join(context['content_types'])}
TARGET AUDIENCE: {context['target_audience']}

For each content idea, provide:
1. title: Compelling, SEO-optimized title
2. description: Brief description of the content
3. content_type: One of the specified content types
4. primary_keyword: Main keyword to target
5. secondary_keywords: 3-5 related keywords
6. content_angle: Unique angle or perspective
7. key_points: 3-5 main points to cover
8. target_audience: Specific audience segment
9. content_outline: Basic structure/outline
10. monetization_opportunities: How to monetize this content

Format as JSON array of objects. Ensure each idea is unique and valuable.
"""
        return prompt
    
    def _format_clusters_for_prompt(self, clusters: List[Dict[str, Any]]) -> str:
        """Format keyword clusters for prompt"""
        if not clusters:
            return "No keyword clusters available"
        
        formatted = []
        for cluster in clusters[:3]:  # Limit to top 3 clusters
            if isinstance(cluster, dict):
                formatted.append(f"- {cluster.get('cluster_name', 'Unknown')}: {cluster.get('primary_keyword', '')}")
        
        return '\n'.join(formatted)
    
    def _format_subtopics_for_prompt(self, subtopics: List[Dict[str, Any]]) -> str:
        """Format subtopics for prompt"""
        if not subtopics:
            return "No subtopics available"
        
        formatted = []
        for subtopic in subtopics[:5]:  # Limit to top 5 subtopics
            if isinstance(subtopic, dict):
                formatted.append(f"- {subtopic.get('name', 'Unknown')}: {subtopic.get('description', '')}")
        
        return '\n'.join(formatted)
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract content ideas"""
        try:
            import json
            
            # Try to find JSON in response
            json_start = response.find('[')
            json_end = response.rfind(']')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = response[json_start:json_end + 1]
                ideas = json.loads(json_str)
                
                # Validate and clean ideas
                cleaned_ideas = []
                for idea in ideas:
                    if isinstance(idea, dict) and idea.get("title"):
                        cleaned_idea = self._clean_idea_data(idea)
                        cleaned_ideas.append(cleaned_idea)
                
                return cleaned_ideas
            
            # Fallback parsing
            return self._fallback_parse_ideas(response)
            
        except Exception as e:
            logger.error("Failed to parse LLM response", error=str(e))
            return []
    
    def _clean_idea_data(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate idea data"""
        return {
            "title": str(idea.get("title", "")),
            "description": str(idea.get("description", "")),
            "content_type": str(idea.get("content_type", ContentType.BLOG_POST.value)),
            "primary_keyword": str(idea.get("primary_keyword", "")),
            "secondary_keywords": idea.get("secondary_keywords", []) if isinstance(idea.get("secondary_keywords"), list) else [],
            "content_angle": str(idea.get("content_angle", "")),
            "key_points": idea.get("key_points", []) if isinstance(idea.get("key_points"), list) else [],
            "target_audience": str(idea.get("target_audience", "")),
            "content_outline": idea.get("content_outline", []) if isinstance(idea.get("content_outline"), list) else [],
            "monetization_opportunities": str(idea.get("monetization_opportunities", ""))
        }
    
    def _fallback_parse_ideas(self, response: str) -> List[Dict[str, Any]]:
        """Fallback parsing for non-JSON responses"""
        # Simple fallback - extract titles and create basic ideas
        lines = response.split('\n')
        ideas = []
        
        for line in lines:
            line = line.strip()
            if line and ('title' in line.lower() or 'idea' in line.lower()):
                # Extract title from line
                title = line.replace('title:', '').replace('idea:', '').strip()
                if title and len(title) > 10:
                    ideas.append({
                        "title": title,
                        "description": f"Content idea about {title}",
                        "content_type": ContentType.BLOG_POST.value,
                        "primary_keyword": title.split()[0] if title.split() else "content",
                        "secondary_keywords": [],
                        "content_angle": "Informative",
                        "key_points": [],
                        "target_audience": "general audience",
                        "content_outline": [],
                        "monetization_opportunities": "Affiliate marketing opportunities"
                    })
        
        return ideas[:5]  # Limit fallback ideas
    
    def _generate_fallback_ideas(
        self, 
        trend_data: Dict[str, Any], 
        clusters_data: List[Dict[str, Any]], 
        max_ideas: int
    ) -> List[Dict[str, Any]]:
        """Generate fallback ideas when LLM fails"""
        ideas = []
        
        # Use trend data to generate ideas
        trends = trend_data.get("trends", [])
        for i, trend in enumerate(trends[:max_ideas]):
            if isinstance(trend, dict):
                keyword = trend.get("keyword", f"trending topic {i+1}")
                ideas.append({
                    "title": f"Complete Guide to {keyword.title()}",
                    "description": f"Comprehensive guide covering all aspects of {keyword}",
                    "content_type": ContentType.BLOG_POST.value,
                    "primary_keyword": keyword,
                    "secondary_keywords": [f"{keyword} guide", f"{keyword} tips", f"{keyword} review"],
                    "content_angle": "Comprehensive guide",
                    "key_points": [
                        f"Introduction to {keyword}",
                        f"Benefits of {keyword}",
                        f"How to get started with {keyword}",
                        f"Best practices for {keyword}",
                        f"Future of {keyword}"
                    ],
                    "target_audience": "beginners and enthusiasts",
                    "content_outline": ["Introduction", "Main Content", "Conclusion"],
                    "monetization_opportunities": "Affiliate products and courses"
                })
        
        return ideas
    
    async def _save_content_idea(
        self,
        user_id: str,
        workflow_session_id: str,
        trend_analysis_id: Optional[str],
        topic_decomposition_id: Optional[str],
        idea_data: Dict[str, Any]
    ) -> ContentIdea:
        """Save content idea to database"""
        try:
            content_idea = ContentIdea(
                id=str(uuid.uuid4()),
                user_id=user_id,
                workflow_session_id=workflow_session_id,
                trend_analysis_id=trend_analysis_id,
                topic_decomposition_id=topic_decomposition_id,
                title=idea_data["title"],
                description=idea_data["description"],
                content_type=ContentType(idea_data["content_type"]),
                status=ContentStatus.DRAFT,
                priority=ContentPriority.MEDIUM,
                target_audience=idea_data["target_audience"],
                content_angle=idea_data["content_angle"],
                key_points=idea_data["key_points"],
                content_outline=idea_data["content_outline"],
                primary_keyword=idea_data["primary_keyword"],
                secondary_keywords=idea_data["secondary_keywords"],
                enhanced_keywords=[],
                monetization_strategy=idea_data["monetization_opportunities"],
                tags=[],
                categories=[]
            )
            
            self.db.add(content_idea)
            self.db.commit()
            self.db.refresh(content_idea)
            
            return content_idea
            
        except Exception as e:
            logger.error("Failed to save content idea", error=str(e))
            self.db.rollback()
            raise
    
    async def _mark_clusters_as_used(self, cluster_ids: List[str], user_id: str) -> None:
        """Mark keyword clusters as used for content generation"""
        try:
            self.db.query(KeywordCluster).filter(
                KeywordCluster.id.in_(cluster_ids),
                KeywordCluster.user_id == user_id
            ).update({"is_used_for_content": True})
            
            self.db.commit()
            
        except Exception as e:
            logger.error("Failed to mark clusters as used", error=str(e))
            self.db.rollback()
    
    async def get_content_ideas(
        self,
        user_id: str,
        workflow_session_id: Optional[str] = None,
        status: Optional[str] = None,
        content_type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get content ideas with optional filtering"""
        try:
            query = self.db.query(ContentIdea).filter(ContentIdea.user_id == user_id)
            
            if workflow_session_id:
                query = query.filter(ContentIdea.workflow_session_id == workflow_session_id)
            
            if status:
                query = query.filter(ContentIdea.status == ContentStatus(status))
            
            if content_type:
                query = query.filter(ContentIdea.content_type == ContentType(content_type))
            
            ideas = query.order_by(ContentIdea.created_at.desc()).offset(offset).limit(limit).all()
            
            return [idea.to_dict() for idea in ideas]
            
        except Exception as e:
            logger.error("Failed to get content ideas", error=str(e))
            raise
    
    async def update_content_idea(
        self,
        idea_id: str,
        user_id: str,
        updates: Dict[str, Any]
    ) -> Optional[ContentIdea]:
        """Update content idea"""
        try:
            idea = self.db.query(ContentIdea).filter(
                ContentIdea.id == idea_id,
                ContentIdea.user_id == user_id
            ).first()
            
            if not idea:
                return None
            
            # Update allowed fields
            allowed_fields = [
                "title", "description", "content_type", "status", "priority",
                "target_audience", "content_angle", "key_points", "content_outline",
                "primary_keyword", "secondary_keywords", "enhanced_keywords",
                "monetization_strategy", "tags", "categories", "user_notes"
            ]
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(idea, field):
                    if field in ["content_type", "status", "priority"]:
                        setattr(idea, field, getattr(ContentType if field == "content_type" else 
                                ContentStatus if field == "status" else ContentPriority, value)(value))
                    else:
                        setattr(idea, field, value)
            
            idea.updated_at = datetime.utcnow()
            self.db.commit()
            
            return idea
            
        except Exception as e:
            logger.error("Failed to update content idea", error=str(e))
            self.db.rollback()
            raise
    
    async def delete_content_idea(self, idea_id: str, user_id: str) -> bool:
        """Delete content idea"""
        try:
            idea = self.db.query(ContentIdea).filter(
                ContentIdea.id == idea_id,
                ContentIdea.user_id == user_id
            ).first()
            
            if not idea:
                return False
            
            self.db.delete(idea)
            self.db.commit()
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete content idea", error=str(e))
            self.db.rollback()
            raise