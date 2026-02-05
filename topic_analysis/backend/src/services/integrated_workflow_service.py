"""
Integrated Workflow Service
Combines affiliate research, trend analysis, and content generation
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog
from ..core.supabase_database import get_supabase_db
from ..core.redis import cache
from .affiliate_research_service import AffiliateResearchService
from .trend_service import TrendService
from .content_service import ContentService

logger = structlog.get_logger()

class IntegratedWorkflowService:
    """Service that integrates affiliate research, trend analysis, and content generation"""
    
    def __init__(self):
        self.db = get_supabase_db()
        self.affiliate_service = AffiliateResearchService()
        self.trend_service = TrendService()
        self.content_service = ContentService()
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    async def complete_research_workflow(
        self,
        search_term: str,
        niche: Optional[str] = None,
        budget_range: Optional[str] = None,
        user_id: Optional[str] = None,
        selected_affiliate_programs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Complete research workflow: Affiliate Research → Trend Analysis → Content Generation
        """
        logger.info("Starting complete research workflow", 
                   search_term=search_term, niche=niche, user_id=user_id)
        
        try:
            # Step 1: Affiliate Research
            logger.info("Step 1: Starting affiliate research")
            affiliate_results = await self.affiliate_service.search_affiliate_programs(
                search_term=search_term,
                niche=niche,
                budget_range=budget_range,
                user_id=user_id
            )
            
            # Extract affiliate programs for trend analysis
            # Check both possible structures: direct programs or nested under data
            affiliate_programs = affiliate_results.get('programs', [])
            if not affiliate_programs:
                affiliate_programs = affiliate_results.get('data', {}).get('programs', [])
            
            logger.info("Debug: affiliate_results structure", 
                       affiliate_results_keys=list(affiliate_results.keys()),
                       data_keys=list(affiliate_results.get('data', {}).keys()) if affiliate_results.get('data') else None,
                       programs_count=len(affiliate_programs))
            
            if not affiliate_programs:
                logger.warning("No affiliate programs found, skipping trend analysis")
                return {
                    "success": False,
                    "message": "No affiliate programs found for trend analysis",
                    "affiliate_results": affiliate_results
                }
            
            # Use selected programs or all programs
            if selected_affiliate_programs:
                selected_programs = [p for p in affiliate_programs if p.get('id') in selected_affiliate_programs]
            else:
                # Select top 5 programs by default
                selected_programs = affiliate_programs[:5]
            
            logger.info("Selected affiliate programs for trend analysis", 
                       count=len(selected_programs))
            
            # Step 2: Trend Analysis
            logger.info("Step 2: Starting trend analysis")
            trend_results = await self.trend_service.discover_trending_topics(
                search_term=search_term,
                selected_programs=[p.get('name', '') for p in selected_programs],
                user_id=user_id
            )
            
            trending_topics = trend_results.get('trending_topics', [])
            if not trending_topics:
                logger.warning("No trending topics found")
                return {
                    "success": False,
                    "message": "No trending topics found",
                    "affiliate_results": affiliate_results,
                    "trend_results": trend_results
                }
            
            # Step 3: Content Generation
            logger.info("Step 3: Starting content generation")
            content_results = await self.content_service.generate_content_ideas(
                search_term=search_term,
                trending_topics=trending_topics,
                affiliate_programs=selected_programs,
                user_id=user_id
            )
            
            # Combine all results
            workflow_result = {
                "success": True,
                "message": "Complete research workflow completed successfully",
                "search_term": search_term,
                "niche": niche,
                "budget_range": budget_range,
                "workflow_steps": {
                    "affiliate_research": {
                        "status": "completed",
                        "programs_found": len(affiliate_programs),
                        "selected_programs": len(selected_programs)
                    },
                    "trend_analysis": {
                        "status": "completed",
                        "trending_topics_found": len(trending_topics)
                    },
                    "content_generation": {
                        "status": "completed",
                        "content_ideas_generated": len(content_results.get('content_ideas', [])),
                        "software_ideas_generated": len(content_results.get('software_ideas', []))
                    }
                },
                "affiliate_results": affiliate_results,
                "trend_results": trend_results,
                "content_results": content_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Cache the complete workflow result
            cache_key = f"workflow:{search_term}:{niche}:{budget_range}"
            cache.set(cache_key, workflow_result, expire=self.cache_ttl)
            
            logger.info("Complete research workflow completed successfully")
            return workflow_result
            
        except Exception as e:
            logger.error("Complete research workflow failed", error=str(e))
            return {
                "success": False,
                "message": f"Workflow failed: {str(e)}",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def generate_content_for_selected_trends(
        self,
        search_term: str,
        selected_trends: List[str],
        affiliate_programs: List[Dict[str, Any]],
        content_type: str = "both",  # "blog", "software", or "both"
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate content ideas for specifically selected trends
        """
        logger.info("Generating content for selected trends", 
                   search_term=search_term, 
                   selected_trends=selected_trends,
                   content_type=content_type)
        
        try:
            # Filter trending topics to only selected ones
            all_trending_topics = []
            for trend_id in selected_trends:
                # This would typically come from a database lookup
                # For now, we'll create a mock structure
                trend_topic = {
                    "id": trend_id,
                    "topic": f"Selected trend: {trend_id}",
                    "search_volume": 1000,
                    "trend_direction": "rising",
                    "competition": "medium",
                    "opportunity_score": 75
                }
                all_trending_topics.append(trend_topic)
            
            # Generate content based on selected trends
            content_results = await self.content_service.generate_content_ideas(
                search_term=search_term,
                trending_topics=all_trending_topics,
                affiliate_programs=affiliate_programs,
                content_type=content_type,
                user_id=user_id
            )
            
            return {
                "success": True,
                "message": "Content generated for selected trends",
                "search_term": search_term,
                "selected_trends": selected_trends,
                "content_results": content_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Content generation for selected trends failed", error=str(e))
            return {
                "success": False,
                "message": f"Content generation failed: {str(e)}",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_workflow_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for workflow results"""
        try:
            pattern = "workflow:*"
            keys = cache.client.keys(pattern)
            return {
                "total_cached_workflows": len(keys),
                "cache_ttl_seconds": self.cache_ttl,
                "cache_ttl_hours": self.cache_ttl / 3600
            }
        except Exception as e:
            logger.warning("Failed to get workflow cache stats", error=str(e))
            return {"error": str(e)}
