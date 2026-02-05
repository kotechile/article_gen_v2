"""
CoSchedule API Integration
Provides content calendar and headline analysis capabilities
"""

import httpx
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class CoScheduleAPI:
    """CoSchedule API client for content calendar and headline analysis"""
    
    def __init__(self):
        self.api_key = settings.COSCHEULE_API_KEY
        self.base_url = "https://api.coschedule.com/v1"
        self.timeout = 60.0
    
    async def analyze_headline(
        self,
        headline: str,
        content_type: str = "blog_post"
    ) -> Dict[str, Any]:
        """
        Analyze headline quality and effectiveness
        
        Args:
            headline: The headline to analyze
            content_type: Type of content (blog_post, social_media, email, etc.)
            
        Returns:
            Headline analysis with scores and recommendations
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/headlines/analyze"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "headline": headline,
                    "content_type": content_type,
                    "include_emotional_analysis": True,
                    "include_power_words": True,
                    "include_readability": True
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_headline_analysis(data, headline)
                
        except Exception as e:
            logger.error(f"CoSchedule headline analysis API error: {e}")
            return {"error": str(e), "headline": headline}
    
    async def generate_headlines(
        self,
        topic: str,
        content_type: str = "blog_post",
        count: int = 10,
        style: str = "engaging"
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple headline options for a topic
        
        Args:
            topic: The topic to generate headlines for
            content_type: Type of content
            count: Number of headlines to generate
            style: Style of headlines (engaging, informative, clickbait, etc.)
            
        Returns:
            List of generated headlines with analysis
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/headlines/generate"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "topic": topic,
                    "content_type": content_type,
                    "count": count,
                    "style": style,
                    "include_analysis": True,
                    "include_emotional_words": True
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_headline_generation(data)
                
        except Exception as e:
            logger.error(f"CoSchedule headline generation API error: {e}")
            return []
    
    async def get_calendar_events(
        self,
        start_date: str,
        end_date: str,
        content_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get calendar events for a date range
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            content_types: List of content types to filter by
            
        Returns:
            List of calendar events
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/calendar/events"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                params = {
                    "start_date": start_date,
                    "end_date": end_date
                }
                
                if content_types:
                    params["content_types"] = ",".join(content_types)
                
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_calendar_events(data)
                
        except Exception as e:
            logger.error(f"CoSchedule calendar events API error: {e}")
            return []
    
    async def create_calendar_event(
        self,
        title: str,
        content_type: str,
        scheduled_date: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new calendar event
        
        Args:
            title: Event title
            content_type: Type of content
            scheduled_date: Scheduled date in YYYY-MM-DD format
            description: Event description
            tags: List of tags
            
        Returns:
            Created event details
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/calendar/events"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "title": title,
                    "content_type": content_type,
                    "scheduled_date": scheduled_date,
                    "description": description or "",
                    "tags": tags or [],
                    "status": "scheduled"
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_calendar_event(data)
                
        except Exception as e:
            logger.error(f"CoSchedule create calendar event API error: {e}")
            return {"error": str(e), "title": title}
    
    async def get_content_ideas(
        self,
        topic: str,
        content_types: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get content ideas for a topic
        
        Args:
            topic: The topic to get ideas for
            content_types: List of content types to filter by
            limit: Maximum number of ideas
            
        Returns:
            List of content ideas
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/content/ideas"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "topic": topic,
                    "content_types": content_types or ["blog_post", "social_media", "email"],
                    "limit": limit,
                    "include_headlines": True,
                    "include_descriptions": True
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_content_ideas(data)
                
        except Exception as e:
            logger.error(f"CoSchedule content ideas API error: {e}")
            return []
    
    async def get_team_performance(
        self,
        start_date: str,
        end_date: str,
        team_members: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get team performance metrics
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            team_members: List of team member IDs to filter by
            
        Returns:
            Team performance metrics
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/analytics/team-performance"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                params = {
                    "start_date": start_date,
                    "end_date": end_date
                }
                
                if team_members:
                    params["team_members"] = ",".join(team_members)
                
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_team_performance(data)
                
        except Exception as e:
            logger.error(f"CoSchedule team performance API error: {e}")
            return {"error": str(e)}
    
    def _process_headline_analysis(self, data: Dict[str, Any], headline: str) -> Dict[str, Any]:
        """Process headline analysis response"""
        analysis = {
            "headline": headline,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_score": 0,
            "emotional_analysis": {},
            "power_words": [],
            "readability": {},
            "recommendations": []
        }
        
        if "analysis" in data:
            analysis_data = data["analysis"]
            
            # Overall score
            analysis["overall_score"] = analysis_data.get("overall_score", 0)
            
            # Emotional analysis
            if "emotional_analysis" in analysis_data:
                analysis["emotional_analysis"] = {
                    "emotional_words": analysis_data["emotional_analysis"].get("emotional_words", []),
                    "emotional_score": analysis_data["emotional_analysis"].get("emotional_score", 0),
                    "sentiment": analysis_data["emotional_analysis"].get("sentiment", "neutral")
                }
            
            # Power words
            if "power_words" in analysis_data:
                analysis["power_words"] = analysis_data["power_words"]
            
            # Readability
            if "readability" in analysis_data:
                analysis["readability"] = {
                    "score": analysis_data["readability"].get("score", 0),
                    "grade_level": analysis_data["readability"].get("grade_level", 0),
                    "character_count": analysis_data["readability"].get("character_count", 0),
                    "word_count": analysis_data["readability"].get("word_count", 0)
                }
            
            # Recommendations
            if "recommendations" in analysis_data:
                analysis["recommendations"] = analysis_data["recommendations"]
        
        return analysis
    
    def _process_headline_generation(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process headline generation response"""
        headlines = []
        
        if "headlines" in data:
            for headline in data["headlines"]:
                headlines.append({
                    "headline": headline.get("headline", ""),
                    "score": headline.get("score", 0),
                    "style": headline.get("style", ""),
                    "emotional_words": headline.get("emotional_words", []),
                    "power_words": headline.get("power_words", []),
                    "character_count": len(headline.get("headline", "")),
                    "created_at": datetime.utcnow().isoformat()
                })
        
        return headlines
    
    def _process_calendar_events(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process calendar events response"""
        events = []
        
        if "events" in data:
            for event in data["events"]:
                events.append({
                    "id": event.get("id", ""),
                    "title": event.get("title", ""),
                    "content_type": event.get("content_type", ""),
                    "scheduled_date": event.get("scheduled_date", ""),
                    "description": event.get("description", ""),
                    "status": event.get("status", ""),
                    "tags": event.get("tags", []),
                    "created_at": event.get("created_at", ""),
                    "updated_at": event.get("updated_at", "")
                })
        
        return events
    
    def _process_calendar_event(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process single calendar event response"""
        return {
            "id": data.get("id", ""),
            "title": data.get("title", ""),
            "content_type": data.get("content_type", ""),
            "scheduled_date": data.get("scheduled_date", ""),
            "description": data.get("description", ""),
            "status": data.get("status", ""),
            "tags": data.get("tags", []),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", "")
        }
    
    def _process_content_ideas(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process content ideas response"""
        ideas = []
        
        if "ideas" in data:
            for idea in data["ideas"]:
                ideas.append({
                    "id": idea.get("id", ""),
                    "title": idea.get("title", ""),
                    "description": idea.get("description", ""),
                    "content_type": idea.get("content_type", ""),
                    "headlines": idea.get("headlines", []),
                    "tags": idea.get("tags", []),
                    "priority": idea.get("priority", "medium"),
                    "created_at": datetime.utcnow().isoformat()
                })
        
        return ideas
    
    def _process_team_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process team performance response"""
        performance = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_content": 0,
            "completed_content": 0,
            "team_members": [],
            "content_types": {},
            "metrics": {}
        }
        
        if "performance" in data:
            perf_data = data["performance"]
            
            # Basic metrics
            performance["total_content"] = perf_data.get("total_content", 0)
            performance["completed_content"] = perf_data.get("completed_content", 0)
            
            # Team members
            if "team_members" in perf_data:
                for member in perf_data["team_members"]:
                    performance["team_members"].append({
                        "id": member.get("id", ""),
                        "name": member.get("name", ""),
                        "content_count": member.get("content_count", 0),
                        "completion_rate": member.get("completion_rate", 0),
                        "average_score": member.get("average_score", 0)
                    })
            
            # Content types
            if "content_types" in perf_data:
                performance["content_types"] = perf_data["content_types"]
            
            # Additional metrics
            if "metrics" in perf_data:
                performance["metrics"] = perf_data["metrics"]
        
        return performance

# Global instance
coschedule_api = CoScheduleAPI()

# Convenience functions
async def analyze_headline(
    headline: str,
    content_type: str = "blog_post"
) -> Dict[str, Any]:
    """Analyze headline quality and effectiveness"""
    return await coschedule_api.analyze_headline(headline, content_type)

async def generate_headlines(
    topic: str,
    content_type: str = "blog_post",
    count: int = 10,
    style: str = "engaging"
) -> List[Dict[str, Any]]:
    """Generate multiple headline options for a topic"""
    return await coschedule_api.generate_headlines(topic, content_type, count, style)

async def get_calendar_events(
    start_date: str,
    end_date: str,
    content_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Get calendar events for a date range"""
    return await coschedule_api.get_calendar_events(start_date, end_date, content_types)

async def create_calendar_event(
    title: str,
    content_type: str,
    scheduled_date: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a new calendar event"""
    return await coschedule_api.create_calendar_event(
        title, content_type, scheduled_date, description, tags
    )

async def get_content_ideas(
    topic: str,
    content_types: Optional[List[str]] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Get content ideas for a topic"""
    return await coschedule_api.get_content_ideas(topic, content_types, limit)

async def get_team_performance(
    start_date: str,
    end_date: str,
    team_members: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Get team performance metrics"""
    return await coschedule_api.get_team_performance(start_date, end_date, team_members)
