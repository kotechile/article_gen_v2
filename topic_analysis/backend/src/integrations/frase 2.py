"""
Frase API Integration
Provides content optimization and topic research capabilities
"""

import httpx
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class FraseAPI:
    """Frase API client for content optimization and topic research"""
    
    def __init__(self):
        self.api_key = settings.frase_api_key
        self.base_url = "https://api.frase.io/v1"
        self.timeout = 60.0
    
    async def get_topic_research(
        self,
        topic: str,
        language: str = "en",
        location: str = "US",
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get topic research data for content planning
        
        Args:
            topic: The topic to research
            language: Language code (e.g., "en", "es", "fr")
            location: Location code (e.g., "US", "GB", "CA")
            limit: Maximum number of results
            
        Returns:
            Topic research data with questions, keywords, and content ideas
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/research/topic"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "topic": topic,
                    "language": language,
                    "location": location,
                    "limit": limit,
                    "include_questions": True,
                    "include_keywords": True,
                    "include_content_ideas": True
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_topic_research(data, topic)
                
        except Exception as e:
            logger.error(f"Frase topic research API error: {e}")
            return {"error": str(e), "topic": topic}
    
    async def get_content_optimization(
        self,
        content: str,
        target_keyword: str,
        language: str = "en",
        location: str = "US"
    ) -> Dict[str, Any]:
        """
        Get content optimization suggestions
        
        Args:
            content: The content to optimize
            target_keyword: The target keyword for optimization
            language: Language code
            location: Location code
            
        Returns:
            Content optimization recommendations
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/optimize/content"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "content": content,
                    "target_keyword": target_keyword,
                    "language": language,
                    "location": location,
                    "include_readability": True,
                    "include_keyword_density": True,
                    "include_semantic_analysis": True
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_content_optimization(data, target_keyword)
                
        except Exception as e:
            logger.error(f"Frase content optimization API error: {e}")
            return {"error": str(e), "target_keyword": target_keyword}
    
    async def get_question_research(
        self,
        topic: str,
        language: str = "en",
        location: str = "US",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get questions people ask about a topic
        
        Args:
            topic: The topic to research questions for
            language: Language code
            location: Location code
            limit: Maximum number of questions
            
        Returns:
            List of questions with metrics
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/research/questions"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "topic": topic,
                    "language": language,
                    "location": location,
                    "limit": limit,
                    "include_metrics": True,
                    "include_answers": True
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_question_research(data)
                
        except Exception as e:
            logger.error(f"Frase question research API error: {e}")
            return []
    
    async def get_content_outline(
        self,
        topic: str,
        target_keyword: str,
        language: str = "en",
        location: str = "US"
    ) -> Dict[str, Any]:
        """
        Generate content outline for a topic
        
        Args:
            topic: The topic to create outline for
            target_keyword: The target keyword
            language: Language code
            location: Location code
            
        Returns:
            Content outline with sections and subsections
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/outline/generate"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "topic": topic,
                    "target_keyword": target_keyword,
                    "language": language,
                    "location": location,
                    "include_keywords": True,
                    "include_questions": True,
                    "include_competitor_analysis": True
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_content_outline(data, topic, target_keyword)
                
        except Exception as e:
            logger.error(f"Frase content outline API error: {e}")
            return {"error": str(e), "topic": topic, "target_keyword": target_keyword}
    
    async def get_competitor_analysis(
        self,
        target_keyword: str,
        language: str = "en",
        location: str = "US",
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze competitors for a target keyword
        
        Args:
            target_keyword: The target keyword to analyze competitors for
            language: Language code
            location: Location code
            limit: Maximum number of competitors to analyze
            
        Returns:
            Competitor analysis with insights
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/competitors/analyze"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "target_keyword": target_keyword,
                    "language": language,
                    "location": location,
                    "limit": limit,
                    "include_content_analysis": True,
                    "include_keyword_analysis": True
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_competitor_analysis(data, target_keyword)
                
        except Exception as e:
            logger.error(f"Frase competitor analysis API error: {e}")
            return {"error": str(e), "target_keyword": target_keyword}
    
    def _process_topic_research(self, data: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Process topic research response"""
        research = {
            "topic": topic,
            "timestamp": datetime.utcnow().isoformat(),
            "questions": [],
            "keywords": [],
            "content_ideas": [],
            "summary": {}
        }
        
        if "research" in data:
            research_data = data["research"]
            
            # Questions
            if "questions" in research_data:
                for question in research_data["questions"]:
                    research["questions"].append({
                        "question": question.get("question", ""),
                        "search_volume": question.get("search_volume", 0),
                        "difficulty": question.get("difficulty", 0),
                        "answer": question.get("answer", ""),
                        "source": question.get("source", "")
                    })
            
            # Keywords
            if "keywords" in research_data:
                for keyword in research_data["keywords"]:
                    research["keywords"].append({
                        "keyword": keyword.get("keyword", ""),
                        "search_volume": keyword.get("search_volume", 0),
                        "difficulty": keyword.get("difficulty", 0),
                        "cpc": keyword.get("cpc", 0),
                        "competition": keyword.get("competition", "UNKNOWN"),
                        "trend": keyword.get("trend", "stable")
                    })
            
            # Content ideas
            if "content_ideas" in research_data:
                research["content_ideas"] = research_data["content_ideas"]
            
            # Summary
            if "summary" in research_data:
                research["summary"] = research_data["summary"]
        
        return research
    
    def _process_content_optimization(self, data: Dict[str, Any], target_keyword: str) -> Dict[str, Any]:
        """Process content optimization response"""
        optimization = {
            "target_keyword": target_keyword,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_score": 0,
            "readability": {},
            "keyword_density": {},
            "semantic_analysis": {},
            "recommendations": [],
            "issues": []
        }
        
        if "optimization" in data:
            opt_data = data["optimization"]
            
            # Overall score
            optimization["overall_score"] = opt_data.get("overall_score", 0)
            
            # Readability
            if "readability" in opt_data:
                optimization["readability"] = {
                    "score": opt_data["readability"].get("score", 0),
                    "grade_level": opt_data["readability"].get("grade_level", 0),
                    "sentences": opt_data["readability"].get("sentences", 0),
                    "words": opt_data["readability"].get("words", 0),
                    "paragraphs": opt_data["readability"].get("paragraphs", 0)
                }
            
            # Keyword density
            if "keyword_density" in opt_data:
                optimization["keyword_density"] = {
                    "target_keyword": opt_data["keyword_density"].get("target_keyword", 0),
                    "semantic_keywords": opt_data["keyword_density"].get("semantic_keywords", {}),
                    "recommended_density": opt_data["keyword_density"].get("recommended_density", 0)
                }
            
            # Semantic analysis
            if "semantic_analysis" in opt_data:
                optimization["semantic_analysis"] = opt_data["semantic_analysis"]
            
            # Recommendations
            if "recommendations" in opt_data:
                optimization["recommendations"] = opt_data["recommendations"]
            
            # Issues
            if "issues" in opt_data:
                optimization["issues"] = opt_data["issues"]
        
        return optimization
    
    def _process_question_research(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process question research response"""
        questions = []
        
        if "questions" in data:
            for question in data["questions"]:
                questions.append({
                    "question": question.get("question", ""),
                    "search_volume": question.get("search_volume", 0),
                    "difficulty": question.get("difficulty", 0),
                    "answer": question.get("answer", ""),
                    "source": question.get("source", ""),
                    "related_keywords": question.get("related_keywords", []),
                    "created_at": datetime.utcnow().isoformat()
                })
        
        return questions
    
    def _process_content_outline(self, data: Dict[str, Any], topic: str, target_keyword: str) -> Dict[str, Any]:
        """Process content outline response"""
        outline = {
            "topic": topic,
            "target_keyword": target_keyword,
            "timestamp": datetime.utcnow().isoformat(),
            "sections": [],
            "keywords": [],
            "questions": [],
            "competitor_insights": {}
        }
        
        if "outline" in data:
            outline_data = data["outline"]
            
            # Sections
            if "sections" in outline_data:
                for section in outline_data["sections"]:
                    outline["sections"].append({
                        "title": section.get("title", ""),
                        "content": section.get("content", ""),
                        "keywords": section.get("keywords", []),
                        "word_count": section.get("word_count", 0),
                        "order": section.get("order", 0)
                    })
            
            # Keywords
            if "keywords" in outline_data:
                outline["keywords"] = outline_data["keywords"]
            
            # Questions
            if "questions" in outline_data:
                outline["questions"] = outline_data["questions"]
            
            # Competitor insights
            if "competitor_insights" in outline_data:
                outline["competitor_insights"] = outline_data["competitor_insights"]
        
        return outline
    
    def _process_competitor_analysis(self, data: Dict[str, Any], target_keyword: str) -> Dict[str, Any]:
        """Process competitor analysis response"""
        analysis = {
            "target_keyword": target_keyword,
            "timestamp": datetime.utcnow().isoformat(),
            "competitors": [],
            "content_analysis": {},
            "keyword_analysis": {},
            "insights": []
        }
        
        if "analysis" in data:
            analysis_data = data["analysis"]
            
            # Competitors
            if "competitors" in analysis_data:
                for competitor in analysis_data["competitors"]:
                    analysis["competitors"].append({
                        "domain": competitor.get("domain", ""),
                        "url": competitor.get("url", ""),
                        "title": competitor.get("title", ""),
                        "description": competitor.get("description", ""),
                        "position": competitor.get("position", 0),
                        "content_score": competitor.get("content_score", 0),
                        "keyword_density": competitor.get("keyword_density", 0),
                        "word_count": competitor.get("word_count", 0)
                    })
            
            # Content analysis
            if "content_analysis" in analysis_data:
                analysis["content_analysis"] = analysis_data["content_analysis"]
            
            # Keyword analysis
            if "keyword_analysis" in analysis_data:
                analysis["keyword_analysis"] = analysis_data["keyword_analysis"]
            
            # Insights
            if "insights" in analysis_data:
                analysis["insights"] = analysis_data["insights"]
        
        return analysis

# Global instance
frase_api = FraseAPI()

# Convenience functions
async def get_topic_research(
    topic: str,
    language: str = "en",
    location: str = "US",
    limit: int = 50
) -> Dict[str, Any]:
    """Get topic research data for content planning"""
    return await frase_api.get_topic_research(topic, language, location, limit)

async def get_content_optimization(
    content: str,
    target_keyword: str,
    language: str = "en",
    location: str = "US"
) -> Dict[str, Any]:
    """Get content optimization suggestions"""
    return await frase_api.get_content_optimization(content, target_keyword, language, location)

async def get_question_research(
    topic: str,
    language: str = "en",
    location: str = "US",
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Get questions people ask about a topic"""
    return await frase_api.get_question_research(topic, language, location, limit)

async def get_content_outline(
    topic: str,
    target_keyword: str,
    language: str = "en",
    location: str = "US"
) -> Dict[str, Any]:
    """Generate content outline for a topic"""
    return await frase_api.get_content_outline(topic, target_keyword, language, location)

async def get_competitor_analysis(
    target_keyword: str,
    language: str = "en",
    location: str = "US",
    limit: int = 10
) -> Dict[str, Any]:
    """Analyze competitors for a target keyword"""
    return await frase_api.get_competitor_analysis(target_keyword, language, location, limit)
