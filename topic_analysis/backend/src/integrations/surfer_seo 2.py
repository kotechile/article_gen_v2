"""
SurferSEO API Integration
Provides SEO content optimization and keyword analysis capabilities
"""

import httpx
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class SurferSEOAPI:
    """SurferSEO API client for SEO content optimization"""
    
    def __init__(self):
        self.api_key = settings.surfer_seo_api_key
        self.base_url = "https://api.surferseo.com/v1"
        self.timeout = 60.0
    
    async def analyze_content(
        self,
        content: str,
        target_keyword: str,
        language: str = "en",
        location: str = "US"
    ) -> Dict[str, Any]:
        """
        Analyze content for SEO optimization
        
        Args:
            content: The content to analyze
            target_keyword: The target keyword for optimization
            language: Language code (e.g., "en", "es", "fr")
            location: Location code (e.g., "US", "GB", "CA")
            
        Returns:
            Content analysis with SEO recommendations
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/content/analyze"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "content": content,
                    "target_keyword": target_keyword,
                    "language": language,
                    "location": location,
                    "include_keyword_density": True,
                    "include_readability": True,
                    "include_semantic_keywords": True
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_content_analysis(data, target_keyword)
                
        except Exception as e:
            logger.error(f"SurferSEO content analysis API error: {e}")
            return {"error": str(e), "target_keyword": target_keyword}
    
    async def get_keyword_suggestions(
        self,
        seed_keyword: str,
        language: str = "en",
        location: str = "US",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get keyword suggestions based on a seed keyword
        
        Args:
            seed_keyword: The seed keyword to generate suggestions from
            language: Language code
            location: Location code
            limit: Maximum number of suggestions
            
        Returns:
            List of keyword suggestions with metrics
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/keywords/suggestions"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "keyword": seed_keyword,
                    "language": language,
                    "location": location,
                    "limit": limit,
                    "include_metrics": True,
                    "include_competition": True
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_keyword_suggestions(data)
                
        except Exception as e:
            logger.error(f"SurferSEO keyword suggestions API error: {e}")
            return []
    
    async def get_content_planner(
        self,
        target_keyword: str,
        language: str = "en",
        location: str = "US"
    ) -> Dict[str, Any]:
        """
        Get content planning suggestions for a target keyword
        
        Args:
            target_keyword: The target keyword to plan content for
            language: Language code
            location: Location code
            
        Returns:
            Content planning recommendations
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/content/planner"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "keyword": target_keyword,
                    "language": language,
                    "location": location,
                    "include_outline": True,
                    "include_questions": True,
                    "include_related_topics": True
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_content_planner(data, target_keyword)
                
        except Exception as e:
            logger.error(f"SurferSEO content planner API error: {e}")
            return {"error": str(e), "target_keyword": target_keyword}
    
    async def get_serp_analyzer(
        self,
        keyword: str,
        language: str = "en",
        location: str = "US",
        depth: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze SERP for a keyword
        
        Args:
            keyword: The keyword to analyze SERP for
            language: Language code
            location: Location code
            depth: Number of results to analyze
            
        Returns:
            SERP analysis with optimization insights
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/serp/analyze"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "keyword": keyword,
                    "language": language,
                    "location": location,
                    "depth": depth,
                    "include_competitors": True,
                    "include_metrics": True
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_serp_analyzer(data, keyword)
                
        except Exception as e:
            logger.error(f"SurferSEO SERP analyzer API error: {e}")
            return {"error": str(e), "keyword": keyword}
    
    async def get_audit_report(
        self,
        url: str,
        language: str = "en",
        location: str = "US"
    ) -> Dict[str, Any]:
        """
        Get SEO audit report for a URL
        
        Args:
            url: The URL to audit
            language: Language code
            location: Location code
            
        Returns:
            SEO audit report with recommendations
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url_endpoint = f"{self.base_url}/audit/report"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "url": url,
                    "language": language,
                    "location": location,
                    "include_technical_seo": True,
                    "include_content_seo": True,
                    "include_off_page_seo": True
                }
                
                response = await client.post(url_endpoint, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_audit_report(data, url)
                
        except Exception as e:
            logger.error(f"SurferSEO audit report API error: {e}")
            return {"error": str(e), "url": url}
    
    def _process_content_analysis(self, data: Dict[str, Any], target_keyword: str) -> Dict[str, Any]:
        """Process content analysis response"""
        analysis = {
            "target_keyword": target_keyword,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_score": 0,
            "keyword_density": {},
            "readability": {},
            "semantic_keywords": [],
            "recommendations": [],
            "technical_issues": []
        }
        
        if "analysis" in data:
            analysis_data = data["analysis"]
            
            # Overall score
            analysis["overall_score"] = analysis_data.get("overall_score", 0)
            
            # Keyword density
            if "keyword_density" in analysis_data:
                analysis["keyword_density"] = {
                    "target_keyword": analysis_data["keyword_density"].get("target_keyword", 0),
                    "semantic_keywords": analysis_data["keyword_density"].get("semantic_keywords", {}),
                    "recommended_density": analysis_data["keyword_density"].get("recommended_density", 0)
                }
            
            # Readability
            if "readability" in analysis_data:
                analysis["readability"] = {
                    "score": analysis_data["readability"].get("score", 0),
                    "grade_level": analysis_data["readability"].get("grade_level", 0),
                    "sentences": analysis_data["readability"].get("sentences", 0),
                    "words": analysis_data["readability"].get("words", 0),
                    "paragraphs": analysis_data["readability"].get("paragraphs", 0)
                }
            
            # Semantic keywords
            if "semantic_keywords" in analysis_data:
                analysis["semantic_keywords"] = analysis_data["semantic_keywords"]
            
            # Recommendations
            if "recommendations" in analysis_data:
                analysis["recommendations"] = analysis_data["recommendations"]
            
            # Technical issues
            if "technical_issues" in analysis_data:
                analysis["technical_issues"] = analysis_data["technical_issues"]
        
        return analysis
    
    def _process_keyword_suggestions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process keyword suggestions response"""
        suggestions = []
        
        if "suggestions" in data:
            for suggestion in data["suggestions"]:
                suggestions.append({
                    "keyword": suggestion.get("keyword", ""),
                    "search_volume": suggestion.get("search_volume", 0),
                    "difficulty": suggestion.get("difficulty", 0),
                    "cpc": suggestion.get("cpc", 0),
                    "competition": suggestion.get("competition", "UNKNOWN"),
                    "trend": suggestion.get("trend", "stable"),
                    "created_at": datetime.utcnow().isoformat()
                })
        
        return suggestions
    
    def _process_content_planner(self, data: Dict[str, Any], target_keyword: str) -> Dict[str, Any]:
        """Process content planner response"""
        planner = {
            "target_keyword": target_keyword,
            "timestamp": datetime.utcnow().isoformat(),
            "outline": [],
            "questions": [],
            "related_topics": [],
            "content_ideas": []
        }
        
        if "planner" in data:
            planner_data = data["planner"]
            
            # Content outline
            if "outline" in planner_data:
                planner["outline"] = planner_data["outline"]
            
            # Questions to answer
            if "questions" in planner_data:
                planner["questions"] = planner_data["questions"]
            
            # Related topics
            if "related_topics" in planner_data:
                planner["related_topics"] = planner_data["related_topics"]
            
            # Content ideas
            if "content_ideas" in planner_data:
                planner["content_ideas"] = planner_data["content_ideas"]
        
        return planner
    
    def _process_serp_analyzer(self, data: Dict[str, Any], keyword: str) -> Dict[str, Any]:
        """Process SERP analyzer response"""
        analysis = {
            "keyword": keyword,
            "timestamp": datetime.utcnow().isoformat(),
            "serp_results": [],
            "competitors": [],
            "metrics": {},
            "insights": []
        }
        
        if "serp_analysis" in data:
            serp_data = data["serp_analysis"]
            
            # SERP results
            if "results" in serp_data:
                for result in serp_data["results"]:
                    analysis["serp_results"].append({
                        "position": result.get("position", 0),
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("description", ""),
                        "domain": result.get("domain", ""),
                        "title_length": len(result.get("title", "")),
                        "description_length": len(result.get("description", ""))
                    })
            
            # Competitors
            if "competitors" in serp_data:
                analysis["competitors"] = serp_data["competitors"]
            
            # Metrics
            if "metrics" in serp_data:
                analysis["metrics"] = serp_data["metrics"]
            
            # Insights
            if "insights" in serp_data:
                analysis["insights"] = serp_data["insights"]
        
        return analysis
    
    def _process_audit_report(self, data: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Process audit report response"""
        report = {
            "url": url,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_score": 0,
            "technical_seo": {},
            "content_seo": {},
            "off_page_seo": {},
            "issues": [],
            "recommendations": []
        }
        
        if "audit" in data:
            audit_data = data["audit"]
            
            # Overall score
            report["overall_score"] = audit_data.get("overall_score", 0)
            
            # Technical SEO
            if "technical_seo" in audit_data:
                report["technical_seo"] = audit_data["technical_seo"]
            
            # Content SEO
            if "content_seo" in audit_data:
                report["content_seo"] = audit_data["content_seo"]
            
            # Off-page SEO
            if "off_page_seo" in audit_data:
                report["off_page_seo"] = audit_data["off_page_seo"]
            
            # Issues
            if "issues" in audit_data:
                report["issues"] = audit_data["issues"]
            
            # Recommendations
            if "recommendations" in audit_data:
                report["recommendations"] = audit_data["recommendations"]
        
        return report

# Global instance
surfer_seo_api = SurferSEOAPI()

# Convenience functions
async def analyze_content(
    content: str,
    target_keyword: str,
    language: str = "en",
    location: str = "US"
) -> Dict[str, Any]:
    """Analyze content for SEO optimization"""
    return await surfer_seo_api.analyze_content(content, target_keyword, language, location)

async def get_keyword_suggestions(
    seed_keyword: str,
    language: str = "en",
    location: str = "US",
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Get keyword suggestions based on a seed keyword"""
    return await surfer_seo_api.get_keyword_suggestions(seed_keyword, language, location, limit)

async def get_content_planner(
    target_keyword: str,
    language: str = "en",
    location: str = "US"
) -> Dict[str, Any]:
    """Get content planning suggestions for a target keyword"""
    return await surfer_seo_api.get_content_planner(target_keyword, language, location)

async def get_serp_analyzer(
    keyword: str,
    language: str = "en",
    location: str = "US",
    depth: int = 10
) -> Dict[str, Any]:
    """Analyze SERP for a keyword"""
    return await surfer_seo_api.get_serp_analyzer(keyword, language, location, depth)

async def get_audit_report(
    url: str,
    language: str = "en",
    location: str = "US"
) -> Dict[str, Any]:
    """Get SEO audit report for a URL"""
    return await surfer_seo_api.get_audit_report(url, language, location)
