"""
DataForSEO API Integration
Provides keyword research and SEO data capabilities
"""

import httpx
import base64
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class DataForSEOAPI:
    """DataForSEO API client for keyword research and SEO data"""
    
    def __init__(self):
        self.api_login = settings.dataforseo_username
        self.api_password = settings.dataforseo_password
        self.base_url = "https://api.dataforseo.com/v3"
        self.timeout = 60.0
        
        # Create basic auth header
        credentials = f"{self.api_login}:{self.api_password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        self.auth_header = f"Basic {encoded_credentials}"
    
    async def get_keyword_ideas(
        self,
        seed_keyword: str,
        language_code: str = "en",
        location_code: int = 2840,  # US
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get keyword ideas based on a seed keyword
        
        Args:
            seed_keyword: The seed keyword to generate ideas from
            language_code: Language code (e.g., "en", "es", "fr")
            location_code: Location code (2840 = US, 2826 = UK, etc.)
            limit: Maximum number of keywords to return
            
        Returns:
            List of keyword ideas with metrics
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/keywords_data/google_ads/keywords_for_keywords/live"
                
                payload = [{
                    "keyword": seed_keyword,
                    "language_code": language_code,
                    "location_code": location_code,
                    "limit": limit,
                    "filters": [
                        ["keyword_info.search_volume", ">", 100],
                        ["keyword_info.competition", "in", ["LOW", "MEDIUM", "HIGH"]]
                    ],
                    "order_by": ["keyword_info.search_volume,desc"]
                }]
                
                headers = {
                    "Authorization": self.auth_header,
                    "Content-Type": "application/json"
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_keyword_ideas(data)
                
        except Exception as e:
            logger.error(f"DataForSEO keyword ideas API error: {e}")
            return []
    
    async def get_keyword_metrics(
        self,
        keywords: List[str],
        language_code: str = "en",
        location_code: int = 2840
    ) -> List[Dict[str, Any]]:
        """
        Get detailed metrics for specific keywords
        
        Args:
            keywords: List of keywords to get metrics for
            language_code: Language code
            location_code: Location code
            
        Returns:
            List of keyword metrics
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/keywords_data/google_ads/search_volume/live"
                
                payload = [{
                    "keywords": keywords,
                    "language_code": language_code,
                    "location_code": location_code
                }]
                
                headers = {
                    "Authorization": self.auth_header,
                    "Content-Type": "application/json"
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_keyword_metrics(data)
                
        except Exception as e:
            logger.error(f"DataForSEO keyword metrics API error: {e}")
            return []
    
    async def get_serp_analysis(
        self,
        keyword: str,
        language_code: str = "en",
        location_code: int = 2840,
        depth: int = 10
    ) -> Dict[str, Any]:
        """
        Get SERP analysis for a keyword
        
        Args:
            keyword: The keyword to analyze
            language_code: Language code
            location_code: Location code
            depth: Number of results to analyze
            
        Returns:
            SERP analysis data
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/serp/google/organic/live/advanced"
                
                payload = [{
                    "keyword": keyword,
                    "language_code": language_code,
                    "location_code": location_code,
                    "depth": depth,
                    "calculate_rectangles": True
                }]
                
                headers = {
                    "Authorization": self.auth_header,
                    "Content-Type": "application/json"
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_serp_analysis(data, keyword)
                
        except Exception as e:
            logger.error(f"DataForSEO SERP analysis API error: {e}")
            return {"error": str(e)}
    
    async def get_related_keywords(
        self,
        keyword: str,
        language_code: str = "en",
        location_code: int = 2840,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get related keywords for a given keyword
        
        Args:
            keyword: The keyword to get related keywords for
            language_code: Language code
            location_code: Location code
            limit: Maximum number of related keywords
            
        Returns:
            List of related keywords
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/keywords_data/google_ads/keywords_for_keywords/live"
                
                payload = [{
                    "keyword": keyword,
                    "language_code": language_code,
                    "location_code": location_code,
                    "limit": limit,
                    "filters": [
                        ["keyword_info.search_volume", ">", 10]
                    ],
                    "order_by": ["keyword_info.search_volume,desc"]
                }]
                
                headers = {
                    "Authorization": self.auth_header,
                    "Content-Type": "application/json"
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_related_keywords(data)
                
        except Exception as e:
            logger.error(f"DataForSEO related keywords API error: {e}")
            return []
    
    async def get_keyword_difficulty(
        self,
        keywords: List[str],
        language_code: str = "en",
        location_code: int = 2840
    ) -> List[Dict[str, Any]]:
        """
        Get keyword difficulty scores
        
        Args:
            keywords: List of keywords to analyze
            language_code: Language code
            location_code: Location code
            
        Returns:
            List of keyword difficulty scores
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/keywords_data/google_ads/keyword_difficulty/live"
                
                payload = [{
                    "keywords": keywords,
                    "language_code": language_code,
                    "location_code": location_code
                }]
                
                headers = {
                    "Authorization": self.auth_header,
                    "Content-Type": "application/json"
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                return self._process_keyword_difficulty(data)
                
        except Exception as e:
            logger.error(f"DataForSEO keyword difficulty API error: {e}")
            return []
    
    def _process_keyword_ideas(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process keyword ideas response"""
        keywords = []
        
        if "tasks" in data and len(data["tasks"]) > 0:
            task = data["tasks"][0]
            if "result" in task and len(task["result"]) > 0:
                for item in task["result"]:
                    keyword_info = item.get("keyword_info", {})
                    keyword_data = item.get("keyword_data", {})
                    
                    keywords.append({
                        "keyword": item.get("keyword", ""),
                        "search_volume": keyword_info.get("search_volume", 0),
                        "competition": keyword_info.get("competition", "UNKNOWN"),
                        "competition_level": keyword_info.get("competition_level", 0),
                        "cpc": keyword_data.get("cpc", 0),
                        "monthly_searches": keyword_data.get("monthly_searches", []),
                        "keyword_difficulty": keyword_data.get("keyword_difficulty", 0),
                        "created_at": datetime.utcnow().isoformat()
                    })
        
        return keywords
    
    def _process_keyword_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process keyword metrics response"""
        metrics = []
        
        if "tasks" in data and len(data["tasks"]) > 0:
            task = data["tasks"][0]
            if "result" in task and len(task["result"]) > 0:
                for item in task["result"]:
                    metrics.append({
                        "keyword": item.get("keyword", ""),
                        "search_volume": item.get("search_volume", 0),
                        "competition": item.get("competition", "UNKNOWN"),
                        "competition_level": item.get("competition_level", 0),
                        "cpc": item.get("cpc", 0),
                        "monthly_searches": item.get("monthly_searches", []),
                        "created_at": datetime.utcnow().isoformat()
                    })
        
        return metrics
    
    def _process_serp_analysis(self, data: Dict[str, Any], keyword: str) -> Dict[str, Any]:
        """Process SERP analysis response"""
        analysis = {
            "keyword": keyword,
            "timestamp": datetime.utcnow().isoformat(),
            "organic_results": [],
            "paid_results": [],
            "related_searches": [],
            "people_also_ask": [],
            "summary": {
                "total_organic_results": 0,
                "total_paid_results": 0,
                "avg_title_length": 0,
                "avg_description_length": 0
            }
        }
        
        if "tasks" in data and len(data["tasks"]) > 0:
            task = data["tasks"][0]
            if "result" in task and len(task["result"]) > 0:
                result = task["result"][0]
                
                # Process organic results
                if "items" in result:
                    for item in result["items"]:
                        if item.get("type") == "organic":
                            analysis["organic_results"].append({
                                "position": item.get("rank_group", 0),
                                "title": item.get("title", ""),
                                "description": item.get("description", ""),
                                "url": item.get("url", ""),
                                "domain": item.get("domain", ""),
                                "title_length": len(item.get("title", "")),
                                "description_length": len(item.get("description", ""))
                            })
                        elif item.get("type") == "paid":
                            analysis["paid_results"].append({
                                "position": item.get("rank_group", 0),
                                "title": item.get("title", ""),
                                "description": item.get("description", ""),
                                "url": item.get("url", ""),
                                "domain": item.get("domain", "")
                            })
                        elif item.get("type") == "related_searches":
                            if "items" in item:
                                for related_item in item["items"]:
                                    analysis["related_searches"].append({
                                        "keyword": related_item.get("keyword", ""),
                                        "search_volume": related_item.get("search_volume", 0)
                                    })
                        elif item.get("type") == "people_also_ask":
                            if "items" in item:
                                for paa_item in item["items"]:
                                    analysis["people_also_ask"].append({
                                        "question": paa_item.get("question", ""),
                                        "answer": paa_item.get("answer", "")
                                    })
                
                # Calculate summary statistics
                organic_results = analysis["organic_results"]
                if organic_results:
                    analysis["summary"]["total_organic_results"] = len(organic_results)
                    analysis["summary"]["avg_title_length"] = sum(
                        r["title_length"] for r in organic_results
                    ) / len(organic_results)
                    analysis["summary"]["avg_description_length"] = sum(
                        r["description_length"] for r in organic_results
                    ) / len(organic_results)
                
                analysis["summary"]["total_paid_results"] = len(analysis["paid_results"])
        
        return analysis
    
    def _process_related_keywords(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process related keywords response"""
        keywords = []
        
        if "tasks" in data and len(data["tasks"]) > 0:
            task = data["tasks"][0]
            if "result" in task and len(task["result"]) > 0:
                for item in task["result"]:
                    keyword_info = item.get("keyword_info", {})
                    keyword_data = item.get("keyword_data", {})
                    
                    keywords.append({
                        "keyword": item.get("keyword", ""),
                        "search_volume": keyword_info.get("search_volume", 0),
                        "competition": keyword_info.get("competition", "UNKNOWN"),
                        "competition_level": keyword_info.get("competition_level", 0),
                        "cpc": keyword_data.get("cpc", 0),
                        "keyword_difficulty": keyword_data.get("keyword_difficulty", 0),
                        "created_at": datetime.utcnow().isoformat()
                    })
        
        return keywords
    
    def _process_keyword_difficulty(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process keyword difficulty response"""
        difficulties = []
        
        if "tasks" in data and len(data["tasks"]) > 0:
            task = data["tasks"][0]
            if "result" in task and len(task["result"]) > 0:
                for item in task["result"]:
                    difficulties.append({
                        "keyword": item.get("keyword", ""),
                        "keyword_difficulty": item.get("keyword_difficulty", 0),
                        "competition": item.get("competition", "UNKNOWN"),
                        "competition_level": item.get("competition_level", 0),
                        "search_volume": item.get("search_volume", 0),
                        "cpc": item.get("cpc", 0),
                        "created_at": datetime.utcnow().isoformat()
                    })
        
        return difficulties

# Global instance
dataforseo_api = DataForSEOAPI()

# Convenience functions
async def get_keyword_ideas(seed_keyword: str, language_code: str = "en", location_code: int = 2840, limit: int = 100) -> List[Dict[str, Any]]:
    """Get keyword ideas based on a seed keyword"""
    return await dataforseo_api.get_keyword_ideas(seed_keyword, language_code, location_code, limit)

async def get_keyword_metrics(keywords: List[str], language_code: str = "en", location_code: int = 2840) -> List[Dict[str, Any]]:
    """Get detailed metrics for specific keywords"""
    return await dataforseo_api.get_keyword_metrics(keywords, language_code, location_code)

async def get_serp_analysis(keyword: str, language_code: str = "en", location_code: int = 2840, depth: int = 10) -> Dict[str, Any]:
    """Get SERP analysis for a keyword"""
    return await dataforseo_api.get_serp_analysis(keyword, language_code, location_code, depth)

async def get_related_keywords(keyword: str, language_code: str = "en", location_code: int = 2840, limit: int = 50) -> List[Dict[str, Any]]:
    """Get related keywords for a given keyword"""
    return await dataforseo_api.get_related_keywords(keyword, language_code, location_code, limit)

async def get_keyword_difficulty(keywords: List[str], language_code: str = "en", location_code: int = 2840) -> List[Dict[str, Any]]:
    """Get keyword difficulty scores"""
    return await dataforseo_api.get_keyword_difficulty(keywords, language_code, location_code)
