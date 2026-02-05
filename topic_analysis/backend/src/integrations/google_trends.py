"""
Google Trends API Integration
Provides trend data fetching and analysis capabilities
"""

import httpx
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class GoogleTrendsAPI:
    """Google Trends API client for fetching trend data"""
    
    def __init__(self):
        self.api_key = settings.GOOGLE_TRENDS_API_KEY
        self.base_url = "https://trends.googleapis.com/trends/api"
        self.timeout = 30.0
        
    async def get_trend_data(
        self,
        keyword: str,
        geo: str = "US",
        timeframe: str = "today 12-m",
        category: int = 0
    ) -> Dict[str, Any]:
        """
        Fetch trend data for a specific keyword
        
        Args:
            keyword: The keyword to search for
            geo: Geographic location (e.g., "US", "GB", "CA")
            timeframe: Time period for data (e.g., "today 12-m", "today 5-y")
            category: Category ID (0 = all categories)
            
        Returns:
            Dict containing trend data
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Google Trends API endpoint
                url = f"{self.base_url}/explore"
                
                params = {
                    "hl": "en-US",
                    "tz": "-480",
                    "req": {
                        "comparisonItem": [{
                            "keyword": keyword,
                            "geo": geo,
                            "time": timeframe
                        }],
                        "category": category,
                        "property": ""
                    },
                    "keywordType": "QUERY",
                    "metric": ["TOP", "RISING"],
                    "granularTimeUnit": "month",
                    "restrictedCountryCode": geo
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                response = await client.post(url, json=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                # Process the response data
                processed_data = self._process_trend_data(data, keyword)
                
                logger.info(f"Successfully fetched trend data for keyword: {keyword}")
                return processed_data
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching trend data for {keyword}: {e}")
            raise Exception(f"Failed to fetch trend data: {e}")
        except Exception as e:
            logger.error(f"Error fetching trend data for {keyword}: {e}")
            raise Exception(f"Trend data fetch failed: {e}")
    
    async def get_related_queries(
        self,
        keyword: str,
        geo: str = "US"
    ) -> List[Dict[str, Any]]:
        """
        Get related queries for a keyword
        
        Args:
            keyword: The keyword to get related queries for
            geo: Geographic location
            
        Returns:
            List of related queries with their search volumes
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/relatedqueries"
                
                params = {
                    "hl": "en-US",
                    "tz": "-480",
                    "req": {
                        "comparisonItem": [{
                            "keyword": keyword,
                            "geo": geo,
                            "time": "today 12-m"
                        }],
                        "category": 0,
                        "property": ""
                    }
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                response = await client.post(url, json=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                # Process related queries
                related_queries = self._process_related_queries(data)
                
                logger.info(f"Successfully fetched {len(related_queries)} related queries for {keyword}")
                return related_queries
                
        except Exception as e:
            logger.error(f"Error fetching related queries for {keyword}: {e}")
            raise Exception(f"Related queries fetch failed: {e}")
    
    async def get_interest_over_time(
        self,
        keywords: List[str],
        geo: str = "US",
        timeframe: str = "today 12-m"
    ) -> Dict[str, Any]:
        """
        Get interest over time for multiple keywords
        
        Args:
            keywords: List of keywords to compare
            geo: Geographic location
            timeframe: Time period for data
            
        Returns:
            Dict containing interest over time data
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/explore"
                
                # Create comparison items for multiple keywords
                comparison_items = []
                for keyword in keywords:
                    comparison_items.append({
                        "keyword": keyword,
                        "geo": geo,
                        "time": timeframe
                    })
                
                params = {
                    "hl": "en-US",
                    "tz": "-480",
                    "req": {
                        "comparisonItem": comparison_items,
                        "category": 0,
                        "property": ""
                    },
                    "keywordType": "QUERY",
                    "metric": ["TIMESERIES"],
                    "granularTimeUnit": "month"
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                response = await client.post(url, json=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                # Process interest over time data
                interest_data = self._process_interest_over_time(data, keywords)
                
                logger.info(f"Successfully fetched interest over time for {len(keywords)} keywords")
                return interest_data
                
        except Exception as e:
            logger.error(f"Error fetching interest over time: {e}")
            raise Exception(f"Interest over time fetch failed: {e}")
    
    def _process_trend_data(self, raw_data: Dict[str, Any], keyword: str) -> Dict[str, Any]:
        """Process raw Google Trends API response"""
        try:
            processed = {
                "keyword": keyword,
                "timestamp": datetime.utcnow().isoformat(),
                "interest_over_time": [],
                "related_queries": [],
                "rising_queries": [],
                "top_queries": [],
                "summary": {
                    "average_interest": 0,
                    "peak_interest": 0,
                    "trend_direction": "stable",
                    "volatility": 0
                }
            }
            
            # Extract interest over time data
            if "default" in raw_data and "timelineData" in raw_data["default"]:
                timeline_data = raw_data["default"]["timelineData"]
                interest_values = []
                
                for item in timeline_data:
                    interest_value = item.get("value", [0])[0] if item.get("value") else 0
                    processed["interest_over_time"].append({
                        "date": item.get("formattedTime", ""),
                        "value": interest_value
                    })
                    interest_values.append(interest_value)
                
                # Calculate summary statistics
                if interest_values:
                    processed["summary"]["average_interest"] = sum(interest_values) / len(interest_values)
                    processed["summary"]["peak_interest"] = max(interest_values)
                    
                    # Determine trend direction
                    if len(interest_values) >= 2:
                        recent_avg = sum(interest_values[-3:]) / min(3, len(interest_values))
                        older_avg = sum(interest_values[:3]) / min(3, len(interest_values))
                        
                        if recent_avg > older_avg * 1.1:
                            processed["summary"]["trend_direction"] = "rising"
                        elif recent_avg < older_avg * 0.9:
                            processed["summary"]["trend_direction"] = "falling"
                    
                    # Calculate volatility (standard deviation)
                    variance = sum((x - processed["summary"]["average_interest"]) ** 2 for x in interest_values) / len(interest_values)
                    processed["summary"]["volatility"] = variance ** 0.5
            
            # Extract related queries
            if "default" in raw_data and "relatedQueries" in raw_data["default"]:
                related_queries = raw_data["default"]["relatedQueries"]
                if "rankedList" in related_queries:
                    for ranked_list in related_queries["rankedList"]:
                        if "rankedKeyword" in ranked_list:
                            for item in ranked_list["rankedKeyword"]:
                                query_data = {
                                    "query": item.get("query", ""),
                                    "value": item.get("value", 0),
                                    "formatted_value": item.get("formattedValue", "0")
                                }
                                
                                if ranked_list.get("rankedKeyword", [{}])[0].get("link") == "RISING":
                                    processed["rising_queries"].append(query_data)
                                else:
                                    processed["top_queries"].append(query_data)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing trend data: {e}")
            return {
                "keyword": keyword,
                "timestamp": datetime.utcnow().isoformat(),
                "error": f"Failed to process trend data: {e}",
                "interest_over_time": [],
                "related_queries": [],
                "rising_queries": [],
                "top_queries": [],
                "summary": {
                    "average_interest": 0,
                    "peak_interest": 0,
                    "trend_direction": "unknown",
                    "volatility": 0
                }
            }
    
    def _process_related_queries(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process related queries data"""
        try:
            related_queries = []
            
            if "default" in raw_data and "relatedQueries" in raw_data["default"]:
                related_queries_data = raw_data["default"]["relatedQueries"]
                if "rankedList" in related_queries_data:
                    for ranked_list in related_queries_data["rankedList"]:
                        if "rankedKeyword" in ranked_list:
                            for item in ranked_list["rankedKeyword"]:
                                related_queries.append({
                                    "query": item.get("query", ""),
                                    "value": item.get("value", 0),
                                    "formatted_value": item.get("formattedValue", "0"),
                                    "link": item.get("link", "")
                                })
            
            return related_queries
            
        except Exception as e:
            logger.error(f"Error processing related queries: {e}")
            return []
    
    def _process_interest_over_time(self, raw_data: Dict[str, Any], keywords: List[str]) -> Dict[str, Any]:
        """Process interest over time data for multiple keywords"""
        try:
            processed = {
                "keywords": keywords,
                "timestamp": datetime.utcnow().isoformat(),
                "timeline": [],
                "comparison": {}
            }
            
            if "default" in raw_data and "timelineData" in raw_data["default"]:
                timeline_data = raw_data["default"]["timelineData"]
                
                for item in timeline_data:
                    time_point = {
                        "date": item.get("formattedTime", ""),
                        "values": {}
                    }
                    
                    if "value" in item and len(item["value"]) == len(keywords):
                        for i, keyword in enumerate(keywords):
                            time_point["values"][keyword] = item["value"][i]
                    
                    processed["timeline"].append(time_point)
                
                # Calculate comparison metrics
                for keyword in keywords:
                    values = [point["values"].get(keyword, 0) for point in processed["timeline"]]
                    if values:
                        processed["comparison"][keyword] = {
                            "average": sum(values) / len(values),
                            "peak": max(values),
                            "total": sum(values)
                        }
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing interest over time: {e}")
            return {
                "keywords": keywords,
                "timestamp": datetime.utcnow().isoformat(),
                "timeline": [],
                "comparison": {},
                "error": f"Failed to process interest over time: {e}"
            }

# Global instance - lazy initialization
google_trends_api = None

def get_google_trends_api():
    """Get Google Trends API instance with lazy initialization"""
    global google_trends_api
    if google_trends_api is None:
        google_trends_api = GoogleTrendsAPI()
    return google_trends_api

# Convenience functions
async def get_trend_data(keyword: str, geo: str = "US", timeframe: str = "today 12-m") -> Dict[str, Any]:
    """Get trend data for a keyword"""
    api = get_google_trends_api()
    return await api.get_trend_data(keyword, geo, timeframe)

async def get_related_queries(keyword: str, geo: str = "US") -> List[Dict[str, Any]]:
    """Get related queries for a keyword"""
    api = get_google_trends_api()
    return await api.get_related_queries(keyword, geo)

async def get_interest_over_time(keywords: List[str], geo: str = "US", timeframe: str = "today 12-m") -> Dict[str, Any]:
    """Get interest over time for multiple keywords"""
    api = get_google_trends_api()
    return await api.get_interest_over_time(keywords, geo, timeframe)
