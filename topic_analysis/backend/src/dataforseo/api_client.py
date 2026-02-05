"""
DataForSEO API client for trend analysis and keyword research.

This module provides the main API client for interacting with DataForSEO APIs.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import httpx
from httpx import AsyncClient, Response, HTTPStatusError, TimeoutException

from ..models.api_credentials import APICredentials
from ..models.trend_data import TrendData, TimeSeriesPoint, Demographics
from ..models.keyword_data import KeywordData, Trends, MonthlyDataPoint, CompetitionLevel, IntentType, TrendDirection

logger = logging.getLogger(__name__)


class DataForSEOError(Exception):
    """Base exception for DataForSEO API errors."""
    pass


class RateLimitError(DataForSEOError):
    """Exception raised when rate limit is exceeded."""
    pass


class QuotaExceededError(DataForSEOError):
    """Exception raised when quota is exceeded."""
    pass


class APIUnavailableError(DataForSEOError):
    """Exception raised when API is unavailable."""
    pass


class DataForSEOClient:
    """Client for DataForSEO API integration."""
    
    def __init__(self, credentials: APICredentials, timeout: int = 30):
        """
        Initialize DataForSEO client.
        
        Args:
            credentials: API credentials
            timeout: Request timeout in seconds
        """
        self.credentials = credentials
        self.timeout = timeout
        self.base_url = credentials.base_url
        self.api_key = credentials.key_value
        self.headers = {
            "Authorization": f"Basic {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Make HTTP request to DataForSEO API with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            retries: Number of retries
            
        Returns:
            API response data
            
        Raises:
            RateLimitError: When rate limit is exceeded
            QuotaExceededError: When quota is exceeded
            APIUnavailableError: When API is unavailable
        """
        url = f"{self.base_url}{endpoint}"
        
        async with AsyncClient(timeout=self.timeout) as client:
            for attempt in range(retries + 1):
                try:
                    if method.upper() == "GET":
                        response = await client.get(url, headers=self.headers, params=data)
                    else:
                        response = await client.request(
                            method, url, headers=self.headers, json=data
                        )
                    
                    response.raise_for_status()
                    return response.json()
                    
                except HTTPStatusError as e:
                    if e.response.status_code == 429:
                        if attempt < retries:
                            wait_time = 2 ** attempt + 1  # Exponential backoff
                            logger.warning(f"Rate limit exceeded, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        raise RateLimitError(f"Rate limit exceeded: {e.response.text}")
                    
                    elif e.response.status_code == 402:
                        raise QuotaExceededError(f"Quota exceeded: {e.response.text}")
                    
                    elif e.response.status_code >= 500:
                        if attempt < retries:
                            wait_time = 2 ** attempt + 1
                            logger.warning(f"Server error {e.response.status_code}, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        raise APIUnavailableError(f"API unavailable: {e.response.text}")
                    
                    else:
                        raise DataForSEOError(f"API error {e.response.status_code}: {e.response.text}")
                
                except TimeoutException:
                    if attempt < retries:
                        wait_time = 2 ** attempt + 1
                        logger.warning(f"Request timeout, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    raise APIUnavailableError("Request timeout")
                
                except Exception as e:
                    if attempt < retries:
                        wait_time = 2 ** attempt + 1
                        logger.warning(f"Unexpected error, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    raise DataForSEOError(f"Unexpected error: {e}")
    
    async def get_trends_data(
        self, 
        keywords: List[str], 
        location: str = "United States",
        time_range: str = "12m"
    ) -> Dict[str, Any]:
        """
        Get trend data for keywords.
        
        Args:
            keywords: List of keywords to analyze
            location: Geographic location
            time_range: Time range for analysis
            
        Returns:
            Trend data from DataForSEO API
        """
        endpoint = "/v3/dataforseo_trends/explore/live"
        
        data = {
            "keywords": keywords,
            "location_name": location,
            "date_from": self._get_date_from(time_range),
            "date_to": datetime.now().strftime("%Y-%m-%d")
        }
        
        return await self._make_request("POST", endpoint, data)
    
    async def get_subregion_interests(
        self, 
        keywords: List[str], 
        location: str = "United States"
    ) -> Dict[str, Any]:
        """
        Get subregion interests for keywords.
        
        Args:
            keywords: List of keywords to analyze
            location: Geographic location
            
        Returns:
            Subregion interests data from DataForSEO API
        """
        endpoint = "/v3/dataforseo_trends/subregion_interests/live"
        
        data = {
            "keywords": keywords,
            "location_name": location
        }
        
        return await self._make_request("POST", endpoint, data)
    
    async def get_keyword_ideas(
        self, 
        keywords: List[str], 
        language_code: str = "en",
        location_code: int = 2840  # United States
    ) -> Dict[str, Any]:
        """
        Get keyword ideas from DataForSEO Labs API.
        
        Args:
            keywords: List of seed keywords
            language_code: Language code
            location_code: Location code
            
        Returns:
            Keyword ideas data from DataForSEO API
        """
        endpoint = "/v3/dataforseo_labs/google/keyword_ideas/live"
        
        data = {
            "keywords": keywords,
            "language_code": language_code,
            "location_code": location_code,
            "limit": 1000
        }
        
        return await self._make_request("POST", endpoint, data)
    
    async def get_keyword_suggestions(
        self, 
        keywords: List[str], 
        language_code: str = "en",
        location_code: int = 2840  # United States
    ) -> Dict[str, Any]:
        """
        Get keyword suggestions from DataForSEO Labs API.
        
        Args:
            keywords: List of seed keywords
            language_code: Language code
            location_code: Location code
            
        Returns:
            Keyword suggestions data from DataForSEO API
        """
        endpoint = "/v3/dataforseo_labs/google/keyword_suggestions/live"
        
        data = {
            "keywords": keywords,
            "language_code": language_code,
            "location_code": location_code,
            "limit": 1000
        }
        
        return await self._make_request("POST", endpoint, data)
    
    def _get_date_from(self, time_range: str) -> str:
        """Get date_from based on time range."""
        now = datetime.now()
        
        if time_range == "1m":
            return (now.replace(month=now.month-1) if now.month > 1 else now.replace(year=now.year-1, month=12)).strftime("%Y-%m-%d")
        elif time_range == "3m":
            return (now.replace(month=now.month-3) if now.month > 3 else now.replace(year=now.year-1, month=now.month+9)).strftime("%Y-%m-%d")
        elif time_range == "6m":
            return (now.replace(month=now.month-6) if now.month > 6 else now.replace(year=now.year-1, month=now.month+6)).strftime("%Y-%m-%d")
        elif time_range == "24m":
            return (now.replace(year=now.year-2)).strftime("%Y-%m-%d")
        else:  # 12m default
            return (now.replace(year=now.year-1)).strftime("%Y-%m-%d")
    
    def _parse_trend_data(self, api_response: Dict[str, Any]) -> List[TrendData]:
        """Parse trend data from API response."""
        trend_data_list = []
        
        if "tasks" not in api_response:
            return trend_data_list
        
        for task in api_response["tasks"]:
            if "result" not in task:
                continue
                
            for result in task["result"]:
                keyword = result.get("keyword", "")
                location = result.get("location_name", "")
                
                # Parse time series data
                time_series = []
                if "items" in result:
                    for item in result["items"]                    :
                        time_series.append(TimeSeriesPoint(
                            date=item.get("date", ""),
                            value=item.get("value", 0)
                        ))
                
                # Parse demographics
                demographics = None
                if "demographics" in result:
                    demo_data = result["demographics"]
                    demographics = Demographics(
                        age_groups=demo_data.get("age_groups"),
                        gender=demo_data.get("gender")
                    )
                
                # Parse related queries
                related_queries = result.get("related_queries", [])
                
                trend_data = TrendData(
                    keyword=keyword,
                    location=location,
                    time_series=time_series,
                    demographics=demographics,
                    related_queries=related_queries
                )
                
                trend_data_list.append(trend_data)
        
        return trend_data_list
    
    def _parse_keyword_data(self, api_response: Dict[str, Any]) -> List[KeywordData]:
        """Parse keyword data from API response."""
        keyword_data_list = []
        
        if "tasks" not in api_response:
            return keyword_data_list
        
        for task in api_response["tasks"]:
            if "result" not in task:
                continue
                
            for result in task["result"]:
                keyword = result.get("keyword", "")
                search_volume = result.get("search_volume", 0)
                keyword_difficulty = result.get("keyword_difficulty", 0)
                cpc = result.get("cpc", 0)
                competition = result.get("competition", 0)
                
                # Parse competition level
                competition_level = None
                if competition is not None:
                    if competition < 0.33:
                        competition_level = CompetitionLevel.LOW
                    elif competition < 0.66:
                        competition_level = CompetitionLevel.MEDIUM
                    else:
                        competition_level = CompetitionLevel.HIGH
                
                # Parse trends
                trends = None
                if "trends" in result:
                    trend_data = result["trends"]
                    monthly_data = []
                    
                    if "monthly_data" in trend_data:
                        for month_data in trend_data["monthly_data"]:
                            monthly_data.append(MonthlyDataPoint(
                                month=month_data.get("month", ""),
                                volume=month_data.get("volume", 0)
                            ))
                    
                    trend_direction = None
                    if "trend_direction" in trend_data:
                        trend_direction = TrendDirection(trend_data["trend_direction"])
                    
                    trends = Trends(
                        monthly_data=monthly_data,
                        trend_direction=trend_direction,
                        trend_percentage=trend_data.get("trend_percentage")
                    )
                
                # Parse related keywords
                related_keywords = result.get("related_keywords", [])
                
                # Parse intent
                intent = None
                if "intent" in result:
                    intent = IntentType(result["intent"])
                
                keyword_data = KeywordData(
                    keyword=keyword,
                    search_volume=search_volume,
                    keyword_difficulty=keyword_difficulty,
                    cpc=cpc,
                    competition=competition,
                    competition_level=competition_level,
                    trends=trends,
                    related_keywords=related_keywords,
                    intent=intent
                )
                
                keyword_data_list.append(keyword_data)
        
        return keyword_data_list


async def get_api_credentials() -> APICredentials:
    """
    Get API credentials from Supabase.
    
    Returns:
        API credentials for DataForSEO
        
    Raises:
        DataForSEOError: If credentials cannot be retrieved
    """
    # This would be implemented to fetch from Supabase
    # For now, return a mock implementation
    return APICredentials(
        id="dataforseo_001",
        provider="dataforseo",
        base_url="https://api.dataforseo.com",
        key_value="your_api_key_here",
        is_active=True,
        rate_limit=1000,
        quota_used=0,
        quota_limit=10000
    )
