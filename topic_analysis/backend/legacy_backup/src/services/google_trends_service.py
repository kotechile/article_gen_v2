"""
Google Trends Service
Handles Google Trends API integration with fallback mechanisms
"""

import asyncio
import aiohttp
import json
import csv
import io
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import structlog
from sqlalchemy.orm import Session
from ..core.redis import cache
from ..services.api_key_service import APIKeyService

logger = structlog.get_logger()

class GoogleTrendsService:
    """Service for Google Trends data retrieval with multiple fallback options"""
    
    def __init__(self, db: Session):
        self.db = db
        self.api_key_service = APIKeyService(db)
        self.cache_ttl = 86400  # 24 hours cache TTL
        self.rate_limit_delay = 2  # 2 seconds between requests
        
        # API endpoints
        self.google_trends_url = "https://trends.google.com/trends/api"
        self.serpapi_url = "https://serpapi.com/search"
        self.dataforseo_url = "https://api.dataforseo.com/v3"
        
        # Request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    async def get_trend_data(
        self, 
        keywords: List[str], 
        user_id: str,
        timeframe: str = "12m",
        geo: str = "US",
        category: int = 0
    ) -> Dict[str, Any]:
        """
        Get trend data for keywords using multiple fallback methods
        
        Args:
            keywords: List of keywords to analyze
            user_id: User ID for API key retrieval
            timeframe: Time period (1h, 4h, 1d, 7d, 30d, 90d, 12m, 5y, all)
            geo: Geographic location (US, GB, etc.)
            category: Google Trends category ID
            
        Returns:
            Dict containing trend data for all keywords
        """
        try:
            logger.info("Starting trend data retrieval", 
                       keywords=keywords, 
                       user_id=user_id,
                       timeframe=timeframe,
                       geo=geo)
            
            # Check cache first
            cache_key = f"trends:{user_id}:{hash(tuple(keywords))}:{timeframe}:{geo}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info("Cache hit for trend data", cache_key=cache_key)
                return cached_result
            
            # Try multiple methods in order of preference
            result = None
            
            # Method 1: Google Trends Website API
            try:
                result = await self._get_google_trends_data(keywords, timeframe, geo, category)
                if result:
                    logger.info("Google Trends API successful", keywords=keywords)
            except Exception as e:
                logger.warning("Google Trends API failed", error=str(e))
            
            # Method 2: SerpAPI (if Google Trends fails)
            if not result:
                try:
                    result = await self._get_serpapi_data(keywords, user_id, timeframe, geo)
                    if result:
                        logger.info("SerpAPI successful", keywords=keywords)
                except Exception as e:
                    logger.warning("SerpAPI failed", error=str(e))
            
            # Method 3: DataForSEO (if SerpAPI fails)
            if not result:
                try:
                    result = await self._get_dataforseo_data(keywords, user_id, timeframe, geo)
                    if result:
                        logger.info("DataForSEO successful", keywords=keywords)
                except Exception as e:
                    logger.warning("DataForSEO failed", error=str(e))
            
            # Method 4: Fallback to mock data
            if not result:
                logger.warning("All APIs failed, using fallback data", keywords=keywords)
                result = self._get_fallback_data(keywords, timeframe, geo)
            
            # Cache the result
            await self._set_cached_result(cache_key, result)
            
            logger.info("Trend data retrieval completed", 
                       keywords=keywords, 
                       data_points=len(result.get("trends", [])))
            
            return result
            
        except Exception as e:
            logger.error("Failed to get trend data", 
                        keywords=keywords, 
                        user_id=user_id,
                        error=str(e))
            raise
    
    async def _get_google_trends_data(
        self, 
        keywords: List[str], 
        timeframe: str, 
        geo: str, 
        category: int
    ) -> Optional[Dict[str, Any]]:
        """Get data from Google Trends website API"""
        try:
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            # Prepare request parameters
            params = {
                'hl': 'en-US',
                'tz': '-480',
                'req': json.dumps({
                    'comparisonItem': [
                        {
                            'keyword': keyword,
                            'geo': geo,
                            'time': self._get_timeframe_string(timeframe)
                        } for keyword in keywords
                    ],
                    'category': category,
                    'property': ''
                }),
                'tz': '-480'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.google_trends_url}/explore",
                    params=params,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        # Parse the response (Google Trends returns JSONP)
                        if text.startswith(')]}\'\n'):
                            json_text = text[5:]  # Remove JSONP wrapper
                            data = json.loads(json_text)
                            return self._parse_google_trends_response(data, keywords)
                    else:
                        logger.warning("Google Trends API returned error", 
                                     status=response.status)
                        return None
                        
        except Exception as e:
            logger.error("Google Trends API error", error=str(e))
            return None
    
    async def _get_serpapi_data(
        self, 
        keywords: List[str], 
        user_id: str, 
        timeframe: str, 
        geo: str
    ) -> Optional[Dict[str, Any]]:
        """Get data from SerpAPI"""
        try:
            # Get SerpAPI key
            api_key = self.api_key_service.get_decrypted_api_key(user_id, "serpapi")
            if not api_key:
                logger.warning("No SerpAPI key found for user", user_id=user_id)
                return None
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            trends_data = []
            
            async with aiohttp.ClientSession() as session:
                for keyword in keywords:
                    params = {
                        'api_key': api_key,
                        'engine': 'google_trends',
                        'q': keyword,
                        'geo': geo,
                        'date': self._get_serpapi_timeframe(timeframe)
                    }
                    
                    async with session.get(
                        self.serpapi_url,
                        params=params,
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            trend_data = self._parse_serpapi_response(data, keyword)
                            if trend_data:
                                trends_data.append(trend_data)
                        else:
                            logger.warning("SerpAPI returned error", 
                                         status=response.status,
                                         keyword=keyword)
            
            if trends_data:
                return {
                    "trends": trends_data,
                    "source": "serpapi",
                    "timeframe": timeframe,
                    "geo": geo,
                    "retrieved_at": datetime.utcnow().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error("SerpAPI error", error=str(e))
            return None
    
    async def _get_dataforseo_data(
        self, 
        keywords: List[str], 
        user_id: str, 
        timeframe: str, 
        geo: str
    ) -> Optional[Dict[str, Any]]:
        """Get data from DataForSEO"""
        try:
            # Get DataForSEO credentials
            api_key = self.api_key_service.get_decrypted_api_key(user_id, "dataforseo")
            if not api_key:
                logger.warning("No DataForSEO key found for user", user_id=user_id)
                return None
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            trends_data = []
            
            async with aiohttp.ClientSession() as session:
                for keyword in keywords:
                    payload = {
                        "keywords": [keyword],
                        "location_name": geo,
                        "language_name": "English",
                        "date_from": self._get_dataforseo_date_from(timeframe),
                        "date_to": datetime.utcnow().strftime("%Y-%m-%d")
                    }
                    
                    headers = {
                        **self.headers,
                        'Authorization': f'Basic {api_key}'
                    }
                    
                    async with session.post(
                        f"{self.dataforseo_url}/keywords_data/google_trends/explore/live",
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            trend_data = self._parse_dataforseo_response(data, keyword)
                            if trend_data:
                                trends_data.append(trend_data)
                        else:
                            logger.warning("DataForSEO returned error", 
                                         status=response.status,
                                         keyword=keyword)
            
            if trends_data:
                return {
                    "trends": trends_data,
                    "source": "dataforseo",
                    "timeframe": timeframe,
                    "geo": geo,
                    "retrieved_at": datetime.utcnow().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error("DataForSEO error", error=str(e))
            return None
    
    def _get_fallback_data(self, keywords: List[str], timeframe: str, geo: str) -> Dict[str, Any]:
        """Generate fallback data when all APIs fail"""
        trends_data = []
        
        for i, keyword in enumerate(keywords):
            # Generate mock trend data
            base_volume = 1000 + (i * 500)
            trend_score = 50 + (i * 10)
            
            trends_data.append({
                "keyword": keyword,
                "search_volume": base_volume,
                "trend_score": trend_score,
                "growth_rate": (i % 3 - 1) * 10,  # -10, 0, or 10
                "seasonality": "stable",
                "related_queries": [
                    f"{keyword} guide",
                    f"{keyword} tips",
                    f"{keyword} review"
                ],
                "time_series": self._generate_mock_time_series(timeframe),
                "source": "fallback"
            })
        
        return {
            "trends": trends_data,
            "source": "fallback",
            "timeframe": timeframe,
            "geo": geo,
            "retrieved_at": datetime.utcnow().isoformat(),
            "note": "Fallback data - APIs unavailable"
        }
    
    def _parse_google_trends_response(self, data: Dict[str, Any], keywords: List[str]) -> Dict[str, Any]:
        """Parse Google Trends API response"""
        trends_data = []
        
        try:
            # Extract trend data from Google's response format
            if "default" in data and "timelineData" in data["default"]:
                timeline_data = data["default"]["timelineData"]
                
                for i, keyword in enumerate(keywords):
                    keyword_data = {
                        "keyword": keyword,
                        "search_volume": self._extract_search_volume(data, i),
                        "trend_score": self._extract_trend_score(timeline_data, i),
                        "growth_rate": self._calculate_growth_rate(timeline_data, i),
                        "seasonality": self._analyze_seasonality(timeline_data, i),
                        "related_queries": self._extract_related_queries(data, i),
                        "time_series": self._extract_time_series(timeline_data, i),
                        "source": "google_trends"
                    }
                    trends_data.append(keyword_data)
            
            return {
                "trends": trends_data,
                "source": "google_trends",
                "retrieved_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to parse Google Trends response", error=str(e))
            return None
    
    def _parse_serpapi_response(self, data: Dict[str, Any], keyword: str) -> Optional[Dict[str, Any]]:
        """Parse SerpAPI response"""
        try:
            if "interest_over_time" in data:
                interest_data = data["interest_over_time"]
                
                return {
                    "keyword": keyword,
                    "search_volume": interest_data.get("values", [{}])[0].get("value", 0),
                    "trend_score": self._calculate_trend_score(interest_data.get("values", [])),
                    "growth_rate": self._calculate_growth_rate_from_values(interest_data.get("values", [])),
                    "seasonality": "stable",
                    "related_queries": [q.get("query", "") for q in data.get("related_queries", [])[:5]],
                    "time_series": interest_data.get("values", []),
                    "source": "serpapi"
                }
            
            return None
            
        except Exception as e:
            logger.error("Failed to parse SerpAPI response", error=str(e))
            return None
    
    def _parse_dataforseo_response(self, data: Dict[str, Any], keyword: str) -> Optional[Dict[str, Any]]:
        """Parse DataForSEO response"""
        try:
            if "tasks" in data and len(data["tasks"]) > 0:
                task_data = data["tasks"][0]
                
                return {
                    "keyword": keyword,
                    "search_volume": task_data.get("search_volume", 0),
                    "trend_score": task_data.get("trend_score", 50),
                    "growth_rate": task_data.get("growth_rate", 0),
                    "seasonality": "stable",
                    "related_queries": task_data.get("related_queries", [])[:5],
                    "time_series": task_data.get("time_series", []),
                    "source": "dataforseo"
                }
            
            return None
            
        except Exception as e:
            logger.error("Failed to parse DataForSEO response", error=str(e))
            return None
    
    def _get_timeframe_string(self, timeframe: str) -> str:
        """Convert timeframe to Google Trends format"""
        timeframe_map = {
            "1h": "now 1-H",
            "4h": "now 4-H",
            "1d": "now 1-d",
            "7d": "now 7-d",
            "30d": "now 30-d",
            "90d": "now 3-m",
            "12m": "now 12-m",
            "5y": "now 5-y",
            "all": "all"
        }
        return timeframe_map.get(timeframe, "now 12-m")
    
    def _get_serpapi_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to SerpAPI format"""
        timeframe_map = {
            "1h": "now 1-H",
            "4h": "now 4-H",
            "1d": "now 1-d",
            "7d": "now 7-d",
            "30d": "now 30-d",
            "90d": "now 3-m",
            "12m": "now 12-m",
            "5y": "now 5-y",
            "all": "all"
        }
        return timeframe_map.get(timeframe, "now 12-m")
    
    def _get_dataforseo_date_from(self, timeframe: str) -> str:
        """Convert timeframe to DataForSEO date format"""
        now = datetime.utcnow()
        timeframe_map = {
            "1h": (now - timedelta(hours=1)).strftime("%Y-%m-%d"),
            "4h": (now - timedelta(hours=4)).strftime("%Y-%m-%d"),
            "1d": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
            "7d": (now - timedelta(days=7)).strftime("%Y-%m-%d"),
            "30d": (now - timedelta(days=30)).strftime("%Y-%m-%d"),
            "90d": (now - timedelta(days=90)).strftime("%Y-%m-%d"),
            "12m": (now - timedelta(days=365)).strftime("%Y-%m-%d"),
            "5y": (now - timedelta(days=1825)).strftime("%Y-%m-%d"),
            "all": "2015-01-01"
        }
        return timeframe_map.get(timeframe, (now - timedelta(days=365)).strftime("%Y-%m-%d"))
    
    def _extract_search_volume(self, data: Dict[str, Any], keyword_index: int) -> int:
        """Extract search volume from Google Trends data"""
        try:
            if "default" in data and "averages" in data["default"]:
                averages = data["default"]["averages"]
                if keyword_index < len(averages):
                    return averages[keyword_index]
            return 0
        except:
            return 0
    
    def _extract_trend_score(self, timeline_data: List[Dict], keyword_index: int) -> int:
        """Extract trend score from timeline data"""
        try:
            if timeline_data and keyword_index < len(timeline_data[0].get("value", [])):
                values = [point.get("value", [0])[keyword_index] for point in timeline_data[-7:]]
                return sum(values) // len(values) if values else 50
            return 50
        except:
            return 50
    
    def _calculate_growth_rate(self, timeline_data: List[Dict], keyword_index: int) -> float:
        """Calculate growth rate from timeline data"""
        try:
            if len(timeline_data) < 2:
                return 0.0
            
            recent_values = [point.get("value", [0])[keyword_index] for point in timeline_data[-7:]]
            older_values = [point.get("value", [0])[keyword_index] for point in timeline_data[-14:-7]]
            
            if not recent_values or not older_values:
                return 0.0
            
            recent_avg = sum(recent_values) / len(recent_values)
            older_avg = sum(older_values) / len(older_values)
            
            if older_avg == 0:
                return 0.0
            
            return ((recent_avg - older_avg) / older_avg) * 100
        except:
            return 0.0
    
    def _analyze_seasonality(self, timeline_data: List[Dict], keyword_index: int) -> str:
        """Analyze seasonality pattern"""
        try:
            if len(timeline_data) < 30:
                return "stable"
            
            values = [point.get("value", [0])[keyword_index] for point in timeline_data[-30:]]
            if not values:
                return "stable"
            
            # Simple seasonality detection
            variance = sum((v - sum(values)/len(values))**2 for v in values) / len(values)
            if variance < 100:
                return "stable"
            elif variance < 500:
                return "moderate"
            else:
                return "high"
        except:
            return "stable"
    
    def _extract_related_queries(self, data: Dict[str, Any], keyword_index: int) -> List[str]:
        """Extract related queries from Google Trends data"""
        try:
            if "default" in data and "relatedQueries" in data["default"]:
                queries = data["default"]["relatedQueries"]
                if keyword_index < len(queries):
                    return [q.get("query", "") for q in queries[keyword_index].get("value", [])[:5]]
            return []
        except:
            return []
    
    def _extract_time_series(self, timeline_data: List[Dict], keyword_index: int) -> List[Dict]:
        """Extract time series data"""
        try:
            time_series = []
            for point in timeline_data:
                if keyword_index < len(point.get("value", [])):
                    time_series.append({
                        "date": point.get("formattedTime", ""),
                        "value": point.get("value", [0])[keyword_index]
                    })
            return time_series
        except:
            return []
    
    def _calculate_trend_score(self, values: List[Dict]) -> int:
        """Calculate trend score from values"""
        try:
            if not values:
                return 50
            
            numeric_values = [v.get("value", 0) for v in values if isinstance(v.get("value"), (int, float))]
            if not numeric_values:
                return 50
            
            return sum(numeric_values) // len(numeric_values)
        except:
            return 50
    
    def _calculate_growth_rate_from_values(self, values: List[Dict]) -> float:
        """Calculate growth rate from values"""
        try:
            if len(values) < 2:
                return 0.0
            
            recent = values[-7:] if len(values) >= 7 else values[-len(values)//2:]
            older = values[-14:-7] if len(values) >= 14 else values[:len(values)//2]
            
            recent_avg = sum(v.get("value", 0) for v in recent) / len(recent)
            older_avg = sum(v.get("value", 0) for v in older) / len(older)
            
            if older_avg == 0:
                return 0.0
            
            return ((recent_avg - older_avg) / older_avg) * 100
        except:
            return 0.0
    
    def _generate_mock_time_series(self, timeframe: str) -> List[Dict]:
        """Generate mock time series data"""
        time_series = []
        now = datetime.utcnow()
        
        # Generate data points based on timeframe
        if timeframe == "1h":
            for i in range(12):  # 5-minute intervals
                time_series.append({
                    "date": (now - timedelta(minutes=i*5)).strftime("%Y-%m-%d %H:%M"),
                    "value": 50 + (i % 3) * 10
                })
        elif timeframe == "7d":
            for i in range(7):  # Daily
                time_series.append({
                    "date": (now - timedelta(days=i)).strftime("%Y-%m-%d"),
                    "value": 50 + (i % 3) * 10
                })
        else:  # Default to monthly
            for i in range(12):  # Monthly
                time_series.append({
                    "date": (now - timedelta(days=i*30)).strftime("%Y-%m-%d"),
                    "value": 50 + (i % 3) * 10
                })
        
        return time_series
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result from Redis"""
        try:
            cached_data = cache.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning("Failed to get cached result", cache_key=cache_key, error=str(e))
            return None
    
    async def _set_cached_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache result in Redis"""
        try:
            cache.setex(cache_key, self.cache_ttl, json.dumps(result))
        except Exception as e:
            logger.warning("Failed to cache result", cache_key=cache_key, error=str(e))
