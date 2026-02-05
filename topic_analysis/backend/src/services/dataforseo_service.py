"""
DataForSEO Service Implementation

A reusable service class for DataForSEO API that uses the 2-call approach
(POST to create task, GET to poll results) for both related keywords and 
keyword ideas endpoints. Supports both production and sandbox environments
by fetching credentials and base_url from Supabase.
"""

import asyncio
import base64
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import httpx
from httpx import AsyncClient, HTTPError, TimeoutException

from ..dataforseo.database import DataForSEORepository, DatabaseManager
from ..models.api_credentials import APICredentials

logger = logging.getLogger(__name__)


class DataForSEOService:
    """Service class for DataForSEO API with 2-call approach"""
    
    def __init__(self):
        self.base_url: Optional[str] = None
        self.username: Optional[str] = None
        self.password: Optional[str] = None
        self.client: Optional[AsyncClient] = None
        self.db_manager: Optional[DatabaseManager] = None
        self.repository: Optional[DataForSEORepository] = None
        
    async def initialize(self):
        """Initialize service with credentials from Supabase"""
        try:
            # Initialize database manager
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # Initialize repository
            self.repository = DataForSEORepository(self.db_manager)
            
            # Get credentials from Supabase
            credentials = await self.repository.get_api_credentials("dataforseo")
            
            if not credentials:
                raise ValueError("DataForSEO API credentials not found in database. Please add credentials to the 'api_keys' table in Supabase.")
            
            # Get raw credentials to ensure we use the exact values from Supabase
            raw_credentials = await self.repository.get_raw_api_credentials("dataforseo")
            if raw_credentials:
                self.base_url = raw_credentials.get("base_url")
                if not self.base_url:
                    raise ValueError("DataForSEO base_url not found in database")
                self.username = raw_credentials.get("user_name", "")
                self.password = raw_credentials.get("password", "")
                logger.info(f"DEBUG - Raw credentials from database: base_url={self.base_url}, username={self.username}")
                logger.info(f"Using raw credentials - Base URL: {self.base_url}, Username: {self.username}")
            else:
                # Fallback to structured credentials
                self.base_url = credentials.base_url
                self.username, self.password = self._decode_credentials(credentials.key_value)
                logger.info(f"Using structured credentials - Base URL: {self.base_url}")
            
            # Create HTTP client
            logger.info(f"DEBUG - Creating AsyncClient with base_url: {self.base_url}")
            logger.info(f"DEBUG - Auth credentials - Username: {self.username}")
            self.client = AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                auth=(self.username, self.password),
                headers={"Content-Type": "application/json"}
            )
            
            logger.info(f"DEBUG - AsyncClient created successfully")
            logger.info(f"DEBUG - Client base_url: {self.client.base_url}")
            logger.info(f"DEBUG - Client will make requests to: {self.client.base_url}")
            logger.info(f"DataForSEO service initialized - Base URL: {self.base_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DataForSEO service: {e}")
            raise
    
    def _decode_credentials(self, key_value: str) -> Tuple[str, str]:
        """Decode base64-encoded credentials"""
        try:
            decoded_creds = base64.b64decode(key_value).decode('utf-8')
            username, password = decoded_creds.split(':', 1)
            return username, password
        except Exception as e:
            raise ValueError(f"Failed to decode credentials: {e}")
    
    async def close(self):
        """Close HTTP client and database connections"""
        if self.client:
            await self.client.aclose()
        if self.db_manager:
            await self.db_manager.close()
        logger.info("DataForSEO service closed")
    
    async def _post_task(self, endpoint: str, payload: List[Dict[str, Any]]) -> str:
        """POST task to DataForSEO API and return task_id"""
        try:
            url = f"/dataforseo_labs/google/{endpoint}/live"
            logger.info(f"POST task to {url} with {len(payload)} items")
            
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"POST response status: {result.get('status_code')}")
            tasks = result.get("tasks", [])
            
            if not tasks:
                raise RuntimeError(f"No tasks returned in POST response: {result}")
            
            task_id = tasks[0]["id"]
            logger.info(f"Task posted successfully. Task ID: {task_id}")
            
            # Check if the task is already completed (status_code 20000)
            if tasks[0].get("status_code") == 20000:
                logger.info("Task already completed, returning results directly")
                # Return the full response structure, not just the task
                return result
            else:
                logger.info("Task not completed, will need to poll")
                return task_id
            
        except HTTPError as e:
            logger.error(f"HTTP error posting task: {e}")
            raise
        except Exception as e:
            logger.error(f"Error posting task: {e}")
            raise
    
    async def _get_task_result(
        self, 
        endpoint: str, 
        task_id: str, 
        timeout: int = 180, 
        poll_interval: float = 1.0
    ) -> Dict[str, Any]:
        """Poll GET until task status_code == 20000 (Ok) or timeout"""
        try:
            url = f"/dataforseo_labs/google/{endpoint}/live/task_get/{task_id}"
            start_time = time.time()
            
            logger.info(f"Polling task {task_id} with timeout {timeout}s")
            logger.info(f"Polling URL: {self.base_url}{url}")
            
            while time.time() - start_time < timeout:
                logger.info(f"Attempting to get task result from: {self.base_url}{url}")
                response = await self.client.get(url)
                logger.info(f"Response status: {response.status_code}")
                response.raise_for_status()
                
                result = response.json()
                
                # Check status code (can be at top level or in tasks array)
                status_code = result.get("status_code")
                if status_code is None:
                    tasks = result.get("tasks", [])
                    if tasks:
                        status_code = tasks[0].get("status_code")
                
                if status_code == 20000:  # Success
                    logger.info(f"Task {task_id} completed successfully")
                    return result
                elif status_code and status_code != 20000:
                    # Task failed
                    error_msg = result.get("status_message", f"Task failed with status code: {status_code}")
                    raise RuntimeError(f"Task {task_id} failed: {error_msg}")
                
                # Task still processing, wait and retry
                await asyncio.sleep(poll_interval)
            
            # Timeout reached
            raise TimeoutError(f"Task {task_id} not ready after {timeout} seconds")
            
        except HTTPError as e:
            logger.error(f"HTTP error polling task {task_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error polling task {task_id}: {e}")
            raise
    
    async def get_related_keywords(
        self,
        keywords: List[str],
        location_code: int = 2840,
        language_code: str = "en",
        depth: int = 2,  # Changed default to 2 as requested
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get related keywords using DataForSEO Labs API with single-step live endpoint
        
        Args:
            keywords: List of seed keywords
            location_code: Location code (2840 = United States)
            language_code: Language code (en = English)
            depth: Search depth (2 = related to related for better results)
            limit: Maximum keywords per seed
            
        Returns:
            List of related keyword data
        """
        try:
            if not self.client:
                await self.initialize()
            
            # Process all keywords regardless of length
            # DataForSEO API can handle multi-word keywords
            filtered_keywords = keywords
            logger.info(f"Processing {len(filtered_keywords)} keywords for related keywords")
            
            # Process each keyword individually (not in batch)
            all_related_keywords = []
            
            for keyword in filtered_keywords:
                try:
                    logger.info(f"Getting related keywords for: '{keyword}'")
                    
                    # Create payload for single keyword with enhanced data collection
                    payload = [{
                        "keyword": keyword,
                        "location_code": location_code,
                        "language_code": language_code,
                        "depth": depth,
                        "limit": limit,
                        "include_serp_info": True,  # Enable SERP info for more data
                        "include_clickstream_data": True,  # Enable clickstream data for demographics
                        "include_keyword_properties": True,  # Enable keyword properties
                        "include_search_intent": True,  # Enable search intent data
                        "include_backlinks_info": True,  # Enable backlinks data
                        "tag": f"seed-{keyword}"
                    }]
                    
                    # Make direct POST to live endpoint - single step
                    url = "/dataforseo_labs/google/related_keywords/live"
                    logger.info(f"Making single-step API call to {url}")
                    logger.info(f"DEBUG - Full request URL: {self.client.base_url}{url}")
                    logger.info(f"DEBUG - Using base_url: {self.client.base_url}")
                    
                    response = await self.client.post(url, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    logger.info(f"✅ Got immediate results for '{keyword}' from live endpoint")
                    
                    # Parse the results for this keyword
                    keyword_related = self._parse_related_keywords_result(result, keyword)
                    all_related_keywords.extend(keyword_related)
                    
                    logger.info(f"✅ Found {len(keyword_related)} related keywords for '{keyword}'")
                    
                except Exception as e:
                    logger.error(f"Error processing keyword '{keyword}': {e}")
                    continue
            
            logger.info(f"Total related keywords found: {len(all_related_keywords)}")
            return all_related_keywords

        except Exception as e:
            logger.error(f"Error getting related keywords: {e}")
            raise
    
    def _generate_mock_related_keywords(self, seed_keyword: str, count: int, template: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate additional mock related keywords when sandbox returns limited results
        
        Args:
            seed_keyword: The original seed keyword
            count: Number of additional keywords to generate
            template: Template keyword data to base mock data on
            
        Returns:
            List of mock related keyword data
        """
        import random
        import hashlib
        from datetime import datetime
        
        mock_keywords = []
        
        # Generate variations of the seed keyword based on word count
        word_count = len(seed_keyword.split())
        if word_count == 1:
            # Single word - create more variations
            variations = [
                f"{seed_keyword} ideas",
                f"{seed_keyword} tips",
                f"{seed_keyword} guide",
                f"{seed_keyword} examples",
                f"{seed_keyword} benefits",
                f"{seed_keyword} cost",
                f"{seed_keyword} reviews",
                f"{seed_keyword} comparison",
                f"best {seed_keyword}",
                f"how to {seed_keyword}",
                f"{seed_keyword} vs",
                f"{seed_keyword} alternatives",
                f"{seed_keyword} for beginners",
                f"{seed_keyword} tutorial",
                f"{seed_keyword} software",
                f"{seed_keyword} tools"
            ]
        else:
            # Two words - create more targeted variations
            variations = [
                f"{seed_keyword} ideas",
                f"{seed_keyword} tips",
                f"{seed_keyword} guide",
                f"{seed_keyword} examples",
                f"{seed_keyword} benefits",
                f"{seed_keyword} cost",
                f"{seed_keyword} reviews",
                f"{seed_keyword} comparison",
                f"best {seed_keyword}",
                f"how to {seed_keyword}",
                f"{seed_keyword} vs",
                f"{seed_keyword} alternatives",
                f"{seed_keyword} for beginners",
                f"{seed_keyword} tutorial"
            ]
        
        # Use template data if available, otherwise create basic structure
        base_data = template or {
            "search_volume": 1000,
            "cpc": 1.50,
            "competition": 0.5,
            "competition_level": "MEDIUM",
            "difficulty": 30,
            "keyword_difficulty": 30,
            "main_intent": "INFORMATIONAL",
            "intent_type": "INFORMATIONAL"
        }
        
        for i in range(min(count, len(variations))):
            variation = variations[i]
            
            # Generate unique data based on keyword hash
            keyword_hash = int(hashlib.md5(variation.encode()).hexdigest()[:8], 16)
            
            mock_keyword = {
                "seed_keyword": seed_keyword,
                "keyword": variation,
                "related_keyword": variation,
                "search_volume": base_data.get("search_volume", 1000) + (keyword_hash % 5000),
                "cpc": round(base_data.get("cpc", 1.50) + (keyword_hash % 100) / 100, 2),
                "competition": round(base_data.get("competition", 0.5) + (keyword_hash % 50) / 100, 2),
                "competition_level": random.choice(["LOW", "MEDIUM", "HIGH"]),
                "low_top_of_page_bid": round(base_data.get("cpc", 1.50) * 0.7, 2),
                "high_top_of_page_bid": round(base_data.get("cpc", 1.50) * 1.5, 2),
                "categories": [10007, 10878, 12171],
                "monthly_searches": [],
                "last_updated_time": datetime.utcnow().isoformat(),
                "core_keyword": None,
                "synonym_clustering_algorithm": None,
                "difficulty": base_data.get("difficulty", 30) + (keyword_hash % 40),
                "keyword_difficulty": base_data.get("keyword_difficulty", 30) + (keyword_hash % 40),
                "detected_language": "en",
                "is_another_language": False,
                "main_intent": base_data.get("main_intent", "INFORMATIONAL"),
                "intent_type": base_data.get("intent_type", "INFORMATIONAL"),
                "foreign_intent": [],
                "search_intent_last_updated_time": None,
                "monthly_trend": (keyword_hash % 20) - 10,
                "quarterly_trend": (keyword_hash % 30) - 15,
                "yearly_trend": (keyword_hash % 40) - 20,
                "search_volume_trend": [],
                "clickstream_search_volume": None,
                "clickstream_last_updated_time": None,
                "clickstream_gender_distribution": {},
                "clickstream_age_distribution": {},
                "clickstream_monthly_searches": [],
                "serp_se_type": None,
                "serp_check_url": None,
                "serp_item_types": [],
                "se_results_count": None,
                "serp_last_updated_time": None,
                "serp_previous_updated_time": None,
                "avg_backlinks": None,
                "avg_dofollow": None,
                "avg_referring_pages": None,
                "avg_referring_domains": None,
                "avg_referring_main_domains": None,
                "avg_rank": None,
                "avg_main_domain_rank": None,
                "backlinks_last_updated_time": None,
                "normalized_bing_search_volume": None,
                "normalized_bing_is_normalized": None,
                "normalized_bing_last_updated_time": None,
                "normalized_bing_monthly_searches": [],
                "normalized_clickstream_search_volume": None,
                "normalized_clickstream_is_normalized": None,
                "normalized_clickstream_last_updated_time": None,
                "normalized_clickstream_monthly_searches": [],
                "depth": 0,
                "related_keywords": [],
                "trend_percentage": 0,
                "priority_score": 0,
                "source": "related_keywords",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            mock_keywords.append(mock_keyword)
        
        logger.info(f"Generated {len(mock_keywords)} mock related keywords for '{seed_keyword}'")
        return mock_keywords
    
    def _parse_related_keywords_result(self, result: Dict[str, Any], seed_keyword: str) -> List[Dict[str, Any]]:
        """
        Parse DataForSEO related keywords result for a single keyword with all available fields
        
        Args:
            result: DataForSEO API result
            seed_keyword: The original seed keyword
            
        Returns:
            List of parsed related keyword data with comprehensive fields
        """
        related_keywords = []
        
        if not result:
            logger.error(f"No result to parse for keyword: {seed_keyword}")
            return related_keywords

        tasks = result.get("tasks", [])
        if not tasks:
            logger.warning(f"No tasks found in result for keyword: {seed_keyword}")
            return related_keywords

        for task in tasks:
            task_results = task.get("result", [])
            if task_results is None:
                task_results = []

            for task_result in task_results:
                items = task_result.get("items", [])
                if items is None:
                    items = []

                for item in items:
                    # Get all available data sections
                    keyword_data = item.get("keyword_data", {})
                    keyword_info = keyword_data.get("keyword_info", {}) if keyword_data else {}
                    keyword_properties = keyword_data.get("keyword_properties", {}) if keyword_data else {}
                    search_intent_info = item.get("search_intent_info", {})
                    search_volume_trend = keyword_info.get("search_volume_trend", {}) if keyword_info else {}
                    avg_backlinks_info = item.get("avg_backlinks_info", {})
                    clickstream_keyword_info = item.get("clickstream_keyword_info", {})
                    serp_info = item.get("serp_info", {})
                    keyword_info_normalized_with_bing = item.get("keyword_info_normalized_with_bing", {})
                    keyword_info_normalized_with_clickstream = item.get("keyword_info_normalized_with_clickstream", {})

                    # Safety checks for None values
                    if keyword_data is None:
                        keyword_data = {}
                    if keyword_info is None:
                        keyword_info = {}
                    if keyword_properties is None:
                        keyword_properties = {}
                    if search_intent_info is None:
                        search_intent_info = {}
                    if search_volume_trend is None:
                        search_volume_trend = {}
                    if avg_backlinks_info is None:
                        avg_backlinks_info = {}
                    if clickstream_keyword_info is None:
                        clickstream_keyword_info = {}
                    if serp_info is None:
                        serp_info = {}
                    if keyword_info_normalized_with_bing is None:
                        keyword_info_normalized_with_bing = {}
                    if keyword_info_normalized_with_clickstream is None:
                        keyword_info_normalized_with_clickstream = {}

                    # Get the related keywords from the related_keywords array
                    related_keywords_list = item.get("related_keywords", [])
                    
                    # Debug: Log what data is available for this item
                    if len(related_keywords) < 3:  # Only debug first few
                        logger.info(f"DEBUG - Processing item for '{seed_keyword}':")
                        logger.info(f"DEBUG - Full item structure: {item}")
                        logger.info(f"DEBUG - related_keywords_list: {related_keywords_list}")
                        logger.info(f"DEBUG - keyword_properties BEFORE safety check: {keyword_properties}")
                        logger.info(f"DEBUG - keyword_difficulty BEFORE safety check: {keyword_properties.get('keyword_difficulty') if keyword_properties else 'keyword_properties is None/empty'}")
                        logger.info(f"DEBUG - keyword_info: {keyword_info}")
                        logger.info(f"DEBUG - keyword_difficulty from properties: {keyword_properties.get('keyword_difficulty')}")
                        logger.info(f"DEBUG - keyword_info keys: {list(keyword_info.keys()) if keyword_info else 'None'}")
                        logger.info(f"DEBUG - keyword_properties keys: {list(keyword_properties.keys()) if keyword_properties else 'None'}")
                        logger.info(f"DEBUG - search_intent_info: {search_intent_info}")
                        logger.info(f"DEBUG - search_intent_info keys: {list(search_intent_info.keys()) if search_intent_info else 'None'}")
                    
                    # Process each related keyword from the related_keywords array
                    if related_keywords_list:
                        for related_keyword in related_keywords_list:
                            if related_keyword and related_keyword.strip():  # Skip empty strings
                                # Debug difficulty mapping
                                if len(related_keywords) < 3:  # Only debug first few
                                    logger.info(f"DEBUG - Mapping difficulty for '{related_keyword}':")
                                    logger.info(f"DEBUG - keyword_properties at mapping time: {keyword_properties}")
                                    logger.info(f"DEBUG - keyword_difficulty value: {keyword_properties.get('keyword_difficulty')}")
                                
                                related_keywords.append({
                                # Basic identification
                                "seed_keyword": seed_keyword,
                                "keyword": related_keyword,
                                "related_keyword": related_keyword,
                                
                                # Core keyword_info fields
                                "search_volume": keyword_info.get("search_volume"),
                                "cpc": keyword_info.get("cpc"),
                                "competition": keyword_info.get("competition"),
                                "competition_level": keyword_info.get("competition_level"),
                                "low_top_of_page_bid": keyword_info.get("low_top_of_page_bid"),
                                "high_top_of_page_bid": keyword_info.get("high_top_of_page_bid"),
                                "categories": keyword_info.get("categories", []),
                                "monthly_searches": keyword_info.get("monthly_searches", []),
                                "last_updated_time": keyword_info.get("last_updated_time"),
                                
                                # keyword_properties fields
                                "core_keyword": keyword_properties.get("core_keyword"),
                                "synonym_clustering_algorithm": keyword_properties.get("synonym_clustering_algorithm"),
                                "difficulty": keyword_properties.get("keyword_difficulty") or 0,  # Default to 0 if not available
                                "keyword_difficulty": keyword_properties.get("keyword_difficulty") or 0,  # Default to 0 if not available
                                "detected_language": keyword_properties.get("detected_language"),
                                "is_another_language": keyword_properties.get("is_another_language"),
                                
                                # search_intent_info fields
                                "main_intent": search_intent_info.get("main_intent") or keyword_info.get("main_intent"),
                                "intent_type": search_intent_info.get("intent_type") or keyword_info.get("intent_type"),
                                "foreign_intent": search_intent_info.get("foreign_intent", []),
                                "search_intent_last_updated_time": search_intent_info.get("last_updated_time"),
                                
                                # search_volume_trend
                                "monthly_trend": search_volume_trend.get("monthly"),
                                "quarterly_trend": search_volume_trend.get("quarterly"),
                                "yearly_trend": search_volume_trend.get("yearly"),
                                
                                # clickstream_keyword_info
                                "clickstream_search_volume": clickstream_keyword_info.get("search_volume"),
                                "clickstream_last_updated_time": clickstream_keyword_info.get("last_updated_time"),
                                "clickstream_gender_distribution": clickstream_keyword_info.get("gender_distribution", {}),
                                "clickstream_age_distribution": clickstream_keyword_info.get("age_distribution", {}),
                                "clickstream_monthly_searches": clickstream_keyword_info.get("monthly_searches", []),
                                
                                # serp_info
                                "serp_se_type": serp_info.get("se_type"),
                                "serp_check_url": serp_info.get("check_url"),
                                "serp_item_types": serp_info.get("serp_item_types", []),
                                "se_results_count": serp_info.get("se_results_count"),
                                "serp_last_updated_time": serp_info.get("last_updated_time"),
                                "serp_previous_updated_time": serp_info.get("previous_updated_time"),
                                
                                # avg_backlinks_info
                                "avg_backlinks": avg_backlinks_info.get("backlinks"),
                                "avg_dofollow": avg_backlinks_info.get("dofollow"),
                                "avg_referring_pages": avg_backlinks_info.get("referring_pages"),
                                "avg_referring_domains": avg_backlinks_info.get("referring_domains"),
                                "avg_referring_main_domains": avg_backlinks_info.get("referring_main_domains"),
                                "avg_rank": avg_backlinks_info.get("rank"),
                                "avg_main_domain_rank": avg_backlinks_info.get("main_domain_rank"),
                                "backlinks_last_updated_time": avg_backlinks_info.get("last_updated_time"),
                                
                                # keyword_info_normalized_with_bing
                                "normalized_bing_search_volume": keyword_info_normalized_with_bing.get("search_volume"),
                                "normalized_bing_is_normalized": keyword_info_normalized_with_bing.get("is_normalized"),
                                "normalized_bing_last_updated_time": keyword_info_normalized_with_bing.get("last_updated_time"),
                                "normalized_bing_monthly_searches": keyword_info_normalized_with_bing.get("monthly_searches", []),
                                
                                # keyword_info_normalized_with_clickstream
                                "normalized_clickstream_search_volume": keyword_info_normalized_with_clickstream.get("search_volume"),
                                "normalized_clickstream_is_normalized": keyword_info_normalized_with_clickstream.get("is_normalized"),
                                "normalized_clickstream_last_updated_time": keyword_info_normalized_with_clickstream.get("last_updated_time"),
                                "normalized_clickstream_monthly_searches": keyword_info_normalized_with_clickstream.get("monthly_searches", []),
                                
                                # Additional fields
                                "depth": item.get("depth"),
                                "source": "related_keywords",  # Mark as related keywords
                                "created_at": datetime.utcnow().isoformat()
                            })
                    else:
                        # If no related_keywords array, fall back to the main keyword
                        main_keyword = keyword_data.get("keyword", "")
                        if main_keyword and main_keyword.strip():
                            related_keywords.append({
                                # Basic identification
                                "seed_keyword": seed_keyword,
                                "keyword": main_keyword,
                                "related_keyword": main_keyword,
                                
                                # Core keyword_info fields
                                "search_volume": keyword_info.get("search_volume"),
                                "cpc": keyword_info.get("cpc"),
                                "competition": keyword_info.get("competition"),
                                "competition_level": keyword_info.get("competition_level"),
                                "low_top_of_page_bid": keyword_info.get("low_top_of_page_bid"),
                                "high_top_of_page_bid": keyword_info.get("high_top_of_page_bid"),
                                "categories": keyword_info.get("categories", []),
                                "monthly_searches": keyword_info.get("monthly_searches", []),
                                "last_updated_time": keyword_info.get("last_updated_time"),
                                
                                # keyword_properties fields
                                "core_keyword": keyword_properties.get("core_keyword"),
                                "synonym_clustering_algorithm": keyword_properties.get("synonym_clustering_algorithm"),
                                "difficulty": keyword_properties.get("keyword_difficulty") or 0,
                                "keyword_difficulty": keyword_properties.get("keyword_difficulty") or 0,
                                "detected_language": keyword_properties.get("detected_language"),
                                "is_another_language": keyword_properties.get("is_another_language"),
                                
                                # search_intent_info fields
                                "main_intent": search_intent_info.get("main_intent") or keyword_info.get("main_intent"),
                                "intent_type": search_intent_info.get("intent_type") or keyword_info.get("intent_type"),
                                "foreign_intent": search_intent_info.get("foreign_intent", []),
                                "search_intent_last_updated_time": search_intent_info.get("last_updated_time"),
                                
                                # Additional fields
                                "depth": item.get("depth"),
                                "source": "related_keywords",
                                "created_at": datetime.utcnow().isoformat()
                            })

        return related_keywords
    
    async def get_keyword_ideas(
        self,
        seed_keywords: List[str],
        location_code: int = 2840,
        language_code: str = "en",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get keyword ideas using DataForSEO Labs API with single-step live mode
        
        Args:
            seed_keywords: List of seed keywords (up to 200 per request)
            location_code: Location code (2840 = United States)
            language_code: Language code (en = English)
            limit: Maximum keywords to return
            
        Returns:
            List of keyword idea data
        """
        try:
            if not self.client:
                logger.info("Initializing DataForSEO service...")
                await self.initialize()
                logger.info("DataForSEO service initialized successfully")
            
            # Build payload with enhanced data collection (DataForSEO expects array format)
            payload = [{
                "keywords": seed_keywords,
                "location_code": location_code,
                "language_code": language_code,
                "limit": limit,
                "include_serp_info": True,  # Enable SERP info for more data
                "include_clickstream_data": True,  # Enable clickstream data for demographics
                "include_keyword_properties": True,  # Enable keyword properties (Difficulty)
                "closely_variants": False,
                "ignore_synonyms": False,
                "tag": f"keyword-ideas-{int(time.time())}"
            }]
            
            # Use single-step live endpoint directly
            url = "/dataforseo_labs/google/keyword_ideas/live"
            logger.info(f"Making single-step live request to {url} with {len(seed_keywords)} keywords")
            logger.info(f"DEBUG - Client base_url before request: {self.client.base_url}")
            logger.info(f"DEBUG - Full URL will be: {self.client.base_url}{url}")
            logger.info(f"DEBUG - Auth username: {self.username}")
            logger.info(f"DEBUG - Using sandbox URL: {self.base_url}")
            
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Live response status: {result.get('status_code') if result else 'None'}")
            logger.info(f"Response type: {type(result)}")
            logger.info(f"Response is None: {result is None}")
            logger.info(f"Response content: {result}")
            
            # Check if request was successful
            if result.get("status_code") != 20000:
                error_msg = result.get('status_message', 'Unknown error')
                logger.error(f"DataForSEO API error: {error_msg}")
                logger.error(f"Full response: {result}")
                raise RuntimeError(f"DataForSEO API error: {error_msg}")
            
            # Parse response: tasks[].result[].items[].keyword_info
            keyword_ideas = []
            if not result:
                logger.error("No result returned from DataForSEO API")
                return []
            
            # Log the full response structure for debugging (truncated)
            if result is None:
                logger.error("DataForSEO response is None")
                return []
            
            logger.info(f"DataForSEO response structure: {type(result)} with keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Simple debug logging
            logger.info(f"DataForSEO response received, processing tasks...")
            logger.info(f"Result type: {type(result)}")
            
            # Safe key access
            if isinstance(result, dict):
                logger.info(f"Result keys: {list(result.keys())}")
            else:
                logger.error(f"Result is not a dict: {result}")
                return []
                
            tasks = result.get("tasks", [])
            logger.info(f"Tasks type: {type(tasks)}, length: {len(tasks) if isinstance(tasks, list) else 'Not a list'}")
            if not tasks:
                logger.warning("No tasks found in result")
                return []
            
            for task in tasks:
                if task is None:
                    continue
                    
                task_results = task.get("result", [])
                if task_results is None:
                    task_results = []
                
                for task_result in task_results:
                    if task_result is None:
                        continue
                        
                    seed_keywords_list = task_result.get("seed_keywords", [])
                    
                    items = task_result.get("items", [])
                    if items is None:
                        items = []
                    
                    for item in items:
                        if item is None:
                            continue
                        
                        # Check if item is a dictionary
                        if not isinstance(item, dict):
                            logger.warning(f"Item is not a dictionary: {type(item)} - {item}")
                            continue
                        
                        try:
                            # Extract basic keyword info
                            keyword = item.get("keyword")
                            
                            # Safely extract nested objects with None checks
                            keyword_info = item.get("keyword_info") or {}
                            keyword_properties = item.get("keyword_properties") or {}
                            search_intent_info = item.get("search_intent_info") or {}
                            avg_backlinks_info = item.get("avg_backlinks_info") or {}
                            serp_info = item.get("serp_info") or {}
                            clickstream_keyword_info = item.get("clickstream_keyword_info") or {}
                            keyword_info_normalized_with_bing = item.get("keyword_info_normalized_with_bing") or {}
                            keyword_info_normalized_with_clickstream = item.get("keyword_info_normalized_with_clickstream") or {}
                            
                            # Safely extract arrays with None checks
                            monthly_searches = keyword_info.get("monthly_searches") or []
                            categories = keyword_info.get("categories") or []
                            foreign_intent = search_intent_info.get("foreign_intent") or []
                            
                            # Safely extract search_volume_trend
                            search_volume_trend = keyword_info.get("search_volume_trend") or {}
                            
                            # Debug logging for first few items
                            if len(keyword_ideas) < 3:
                                logger.info(f"DEBUG - Item {len(keyword_ideas)}: keyword='{keyword}'")
                                logger.info(f"DEBUG - keyword_info: {keyword_info}")
                                logger.info(f"DEBUG - keyword_properties: {keyword_properties}")
                                logger.info(f"DEBUG - search_intent_info: {search_intent_info}")
                            
                            keyword_ideas.append({
                            # Basic keyword info
                            "seed_keywords": seed_keywords_list,
                            "keyword": keyword,
                            
                            # Core keyword metrics from keyword_info
                            "search_volume": keyword_info.get("search_volume"),
                            "cpc": keyword_info.get("cpc"),
                            "competition": keyword_info.get("competition"),
                            "competition_level": keyword_info.get("competition_level"),
                            "low_top_of_page_bid": keyword_info.get("low_top_of_page_bid"),
                            "high_top_of_page_bid": keyword_info.get("high_top_of_page_bid"),
                            "last_updated_time": keyword_info.get("last_updated_time"),
                            
                            # Keyword properties (with correct field names)
                            "difficulty": keyword_properties.get("keyword_difficulty"),
                            "keyword_difficulty": keyword_properties.get("keyword_difficulty"),  # Frontend expects this
                            "core_keyword": keyword_properties.get("core_keyword"),
                            "synonym_clustering_algorithm": keyword_properties.get("synonym_clustering_algorithm"),
                            "detected_language": keyword_properties.get("detected_language"),
                            "is_another_language": keyword_properties.get("is_another_language"),
                            
                            # Search intent info
                            "main_intent": search_intent_info.get("main_intent"),
                            "intent_type": search_intent_info.get("main_intent"),
                            "foreign_intent": foreign_intent,
                            
                            # Search volume trends
                            "monthly_trend": search_volume_trend.get("monthly"),
                            "quarterly_trend": search_volume_trend.get("quarterly"),
                            "yearly_trend": search_volume_trend.get("yearly"),
                            
                            # Backlink data
                            "avg_backlinks": avg_backlinks_info.get("backlinks"),
                            "avg_referring_domains": avg_backlinks_info.get("referring_domains"),
                            "avg_referring_pages": avg_backlinks_info.get("referring_pages"),
                            "avg_dofollow_links": avg_backlinks_info.get("dofollow"),
                            "avg_rank": avg_backlinks_info.get("rank"),
                            "avg_main_domain_rank": avg_backlinks_info.get("main_domain_rank"),
                            "backlinks_last_updated_time": avg_backlinks_info.get("last_updated_time"),
                            
                            # Additional data
                            "categories": categories,
                            "monthly_searches": monthly_searches,
                            
                            # SERP info (if available)
                            "serp_item_types": serp_info.get("serp_item_types", []),
                            "se_results_count": serp_info.get("se_results_count"),
                            "check_url": serp_info.get("check_url"),
                            "serp_last_updated_time": serp_info.get("last_updated_time"),
                            "serp_previous_updated_time": serp_info.get("previous_updated_time"),
                            
                            # Clickstream data (if available)
                            "clickstream_search_volume": clickstream_keyword_info.get("search_volume"),
                            "clickstream_last_updated_time": clickstream_keyword_info.get("last_updated_time"),
                            "clickstream_gender_distribution": clickstream_keyword_info.get("gender_distribution", {}),
                            "clickstream_age_distribution": clickstream_keyword_info.get("age_distribution", {}),
                            "clickstream_monthly_searches": clickstream_keyword_info.get("monthly_searches", []),
                            
                            # Normalized data with Bing
                            "normalized_bing_search_volume": keyword_info_normalized_with_bing.get("search_volume"),
                            "normalized_bing_is_normalized": keyword_info_normalized_with_bing.get("is_normalized"),
                            "normalized_bing_last_updated_time": keyword_info_normalized_with_bing.get("last_updated_time"),
                            "normalized_bing_monthly_searches": keyword_info_normalized_with_bing.get("monthly_searches", []),
                            
                            # Normalized data with Clickstream
                            "normalized_clickstream_search_volume": keyword_info_normalized_with_clickstream.get("search_volume"),
                            "normalized_clickstream_is_normalized": keyword_info_normalized_with_clickstream.get("is_normalized"),
                            "normalized_clickstream_last_updated_time": keyword_info_normalized_with_clickstream.get("last_updated_time"),
                            "normalized_clickstream_monthly_searches": keyword_info_normalized_with_clickstream.get("monthly_searches", []),
                            
                            # Additional fields for compatibility
                            "depth": 1,  # Default depth for keyword ideas
                            "intent_type": search_intent_info.get("main_intent", "COMMERCIAL"),
                            "competition_value": int(float(keyword_info.get("competition", 0)) * 100) if keyword_info.get("competition") is not None else 0,  # Convert to integer (0-100 range)
                            
                            # Metadata
                            "created_at": datetime.utcnow().isoformat(),
                            "source": "keyword_ideas"
                        })
                        
                        except Exception as item_error:
                            logger.warning(f"Error processing item: {item_error}")
                            logger.warning(f"Item data: {item}")
                            continue
            
            logger.info(f"Retrieved {len(keyword_ideas)} keyword ideas for {len(seed_keywords)} seed keywords")
            return keyword_ideas
            
        except Exception as e:
            logger.error(f"Error getting keyword ideas: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Re-raise the exception to get proper error details
            raise


    async def get_google_trends_explore(
        self,
        keywords: List[str],
        location_name: str = "United States",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        type: str = "web",
        category_code: int = 0
    ) -> Dict[str, Any]:
        """
        Get Google Trends "Interest over time" data using the task-based approach
        
        Args:
            keywords: List of keywords (up to 5 for comparison)
            location_name: Full location name (e.g. "United States")
            date_from: Start date (YYYY-MM-DD or relative like "now 7-d")
            date_to: End date (YYYY-MM-DD)
            type: Search type (web, news, images, froogle, youtube)
            category_code: Google Trends category code
            
        Returns:
            Dictionary mapping keywords to their interest scores over time
        """
        try:
            if not self.client:
                await self.initialize()
            
            # Use current date as default to if not provided
            if not date_to:
                date_to = datetime.utcnow().strftime("%Y-%m-%d")
            # Use 12 months ago as default from if not provided
            if not date_from:
                # Naive 12 months back
                from datetime import timedelta
                date_from = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            payload = [{
                "keywords": keywords,
                "location_name": location_name,
                "type": type,
                "date_from": date_from,
                "date_to": date_to,
                "category_code": category_code,
                "item_types": ["google_trends_graph"]
            }]
            
            # Post task
            post_url = "/keywords_data/google_trends/explore/task_post"
            logger.info(f"Posting Google Trends task for {keywords} to {post_url}")
            
            response = await self.client.post(post_url, json=payload)
            response.raise_for_status()
            post_result = response.json()
            
            if post_result.get("status_code") != 20000:
                raise RuntimeError(f"DataForSEO error (POST): {post_result.get('status_message')}")
            
            task = post_result.get("tasks", [])[0]
            task_id = task.get("id")
            
            # Immediate check: if task is already completed (unlikely but possible)
            if task.get("status_code") == 20000:
                return self._parse_trends_explore_result(post_result)
            
            # Poll for result
            return await self._get_trends_explore_result(task_id)
            
        except Exception as e:
            logger.error(f"Error in get_google_trends_explore: {e}")
            raise

    async def _get_trends_explore_result(self, task_id: str, timeout: int = 120) -> Dict[str, Any]:
        """Poll for Google Trends explore results"""
        url = f"/keywords_data/google_trends/explore/task_get/{task_id}"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = await self.client.get(url)
            response.raise_for_status()
            result = response.json()
            
            task = result.get("tasks", [])[0]
            status_code = task.get("status_code")
            
            if status_code == 20000:
                return self._parse_trends_explore_result(result)
            elif status_code not in [20000, 20100, 40601]: # 40601 is "Task Handed"
                raise RuntimeError(f"Task {task_id} failed: {task.get('status_message')}")
            
            await asyncio.sleep(5) # Google Trends is slow, poll every 5s
            
        raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")

    def _parse_trends_explore_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Google Trends explore task result"""
        trends_data = {}
        for task in result.get("tasks", []):
            for task_result in task.get("result", []):
                for item in task_result.get("items", []):
                    if item.get("type") == "google_trends_graph" and "data" in item:
                        # Map back to keywords
                        keywords = task_result.get("keywords", [])
                        for i, keyword in enumerate(keywords):
                            points = []
                            for data_point in item["data"]:
                                if not data_point.get("missing_data", False):
                                    timestamp = data_point.get("timestamp")
                                    # values is a list corresponding to the keyword index
                                    val = data_point.get("values", [0])[i] if i < len(data_point.get("values", [])) else 0
                                    points.append({
                                        "timestamp": timestamp,
                                        "date": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d") if timestamp else None,
                                        "value": val
                                    })
                            trends_data[keyword] = points
        return trends_data

    async def get_search_volume_history(
        self,
        keywords: List[str],
        location_code: int = 2840,
        language_code: str = "en"
    ) -> Dict[str, Any]:
        """
        Get 12-month search volume history for a list of keywords
        
        Args:
            keywords: List of keywords (up to 700)
            location_code: Location code (2840 = US)
            language_code: Language code (en = English)
            
        Returns:
            Dictionary mapping keywords to their historical search volume data
        """
        try:
            if not self.client:
                await self.initialize()
            
            # DataForSEO search_volume/live expects "keywords" as an array
            payload = [{
                "keywords": keywords,
                "location_code": location_code,
                "language_code": language_code,
                "include_search_volume_trend": True
            }]
            
            url = "/dataforseo_labs/google/historical_search_volume/live"
            logger.info(f"Fetching search volume history for {len(keywords)} keywords")
            
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status_code") != 20000:
                raise RuntimeError(f"DataForSEO error: {result.get('status_message')}")
            
            # Parse results
            history_data = {}
            tasks = result.get("tasks") or []
            if not tasks:
                 logger.warning(f"DataForSEO returned no tasks in result: {result.keys()}")
            
            for task in tasks:
                results = task.get("result") or []
                if not results:
                     logger.warning(f"DataForSEO task has no result: {task.keys()}")
                
                for task_result in results:
                    # In some endpoints it is "items", in others it might be slightly different
                    # For search_volume/live it is usually items
                    items = task_result.get("items") or []
                    if items is None:
                        logger.warning("DataForSEO items is None")
                        items = []
                        
                    for item in items:
                        keyword = item.get("keyword")
                        if keyword:
                            history_data[keyword] = {
                                "search_volume": item.get("search_volume"),
                                "monthly_searches": item.get("monthly_searches", []),
                                "search_volume_trend": item.get("search_volume_trend"),
                                "competition": item.get("competition"),
                                "cpc": item.get("cpc")
                            }
            
            return history_data
            
        except Exception as e:
            logger.error(f"Error getting search volume history: {e}")
            raise

# Global service instance
dataforseo_service = DataForSEOService()
