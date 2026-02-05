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
        self.api_login = settings.DATAFORSEO_API_LOGIN
        self.api_password = settings.DATAFORSEO_API_PASSWORD
        self.base_url = "https://api.dataforseo.com/v3"
        self.timeout = 60.0
        
        self._auth_header = None # Lazy loaded
        
    @property
    def auth_header(self):
        """Lazy load auth header to safe-guard against early DB access"""
        if self._auth_header:
            return self._auth_header
            
        # Determine Auth Header (First Access)
        # 1. Try Settings (Primary) - Only if NOT default 'demo'
        if self.api_login and self.api_login != "demo" and self.api_password and self.api_password != "demo":
            credentials = f"{self.api_login}:{self.api_password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            self._auth_header = f"Basic {encoded_credentials}"
            logger.info("Using DataForSEO credentials from Environment Variables.")
        else:
            # 2. Try Supabase (Fallback)
            logger.info("Environment credentials are 'demo'. Attempting to fetch from Supabase...")
            self._auth_header = self._fetch_credentials_from_supabase()
            
            if not self._auth_header:
                logger.warning("DataForSEO credentials not found in Supabase. Falling back to demo/mock mode.")
                self._auth_header = "Basic ZGVtbzpkZW1v" 
            else:
                logger.info("Successfully loaded DataForSEO credentials from Supabase.")
        
        return self._auth_header

    def _fetch_credentials_from_supabase(self) -> Optional[str]:
        """Fetch DataForSEO key from Supabase api_keys table"""
        from src.database.supabase_client import get_supabase_client
        
        try:
            supabase = get_supabase_client()
            response = supabase.table("api_keys").select("*").eq("provider", "dataforseo").execute()
            
            if response.data and len(response.data) > 0:
                key_entry = response.data[0]
                logger.info(f"DEBUG: Found DataForSEO key entry: {key_entry.keys()}")
                
                # Check for structured keys (login/password) vs single key
                if 'key_value' in key_entry and key_entry['key_value']:
                     # Assume key_value IS the base64 string
                     kv = key_entry['key_value']
                     # Validate if it looks like Basic 
                     if kv.startswith("Basic "):
                         return kv
                     # Assume it's the raw b64
                     return f"Basic {kv}"
                     
            logger.warning("DEBUG: No DataForSEO key found in api_keys table.")
            return None
        except Exception as e:
            logger.error(f"Supabase fetch error: {e}")
            return None

    async def _make_request_with_retry(self, client, url, payload, headers, max_retries=5):
        """
        Make API request with retry logic for Rate Limits (40202)
        Uses exponential backoff with jitter.
        """
        import random
        
        for attempt in range(max_retries + 1):
            # Non-blocking jitter
            jitter = random.uniform(0.5, 1.5)
            await asyncio.sleep(jitter)
            
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Check for DataForSEO-specific Rate Limit code in JSON body
                if "tasks" in data and data["tasks"]:
                    task_status = data["tasks"][0].get("status_code")
                    if task_status == 40202: # Rate Limit Exceeded
                        if attempt < max_retries:
                            # Exponential backoff: 2s, 4s, 8s, 16s, 32s + jitter
                            backoff = (2 ** (attempt + 1)) + random.uniform(0, 1)
                            logger.warning(f"DataForSEO Rate Limit (40202). Retrying in {backoff:.2f}s... (Attempt {attempt+1}/{max_retries})")
                            await asyncio.sleep(backoff)
                            continue
                        else:
                            logger.error("DataForSEO Rate Limit exhausted all retries.")
                            return data # Return error data to let caller handle regular failure
                    
                # If success or other error, return data
                return data
                
            except httpx.HTTPError as e:
                # Handle actual HTTP 429 Too Many Requests if they occur
                if e.response and e.response.status_code == 429:
                     if attempt < max_retries:
                        backoff = (2 ** (attempt + 1)) + random.uniform(0, 1)
                        logger.warning(f"HTTP 429 Rate Limit. Retrying in {backoff:.2f}s... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(backoff)
                        continue
                raise e # Re-raise if not 429 or retries exhausted
                
        return {} # Should not be reached

    
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
                
                data = await self._make_request_with_retry(client, url, payload, headers)
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
                
                data = await self._make_request_with_retry(client, url, payload, headers)
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
                
                data = await self._make_request_with_retry(client, url, payload, headers)
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
                    "filters": [],
                    "order_by": ["keyword_info.search_volume,desc"]
                }]
                
                headers = {
                    "Authorization": self.auth_header,
                    "Content-Type": "application/json"
                }
                
                data = await self._make_request_with_retry(client, url, payload, headers)
                keywords_result = self._process_related_keywords(data)
                if not keywords_result:
                    logger.warning(f"DataForSEO returned 0 related keywords. Raw Response: {str(data)[:1000]}")
                return keywords_result
                
        except Exception as e:
            logger.error(f"DataForSEO related keywords API error: {e}")
            return []

    async def get_related_keywords_standard(
        self,
        seeds: List[str],
        language_code: str = "en",
        location_code: int = 2840,
        limit_per_seed: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get related keywords using the STANDARD (Queue-based) API.
        Flow: Post Task -> Poll for Results -> Process
        Bypasses 12 req/min limit of Live endpoint.
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client: # Short timeout for Post/Get calls themselves
                base_url = f"{self.base_url}/keywords_data/google_ads/keywords_for_keywords"
                
                # 1. POST TASK
                post_url = f"{base_url}/task_post"
                
                # Payload: array of task objects
                # "keywords" field in task_post for keywords_for_keywords takes an array of seeds!
                # Limit: 20 keywords per task object for keywords_for_keywords? 
                # User said: "Each task object can contain up to... 20 keywords (for Keywords for Keywords suggestions)."
                # So we chunk seeds into groups of 20 max.
                
                # Create ONE task per seed keyword for maximum expansion relevance
                # We can post up to 100 tasks in a single request.
                
                task_ids = []
                headers = {
                    "Authorization": self.auth_header,
                    "Content-Type": "application/json"
                }

                post_payload = []
                for seed in seeds:
                    post_payload.append({
                        "keywords": [seed], # Must be array for keywords_for_keywords
                        "language_code": language_code,
                        "location_code": location_code,
                        "limit": limit_per_seed,
                        "order_by": ["keyword_info.search_volume,desc"]
                    })
                    
                    # Safety break if > 100 seeds (API limit per POST is usually 100)
                    if len(post_payload) >= 100:
                        break
                
                logger.info(f"Posting Standard Task with {len(post_payload)} sub-tasks (1 per seed)...")
                response = await client.post(post_url, json=post_payload, headers=headers)
                response.raise_for_status()
                post_data = response.json()
                
                if "tasks" in post_data:
                    for task in post_data["tasks"]:
                        if task.get("id"):
                            task_ids.append(task["id"])
                        else:
                            logger.error(f"Task Post failed for item: {task}")
                            
                if not task_ids:
                    logger.error("No Task IDs received from Standard API post.")
                    return []
                
                logger.info(f"Tasks Posted. IDs: {task_ids}. Polling for results...")
                
                # 2. POLL FOR RESULTS
                # Check status every X seconds
                # Max wait time? Frontend timeout is 4 mins (240s). backend safe limit 200s?
                max_wait_time = 300 
                start_time = asyncio.get_event_loop().time()
                
                completed_results = []
                pending_ids = list(task_ids)
                
                while pending_ids:
                    # Check timeout
                    if asyncio.get_event_loop().time() - start_time > max_wait_time:
                        logger.error("Timed out waiting for Standard API results.")
                        break
                        
                    # Poll interval
                    await asyncio.sleep(5) 
                    
                    for tid in list(pending_ids):
                        get_url = f"{base_url}/task_get/{tid}"
                        
                        try:
                            res_get = await client.get(get_url, headers=headers)
                            res_get.raise_for_status()
                            res_data = res_get.json()
                            
                            if "tasks" in res_data and res_data["tasks"]:
                                task_res = res_data["tasks"][0]
                                status = task_res.get("status_message")
                                
                                # Verbose logging to catch "Task In Queue" vs "In Queue" mismatches
                                # logger.info(f"Task {tid} poll status: {status}") 
                                
                                if status == "Ok.":
                                    # Finished!
                                    completed_results.append(task_res)
                                    pending_ids.remove(tid)
                                elif status in ["In Queue", "Task In Queue", "Active", "Running"]:
                                    # Still waiting
                                    continue
                                else:
                                    # Error?
                                    logger.warning(f"Task {tid} unexpected status: {status}")
                                    pending_ids.remove(tid) # Remove to avoid infinite loop on error
                            else:
                                logger.warning(f"Empty response for Task {tid}")
                                pending_ids.remove(tid)
                                
                        except Exception as e:
                            logger.error(f"Error polling task {tid}: {e}")
                            # Keep trying?
                            pass
                            
                # 3. PROCESS RESULTS
                all_keywords = []
                for task in completed_results:
                    if task.get("result"):
                        for item in task["result"]:
                            all_keywords.append({
                                "keyword": item.get("keyword", ""),
                                "search_volume": item.get("search_volume", 0),
                                "competition": item.get("competition", "UNKNOWN"),
                                "competition_level": item.get("competition_level", 0),
                                "cpc": item.get("cpc", 0), # Can be null
                                "keyword_difficulty": item.get("keyword_difficulty", 0),
                                "created_at": datetime.utcnow().isoformat()
                            })
                return all_keywords
        
        except Exception as e:
            logger.error(f"DataForSEO Standard API error: {e}")
            return []

    async def get_bulk_metrics_standard(
        self,
        keywords: List[str],
        language_code: str = "en",
        location_code: int = 2840
    ) -> List[Dict[str, Any]]:
        """
        Get Search Volume & CPC for a list of keywords using Standard (Queue) API.
        Robust alternative to 'Related Keywords' which returns empty for specific long-tails.
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                base_url = f"{self.base_url}/keywords_data/google_ads/search_volume"
                post_url = f"{base_url}/task_post"
                
                # Payload: up to 700-1000 keywords per task? 
                # Docs: 1000 keywords max per task.
                chunk_size = 700 
                chunked_kws = [keywords[i:i + chunk_size] for i in range(0, len(keywords), chunk_size)]
                
                task_ids = []
                headers = {
                    "Authorization": self.auth_header,
                    "Content-Type": "application/json"
                }

                # Post Tasks
                payload = []
                for chunk in chunked_kws:
                    payload.append({
                        "keywords": chunk, 
                        "language_code": language_code,
                        "location_code": location_code,
                        "include_adult_keywords": True
                    })
                
                logger.info(f"Posting Bulk Volume Task ({len(keywords)} kws)...")
                response = await client.post(post_url, json=payload, headers=headers)
                response.raise_for_status()
                post_data = response.json()
                
                if "tasks" in post_data:
                     for t in post_data["tasks"]:
                         if t.get("id"): task_ids.append(t["id"])
                
                if not task_ids:
                    logger.error("No Task IDs for Bulk Volume.")
                    return []
                
                # Poll
                max_wait_time = 300
                start_time = asyncio.get_event_loop().time()
                completed_results = []
                pending_ids = list(task_ids)
                
                logger.info(f"Polling Bulk Volume (IDs: {task_ids})...")
                while pending_ids:
                    if asyncio.get_event_loop().time() - start_time > max_wait_time:
                         logger.error("Timeout polling Bulk Volume")
                         break
                    await asyncio.sleep(3)
                    
                    for tid in list(pending_ids):
                        get_url = f"{base_url}/task_get/{tid}"
                        try:
                            res = await client.get(get_url, headers=headers)
                            data = res.json()
                            if "tasks" in data:
                                t_res = data["tasks"][0]
                                status = t_res.get("status_message")
                                if status == "Ok.":
                                    completed_results.append(t_res)
                                    pending_ids.remove(tid)
                                elif status in ["In Queue", "Task In Queue", "Active", "Running"]:
                                    continue
                                else:
                                    logger.warning(f"Task {tid} status: {status}")
                                    pending_ids.remove(tid)
                        except:
                            pending_ids.remove(tid)
                
                # Process
                results = []
                for t in completed_results:
                    if t.get("result"):
                        # 'result' for search_volume is a list of results (one per item or task).
                        # Based on debug script: result is list, items are dicts directly?
                        # Debug output: RAW RESULT TYPE: <class 'dict'>. First item keys: ['keyword', 'search_volume', ...]
                        # Wait, the debug output said: RAW RESULT TYPE: <class 'dict'>
                        # RESULT KEYS: ['keyword', 'search_volume', ...]
                        # So t['result'] is a LIST of these dicts?
                        # Let's handle both cases.
                        
                        raw_result_list = t["result"]
                        if not raw_result_list: continue

                        for item in raw_result_list:
                            # Case 1: Item is the keyword object directly (common in some endpoints)
                            if isinstance(item, dict) and 'search_volume' in item:
                                if len(results) == 0:
                                    logger.info(f"DEBUG: Bulk Vol Item Keys: {list(item.keys())}")
                                    logger.info(f"DEBUG: Bulk Vol 1st Item Val: {item}")
                                results.append({
                                    "keyword": item.get("keyword"),
                                    "search_volume": item.get("search_volume", 0),
                                    "cpc": item.get("cpc", 0),
                                    "competition": item.get("competition", "UNKNOWN")
                                })
                            # Case 2: Item has nested 'items' (common in keywords_for_keywords)
                            elif isinstance(item, dict) and 'items' in item:
                                for sub in item['items']:
                                    if isinstance(sub, dict):
                                         results.append({
                                             "keyword": sub.get("keyword"),
                                             "search_volume": sub.get("search_volume", 0),
                                             "cpc": sub.get("cpc", 0),
                                             "competition": sub.get("competition", "UNKNOWN")
                                         })
                return results

        except Exception as e:
            logger.error(f"Bulk Volume API error: {e}")
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
                # Correct endpoint for bulk KD (DataForSEO Labs). 
                # Note: The old 'keywords_data/google_ads/keyword_difficulty/live' was 404 or deprecated.
                url = f"{self.base_url}/dataforseo_labs/google/bulk_keyword_difficulty/live"
                
                payload = [{
                    "keywords": keywords,
                    "language_code": language_code,
                    "location_code": location_code
                }]
                
                headers = {
                    "Authorization": self.auth_header,
                    "Content-Type": "application/json"
                }
                
                data = await self._make_request_with_retry(client, url, payload, headers)
                return self._process_keyword_difficulty(data)
                
        except Exception as e:
            logger.error(f"DataForSEO keyword difficulty API error: {e}")
            return []
    

    async def get_keyword_trends(
        self,
        keywords: List[str],
        language_code: str = "en",
        location_code: int = 2840,
        date_from: str = None,
        date_to: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get Google Trends data for keywords
        
        Args:
            keywords: List of keywords to analyze
            language_code: Language code
            location_code: Location code
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            
        Returns:
            List of trend data
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/keywords_data/google_trends/explore/live"
                
                payload = [{
                    "keywords": keywords,
                    "language_code": language_code,
                    "location_code": location_code,
                    "date_from": date_from,
                    "date_to": date_to
                }]
                
                headers = {
                    "Authorization": self.auth_header,
                    "Content-Type": "application/json"
                }
                
                data = await self._make_request_with_retry(client, url, payload, headers)
                return self._process_keyword_trends(data)
                
        except Exception as e:
            logger.error(f"DataForSEO keyword trends API error: {e}")
            return []

    def _process_keyword_trends(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process keyword trends response"""
        trends = []
        
        if "tasks" in data and data["tasks"]:
            task = data["tasks"][0]
            if task.get("result"):
                for item in task["result"]:
                    # Adjust based on actual API response structure for trends
                    # This is a generic extraction, detailed structure depends on DataForSEO specific endpoint response
                    trends.append({
                        "keywords": item.get("keywords", []),
                        "items": item.get("items", []),
                        "created_at": datetime.utcnow().isoformat()
                    })
        
        return trends
    
    def _process_keyword_ideas(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process keyword ideas response"""
        keywords = []
        
        if "tasks" in data and data["tasks"]:
            task = data["tasks"][0]
            if task.get("result"):
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
        
        if "tasks" in data and data["tasks"]:
            task = data["tasks"][0]
            if task.get("result"):
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
        
        if "tasks" in data and data["tasks"]:
            task = data["tasks"][0]
            if task.get("result"):
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
        
        if "tasks" in data and data["tasks"]:
            task = data["tasks"][0]
            if task.get("result"):
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
        
        if "tasks" in data and data["tasks"]:
            task = data["tasks"][0]
            if task.get("result"):
                # Handle DataForSEO Labs 'bulk_keyword_difficulty' structure
                # Structure: task['result'][0]['items'] -> list of {keyword, keyword_difficulty, ...}
                first_result = task['result'][0]
                logger.info(f"DEBUG KD Response Result Item 0 Keys: {first_result.keys()}")
                
                items = first_result.get("items", [])
                # If items is empty or missing, check if it's the old flat structure (unlikely now)
                if not items and isinstance(first_result, dict): 
                     # Fallback if structure varies
                     items = task['result'] 

                for item in items:
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

async def get_keyword_trends(keywords: List[str], language_code: str = "en", location_code: int = 2840, date_from: str = None, date_to: str = None) -> List[Dict[str, Any]]:
    """Get keyword trends"""
    return await dataforseo_api.get_keyword_trends(keywords, language_code, location_code, date_from, date_to)

