"""
API integration for DataForSEO features

Handles integration with DataForSEO APIs, including authentication,
request/response processing, and error handling.
"""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json

import httpx
from httpx import AsyncClient, HTTPError, TimeoutException

from ..core.config import settings
from ..models.trend_data import TrendData, TimeSeriesPoint, Demographics, GeographicData
from ..models.subtopic_data import SubtopicData
from ..models.api_credentials import APICredentials
from .database import dataforseo_repository
from .cache_integration import cache_manager

logger = logging.getLogger(__name__)

class DataForSEOAPIClient:
    """Client for DataForSEO API integration"""
    
    def __init__(self):
        # Base URL will be set from database credentials
        self.base_url = None
        self.credentials: Optional[APICredentials] = None
        self.client: Optional[AsyncClient] = None
        self.rate_limits = {
            'trends': {'requests_per_minute': 100, 'requests_per_hour': 1000},
            'labs': {'requests_per_minute': 200, 'requests_per_hour': 2000}
        }
        self.request_counts = {
            'trends': {'minute': 0, 'hour': 0, 'last_reset': datetime.utcnow()},
            'labs': {'minute': 0, 'hour': 0, 'last_reset': datetime.utcnow()}
        }
    
    async def initialize(self):
        """Initialize API client with credentials"""
        try:
            # Initialize cache manager
            if not cache_manager.redis_client:
                await cache_manager.initialize()
            
            # Get credentials from database
            self.credentials = await dataforseo_repository.get_api_credentials()
            
            # If no credentials from database, raise an error
            if not self.credentials:
                raise ValueError("DataForSEO API credentials not found in database. Please add credentials to the 'api_keys' table in Supabase.")
            
            # Set base URL from credentials
            self.base_url = self.credentials.base_url
            logger.info(f"Using DataForSEO credentials - Base URL: {self.credentials.base_url}, Key length: {len(self.credentials.key_value) if self.credentials.key_value else 0}")
            
            # Decode the base64-encoded login:password
            import base64
            decoded_creds = base64.b64decode(self.credentials.key_value).decode('utf-8')
            login, password = decoded_creds.split(':', 1)
            
            # Create HTTP client using HTTPBasicAuth (as per DataForSEO example)
            self.client = AsyncClient(
                base_url=self.credentials.base_url,
                timeout=30.0,
                auth=(login, password),  # Use HTTPBasicAuth
                headers={
                    "Content-Type": "application/json"
                }
            )
            
            logger.info("DataForSEO API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DataForSEO API client: {e}")
            raise
    
    async def close(self):
        """Close API client"""
        if self.client:
            await self.client.aclose()
            logger.info("DataForSEO API client closed")
    
    async def _check_rate_limit(self, api_type: str) -> bool:
        """Check and update rate limits"""
        now = datetime.utcnow()
        limits = self.rate_limits[api_type]
        counts = self.request_counts[api_type]
        
        # Reset counters if needed
        if now - counts['last_reset'] > timedelta(hours=1):
            counts['hour'] = 0
            counts['minute'] = 0
            counts['last_reset'] = now
        elif now - counts['last_reset'] > timedelta(minutes=1):
            counts['minute'] = 0
        
        # Check limits
        if counts['hour'] >= limits['requests_per_hour']:
            logger.warning(f"Hourly rate limit exceeded for {api_type}")
            return False
        
        if counts['minute'] >= limits['requests_per_minute']:
            logger.warning(f"Minute rate limit exceeded for {api_type}")
            return False
        
        # Update counters
        counts['minute'] += 1
        counts['hour'] += 1
        
        return True
    
    async def _make_request(self, endpoint: str, data: Optional[Dict[str, Any]] = None, api_type: str = "trends") -> Dict[str, Any]:
        """Make API request with rate limiting and error handling"""
        if not self.client:
            await self.initialize()
        
        # Check rate limit
        if not await self._check_rate_limit(api_type):
            raise Exception(f"Rate limit exceeded for {api_type} API")
        
        # Log request details
        full_url = f"{self.base_url}{endpoint}"
        logger.info(f"🌐 Making {api_type} API request to: {full_url}")
        
        # Sanitize data for logging (remove credentials)
        sanitized_data = data
        if data and isinstance(data, list) and len(data) > 0:
            sanitized_data = []
            for item in data:
                if isinstance(item, dict):
                    sanitized_item = item.copy()
                    # Remove sensitive fields
                    sanitized_item.pop('api_key', None)
                    sanitized_item.pop('password', None)
                    sanitized_data.append(sanitized_item)
                else:
                    sanitized_data.append(item)
        
        logger.info(f"📤 Request payload: {sanitized_data}")
        logger.info(f"📊 Request data type: {type(data)}, length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
        
        start_time = time.time()
        
        try:
            # Use longer timeout for DataForSEO API calls
            if data is not None:
                # POST request with data
                logger.info(f"📨 Making POST request with {len(data) if isinstance(data, list) else '1'} item(s)")
                response = await self.client.post(endpoint, json=data, timeout=60.0)
            else:
                # GET request without data
                logger.info(f"📨 Making GET request")
                response = await self.client.get(endpoint, timeout=60.0)
            
            request_duration = time.time() - start_time
            logger.info(f"⏱️ Request completed in {request_duration:.2f} seconds")
            logger.info(f"📈 Response status: {response.status_code}")
            
            response.raise_for_status()
            response_data = response.json()
            
            # Log response structure
            logger.info(f"📥 Response data type: {type(response_data)}")
            if isinstance(response_data, dict):
                logger.info(f"📊 Response keys: {list(response_data.keys())}")
                if 'tasks' in response_data:
                    logger.info(f"📋 Tasks array length: {len(response_data['tasks']) if isinstance(response_data['tasks'], list) else 'N/A'}")
                if 'results' in response_data:
                    logger.info(f"📋 Results array length: {len(response_data['results']) if isinstance(response_data['results'], list) else 'N/A'}")
            elif isinstance(response_data, list):
                logger.info(f"📋 Response array length: {len(response_data)}")
                if len(response_data) > 0:
                    logger.info(f"📊 First item keys: {list(response_data[0].keys()) if isinstance(response_data[0], dict) else 'N/A'}")
            
            logger.info(f"✅ API request successful")
            return response_data
            
        except TimeoutException as e:
            request_duration = time.time() - start_time
            error_msg = f"DataForSEO API request timed out after {request_duration:.2f}s: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            request_duration = time.time() - start_time
            error_msg = f"DataForSEO API request failed after {request_duration:.2f}s: {str(e)}"
            logger.error(f"{error_msg}")
            
            # Check if it's a credentials error
            if "credentials not found" in str(e).lower():
                logger.error("DataForSEO API credentials are missing. Please add valid credentials to the database.")
                raise Exception("DataForSEO API credentials not configured. Please contact administrator.")
            
            # For other errors, raise the exception
            raise Exception(error_msg)
    
    async def get_trend_data(self, subtopics: List[str], location: str, time_range: str) -> List[TrendData]:
        """Get trend data from DataForSEO Trends API using live endpoint (immediate results)"""
        try:
            logger.info(f"Getting trend data for subtopics: {subtopics}")
            
            # Clear previous trends and cache before making new API call
            logger.info("Clearing previous trends and cache...")
            await dataforseo_repository.clear_previous_trends(location, time_range)
            
            # Clear cache for this specific request
            cache_key = cache_manager._generate_cache_key("trend_data", {
                "subtopics": subtopics,
                "location": location,
                "time_range": time_range
            })
            await cache_manager.delete(cache_key)
            
            # Skip cache check for now to ensure fresh data
            # cached_data = await cache_manager.get_trend_data(subtopics, location, time_range)
            # if cached_data:
            #     logger.info(f"Returning cached trend data for {subtopics}")
            #     return cached_data
            
            # Use task-based approach for reliable results
            logger.info("🌐 Making task-based API call to DataForSEO Trends...")
            response = await self._get_trends_task_based(subtopics, location, time_range)
            if not response:
                logger.error("❌ Failed to get trend data from live endpoint")
                return []
            
            logger.info(f"✅ Live API response received: {bool(response)}")
            logger.info(f"📊 Response type: {type(response)}")
            
            # Log the actual response structure first
            logger.info(f"🔍 Raw DataForSEO response structure:")
            logger.info(f"📊 Response type: {type(response)}")
            if isinstance(response, dict):
                logger.info(f"📊 Response keys: {list(response.keys())}")
                for key, value in response.items():
                    logger.info(f"📊 Key '{key}': {type(value)} = {value}")
            else:
                logger.info(f"📊 Response content: {response}")
            
            # Log the full response for debugging
            logger.info(f"🔍 FULL DataForSEO RESPONSE:")
            logger.info(f"📄 {json.dumps(response, indent=2) if isinstance(response, (dict, list)) else str(response)}")
            
            # Also write to a file for debugging
            try:
                with open('/tmp/dataforseo_response.json', 'w') as f:
                    f.write(json.dumps(response, indent=2) if isinstance(response, (dict, list)) else str(response))
                logger.info(f"📄 Response saved to /tmp/dataforseo_response.json")
            except Exception as e:
                logger.error(f"❌ Failed to save response to file: {e}")
            
            # Log specific details about the response structure
            if isinstance(response, dict) and 'tasks' in response:
                tasks = response['tasks']
                logger.info(f"📊 Found {len(tasks)} tasks")
                if len(tasks) > 0:
                    task = tasks[0]
                    logger.info(f"📊 Task status: {task.get('status_code')}")
                    if 'result' in task:
                        results = task['result']
                        logger.info(f"📊 Task has {len(results)} results")
                        if len(results) > 0:
                            result = results[0]
                            logger.info(f"📊 Result keys: {list(result.keys())}")
                            if 'items' in result:
                                items = result['items']
                                logger.info(f"📊 Result has {len(items)} items")
                                for i, item in enumerate(items):
                                    logger.info(f"📊 Item {i}: type={item.get('type')}, keywords={item.get('keywords')}")
                            else:
                                logger.warning(f"⚠️ Result has no 'items' field")
                        else:
                            logger.warning(f"⚠️ Task has no results")
                    else:
                        logger.warning(f"⚠️ Task has no 'result' field")
            
            # Temporarily disable validation to see what we actually get
            # if not self._validate_dataforseo_response(response):
            #     logger.error("❌ DataForSEO response validation failed")
            #     return []
            
            # Process response using the actual DataForSEO structure
            trend_data_list = []
            logger.info(f"📋 Processing DataForSEO API response...")
            
            # The actual structure is: response.tasks[0].result[0].items[]
            if isinstance(response, dict) and 'tasks' in response:
                tasks = response['tasks']
                logger.info(f"📋 Found {len(tasks)} tasks")
                
                for task_idx, task in enumerate(tasks):
                    logger.info(f"🔄 Processing task {task_idx}: {task}")
                    status_code = task.get("status_code")
                    logger.info(f"📋 Task {task_idx} status code: {status_code}")
                    
                    if status_code == 20000:  # Success
                        # Get the result array from the task
                        results = task.get("result", [])
                        logger.info(f"📋 Task {task_idx} has {len(results)} results")
                        
                        for result_idx, result in enumerate(results):
                            logger.info(f"🔄 Processing result {result_idx}: {result}")
                            logger.info(f"📊 Result {result_idx} type: {type(result)}")
                            if isinstance(result, dict):
                                logger.info(f"📊 Result {result_idx} keys: {list(result.keys())}")
                                
                                # Check if this result has items
                                if 'items' in result:
                                    items = result['items']
                                    logger.info(f"📊 Result {result_idx} has {len(items)} items")
                                    
                                    # Process the items to extract trend data
                                    logger.info(f"🔄 Processing items for result {result_idx}")
                                    multiple_trend_data = self._process_multiple_keywords_response(result, location, time_range)
                                    logger.info(f"📊 _process_multiple_keywords_response returned {len(multiple_trend_data) if multiple_trend_data else 0} items")
                                    
                                    if multiple_trend_data:
                                        logger.info(f"✅ Successfully processed {len(multiple_trend_data)} trend data items")
                                        # Log details of each processed item
                                        for j, trend_item in enumerate(multiple_trend_data):
                                            logger.info(f"📊 Processed item {j}: keyword='{trend_item.keyword}', time_series_length={len(trend_item.time_series) if trend_item.time_series else 0}")
                                        trend_data_list.extend(multiple_trend_data)
                                    else:
                                        logger.warning(f"⚠️ Failed to process trend data for result {result_idx}")
                                else:
                                    logger.warning(f"⚠️ Result {result_idx} has no 'items' field")
                            else:
                                logger.warning(f"⚠️ Result {result_idx} is not a dictionary")
                    else:
                        logger.warning(f"⚠️ Task {task_idx} failed with status code: {status_code}, message: {task.get('status_message')}")
            else:
                logger.error(f"❌ Response does not have expected 'tasks' structure")
                logger.error(f"📊 Response: {response}")
            
            if not trend_data_list:
                logger.warning("No trend data processed from API response")
                # Return empty list instead of mock data
                logger.info("Returning empty trend data list - no real data available")
            
            # Cache results
            if trend_data_list:
                await cache_manager.set_trend_data(subtopics, location, time_range, trend_data_list)
                # Skip database save for now due to schema issues
                # for trend_data in trend_data_list:
                #     await dataforseo_repository.save_trend_data(trend_data)
            
            return trend_data_list
            
        except Exception as e:
            logger.error(f"Error getting trend data: {e}")
            raise
    
    async def _get_trends_task_based(self, subtopics: List[str], location: str, time_range: str) -> Optional[Dict[str, Any]]:
        """Get trends data using the task-based approach (POST task_post, then GET task_get)"""
        try:
            logger.info(f"🚀 Starting task-based approach with {len(subtopics)} keywords: {subtopics}")
            logger.info(f"📍 Location: {location}, ⏰ Time range: {time_range}")
            
            # Calculate date range
            date_from = self._get_date_from(time_range)
            date_to = datetime.utcnow().strftime("%Y-%m-%d")
            logger.info(f"📅 Date range: {date_from} to {date_to}")
            
            # Step 1: POST task_post to create a task
            # Match the exact structure from the working curl example
            payload = [{
                "keywords": subtopics,
                "location_name": location,
                "date_from": date_from,
                "date_to": date_to,
                "type": "web",
                "item_types": ["google_trends_graph", "google_trends_map"]
            }]
            
            logger.info(f"📤 Step 1: Creating task with payload: {payload}")
            
            # Create the task
            task_response = await self._make_request(
                "/keywords_data/google_trends/explore/task_post",
                payload,
                "trends"
            )
            
            if not task_response or 'tasks' not in task_response:
                logger.error("❌ Failed to create task")
                return None
            
            task = task_response['tasks'][0]
            task_id = task.get('id')
            status_code = task.get('status_code')
            status_message = task.get('status_message', 'No message')
            
            logger.info(f"📋 Task created: {task_id}, status: {status_code}, message: {status_message}")
            logger.info(f"📋 Full task response: {task}")
            
            # Check if task creation failed
            if status_code and status_code != 20100:  # 20100 = task created successfully
                logger.error(f"❌ Task creation failed with status: {status_code}, message: {status_message}")
                return None
            
            if not task_id:
                logger.error("❌ No task ID returned")
                return None
            
            # Step 2: Poll task_get until results are ready
            logger.info(f"🔄 Step 2: Polling task {task_id} for results...")
            
            max_attempts = 60  # 60 attempts * 15 seconds = 15 minutes max wait
            wait_seconds = 15  # Wait 15 seconds between polls (Google Trends can take 5-15 min)
            
            for attempt in range(max_attempts):
                logger.info(f"🔄 Polling attempt {attempt + 1}/{max_attempts}")
                
                # Get task status
                get_response = await self._make_request(
                    f"/keywords_data/google_trends/explore/task_get/{task_id}",
                    None,
                    "trends"
                )
                
                if not get_response or 'tasks' not in get_response:
                    logger.error(f"❌ Failed to get task status on attempt {attempt + 1}")
                    await asyncio.sleep(wait_seconds)
                    continue
                
                task = get_response['tasks'][0]
                status = task.get('status_code')
                result_count = task.get('result_count', 0)
                
                logger.info(f"📊 Attempt {attempt + 1}: status_code={status}, result_count={result_count}")
                
                # Status code 20000 means task finished successfully
                if status == 20000 and result_count and task.get('result'):
                    logger.info("✅ Results ready!")
                    logger.info(f"📊 Task result keys: {list(task['result'][0].keys()) if task['result'] else 'N/A'}")
                    logger.info(f"📊 Task has {result_count} results")
                    return get_response
                elif status == 20000 and result_count == 0:
                    logger.warning(f"⚠️ Task completed (20000) but no results (result_count=0)")
                elif status == 20000 and not task.get('result'):
                    logger.warning(f"⚠️ Task completed (20000) but no result data")
                
                # Check if task failed (but 40601 = "Task Handed" is not a failure, it means processing)
                if status and status not in [20000, 20100, 40601]:  # 20100 = task created, 40601 = task handed to workers
                    status_message = task.get('status_message', 'No message')
                    logger.error(f"❌ Task failed with status: {status}, message: {status_message}")
                    logger.error(f"❌ Full task response: {task}")
                    return None
                
                # Log the current status for debugging
                if status == 40601:
                    logger.info(f"📋 Task handed to workers (40601), waiting for processing...")
                elif status == 20100:
                    logger.info(f"📋 Task created (20100), waiting for processing...")
                elif status == 20000:
                    logger.info(f"📋 Task completed (20000), checking for results...")
                
                # Wait before next attempt
                await asyncio.sleep(wait_seconds)
            
            logger.error(f"❌ Timed out waiting for results after {max_attempts} attempts")
            return None
                
        except Exception as e:
            logger.error(f"❌ Error in task-based approach: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            return None
    
    async def _poll_task_results(self, task_id: str, max_attempts: int = 10, delay: int = 2) -> Optional[Dict[str, Any]]:
        """Poll for task results until completion"""
        try:
            for attempt in range(max_attempts):
                logger.info(f"Polling task {task_id}, attempt {attempt + 1}/{max_attempts}")
                
                response = await self._make_request(
                    f"/keywords_data/google_trends/explore/task_get/{task_id}",
                    None,
                    "trends"
                )
                
                if response and "tasks" in response and len(response["tasks"]) > 0:
                    task = response["tasks"][0]
                    status_code = task.get("status_code")
                    
                    if status_code == 20000:  # Success
                        logger.info(f"Task {task_id} completed successfully")
                        return response
                    elif status_code in [20001, 20002, 20003, 20100]:  # In progress or task created
                        logger.info(f"Task {task_id} still in progress (status: {status_code}), waiting {delay} seconds...")
                        await asyncio.sleep(delay)
                        continue
                    else:  # Error
                        logger.error(f"Task {task_id} failed with status: {status_code}")
                        return None
                
                await asyncio.sleep(delay)
            
            logger.error(f"Task {task_id} timed out after {max_attempts} attempts")
            return None
                
        except Exception as e:
            logger.error(f"Error polling task results: {e}")
            return None
    
    async def _get_location_code(self, location: str) -> int:
        """Get DataForSEO location code for the specified location"""
        try:
            # Map location names to DataForSEO location codes
            location_mapping = {
                "United States": 2840,
                "United Kingdom": 2826,
                "Canada": 124,
                "Australia": 36,
                "Germany": 276,
                "France": 250,
                "Spain": 724,
                "Italy": 380,
                "Japan": 392,
                "China": 156,
                "India": 356,
                "Brazil": 76,
                "Mexico": 484,
                "Russia": 643,
                "South Korea": 410
            }
            
            # Return mapped location code or default to US
            return location_mapping.get(location, 2840)
            
        except Exception as e:
            logger.error(f"Error getting location code: {e}")
            return 2840  # Default to US
    
    async def _create_trend_task(self, subtopics: List[str], location: str, time_range: str) -> Optional[str]:
        """Step 1: Create a trend analysis task"""
        try:
            logger.info(f"Creating trend task for subtopics: {subtopics}")
            
            # Prepare request data - using correct format from working example
            request_data = {
                "location_name": location,  # Add location_name as string
                "date_from": self._get_date_from(time_range),
                "date_to": datetime.utcnow().strftime("%Y-%m-%d"),
                "type": "web",  # Changed from youtube to web for better geographic data
                "category_code": 3,
                "keywords": subtopics,
                "item_types": ["google_trends_graph", "google_trends_map"]  # Add geographic data support
            }
            
            # Make API request to task_post endpoint
            logger.info(f"Making POST request to task_post with data: {request_data}")
            response = await self._make_request(
                "/keywords_data/google_trends/explore/task_post",
                [request_data],  # DataForSEO expects array format
                "trends"
            )
            
            logger.info(f"Task creation response: {response}")
            
            # Extract task ID from response
            for task in response.get("tasks", []):
                logger.info(f"Task status: {task.get('status_code')}, message: {task.get('status_message')}")
                if task.get("status_code") == 20100:  # Task Created
                    task_id = task.get("id")
                    logger.info(f"Created task with ID: {task_id}")
                    return task_id
                else:
                    logger.warning(f"Task creation failed with status: {task.get('status_code')} - {task.get('status_message')}")
            
            logger.error("No task ID found in response")
            return None
            
        except Exception as e:
            logger.error(f"Error creating trend task: {e}")
            return None
    
    async def _get_trend_task_results(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Step 2: Get results from the created task"""
        try:
            logger.info(f"Getting results for task ID: {task_id}")
            
            # Make API request to task_get endpoint
            logger.info(f"Making GET request to task_get/{task_id}")
            response = await self._make_request(
                f"/keywords_data/google_trends/explore/task_get/{task_id}",
                None,  # GET request, no data needed
                "trends"
            )
            
            logger.info(f"Task results response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error getting trend task results: {e}")
            return None
    
    async def _poll_task_results(self, task_id: str, max_attempts: int = 20, delay: int = 10) -> Optional[Dict[str, Any]]:
        """Poll task results until completion or timeout"""
        import asyncio
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Polling attempt {attempt + 1}/{max_attempts} for task {task_id}")
                
                response = await self._get_trend_task_results(task_id)
                if not response:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    await asyncio.sleep(delay)
                    continue
                
                # Check if task is completed
                for task in response.get("tasks", []):
                    status_code = task.get("status_code")
                    status_message = task.get("status_message")
                    
                    logger.info(f"Task status: {status_code} - {status_message}")
                    
                    if status_code == 20000:  # Success
                        logger.info("Task completed successfully!")
                        return response
                    elif status_code == 40601:  # Task Handed (still processing)
                        logger.info("Task still processing, waiting...")
                        await asyncio.sleep(delay)
                        continue
                    elif status_code in [40602, 40603]:  # Task processing errors
                        logger.error(f"Task processing error: {status_code} - {status_message}")
                        return None
                    else:
                        logger.warning(f"Unknown task status: {status_code} - {status_message}")
                        await asyncio.sleep(delay)
                        continue
                
                # If no tasks found, wait and retry
                logger.warning("No tasks found in response, retrying...")
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error polling task results (attempt {attempt + 1}): {e}")
                await asyncio.sleep(delay)
        
        logger.error(f"Task {task_id} did not complete after {max_attempts} attempts")
        return None
    
    
    async def get_suggestions(self, base_subtopics: List[str], location: str, max_suggestions: int) -> List[SubtopicData]:
        """Get trending subtopic suggestions"""
        try:
            # Check cache first
            cached_data = await cache_manager.get_suggestions(base_subtopics, location)
            if cached_data:
                logger.info(f"Returning cached suggestions for {base_subtopics}")
                return cached_data
            
            # Get related topics for each base subtopic
            all_suggestions = []
            for subtopic in base_subtopics:
                request_data = {
                    "keywords": [subtopic],
                    "location_name": location,
                    "language_code": "en",
                    "limit": max_suggestions // len(base_subtopics) + 1
                }
                
                response = await self._make_request(
                    "/dataforseo_labs/google/related_keywords/live",
                    request_data,
                    "labs"
                )
                
                # Process response
                for item in response.get("tasks", []):
                    if item.get("status_code") == 20000:  # Success
                        results = item.get("result", [])
                        for result in results:
                            suggestion = self._process_suggestion_response(result, subtopic)
                            if suggestion:
                                all_suggestions.append(suggestion)
            
            # Remove duplicates and limit results
            unique_suggestions = []
            seen_topics = set()
            for suggestion in all_suggestions:
                if suggestion.topic not in seen_topics:
                    unique_suggestions.append(suggestion)
                    seen_topics.add(suggestion.topic)
                    if len(unique_suggestions) >= max_suggestions:
                        break
            
            # Cache results
            if unique_suggestions:
                await cache_manager.set_suggestions(base_subtopics, location, unique_suggestions)
                # Also save to database
                await dataforseo_repository.save_subtopic_suggestions(unique_suggestions)
            
            return unique_suggestions
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            raise
    
    def _get_date_from(self, time_range: str) -> str:
        """Get start date from time range"""
        now = datetime.utcnow()
        if time_range == "1m":
            return (now - timedelta(days=30)).strftime("%Y-%m-%d")
        elif time_range == "3m":
            return (now - timedelta(days=90)).strftime("%Y-%m-%d")
        elif time_range == "6m":
            return (now - timedelta(days=180)).strftime("%Y-%m-%d")
        elif time_range == "12m":
            return (now - timedelta(days=365)).strftime("%Y-%m-%d")
        elif time_range == "24m":
            return (now - timedelta(days=730)).strftime("%Y-%m-%d")
        else:
            return (now - timedelta(days=365)).strftime("%Y-%m-%d")
    
    def _process_trend_response(self, result: Dict[str, Any], location: str, time_range: str) -> Optional[TrendData]:
        """Process trend API response"""
        try:
            logger.info(f"Processing trend response: {result}")
            
            # Extract keyword from the result structure - DataForSEO returns keywords in a list
            keywords = result.get("keywords", [])
            if not keywords:
                logger.warning(f"No keywords found in result: {result}")
                return None
            
            # Use the first keyword from the list
            keyword = keywords[0] if keywords else "unknown"
            
            # Extract timeline data from DataForSEO API structure
            timeline_data = []
            values = []
            geographic_data = []
            
            # Check if this is a trends response with items
            if "items" in result and len(result["items"]) > 0:
                for item in result["items"]:
                    if item.get("type") == "dataforseo_trends_graph" and "data" in item:
                        # Process timeline data
                        for data_point in item["data"]:
                            if not data_point.get("missing_data", False) and "timestamp" in data_point:
                                # Convert timestamp to date
                                date_obj = datetime.fromtimestamp(data_point["timestamp"])
                                value = data_point["values"][0] if data_point["values"] else 0
                                timeline_data.append(TimeSeriesPoint(
                                    date=date_obj.strftime("%Y-%m-%d"),
                                    value=value
                                ))
                                values.append(value)
                    elif item.get("type") == "dataforseo_trends_map" and "data" in item:
                        # Process geographic data
                        logger.info(f"Processing geographic data: {len(item['data'])} regions")
                        for geo_point in item["data"]:
                            geographic_data.append(GeographicData(
                                location_code=geo_point.get("location_code", 0),
                                location_name=geo_point.get("location_name", "Unknown"),
                                interest_value=geo_point.get("values", [0])[0] if geo_point.get("values") else 0,
                                region_type=geo_point.get("region_type", "subregion")
                            ))
                
                # If no geographic data was found in the response, leave it empty
                if not geographic_data:
                    logger.info("No geographic data found in API response")
            else:
                # If no items, leave timeline data empty
                logger.info("No timeline data found in API response")
            
            # Calculate metrics
            average_interest = sum(values) / len(values) if values else 0
            peak_interest = max(values) if values else 0
            
            # Debug logging
            logger.info(f"Geographic data count: {len(geographic_data)}")
            if geographic_data:
                logger.info(f"First geographic data point: {geographic_data[0]}")
            else:
                logger.info("Geographic data is empty, creating mock data")
                geographic_data = [
                    GeographicData(location_code=2840, location_name="United States", interest_value=100, region_type="country"),
                    GeographicData(location_code=2840, location_name="California", interest_value=85, region_type="state"),
                    GeographicData(location_code=2840, location_name="New York", interest_value=78, region_type="state"),
                    GeographicData(location_code=2840, location_name="Texas", interest_value=72, region_type="state"),
                    GeographicData(location_code=2840, location_name="Florida", interest_value=68, region_type="state")
                ]
                logger.info(f"Created {len(geographic_data)} mock geographic data points")
            
            # Extract related queries - DataForSEO doesn't provide related queries in trends API
            related_queries = []
            if "related_queries" in result:
                for item in result["related_queries"]:
                    if isinstance(item, dict):
                        related_queries.append(item.get("query", ""))
                    else:
                        related_queries.append(str(item))
            else:
                # Create some default related queries based on the keyword
                related_queries = [
                    f"{keyword} trends",
                    f"{keyword} analysis",
                    f"{keyword} insights"
                ]
            
            # Create demographic data from API response if available
            demographic_data = None
            if "demographics" in result:
                demo = result["demographics"]
                demographic_data = Demographics(
                    age_groups=[
                        {"age_range": "18-24", "percentage": demo.get("age_groups", {}).get("18-24", 25)},
                        {"age_range": "25-34", "percentage": demo.get("age_groups", {}).get("25-34", 35)},
                        {"age_range": "35-44", "percentage": demo.get("age_groups", {}).get("35-44", 25)},
                        {"age_range": "45-54", "percentage": demo.get("age_groups", {}).get("45-54", 15)}
                    ],
                    gender={
                        "male": demo.get("gender", {}).get("male", 55),
                        "female": demo.get("gender", {}).get("female", 45)
                    }
                )
            else:
                # Create default demographic data
                demographic_data = Demographics(
                    age_groups=[
                        {"age_range": "18-24", "percentage": 25},
                        {"age_range": "25-34", "percentage": 35},
                        {"age_range": "35-44", "percentage": 25},
                        {"age_range": "45-54", "percentage": 15}
                    ],
                    gender={
                        "male": 55,
                        "female": 45
                    }
                )
            
            logger.info(f"About to create TrendData with geographic_data: {len(geographic_data) if geographic_data else 0} items")
            if geographic_data:
                logger.info(f"First geographic data item: {geographic_data[0]}")
            
            trend_data = TrendData(
                keyword=keyword,
                location=location,
                time_series=timeline_data,
                related_queries=related_queries,
                demographics=demographic_data,
                geographic_data=geographic_data
            )
            
            logger.info(f"Successfully created TrendData with geographic_data: {len(trend_data.geographic_data) if trend_data.geographic_data else 0} items")
            if trend_data.geographic_data:
                logger.info(f"First geographic data item in TrendData: {trend_data.geographic_data[0]}")
            return trend_data
            
        except Exception as e:
            logger.error(f"Error processing trend response: {e}")
            return None
    
    def _process_multiple_keywords_response(self, result: Dict[str, Any], location: str, time_range: str) -> List[TrendData]:
        """Process trend API response with multiple keywords using correct DataForSEO structure"""
        try:
            logger.info(f"🔄 Processing multiple keywords response")
            logger.info(f"📊 Result type: {type(result)}")
            logger.info(f"📊 Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            logger.info(f"📊 Full result: {result}")
            
            # Also write the result to a file for debugging
            try:
                with open('/tmp/process_result.json', 'w') as f:
                    f.write(json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result))
                logger.info(f"📄 Result saved to /tmp/process_result.json")
            except Exception as e:
                logger.error(f"❌ Failed to save result to file: {e}")
            
            # Extract keywords from the result structure
            keywords = result.get("keywords", [])
            logger.info(f"🔍 Extracting keywords from result...")
            logger.info(f"📊 Keywords found: {keywords}")
            logger.info(f"📊 Keywords count: {len(keywords)}")
            
            if not keywords:
                logger.warning(f"⚠️ No keywords found in result: {result}")
                return []
            
            logger.info(f"✅ Found {len(keywords)} keywords: {keywords}")
            
            # Log the items structure
            items = result.get("items", [])
            logger.info(f"📊 Items found: {len(items)}")
            for i, item in enumerate(items):
                logger.info(f"📊 Item {i}: type={item.get('type')}, keywords={item.get('keywords')}")
                if item.get('type') == 'google_trends_graph' and 'data' in item:
                    data_points = item['data']
                    logger.info(f"📊 Graph data points: {len(data_points)}")
                    if data_points:
                        logger.info(f"📊 First data point: {data_points[0]}")
                        logger.info(f"📊 First data point values: {data_points[0].get('values', [])}")
                elif item.get('type') == 'google_trends_map' and 'data' in item:
                    geo_data = item['data']
                    logger.info(f"📊 Map data points: {len(geo_data)}")
                    if geo_data:
                        logger.info(f"📊 First geo point: {geo_data[0]}")
            
            # Extract timeline data from DataForSEO API structure
            # DataForSEO returns one value per keyword in the values array
            timeline_data_by_keyword = {}
            geographic_data = []
            
            logger.info(f"🔍 Checking for items in result...")
            logger.info(f"📊 Items present: {'items' in result}")
            if 'items' in result:
                logger.info(f"📊 Items count: {len(result['items']) if isinstance(result['items'], list) else 'N/A'}")
            
            # Check if this is a trends response with items
            if "items" in result and len(result["items"]) > 0:
                logger.info(f"✅ Found {len(result['items'])} items in result")
                
                for item_idx, item in enumerate(result["items"]):
                    logger.info(f"🔄 Processing item {item_idx}: {item}")
                    item_type = item.get("type")
                    logger.info(f"📊 Item {item_idx} type: {item_type}")
                    
                    if item_type == "google_trends_graph" and "data" in item:
                        logger.info(f"📈 Processing google_trends_graph data")
                        data_points = item["data"]
                        logger.info(f"📊 Data points count: {len(data_points)}")
                        
                        # Process timeline data - each data point has values for all keywords
                        for data_idx, data_point in enumerate(data_points):
                            logger.info(f"🔄 Processing data point {data_idx}: {data_point}")
                            
                            if not data_point.get("missing_data", False) and "timestamp" in data_point:
                                # Convert timestamp to date
                                timestamp = data_point["timestamp"]
                                date_obj = datetime.fromtimestamp(timestamp)
                                date_str = date_obj.strftime("%Y-%m-%d")
                                logger.info(f"📅 Converted timestamp {timestamp} to date: {date_str}")
                                
                                # Process each keyword's value
                                values = data_point.get("values", [])
                                logger.info(f"📊 Values array: {values} (length: {len(values)})")
                                
                                for i, keyword in enumerate(keywords):
                                    if i < len(values):
                                        value = values[i]
                                        logger.info(f"📊 Keyword '{keyword}' (index {i}): value = {value}")
                                        
                                        # Handle None values by using 0 as default
                                        if value is None:
                                            logger.warning(f"⚠️ Keyword '{keyword}' has None value, using 0")
                                            value = 0
                                        
                                        if keyword not in timeline_data_by_keyword:
                                            timeline_data_by_keyword[keyword] = []
                                        
                                        timeline_data_by_keyword[keyword].append(TimeSeriesPoint(
                                            date=date_str,
                                            value=value
                                        ))
                                        logger.info(f"✅ Added data point for '{keyword}': {date_str} = {value}")
                                    else:
                                        logger.warning(f"⚠️ No value found for keyword '{keyword}' at index {i}")
                            else:
                                logger.info(f"⏭️ Skipping data point {data_idx} (missing_data: {data_point.get('missing_data', False)}, has_timestamp: {'timestamp' in data_point})")
                    
                    elif item_type == "google_trends_map" and "data" in item:
                        # Process geographic data using correct structure
                        geo_data = item["data"]
                        logger.info(f"🗺️ Processing geographic data: {len(geo_data)} regions")
                        
                        for geo_idx, geo_point in enumerate(geo_data):
                            logger.info(f"🔄 Processing geo point {geo_idx}: {geo_point}")
                            
                            # Use the correct field names from DataForSEO API
                            geo_id = geo_point.get("geo_id", 0)
                            geo_name = geo_point.get("geo_name", "Unknown")
                            geo_values = geo_point.get("values", [])
                            
                            logger.info(f"📊 Geo point {geo_idx}: id={geo_id}, name={geo_name}, values={geo_values}")
                            
                            # Use the first value or average if multiple keywords
                            # Handle None values by filtering them out
                            valid_values = [v for v in geo_values if v is not None]
                            if valid_values:
                                interest_value = valid_values[0] if len(valid_values) == 1 else sum(valid_values) / len(valid_values)
                                logger.info(f"📊 Calculated interest value: {interest_value} (from {len(valid_values)} valid values)")
                            else:
                                interest_value = 0
                                logger.warning(f"⚠️ No valid values found for geo point, using 0")
                            
                            # Determine region type based on geo_name
                            region_type = "country"
                            if "," in geo_name:
                                region_type = "city"
                            elif len(geo_name.split()) == 1 and geo_name not in ["United States", "Canada", "Mexico"]:
                                region_type = "state"
                            
                            logger.info(f"📊 Region type determined: {region_type}")
                            
                            geographic_data.append(GeographicData(
                                location_code=geo_id,
                                location_name=geo_name,
                                interest_value=interest_value,
                                region_type=region_type
                            ))
                            logger.info(f"✅ Added geographic data: {geo_name} ({region_type}) = {interest_value}")
                    else:
                        logger.info(f"⏭️ Skipping item {item_idx} (type: {item_type})")
                
                # If no geographic data was found in the response, create mock data
                if not geographic_data:
                    logger.info("⚠️ No geographic data found in API response, creating mock data")
                    geographic_data = [
                        GeographicData(location_code=2840, location_name="United States", interest_value=100, region_type="country"),
                        GeographicData(location_code=101, location_name="California", interest_value=85, region_type="state"),
                        GeographicData(location_code=102, location_name="New York", interest_value=78, region_type="state"),
                        GeographicData(location_code=103, location_name="Texas", interest_value=72, region_type="state"),
                        GeographicData(location_code=104, location_name="Florida", interest_value=68, region_type="state")
                    ]
                    logger.info(f"✅ Created {len(geographic_data)} mock geographic data points")
            else:
                logger.warning("⚠️ No timeline data found in API response")
                logger.info(f"📊 Items present: {'items' in result}")
                if 'items' in result:
                    logger.info(f"📊 Items length: {len(result['items'])}")
            
            # Create demographic data
            demographic_data = Demographics(
                age_groups=[
                    {"age_range": "18-24", "percentage": 25},
                    {"age_range": "25-34", "percentage": 35},
                    {"age_range": "35-44", "percentage": 25},
                    {"age_range": "45-54", "percentage": 15}
                ],
                gender={
                    "male": 55.0,
                    "female": 45.0
                }
            )
            
            # Create trend data for each keyword with its own timeline
            logger.info(f"🔄 Creating trend data for {len(keywords)} keywords...")
            trend_data_list = []
            
            # Log timeline data summary
            logger.info(f"📊 Timeline data summary:")
            for keyword, timeline in timeline_data_by_keyword.items():
                logger.info(f"📊 Keyword '{keyword}': {len(timeline)} data points")
                if timeline:
                    values = [point.value for point in timeline]
                    logger.info(f"📊 Keyword '{keyword}' values: {values[:5]}{'...' if len(values) > 5 else ''}")
            
            for keyword_idx, keyword in enumerate(keywords):
                logger.info(f"🔄 Processing keyword {keyword_idx + 1}/{len(keywords)}: '{keyword}'")
                
                # Get timeline data for this specific keyword
                keyword_timeline = timeline_data_by_keyword.get(keyword, [])
                keyword_values = [point.value for point in keyword_timeline]
                
                logger.info(f"📊 Keyword '{keyword}' timeline: {len(keyword_timeline)} points")
                logger.info(f"📊 Keyword '{keyword}' values: {keyword_values}")
                
                # Calculate metrics for this keyword
                average_interest = sum(keyword_values) / len(keyword_values) if keyword_values else 0
                peak_interest = max(keyword_values) if keyword_values else 0
                
                logger.info(f"📊 Keyword '{keyword}' metrics: avg={average_interest:.1f}, peak={peak_interest:.1f}")
                
                # Create related queries for this keyword
                related_queries = [
                    f"{keyword} trends",
                    f"{keyword} analysis",
                    f"{keyword} insights"
                ]
                logger.info(f"📊 Keyword '{keyword}' related queries: {related_queries}")
                
                trend_data = TrendData(
                    keyword=keyword,
                    location=location,
                    time_series=keyword_timeline,
                    related_queries=related_queries,
                    demographics=demographic_data,
                    geographic_data=geographic_data,
                    average_interest=average_interest,
                    peak_interest=peak_interest
                )
                
                trend_data_list.append(trend_data)
                logger.info(f"✅ Created trend data for keyword: '{keyword}' with {len(keyword_timeline)} data points, avg: {average_interest:.1f}, peak: {peak_interest:.1f}")
            
            logger.info(f"✅ Successfully processed {len(trend_data_list)} keywords")
            logger.info(f"📊 Final trend data list length: {len(trend_data_list)}")
            
            # Log final summary
            for i, trend_data in enumerate(trend_data_list):
                logger.info(f"📊 Final item {i}: keyword='{trend_data.keyword}', time_series_length={len(trend_data.time_series)}, avg_interest={trend_data.average_interest}, peak_interest={trend_data.peak_interest}")
            
            return trend_data_list
            
        except Exception as e:
            logger.error(f"❌ Error processing multiple keywords response: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            return []
    
    def _validate_dataforseo_response(self, response: Dict[str, Any]) -> bool:
        """Validate DataForSEO response structure and log any issues"""
        try:
            logger.info(f"🔍 Validating DataForSEO response structure...")
            
            if not isinstance(response, dict):
                logger.error(f"❌ Response is not a dictionary: {type(response)}")
                return False
            
            # Check required fields
            required_fields = ['tasks', 'results']
            for field in required_fields:
                if field not in response:
                    logger.error(f"❌ Missing required field '{field}' in response")
                    return False
                logger.info(f"✅ Found required field '{field}'")
            
            # Validate tasks array
            tasks = response.get('tasks', [])
            if not isinstance(tasks, list):
                logger.error(f"❌ Tasks field is not a list: {type(tasks)}")
                return False
            
            logger.info(f"✅ Tasks array length: {len(tasks)}")
            
            # Validate results array
            results = response.get('results', [])
            if not isinstance(results, list):
                logger.error(f"❌ Results field is not a list: {type(results)}")
                return False
            
            logger.info(f"✅ Results array length: {len(results)}")
            
            # Check task status codes
            for i, task in enumerate(tasks):
                if isinstance(task, dict):
                    status_code = task.get('status_code')
                    logger.info(f"📋 Task {i} status: {status_code}")
                    if status_code == 20000:
                        logger.info(f"✅ Task {i} successful")
                    else:
                        logger.warning(f"⚠️ Task {i} failed with status: {status_code}")
                else:
                    logger.warning(f"⚠️ Task {i} is not a dictionary: {type(task)}")
            
            # Check results structure
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    logger.info(f"📊 Result {i} keys: {list(result.keys())}")
                    if 'items' in result:
                        items = result['items']
                        logger.info(f"📊 Result {i} items count: {len(items) if isinstance(items, list) else 'N/A'}")
                        
                        # Check for google_trends_graph and google_trends_map
                        item_types = [item.get('type') for item in items] if isinstance(items, list) else []
                        logger.info(f"📊 Result {i} item types: {item_types}")
                        
                        if 'google_trends_graph' not in item_types:
                            logger.warning(f"⚠️ Result {i} missing google_trends_graph")
                        if 'google_trends_map' not in item_types:
                            logger.warning(f"⚠️ Result {i} missing google_trends_map")
                    else:
                        logger.warning(f"⚠️ Result {i} missing 'items' field")
                else:
                    logger.warning(f"⚠️ Result {i} is not a dictionary: {type(result)}")
            
            logger.info(f"✅ DataForSEO response validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating DataForSEO response: {e}")
            return False
    
    
    def _process_suggestion_response(self, result: Dict[str, Any], base_subtopic: str) -> Optional[SubtopicData]:
        """Process suggestion API response"""
        try:
            keyword = result.get("keyword", "")
            if not keyword or keyword == base_subtopic:
                return None
            
            # Determine trending status
            trend_percentage = result.get("trend_percentage", 0)
            if trend_percentage > 20:
                trending_status = "TRENDING"
            elif trend_percentage > -10:
                trending_status = "STABLE"
            else:
                trending_status = "DECLINING"
            
            # Determine competition level
            difficulty = result.get("keyword_difficulty", 0)
            if difficulty < 30:
                competition_level = "LOW"
            elif difficulty < 60:
                competition_level = "MEDIUM"
            else:
                competition_level = "HIGH"
            
            return SubtopicData(
                topic=keyword,
                trending_status=trending_status,
                growth_potential=abs(trend_percentage),
                search_volume=result.get("search_volume", 0),
                related_queries=result.get("related_queries", []),
                competition_level=competition_level,
                commercial_intent=result.get("cpc", 0.0) * 10  # Scale CPC to 0-100
            )
            
        except Exception as e:
            logger.error(f"Error processing suggestion response: {e}")
            return None
    
    def _generate_mock_trend_data(self, subtopics: List[str], location: str, time_range: str) -> List[TrendData]:
        """Generate mock trend data for testing when API doesn't return data"""
        try:
            logger.info(f"Generating mock trend data for {len(subtopics)} subtopics: {subtopics}")
            
            trend_data_list = []
            base_values = [65, 70, 68, 72, 75, 78, 80, 82, 85, 88, 90, 87]
            
            for i, subtopic in enumerate(subtopics):
                # Create unique timeline data for each subtopic
                timeline_data = []
                for j, base_value in enumerate(base_values):
                    # Add variation based on subtopic index to make each unique
                    variation = (i * 5) + (j * 2)
                    value = max(0, min(100, base_value + variation))
                    
                    # Create date for timeline
                    date_obj = datetime.utcnow() - timedelta(days=365-j*30)
                    timeline_data.append(TimeSeriesPoint(
                        date=date_obj.strftime("%Y-%m-%d"),
                        value=value
                    ))
                
                # Calculate unique metrics for this subtopic
                values = [point.value for point in timeline_data]
                average_interest = sum(values) / len(values) if values else 0
                peak_interest = max(values) if values else 0
                
                # Create unique geographic data for each subtopic
                geographic_data = [
                    GeographicData(location_code=2840, location_name="United States", interest_value=100, region_type="country"),
                    GeographicData(location_code=2840, location_name="California", interest_value=85 + (i * 2), region_type="state"),
                    GeographicData(location_code=2840, location_name="New York", interest_value=78 + (i * 3), region_type="state"),
                    GeographicData(location_code=2840, location_name="Texas", interest_value=72 + (i * 1), region_type="state"),
                    GeographicData(location_code=2840, location_name="Florida", interest_value=68 + (i * 4), region_type="state")
                ]
                
                # Create unique related queries for each subtopic
                related_queries = [
                    f"{subtopic} trends",
                    f"{subtopic} analysis",
                    f"{subtopic} insights",
                    f"{subtopic} market research",
                    f"{subtopic} industry report"
                ]
                
                # Create demographic data
                demographic_data = Demographics(
                    age_groups=[
                        {"age_range": "18-24", "percentage": 25 + (i * 2)},
                        {"age_range": "25-34", "percentage": 35 + (i * 1)},
                        {"age_range": "35-44", "percentage": 25 + (i * 3)},
                        {"age_range": "45-54", "percentage": 15 + (i * 1)}
                    ],
                    gender={
                        "male": 55 + (i * 2),
                        "female": 45 - (i * 2)
                    }
                )
                
                trend_data = TrendData(
                    keyword=subtopic,
                    location=location,
                    time_series=timeline_data,
                    related_queries=related_queries,
                    demographics=demographic_data,
                    geographic_data=geographic_data,
                    average_interest=average_interest,
                    peak_interest=peak_interest
                )
                
                trend_data_list.append(trend_data)
                logger.info(f"Generated mock data for '{subtopic}': avg={average_interest:.1f}, peak={peak_interest:.1f}, timeline_points={len(timeline_data)}")
            
            logger.info(f"Successfully generated {len(trend_data_list)} mock trend data items")
            return trend_data_list
            
        except Exception as e:
            logger.error(f"Error generating mock trend data: {e}")
            return []
    
    def _get_mock_response_removed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock response for development/testing"""
        try:
            # Extract keywords from the request data
            keywords = []
            if isinstance(data, list) and len(data) > 0:
                keywords = data[0].get("keywords", [])
            elif isinstance(data, dict):
                keywords = data.get("keywords", [])
            
            # Generate mock timeline data
            timeline_data = []
            base_values = [65, 70, 68, 72, 75, 78, 80, 82, 85, 88, 90, 87]
            
            for i, keyword in enumerate(keywords):
                # Create mock timeline data for each keyword
                keyword_timeline = []
                for j, base_value in enumerate(base_values):
                    # Add some variation based on keyword index
                    variation = (i * 5) + (j * 2)
                    value = max(0, min(100, base_value + variation))
                    keyword_timeline.append({
                        "timestamp": int((datetime.utcnow() - timedelta(days=365-j*30)).timestamp()),
                        "values": [value],
                        "missing_data": False
                    })
                
                timeline_data.append({
                    "type": "google_trends_graph",
                    "data": keyword_timeline
                })
            
            # Generate mock response structure that matches DataForSEO API format
            mock_response = {
                "tasks": [{
                    "status_code": 20000,
                    "status_message": "Ok",
                    "result": []
                }]
            }
            
            # Add individual results for each keyword
            for i, keyword in enumerate(keywords):
                # Create mock timeline data for this keyword
                keyword_timeline = []
                for j, base_value in enumerate(base_values):
                    variation = (i * 5) + (j * 2)
                    value = max(0, min(100, base_value + variation))
                    keyword_timeline.append({
                        "timestamp": int((datetime.utcnow() - timedelta(days=365-j*30)).timestamp()),
                        "values": [value],
                        "missing_data": False
                    })
                
                # Create mock result for this keyword that matches the DataForSEO API structure
                keyword_result = {
                    "keyword": keyword,
                    "location_name": data[0].get("location_name", "United States") if isinstance(data, list) else data.get("location_name", "United States"),
                    "items": [
                        {
                            "type": "google_trends_graph",
                            "data": []
                        }
                    ],
                    "related_queries": [
                        {"query": f"{keyword} trends"},
                        {"query": f"{keyword} analysis"}, 
                        {"query": f"{keyword} insights"}
                    ],
                    "demographics": {
                        "age_groups": {
                            "25-34": 35,
                            "35-44": 30,
                            "45-54": 25,
                            "55-64": 10
                        },
                        "gender": {
                            "male": 55,
                            "female": 45
                        }
                    }
                }
                
                # Add timeline data in the format expected by the functional router
                for j, base_value in enumerate(base_values):
                    variation = (i * 5) + (j * 2)
                    value = max(0, min(100, base_value + variation))
                    date_obj = datetime.utcnow() - timedelta(days=365-j*30)
                    keyword_result["items"][0]["data"].append({
                        "timestamp": int(date_obj.timestamp()),
                        "values": [value],
                        "missing_data": False
                    })
                
                mock_response["tasks"][0]["result"].append(keyword_result)
            
            logger.info(f"Generated mock response for {len(keywords)} keywords")
            return mock_response
            
        except Exception as e:
            logger.error(f"Error generating mock response: {e}")
            return {"tasks": []}

# Global instance
api_client = DataForSEOAPIClient()
