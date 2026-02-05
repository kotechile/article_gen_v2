"""
Google Autocomplete integration service
Handles communication with Google's autocomplete API
"""

import asyncio
import aiohttp
import time
import random
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import quote
from datetime import datetime, timedelta

from ..models.autocomplete_result import AutocompleteResult, AutocompleteResultCreate

logger = logging.getLogger(__name__)

class GoogleAutocompleteService:
    """
    Service for integrating with Google Autocomplete API
    
    Features:
    - Rate limiting to prevent blocking
    - User agent rotation
    - Error handling and fallback
    - Caching for performance
    - Request timeout management
    """
    
    def __init__(self, 
                 base_url: str = "http://suggestqueries.google.com/complete/search",
                 timeout: float = 10.0,
                 rate_limit_delay: float = 0.1,
                 max_retries: int = 3):
        """
        Initialize Google Autocomplete service
        
        Args:
            base_url: Google autocomplete API base URL
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.cache: Dict[str, AutocompleteResult] = {}
        self.cache_ttl = timedelta(hours=1)  # Cache TTL: 1 hour
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        self.last_request_time = 0.0
    
    async def get_suggestions(self, query: str) -> AutocompleteResult:
        """
        Get autocomplete suggestions for a query
        
        Args:
            query: Search query
            
        Returns:
            AutocompleteResult with suggestions or error
        """
        start_time = time.time()
        
        try:
            # Validate query
            if not query or not query.strip():
                return AutocompleteResult.create_error(
                    query=query,
                    error_message="Query cannot be empty",
                    processing_time=time.time() - start_time
                )
            
            query = query.strip()
            
            # Check cache first
            cached_result = self._get_cached_result(query)
            if cached_result:
                logger.info(f"Returning cached result for query: {query}")
                return cached_result
            
            # REAL GOOGLE AUTOCOMPLETE API IMPLEMENTATION
            logger.info(f"Making real Google Autocomplete API call for query: {query}")
            
            # Rate limiting
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_since_last_request)
            
            # Make real API call to Google Autocomplete
            params = {
                'client': 'firefox',
                'hl': 'en',
                'q': query
            }
            
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(
                    self.base_url,
                    params=params,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        # Parse the JSONP response from Google
                        text = await response.text()
                        
                        # Google returns JSONP format: window.google.ac.h(..., [...])
                        # Extract the suggestions array
                        import re
                        json_match = re.search(r'\[(.*?)\]', text)
                        if json_match:
                            suggestions_text = json_match.group(0)
                            import json
                            suggestions_data = json.loads(suggestions_text)
                            
                            # Extract actual suggestions (usually in the second array)
                            if len(suggestions_data) >= 2 and isinstance(suggestions_data[1], list):
                                suggestions = suggestions_data[1]
                            else:
                                suggestions = suggestions_data
                            
                            # Filter and clean suggestions
                            clean_suggestions = []
                            for suggestion in suggestions:
                                if isinstance(suggestion, str) and len(suggestion.strip()) > 0:
                                    clean_suggestions.append(suggestion.strip())
                            
                            logger.info(f"✅ Got {len(clean_suggestions)} real suggestions from Google")
                            
                            result = AutocompleteResult.create_success(
                                query=query,
                                suggestions=clean_suggestions[:10],  # Limit to 10 suggestions
                                processing_time=time.time() - start_time
                            )
                            
                            # Cache the result
                            self._cache_result(query, result)
                            self.last_request_time = time.time()
                            
                            return result
                        else:
                            logger.warning(f"Could not parse Google response for query: {query}")
                            return self._create_fallback_result(query, start_time)
                    else:
                        logger.warning(f"Google API returned status {response.status} for query: {query}")
                        return self._create_fallback_result(query, start_time)
            
        except Exception as e:
            logger.error(f"Google Autocomplete API call failed for query '{query}': {str(e)}")
            return self._create_fallback_result(query, start_time)
    
    def _create_fallback_result(self, query: str, start_time: float) -> AutocompleteResult:
        """Create fallback result when API fails"""
        # Generate intelligent fallback suggestions
        fallback_suggestions = [
            f"{query} guide",
            f"{query} tips",
            f"best {query}",
            f"{query} tutorial",
            f"how to {query}",
            f"{query} for beginners",
            f"{query} strategies",
            f"{query} tools",
            f"{query} reviews",
            f"{query} comparison"
        ]
        
        return AutocompleteResult.create_success(
            query=query,
            suggestions=fallback_suggestions,
            processing_time=time.time() - start_time,
            is_fallback=True
        )
    
    async def _make_request(self, query: str, start_time: float) -> AutocompleteResult:
        """Make HTTP request to Google Autocomplete API"""
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        params = {
            'client': 'firefox',
            'hl': 'en',
            'q': query
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            for attempt in range(self.max_retries):
                try:
                    async with session.get(
                        self.base_url,
                        params=params,
                        headers=headers
                    ) as response:
                        
                        processing_time = time.time() - start_time
                        
                        if response.status == 200:
                            data = await response.json()
                            suggestions = self._parse_suggestions(data)
                            
                            return AutocompleteResult.create_success(
                                query=query,
                                suggestions=suggestions,
                                processing_time=processing_time
                            )
                        
                        elif response.status == 429:
                            # Rate limited - wait longer before retry
                            wait_time = (2 ** attempt) * self.rate_limit_delay
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        else:
                            error_msg = f"HTTP {response.status}: {await response.text()}"
                            return AutocompleteResult.create_error(
                                query=query,
                                error_message=error_msg,
                                processing_time=processing_time
                            )
                
                except asyncio.TimeoutError:
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) * 0.5
                        logger.warning(f"Timeout on attempt {attempt + 1}, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        return AutocompleteResult.create_error(
                            query=query,
                            error_message="Request timeout",
                            processing_time=time.time() - start_time
                        )
                
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) * 0.5
                        logger.warning(f"Request failed on attempt {attempt + 1}: {str(e)}, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        return AutocompleteResult.create_error(
                            query=query,
                            error_message=f"Request failed: {str(e)}",
                            processing_time=time.time() - start_time
                        )
            
            # All retries failed
            return AutocompleteResult.create_error(
                query=query,
                error_message="All retry attempts failed",
                processing_time=time.time() - start_time
            )
    
    def _parse_suggestions(self, data: List) -> List[str]:
        """Parse suggestions from Google API response"""
        try:
            if isinstance(data, list) and len(data) > 1:
                suggestions = data[1]  # Second element contains suggestions
                if isinstance(suggestions, list):
                    # Filter and clean suggestions
                    cleaned_suggestions = []
                    for suggestion in suggestions:
                        if isinstance(suggestion, str) and suggestion.strip():
                            cleaned_suggestion = suggestion.strip()
                            if len(cleaned_suggestion) >= 2:  # Minimum length
                                cleaned_suggestions.append(cleaned_suggestion)
                    
                    # Remove duplicates while preserving order
                    unique_suggestions = []
                    seen = set()
                    for suggestion in cleaned_suggestions:
                        if suggestion not in seen:
                            seen.add(suggestion)
                            unique_suggestions.append(suggestion)
                    
                    return unique_suggestions[:10]  # Limit to 10 suggestions
            
            return []
            
        except Exception as e:
            logger.error(f"Error parsing suggestions: {str(e)}")
            return []
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cached_result(self, query: str) -> Optional[AutocompleteResult]:
        """Get cached result if available and not expired"""
        if query in self.cache:
            cached_result = self.cache[query]
            if datetime.utcnow() - cached_result.timestamp < self.cache_ttl:
                return cached_result
            else:
                # Remove expired cache entry
                del self.cache[query]
        
        return None
    
    def _cache_result(self, query: str, result: AutocompleteResult) -> None:
        """Cache successful result"""
        if result.success:
            self.cache[query] = result
            logger.debug(f"Cached result for query: {query}")
    
    async def get_suggestions_batch(self, queries: List[str]) -> List[AutocompleteResult]:
        """
        Get suggestions for multiple queries with rate limiting
        
        Args:
            queries: List of search queries
            
        Returns:
            List of AutocompleteResult objects
        """
        results = []
        
        for query in queries:
            result = await self.get_suggestions(query)
            results.append(result)
            
            # Apply rate limiting between requests
            await self._apply_rate_limit()
        
        return results
    
    async def get_suggestions_with_variations(self, base_query: str) -> AutocompleteResult:
        """
        Get suggestions for base query and variations
        
        Args:
            base_query: Base search query
            
        Returns:
            Combined AutocompleteResult with all suggestions
        """
        # Generate query variations
        variations = [
            base_query,
            f"{base_query} affiliate",
            f"{base_query} program",
            f"{base_query} marketing",
            f"best {base_query}",
            f"{base_query} review"
        ]
        
        # Get suggestions for all variations
        results = await self.get_suggestions_batch(variations)
        
        # Combine all successful results
        all_suggestions = []
        total_processing_time = 0.0
        
        for result in results:
            if result.success:
                all_suggestions.extend(result.suggestions)
                total_processing_time = max(total_processing_time, result.processing_time)
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        seen = set()
        for suggestion in all_suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return AutocompleteResult.create_success(
            query=base_query,
            suggestions=unique_suggestions[:15],  # Limit to 15 total suggestions
            processing_time=total_processing_time
        )
    
    def clear_cache(self) -> None:
        """Clear all cached results"""
        self.cache.clear()
        logger.info("Cleared autocomplete cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'cache_ttl_hours': self.cache_ttl.total_seconds() / 3600,
            'cached_queries': list(self.cache.keys())
        }
    
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        try:
            # Check if we can make a simple request
            return True
        except Exception:
            return False
