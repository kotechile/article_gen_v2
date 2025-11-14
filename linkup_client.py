"""
Linkup web search client for Content Generator V2.

This module provides integration with Linkup for real-time web search
and information gathering capabilities.
"""

import os
import time
import logging
import requests
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import json
from urllib.parse import urlparse, urljoin

# Import caching utilities
try:
    from .utils.search_cache import SearchCache
except ImportError:
    # Fallback for different import paths
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.search_cache import SearchCache

# Configure logging
logger = logging.getLogger(__name__)

class SearchProvider(Enum):
    """Supported search providers."""
    LINKUP = "linkup"
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"

@dataclass
class LinkupConfig:
    """Configuration for Linkup requests."""
    api_key: str
    endpoint: str = "https://api.linkup.so/v1/search"
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    max_results: int = 10
    language: str = "en"
    region: str = "us"
    safe_search: bool = True

@dataclass
class SearchQuery:
    """Search query parameters."""
    query: str
    depth: Optional[str] = None
    max_results: Optional[int] = None
    language: Optional[str] = None
    region: Optional[str] = None
    safe_search: Optional[bool] = None
    date_range: Optional[str] = None  # e.g., "past_year", "past_month"
    site: Optional[str] = None  # Search within specific site

@dataclass
class SearchResult:
    """Search result."""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    source: str = "web"
    source_type: str = "linkup"
    relevance_score: float = 0.0
    credibility_score: float = 0.0
    freshness_score: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class SearchResponse:
    """Search response."""
    results: List[SearchResult]
    total_results: int
    query_time: float
    provider: str
    success: bool = True
    error: Optional[str] = None

class LinkupClient:
    """
    Linkup web search client.
    """
    
    def __init__(self, config: LinkupConfig, cache_enabled: bool = True):
        """
        Initialize the Linkup client.
        
        Args:
            config: Linkup configuration
            cache_enabled: Whether to enable result caching (default: True)
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.search_cache = SearchCache(enabled=cache_enabled)
        
    def search(self, search_query: SearchQuery) -> SearchResponse:
        """
        Search the web synchronously using requests.
        Caches results to reduce API calls.
        """
        start_time = time.time()
        
        # Check cache first
        depth = search_query.depth or "standard"
        cached_result = self.search_cache.get(
            query=search_query.query,
            depth=depth,
            date_range=search_query.date_range,
            site=search_query.site,
            max_results=search_query.max_results
        )
        
        if cached_result:
            # Reconstruct SearchResponse from cached data
            cached_results = []
            for item in cached_result.get('results', []):
                cached_results.append(SearchResult(**item))
            
            self.logger.info(f"Cache HIT for Linkup search: '{search_query.query[:50]}...'")
            return SearchResponse(
                results=cached_results,
                total_results=cached_result.get('total_results', len(cached_results)),
                query_time=0.001,  # Cached - very fast
                provider="linkup",
                success=True
            )
        
        try:
            # Prepare search parameters
            search_params = {
                "q": search_query.query,
                "depth": search_query.depth or "standard",
                "outputType": "searchResults",
                "num": search_query.max_results or self.config.max_results,
                "lang": search_query.language or self.config.language,
                "region": search_query.region or self.config.region,
                "safe": "1" if (search_query.safe_search if search_query.safe_search is not None else self.config.safe_search) else "0"
            }
            
            # Add optional parameters
            if search_query.date_range:
                search_params["date_range"] = search_query.date_range
            if search_query.site:
                search_params["site"] = search_query.site
            
            # Make the request
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "User-Agent": "ContentGeneratorV2/1.0",
                "Accept": "application/json"
            }
            
            # Use a more robust session configuration to prevent SIGSEGV
            session = requests.Session()
            session.headers.update(headers)
            
            try:
                response = session.post(
                    self.config.endpoint,
                    json=search_params,
                    timeout=self.config.timeout,
                    verify=True,
                    stream=False  # Disable streaming to prevent memory issues
                )
            finally:
                session.close()
            
            if response.status_code == 200:
                data = response.json()
                query_time = time.time() - start_time
                
                # Parse results
                results = self._parse_search_results(data, search_query)
                
                # Cache the successful result
                depth = search_query.depth or "standard"
                cache_data = {
                    'results': [
                        {
                            'title': r.title,
                            'url': r.url,
                            'snippet': r.snippet,
                            'content': r.content,
                            'source': r.source,
                            'source_type': r.source_type,
                            'relevance_score': r.relevance_score,
                            'credibility_score': r.credibility_score,
                            'freshness_score': r.freshness_score,
                            'metadata': r.metadata
                        } for r in results
                    ],
                    'total_results': len(results),
                    'query_time': query_time
                }
                
                self.search_cache.set(
                    query=search_query.query,
                    result=cache_data,
                    depth=depth,
                    date_range=search_query.date_range,
                    site=search_query.site,
                    max_results=search_query.max_results
                )
                
                self.logger.info(f"Web search successful: {len(results)} results in {query_time:.2f}s (cached)")
                
                return SearchResponse(
                    results=results,
                    total_results=len(results),
                    query_time=query_time,
                    provider="linkup",
                    success=True
                )
            else:
                error_text = response.text
                self.logger.error(f"Web search failed: {response.status_code} - {error_text}")
                
                return SearchResponse(
                    results=[],
                    total_results=0,
                    query_time=time.time() - start_time,
                    provider="linkup",
                    success=False,
                    error=f"HTTP {response.status_code}: {error_text}"
                )
        
        except requests.exceptions.Timeout:
            self.logger.error("Web search timeout")
            return SearchResponse(
                results=[],
                total_results=0,
                query_time=time.time() - start_time,
                provider="linkup",
                success=False,
                error="Search timeout"
            )
        
        except Exception as e:
            self.logger.error(f"Web search error: {str(e)}")
            return SearchResponse(
                results=[],
                total_results=0,
                query_time=time.time() - start_time,
                provider="linkup",
                success=False,
                error=str(e)
            )
    
    def _parse_search_results(self, data: Dict[str, Any], search_query: SearchQuery) -> List[SearchResult]:
        """
        Parse search results.
        
        Args:
            data: Raw response data
            search_query: Original search query
            
        Returns:
            List of SearchResult objects
        """
        results = []
        
        try:
            # Handle different response formats
            if "results" in data:
                items = data["results"]
            elif "items" in data:
                items = data["items"]
            elif "webPages" in data and "value" in data["webPages"]:
                items = data["webPages"]["value"]
            else:
                items = data if isinstance(data, list) else []
            
            for item in items:
                # Extract basic information
                title = item.get("title", item.get(
                    "name", ""))
                url = item.get("url", item.get("link", ""))
                snippet = item.get("snippet", item.get(
                    "description", item.get("abstract", "")))
                
                # Extract additional content if available
                content = item.get("content", item.get("body", ""))
                
                # Calculate scores
                relevance_score = self._calculate_relevance_score(
                    title, snippet, search_query.query)
                credibility_score = self._calculate_credibility_score(url, title)
                freshness_score = self._calculate_freshness_score(item)
                
                # Extract metadata
                metadata = {
                    "published_date": item.get("published_date", item.get("datePublished", "")),
                    "author": item.get("author", item.get("creator", "")),
                    "domain": self._extract_domain(url),
                    "language": item.get("language", search_query.language or self.config.language)
                }
                
                result = SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=content,
                    source="web",
                    source_type="linkup",
                    relevance_score=relevance_score,
                    credibility_score=credibility_score,
                    freshness_score=freshness_score,
                    metadata=metadata
                )
                
                results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error parsing search results: {str(e)}")
        
        return results
    
    def _calculate_relevance_score(self, title: str, snippet: str, query: str) -> float:
        """
        Calculate relevance score for a search result.
        
        Args:
            title: Result title
            snippet: Result snippet
            query: Search query
            
        Returns:
            Relevance score (0-1)
        """
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())
        snippet_words = set(snippet.lower().split())
        
        # Calculate word overlap
        title_overlap = len(query_words.intersection(
            title_words)) / len(query_words) if query_words else 0
        snippet_overlap = len(query_words.intersection(
            snippet_words)) / len(query_words) if query_words else 0
        
        # Weight title more heavily than snippet
        relevance = (title_overlap * 0.7) + (snippet_overlap * 0.3)
        
        # Boost for exact phrase matches
        if query.lower() in title.lower():
            relevance += 0.2
        if query.lower() in snippet.lower():
            relevance += 0.1
        
        return min(1.0, relevance)
    
    def _calculate_credibility_score(self, url: str, title: str) -> float:
        """
        Calculate credibility score for a search result.
        
        Args:
            url: Result URL
            title: Result title
            
        Returns:
            Credibility score (0-1)
        """
        credibility = 0.5  # Base credibility
        
        # Boost for authoritative domains
        authoritative_domains = [
            "wikipedia.org", "britannica.com", "edu", "gov", "org",
            "nature.com", "science.org", "ieee.org", "acm.org",
            "springer.com", "elsevier.com", "pubmed.ncbi.nlm.nih.gov",
            "reuters.com", "bbc.com", "cnn.com", "nytimes.com"
        ]
        
        for domain in authoritative_domains:
            if domain in url.lower():
                credibility += 0.2
                break
        
        # Boost for academic sources
        if ".edu" in url or ".gov" in url:
            credibility += 0.3
        
        # Boost for news sources
        news_indicators = ["news", "times", "post", "journal", "herald", "tribune"]
        if any(indicator in url.lower() for indicator in news_indicators):
            credibility += 0.1
        
        # Boost for professional titles
        professional_indicators = [
            "study", "research", "analysis", "report", "journal"]
        if any(indicator in title.lower() for indicator in professional_indicators):
            credibility += 0.1
        
        return min(1.0, credibility)
    
    def _calculate_freshness_score(self, item: Dict[str, Any]) -> float:
        """
        Calculate freshness score for a search result.
        
        Args:
            item: Search result item
            
        Returns:
            Freshness score (0-1)
        """
        freshness = 0.5  # Base freshness
        
        # Extract date information
        date_str = item.get(
            "published_date", item.get("datePublished", item.get("date", "")))
        
        if date_str:
            try:
                # Simple date parsing (this could be enhanced)
                import datetime
                current_year = datetime.datetime.now().year
                
                # Extract year from date string
                year = int(str(date_str)[:4])
                
                # Calculate freshness based on recency
                if year >= current_year:
                    freshness = 1.0
                elif year >= current_year - 1:
                    freshness = 0.8
                elif year >= current_year - 2:
                    freshness = 0.6
                elif year >= current_year - 5:
                    freshness = 0.4
                else:
                    freshness = 0.2
                    
            except:
                pass
        
        return freshness
    
    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "unknown"
    
    def batch_search(self, queries: List[SearchQuery]) -> List[SearchResponse]:
        """
        Execute multiple search queries synchronously.
        
        Args:
            queries: List of search queries
            
        Returns:
            List of SearchResponse objects
        """
        responses = []
        for query in queries:
            responses.append(self.search(query))
        return responses
    
    def get_search_info(self) -> Dict[str, Any]:
        """
        Get information about the search configuration.
        
        Returns:
            Search configuration information
        """
        return {
            "endpoint": self.config.endpoint,
            "max_results": self.config.max_results,
            "language": self.config.language,
            "region": self.config.region,
            "timeout": self.config.timeout
        }

# Factory function for creating Linkup clients
def create_linkup_client(
    api_key: str,
    endpoint: Optional[str] = None,
    cache_enabled: bool = True,
    **kwargs
) -> LinkupClient:
    """
    Create a Linkup client with the specified configuration.
    
    Args:
        api_key: Linkup API key
        endpoint: Optional custom endpoint
        cache_enabled: Whether to enable result caching (default: True)
        **kwargs: Additional configuration options
        
    Returns:
        Configured LinkupClient instance
    """
    config = LinkupConfig(
        api_key=api_key,
        endpoint=endpoint or "https://api.linkup.so/v1/search",
        **kwargs
    )
    
    return LinkupClient(config, cache_enabled=cache_enabled)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    client = create_linkup_client(
        api_key="your-linkup-api-key",
        max_results=5,
        language="en",
        region="us"
    )
    
    query = SearchQuery(
        query="sustainable energy solutions 2024",
        max_results=3,
        date_range="past_year"
    )
    
    try:
        response = client.search(query)
        print(f"Found {response.total_results} results")
        for result in response.results:
            print(f"- {result.title} (relevance: {result.relevance_score:.2f})")
            print(f"  {result.url}")
    except Exception as e:
        print(f"Error: {e}")
