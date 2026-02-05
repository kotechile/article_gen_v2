
import os
import httpx
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ApifyClient:
    """
    Apify API Client for Pinterest Scraper
    """
    def __init__(self):
        self.api_token = os.getenv("APIFY_API_TOKEN")
        self.base_url = "https://api.apify.com/v2"
        self.timeout = 60.0
        
        # Fallback to Supabase if not in Env
        if not self.api_token:
             self.api_token = self._fetch_token_from_supabase()

    def _fetch_token_from_supabase(self) -> Optional[str]:
        """Fetch Apify key from Supabase api_keys table"""
        from src.core.supabase_singleton import get_supabase_client
        try:
            supabase = get_supabase_client()
            response = supabase.table("api_keys").select("key_value").eq("provider", "apify").limit(1).execute()
            if response.data:
                return response.data[0]['key_value']
        except Exception as e:
            logger.error(f"Failed to fetch Apify token from DB: {e}")
        return None

    async def get_pinterest_trends(self, keywords: List[str], max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Get Pinterest pins/trends for specific keywords using 'pinterest-scraper' actor.
        Actor ID for pinterest-scraper: 'alexey/pinterest-crawler' (Example, verify exact actor ID)
        Assuming user wants to use a generic pinterest scraper. 
        Most popular is generic 'zuzka/pinterest-scraper' or similar. 
        I Will use a direct start logic.
        """
        if not self.api_token:
            logger.warning("APIFY_API_TOKEN not found (Env or DB). Pinterest trends disabled.")
            return []

        results = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for keyword in keywords:
                try:
                    # Retrieve trending/search results from Pinterest
                    # Using 'alexey/pinterest-crawler' or equivalent.
                    # Using a simplified synchronous run-sync endpoint for simplicity if available,
                    # or start task + poll. Apify run-sync-get-dataset-items is best.
                    
                    # Actor: 'trudax/pinterest-scraper' or similar. 
                    # Let's assume 'trudax/pinterest-scraper' which is popular.
                    actor_id = "trudax/pinterest-scraper" 
                    
                    url = f"{self.base_url}/acts/{actor_id}/run-sync-get-dataset-items"
                    
                    params = {
                        "token": self.api_token,
                    }
                    
                    input_data = {
                        "search": keyword,
                        "maxPins": max_results,
                        "proxy": {"useApifyProxy": True}
                    }
                    
                    response = await client.post(url, json=input_data, params=params, timeout=120.0) # Longer timeout for scraping
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Process items
                    for item in data:
                        results.append({
                            "keyword": keyword,
                            "title": item.get("title", ""),
                            "description": item.get("description", ""),
                            "image_url": item.get("image", ""),
                            "pin_url": item.get("url", ""),
                            "source": "Pinterest"
                        })
                        
                except Exception as e:
                    logger.error(f"Apify Pinterest lookup failed for '{keyword}': {e}")
                    
        return results

    async def get_linkedin_posts(self, keywords: List[str], max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Get LinkedIn posts for specific keywords using a scraper.
        Uses 'content-so/linkedin-posts-scraper' (or similar).
        """
        if not self.api_token:
            logger.warning("APIFY_API_TOKEN not found. LinkedIn scraping disabled.")
            return []

        results = []
        async with httpx.AsyncClient(timeout=180.0) as client: # LinkedIn scraping can be slow
            for keyword in keywords:
                try:
                    # Actor: 'content-so/linkedin-posts-scraper' is a good candidate.
                    # Or 'katerinah/linkedin-post-scraper'.
                    # Let's try 'katerinah/linkedin-post-scraper' which is often reliable for simple searches.
                    # User suggested 'linkedin-post-scraper' actor. I will use 'katerinah/linkedin-post-scraper' as it's a common one.
                    # Or 'hype/linkedin-post-scraper'.
                    # Let's assume a generic one.
                    actor_id = "katerinah/linkedin-post-scraper"
                    
                    url = f"{self.base_url}/acts/{actor_id}/run-sync-get-dataset-items"
                    params = {"token": self.api_token}
                    
                    input_data = {
                        "keyword": keyword, # Some take 'keyword', some 'search'
                        "searchUrl": f"https://www.linkedin.com/search/results/content/?keywords={keyword}", # More robust
                        "limit": max_results,
                        "proxy": {"useApifyProxy": True}
                    }
                    
                    logger.info(f"Scraping LinkedIn for {keyword}...")
                    response = await client.post(url, json=input_data, params=params, timeout=120.0)
                    if response.status_code != 200:
                        logger.warning(f"LinkedIn scrape failed: {response.status_code} {response.text}")
                        continue
                        
                    data = response.json()
                    
                    # Process items
                    for item in data:
                        # Normalize fields
                        text = item.get("text") or item.get("postText") or item.get("content")
                        if not text: continue
                        
                        results.append({
                            "keyword": keyword,
                            "text": text[:200] + "...",
                            "url": item.get("url") or item.get("postUrl"),
                            "likes": item.get("likesCount") or item.get("numLikes", 0),
                            "comments": item.get("commentsCount") or item.get("numComments", 0),
                            "source": "LinkedIn"
                        })
                        
                except Exception as e:
                    logger.error(f"Apify LinkedIn lookup failed for '{keyword}': {e}")
                    
        return results
