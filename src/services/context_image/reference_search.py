"""
Reference Search Client for Context-Aware Image Generation.

Searches the web for high-fidelity reference photography using:
1. Tavily Search (native image extraction via include_images=True)
2. Linkup Search (using active Supabase key)
3. Fallback providers (SerpAPI, Google CSE, Unsplash)
"""

import logging
import os
import re
import requests
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from supabase_client import get_api_key

logger = logging.getLogger(__name__)


@dataclass
class ReferenceImageItem:
    url: str
    thumbnail_url: Optional[str] = None
    title: Optional[str] = None
    source_domain: Optional[str] = None
    provider: str = "web"
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReferenceSearchClient:
    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        linkup_api_key: Optional[str] = None,
        timeout: int = 15
    ):
        self.tavily_api_key = tavily_api_key or get_api_key("tavily") or os.environ.get("TAVILY_API_KEY")
        self.linkup_api_key = linkup_api_key or get_api_key("linkup") or os.environ.get("LINKUP_API_KEY")
        self.timeout = timeout

    def search_reference_images(self, query: str, max_results: int = 6) -> List[ReferenceImageItem]:
        """
        Search for reference images matching the query across available providers.
        """
        if not query or not query.strip():
            return []

        clean_query = self._sanitize_query(query)
        images: List[ReferenceImageItem] = []

        # 1. Priority: Tavily (direct image search)
        if self.tavily_api_key:
            try:
                images = self._search_tavily(clean_query, max_results=max_results)
                if images:
                    logger.info(f"Retrieved {len(images)} reference images from Tavily for query: '{clean_query}'")
                    return images[:max_results]
            except Exception as e:
                logger.warning(f"Tavily image search failed: {e}. Trying next provider.")

        # 2. Priority: Linkup Search
        if self.linkup_api_key:
            try:
                images = self._search_linkup(clean_query, max_results=max_results)
                if images:
                    logger.info(f"Retrieved {len(images)} reference images via Linkup for query: '{clean_query}'")
                    return images[:max_results]
            except Exception as e:
                logger.warning(f"Linkup reference search failed: {e}.")

        # 3. Fallback: SerpAPI Google Images if key is configured
        serpapi_key = get_api_key("serpapi") or os.environ.get("SERPAPI_API_KEY")
        if serpapi_key:
            try:
                images = self._search_serpapi(clean_query, serpapi_key, max_results=max_results)
                if images:
                    return images[:max_results]
            except Exception as e:
                logger.warning(f"SerpAPI reference search failed: {e}.")

        # 4. Final fallback: Unsplash / Stock image search
        try:
            images = self._search_unsplash_fallback(clean_query, max_results=max_results)
        except Exception as e:
            logger.warning(f"Stock search fallback failed: {e}")

        return images[:max_results]

    def _sanitize_query(self, query: str) -> str:
        # Remove quotes or special operators that might break search
        sanitized = re.sub(r'["\';]', " ", query)
        return " ".join(sanitized.split())

    def _search_tavily(self, query: str, max_results: int = 6) -> List[ReferenceImageItem]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "include_images": True,
            "include_image_descriptions": True,
            "max_results": max_results + 2,
        }
        res = requests.post(url, json=payload, timeout=self.timeout)
        res.raise_for_status()
        data = res.json()

        items: List[ReferenceImageItem] = []
        raw_images = data.get("images", [])

        for img in raw_images:
            img_url = None
            description = None
            if isinstance(img, str):
                img_url = img
            elif isinstance(img, dict):
                img_url = img.get("url")
                description = img.get("description")

            if img_url and self._is_valid_image_url(img_url):
                domain = urlparse(img_url).netloc
                items.append(ReferenceImageItem(
                    url=img_url,
                    thumbnail_url=img_url,
                    title=description or query,
                    source_domain=domain,
                    provider="tavily",
                    score=0.9
                ))

        # Check results array for additional images if needed
        if len(items) < max_results:
            for result in data.get("results", []):
                for res_img in result.get("images", []):
                    img_u = res_img if isinstance(res_img, str) else res_img.get("url")
                    if img_u and self._is_valid_image_url(img_u) and not any(i.url == img_u for i in items):
                        domain = urlparse(img_u).netloc
                        items.append(ReferenceImageItem(
                            url=img_u,
                            thumbnail_url=img_u,
                            title=result.get("title") or query,
                            source_domain=domain,
                            provider="tavily",
                            score=0.85
                        ))

        return items

    def _search_linkup(self, query: str, max_results: int = 6) -> List[ReferenceImageItem]:
        url = "https://api.linkup.so/v1/search"
        headers = {
            "Authorization": f"Bearer {self.linkup_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "depth": "standard",
            "outputType": "searchResults"
        }
        res = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        res.raise_for_status()
        data = res.json()

        items: List[ReferenceImageItem] = []
        results = data.get("results", []) or data.get("searchResults", [])

        for item in results:
            page_url = item.get("url") or ""
            domain = urlparse(page_url).netloc
            title = item.get("name") or item.get("title") or query

            # Check if Linkup returns images directly in metadata
            images = item.get("images") or []
            if isinstance(images, list):
                for img in images:
                    img_u = img if isinstance(img, str) else img.get("url")
                    if img_u and self._is_valid_image_url(img_u):
                        items.append(ReferenceImageItem(
                            url=img_u,
                            thumbnail_url=img_u,
                            title=title,
                            source_domain=domain,
                            provider="linkup",
                            score=0.8
                        ))

            # Inspect page metadata / og:image if available
            metadata = item.get("metadata") or {}
            og_image = metadata.get("og:image") or metadata.get("image")
            if og_image and self._is_valid_image_url(og_image):
                if not any(i.url == og_image for i in items):
                    items.append(ReferenceImageItem(
                        url=og_image,
                        thumbnail_url=og_image,
                        title=title,
                        source_domain=domain,
                        provider="linkup",
                        score=0.75
                    ))

        return items

    def _search_serpapi(self, query: str, api_key: str, max_results: int = 6) -> List[ReferenceImageItem]:
        url = "https://serpapi.com/search.json"
        params = {
            "q": query,
            "engine": "google_images",
            "ijn": "0",
            "api_key": api_key,
            "tbs": "itp:photo"  # Filter for photos
        }
        res = requests.get(url, params=params, timeout=self.timeout)
        res.raise_for_status()
        data = res.json()

        items = []
        for img in data.get("images_results", [])[:max_results]:
            original = img.get("original")
            thumbnail = img.get("thumbnail")
            if original and self._is_valid_image_url(original):
                items.append(ReferenceImageItem(
                    url=original,
                    thumbnail_url=thumbnail or original,
                    title=img.get("title") or query,
                    source_domain=img.get("source"),
                    provider="serpapi",
                    score=0.95
                ))
        return items

    def _search_unsplash_fallback(self, query: str, max_results: int = 6) -> List[ReferenceImageItem]:
        unsplash_key = get_api_key("unsplash") or os.environ.get("UNSPLASH_ACCESS_KEY")
        if not unsplash_key:
            return []

        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {unsplash_key}"}
        params = {"query": query, "per_page": max_results}
        res = requests.get(url, headers=headers, params=params, timeout=self.timeout)
        res.raise_for_status()
        data = res.json()

        items = []
        for photo in data.get("results", []):
            urls = photo.get("urls", {})
            regular = urls.get("regular")
            thumb = urls.get("thumb")
            if regular:
                items.append(ReferenceImageItem(
                    url=regular,
                    thumbnail_url=thumb or regular,
                    title=photo.get("alt_description") or query,
                    source_domain="unsplash.com",
                    provider="unsplash",
                    score=0.7
                ))
        return items

    def _is_valid_image_url(self, url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return False
        # Filter out common tracking or icon pixels
        lower = url.lower()
        if any(bad in lower for bad in ["1x1", "pixel", "icon", "favicon", "logo_small", ".svg"]):
            return False
        return True
