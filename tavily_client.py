"""
Tavily web search client for Content Generator V2.

This client mirrors the Linkup client response shape so evidence collection
can route between providers without changing downstream logic.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from tavily import TavilyClient
except Exception:  # pragma: no cover - import handled at runtime
    TavilyClient = None  # type: ignore

from linkup_client import SearchQuery, SearchResult, SearchResponse

logger = logging.getLogger(__name__)


@dataclass
class TavilyConfig:
    api_key: str
    timeout: int = 30
    max_results: int = 8


class TavilySearchClient:
    def __init__(self, config: TavilyConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def search(self, search_query: SearchQuery) -> SearchResponse:
        start = time.time()
        if TavilyClient is None:
            return SearchResponse(
                results=[],
                total_results=0,
                query_time=time.time() - start,
                provider="tavily",
                success=False,
                error="tavily-python dependency is not installed",
            )

        try:
            client = TavilyClient(api_key=self.config.api_key)
            depth = "advanced" if (search_query.depth or "standard") == "deep" else "basic"
            max_results = search_query.max_results or self.config.max_results

            response = client.search(
                query=search_query.query,
                search_depth=depth,
                max_results=max_results,
            )

            results_raw = response.get("results", []) if isinstance(response, dict) else []
            parsed: List[SearchResult] = []
            for item in results_raw:
                title = item.get("title", "Untitled result")
                url = item.get("url", "")
                snippet = item.get("content", "") or ""
                parsed.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet[:300],
                        content=snippet,
                        source="web",
                        source_type="tavily",
                        relevance_score=0.7,
                        credibility_score=0.65,
                        freshness_score=0.6,
                        metadata={
                            "provider": "tavily",
                            "raw_score": item.get("score"),
                        },
                    )
                )

            return SearchResponse(
                results=parsed,
                total_results=len(parsed),
                query_time=time.time() - start,
                provider="tavily",
                success=True,
            )
        except Exception as exc:
            self.logger.error(f"Tavily search failed: {exc}")
            return SearchResponse(
                results=[],
                total_results=0,
                query_time=time.time() - start,
                provider="tavily",
                success=False,
                error=str(exc),
            )


def create_tavily_client(api_key: str, timeout: int = 30, max_results: int = 8) -> TavilySearchClient:
    return TavilySearchClient(TavilyConfig(api_key=api_key, timeout=timeout, max_results=max_results))

