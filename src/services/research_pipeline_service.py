import logging
import asyncio
from typing import List, Dict, Any, Optional

from src.integrations.dataforseo import dataforseo_api
from src.services.semantic_expansion_service import semantic_expansion_service

logger = logging.getLogger(__name__)

class ResearchPipelineService:
    async def run_pipeline(
        self,
        seed_keyword: str,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        logger.info(f"Starting End-to-End Research Pipeline for: {seed_keyword}")

        # Step 1: Seed Expansion (Wide Net)
        # We fire off SERP, Autocomplete, and Related Searches simultaneously.
        serp_task = dataforseo_api.get_serp_analysis(seed_keyword)
        # We use limit_per_seed=25 to mimic the current limits, but we can expand if needed.
        suggestions_task = dataforseo_api.get_keyword_suggestions_labs_live(
            [seed_keyword],
            limit_per_seed=50,
            return_raw=True
        )
        related_task = dataforseo_api.get_related_keywords_labs_live(
            [seed_keyword],
            limit_per_seed=50,
            return_raw=True
        )

        serp_res, suggestions_res, related_res = await asyncio.gather(
            serp_task, suggestions_task, related_task, return_exceptions=True
        )

        raw_keywords = set()

        # Extract PAA and Related Searches from SERP
        if isinstance(serp_res, dict):
            for paa in serp_res.get("people_also_ask", []):
                q = paa.get("question")
                if q:
                    raw_keywords.add(str(q).strip().lower())
            for rs in serp_res.get("related_searches", []):
                k = rs.get("keyword")
                if k:
                    raw_keywords.add(str(k).strip().lower())

        # Extract Autocomplete
        if isinstance(suggestions_res, dict):
            items = suggestions_res.get("items", [])
            for item in items:
                k = item.get("keyword")
                if k:
                    raw_keywords.add(str(k).strip().lower())

        # Extract Related Keywords
        if isinstance(related_res, dict):
            items = related_res.get("items", [])
            for item in items:
                k = item.get("keyword")
                if k:
                    raw_keywords.add(str(k).strip().lower())
                    
        # Include the seed itself
        raw_keywords.add(seed_keyword.strip().lower())

        if not raw_keywords:
            logger.warning("No keywords found from any expansion source.")
            return []

        # Convert to list for Bulk Metrics
        lookup_terms = list(raw_keywords)
        logger.info(f"Found {len(lookup_terms)} unique keywords from expansion. Fetching bulk metrics...")

        # Step 2: Profitability Expansion & Filtering
        bulk_metrics = await dataforseo_api.get_bulk_metrics_standard(lookup_terms[:1000])  # Cap at 1000 to be safe
        
        filtered_keywords = []
        if bulk_metrics:
            for item in bulk_metrics:
                vol = item.get("search_volume") or 0
                kd = item.get("keyword_difficulty") or 100
                # Filter: KD < 30 and Vol > 30 (Based on user request)
                if kd < 30 and vol > 30:
                    filtered_keywords.append({
                        "keyword": item.get("keyword"),
                        "search_volume": vol,
                        "keyword_difficulty": kd,
                        "cpc": item.get("cpc") or 0.0,
                        "competition": item.get("competition") or "UNKNOWN"
                    })

        if not filtered_keywords:
            logger.warning("No keywords passed the profitability filter.")
            return []

        # Step 3: Semantic Clustering
        # We use the existing SemanticExpansionService to group these keywords
        logger.info(f"Clustering {len(filtered_keywords)} profitable keywords...")
        
        # Sort by volume and keep top 75 for LLM clustering
        filtered_keywords.sort(key=lambda x: x.get("search_volume", 0), reverse=True)
        top_keywords = filtered_keywords[:75]
        
        clusters = await semantic_expansion_service.cluster_keywords(top_keywords)
        
        # We skip verification for now as the user just wants the cluster cards
        
        logger.info(f"Pipeline complete. Generated {len(clusters)} clusters.")
        return clusters

research_pipeline_service = ResearchPipelineService()
