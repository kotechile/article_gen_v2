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
        vol_task = dataforseo_api.get_bulk_metrics_standard(lookup_terms[:1000])
        kd_task = dataforseo_api.get_keyword_difficulty(lookup_terms[:1000])
        
        bulk_metrics_res, kd_metrics_res = await asyncio.gather(vol_task, kd_task, return_exceptions=True)
        
        bulk_metrics = bulk_metrics_res if isinstance(bulk_metrics_res, list) else []
        kd_metrics = kd_metrics_res if isinstance(kd_metrics_res, list) else []

        # Merge them by keyword
        merged_metrics = {}
        for item in bulk_metrics:
            kw = item.get("keyword")
            if kw:
                merged_metrics[kw] = {
                    "keyword": kw,
                    "search_volume": item.get("search_volume") or 0,
                    "cpc": item.get("cpc") or 0.0,
                    "competition": item.get("competition") or "UNKNOWN",
                    "keyword_difficulty": None  # Will be filled by kd_metrics
                }
                
        for item in kd_metrics:
            kw = item.get("keyword")
            if kw:
                if kw not in merged_metrics:
                    merged_metrics[kw] = {
                        "keyword": kw,
                        "search_volume": item.get("search_volume") or 0, # get_keyword_difficulty doesn't return SV, but just in case
                        "cpc": item.get("cpc") or 0.0,
                        "competition": item.get("competition") or "UNKNOWN",
                        "keyword_difficulty": item.get("keyword_difficulty")
                    }
                else:
                    merged_metrics[kw]["keyword_difficulty"] = item.get("keyword_difficulty")
        
        filtered_keywords = []
        for kw, item in merged_metrics.items():
            vol = item.get("search_volume") or 0
            # If KD is missing from Labs, we assume it's low or 0 (or we could assume 100).
            # Usually, un-tracked long tails have low difficulty, but to be safe we can use 0 for filtering or a fallback.
            # Wait, the user asked for KD < 30. If it's missing, maybe it's 0.
            kd = item.get("keyword_difficulty") if item.get("keyword_difficulty") is not None else 0
            
            if kd < 30 and vol > 30:
                filtered_keywords.append({
                    "keyword": kw,
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
