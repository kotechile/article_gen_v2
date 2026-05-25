import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.integrations.dataforseo import dataforseo_api
from src.services.semantic_expansion_service import semantic_expansion_service
from src.services.supabase_service import supabase_service

logger = logging.getLogger(__name__)

class ResearchPipelineService:
    async def extract_and_persist(
        self,
        seed_keyword: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Step 1: Extract keywords using SERP and Keyword Ideas, filter, and persist.
        Returns the validated keywords and a generated run_id.
        """
        logger.info(f"Extracting keyword ideas for seed: {seed_keyword}")

        # 1. SERP Expansion for additional seeds
        serp_res = await dataforseo_api.get_serp_analysis(seed_keyword)
        
        seed_list = {seed_keyword.strip().lower()}
        if isinstance(serp_res, dict):
            for paa in serp_res.get("people_also_ask", []):
                if q := paa.get("question"): seed_list.add(str(q).strip().lower())
            for rs in serp_res.get("related_searches", []):
                if k := rs.get("keyword"): seed_list.add(str(k).strip().lower())

        # 2. Keyword Ideas (DataForSEO Labs)
        # Cap at 200 seeds to avoid payload limits
        seeds_to_expand = list(seed_list)[:200]
        
        ideas = await dataforseo_api.get_keyword_ideas_labs_live(
            keywords=seeds_to_expand,
            limit=50, # Limit per seed
            include_serp_info=False
        )

        filtered_keywords = []
        seen = set()
        
        for item in ideas:
            kw = item.get("keyword")
            if not kw or kw in seen:
                continue
            
            seen.add(kw)
            
            vol = item.get("search_volume") or 0
            kd = item.get("keyword_difficulty") if item.get("keyword_difficulty") is not None else 0
            
            # Filter: KD < 30 and Vol > 30
            if kd < 30 and vol > 30:
                filtered_keywords.append(item)

        # Sort by volume desc
        filtered_keywords.sort(key=lambda x: x.get("search_volume", 0), reverse=True)

        if not filtered_keywords:
            logger.warning("No keywords passed the profitability filter.")
            return {"run_id": None, "keywords": []}

        # 3. Persist to Supabase
        run_id = str(uuid.uuid4())
        topic_id = str(uuid.uuid4()) # Dummy topic_id just for DB constraints if needed
        
        db_payload = []
        for row in filtered_keywords:
            comp_level = row.get("competition_level")
            if isinstance(comp_level, str) and not comp_level.replace('.', '').isdigit():
                comp_index = 0
            else:
                comp_index = int(float(comp_level or 0))
                
            db_payload.append({
                "research_run_id": run_id,
                "topic_id": topic_id,
                "user_id": user_id,
                "keyword": row.get("keyword"),
                "search_volume": int(row.get("search_volume") or 0),
                "cpc": float(row.get("cpc") or 0),
                "competition": row.get("competition"),
                "competition_index": comp_index,
                "keyword_difficulty": float(row.get("keyword_difficulty") or 0),
                "intent_label": row.get("intent")
            })

        # Batch insert using SupabaseService
        try:
            client = supabase_service.get_client()
            for i in range(0, len(db_payload), 100):
                chunk = db_payload[i:i+100]
                client.table("topic_keyword_candidates").insert(chunk).execute()
        except Exception as e:
            logger.warning(f"Failed to persist keywords to Supabase: {e}", exc_info=True)

        logger.info(f"Extracted and persisted {len(filtered_keywords)} keywords for run {run_id}.")
        return {
            "run_id": run_id,
            "keywords": filtered_keywords
        }

    async def cluster_detailed_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 2: Cluster the rich detailed keywords.
        """
        logger.info(f"Clustering {len(keywords)} profitable keywords...")
        
        # Sort by volume and keep top 75 for LLM clustering
        keywords.sort(key=lambda x: x.get("search_volume", 0), reverse=True)
        top_keywords = keywords[:75]
        
        # We need to map it to the structure SemanticExpansionService expects
        cluster_input = []
        for k in top_keywords:
            cluster_input.append({
                "keyword": k.get("keyword"),
                "search_volume": k.get("search_volume"),
                "keyword_difficulty": k.get("keyword_difficulty"),
                "cpc": k.get("cpc"),
                "competition": k.get("competition")
            })
            
        clusters = await semantic_expansion_service.cluster_keywords(cluster_input)
        
        logger.info(f"Pipeline complete. Generated {len(clusters)} clusters.")
        return clusters

    # Keeping old run_pipeline for backwards compatibility just in case
    async def run_pipeline(
        self,
        seed_keyword: str,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        res = await self.extract_and_persist(seed_keyword, user_id)
        if not res["keywords"]:
            return []
        return await self.cluster_detailed_keywords(res["keywords"])

research_pipeline_service = ResearchPipelineService()
