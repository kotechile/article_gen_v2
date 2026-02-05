import math
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID

from src.services.dataforseo_service import DataForSEOService
from src.models.keyword import KeywordBase
from src.models.content_topic import ContentTopicCreate, ContentTopicBase

logger = logging.getLogger(__name__)

class KeywordExpansionService:
    """
    Service for expanding seed keywords, filtering by profitability,
    and clustering into content topics.
    """
    
    def __init__(self):
        self.dataforseo = DataForSEOService()
    
    async def expand_seed_keywords(self, seed_keywords: List[str]) -> List[KeywordBase]:
        """
        Expand seed keywords using DataForSEO API with batching support.
        
        Args:
            seed_keywords: List of seed keywords to expand
            
        Returns:
            List of expanded keywords with metrics
        """
        if not seed_keywords:
            return []
            
        # Dedup seed keywords
        unique_seeds = list(set(seed_keywords))
        
        # DataForSEO limit is 200 keywords per request
        BATCH_SIZE = 200
        all_expanded_keywords = []
        
        # Process in batches
        for i in range(0, len(unique_seeds), BATCH_SIZE):
            batch = unique_seeds[i:i + BATCH_SIZE]
            logger.info(f"Expanding batch of {len(batch)} keywords: {batch[:5]}...")
            
            try:
                # Use single-step live endpoint via existing service method
                # The service method get_keyword_ideas handles the API call details
                results = await self.dataforseo.get_keyword_ideas(
                    seed_keywords=batch,
                    location_code=2840, # US
                    language_code="en"
                )
                
                # Transform dictionary results to Pydantic models
                for item in results:
                    try:
                        # Extract metrics safely
                        search_volume = item.get("search_volume")
                        cpc = item.get("cpc")
                        difficulty = item.get("keyword_difficulty")
                        
                        # Only include keywords that have minimal valid data to be useful
                        if search_volume is None:
                            continue
                            
                        keyword_model = KeywordBase(
                            seed_keyword=item.get("seed_keywords", [""])[0] if item.get("seed_keywords") else "",
                            keyword=item.get("keyword", ""),
                            search_volume=search_volume,
                            cpc=cpc,
                            competition=item.get("competition"),
                            competition_level=item.get("competition_level"),
                            difficulty=difficulty,
                            keyword_difficulty=difficulty,
                            main_intent=item.get("main_intent"),
                            intent_type=item.get("intent_type"),
                            low_top_of_page_bid=item.get("low_top_of_page_bid"),
                            high_top_of_page_bid=item.get("high_top_of_page_bid"),
                            categories=item.get("categories", []),
                            monthly_searches=item.get("monthly_searches", []),
                            core_keyword=item.get("core_keyword"),
                            synonym_clustering_algorithm=item.get("synonym_clustering_algorithm"),
                            detected_language=item.get("detected_language"),
                            is_another_language=item.get("is_another_language", False),
                            monthly_trend=item.get("monthly_trend"),
                            quarterly_trend=item.get("quarterly_trend"),
                            yearly_trend=item.get("yearly_trend"),
                            source="dataforseo_keyword_ideas"
                        )
                        all_expanded_keywords.append(keyword_model)
                    except Exception as e:
                        logger.error(f"Error parsing keyword item: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error expanding keyword batch: {e}")
                # Continue to next batch instead of failing completely
                continue
                
        logger.info(f"Total expanded keywords found: {len(all_expanded_keywords)}")
        return all_expanded_keywords

    def apply_profitability_filter(
        self, 
        keywords: List[KeywordBase], 
        min_volume: int = 100, 
        max_difficulty: int = 50,
        min_cpc: float = 0.05
    ) -> List[KeywordBase]:
        """
        Filter keywords by calculated profitability score.
        Includes fallback logic: if strict filter returns < 5 results,
        it retries with relaxed thresholds (min_volume=10, max_difficulty=85).
        """
        
        def _filter_pass(kws, vol_threshold, kd_threshold):
            passed = []
            for kw in kws:
                vol = kw.search_volume or 0
                kd = kw.keyword_difficulty or kw.difficulty or 0
                
                if vol < vol_threshold:
                    continue
                if kd > kd_threshold:
                    continue
                
                # Calculate Score
                cpc = kw.cpc or 0.0
                log_volume = math.log10(vol) if vol > 0 else 0
                score = (log_volume * cpc) / (kd + 1)
                kw.profitability_score = round(score, 4)
                passed.append(kw)
            return passed

        # 1. Primary Pass (Strict)
        filtered_keywords = _filter_pass(keywords, min_volume, max_difficulty)
        
        # 2. Fallback Pass (Relaxed) - if we have very generic/competitive keywords
        if len(filtered_keywords) < 5:
            logger.info(f"Strict filter returned only {len(filtered_keywords)} results. Retrying with relaxed criteria.")
            # Relaxed: much lower volume allowed, much higher difficulty allowed
            filtered_keywords = _filter_pass(keywords, min_volume=10, max_difficulty=85)
            logger.info(f"Relaxed filter returned {len(filtered_keywords)} results.")
            
        # 3. Sort by Profitability Score Descending
        filtered_keywords.sort(key=lambda x: x.profitability_score or 0, reverse=True)
        
        return filtered_keywords

    def cluster_keywords_by_intent(self, keywords: List[KeywordBase]) -> Dict[str, List[KeywordBase]]:
        """
        Group keywords by their intent type.
        """
        clusters = {
            "commercial": [],
            "informational": [],
            "transactional": [],
            "navigational": [],
            "other": []
        }
        
        for kw in keywords:
            intent = (kw.intent_type or kw.main_intent or "other").lower()
            if "commercial" in intent:
                clusters["commercial"].append(kw)
            elif "informational" in intent:
                clusters["informational"].append(kw)
            elif "transactional" in intent:
                clusters["transactional"].append(kw)
            elif "navigational" in intent:
                clusters["navigational"].append(kw)
            else:
                clusters["other"].append(kw)
                
        return clusters

    def generate_content_topics(
        self, 
        clustered_keywords: Dict[str, List[KeywordBase]], 
        user_id: UUID,
        research_topic_id: UUID, 
        max_topics_per_intent: int = 5
    ) -> List[ContentTopicCreate]:
        """
        Generate strict content topic suggestions from clustered keywords.
        One topic per high-value keyword.
        """
        content_topics = []
        
        for intent, cluster in clustered_keywords.items():
            if not cluster:
                continue
                
            # Take top K keywords from this intent cluster
            top_keywords = cluster[:max_topics_per_intent]
            
            for primary_kw in top_keywords:
                # Find supporting keywords (related logic could be improved here)
                # For now, simplistic: take next 5 keywords from same cluster
                # In real world, we'd use semantic similarity clustering
                supporting_kws = [k for k in cluster if k != primary_kw][:5]
                supporting_indices = [] # We don't have IDs yet, so we can't link them here easily
                # This part is tricky because we haven't saved keywords to DB yet.
                # The caller will need to handle saving keywords first to get IDs, 
                # or we just create the topic structure and link later.
                
                # Construct a blog title based on intent
                title = self._generate_title_template(primary_kw.keyword, intent)
                
                # Estimate aggregated profitability
                # Simple sum of top keyword score + 10% of supporting
                primary_score = primary_kw.profitability_score or 0
                supporting_score = sum((k.profitability_score or 0) for k in supporting_kws) * 0.1
                total_score = primary_score + supporting_score
                
                topic = ContentTopicCreate(
                    research_topic_id=research_topic_id,
                    user_id=user_id,
                    title=title,
                    description=f"Targeting '{primary_kw.keyword}' with {intent} intent.",
                    estimated_profitability_score=round(total_score, 4),
                    total_search_volume=primary_kw.search_volume,
                    average_cpc=primary_kw.cpc,
                    average_difficulty=float(primary_kw.keyword_difficulty or 0),
                    intent_type=intent.capitalize(),
                    status="suggested",
                    priority_score=min(total_score * 10, 100) # Normalize properly in future
                )
                content_topics.append(topic)
                
        # Sort all topics by priority
        content_topics.sort(key=lambda x: x.priority_score or 0, reverse=True)
        return content_topics

    def _generate_title_template(self, keyword: str, intent: str) -> str:
        """Generate a basic SEO title based on keyword and intent"""
        kw_cap = keyword.title()
        if intent == "commercial":
            return f"Best {kw_cap} Reviewed: Top Picks for 2024"
        elif intent == "transactional":
            return f"Buy {kw_cap}: Complete Buyer's Guide & Deals"
        elif intent == "informational":
            return f"What is {kw_cap}? Everything You Need to Know"
        elif intent == "comparison":
            return f"{kw_cap} vs Competitors: Which is Best?"
        else:
            return f"{kw_cap}: In-Depth Analysis"
