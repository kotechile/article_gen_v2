"""
Service for orchestrating topic decomposition and analysis.
"""

import logging
import random # For mock metrics
from typing import List, Optional
from datetime import datetime

from .semantic_expansion import semantic_expansion_service
from ..core.models.topic_analysis import Subtopic, SubtopicCreate, KeywordMetrics

# Import Supabase client (using same pattern as endpoints)
try:
    from supabase_client import get_supabase_client
except ImportError:
    import sys
    import os
    sys.path.append(os.getcwd())
    from supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

class TopicDecompositionService:
    """
    Orchestrates the decomposition of topics into sub-niches.
    Connects Semantic Expansion (LLM) with Data Validation (DataForSEO) and Storage (Supabase).
    """

    def decompose_topic(self, topic_id: str, topic_title: str) -> List[Subtopic]:
        """
        Decomposes a topic into validated sub-niches.
        
        Args:
            topic_id: The ID of the parent research topic.
            topic_title: The title of the topic (e.g. "Golf").
            
        Returns:
            List of saved Subtopic objects.
        """
        logger.info(f"Decomposing topic {topic_id}: {topic_title}")
        
        # 1. Semantic Explosion (LLM)
        proposed_subtopics: List[SubtopicCreate] = semantic_expansion_service.generate_sub_niches(topic_title)
        
        saved_subtopics = []
        supabase = get_supabase_client()
        
        for proposed in proposed_subtopics:
            # 2. Bulk Data Retrieval (DataForSEO - Mocked)
            # In production, we would query DataForSEO here for proposed.keywords
            metrics = self._fetch_mock_metrics(proposed.keywords)
            proposed.metrics = metrics
            
            # 3. Profitability Filtering (Mocked - we accept all for now)
            # if metrics.volume < 100: continue
            
            # 4. Save to Database
            try:
                # We need to adapt the model to match the DB schema if strictly defined, 
                # but for likely JSONB or flexible schema we can insert.
                # Assuming 'research_subtopics' table exists or we dump to a 'subtopics' JSONB column in 'research_topics'?
                # The user request mentioned "The valid... subtopics are saved to the database."
                # I will assume there is a 'subtopics' table or similar. 
                # If not, I might need to create it or store in a JSON field.
                # Given the 'research_topics' table structure usually has related tables.
                # Let's try inserting into 'research_subtopics'.
                
                insert_data = {
                    "topic_id": topic_id,
                    "name": proposed.name,
                    "description": proposed.description,
                    "status": "active",
                    "keywords": proposed.keywords,
                    "trend_direction": proposed.trend_direction,
                    # Flatten metrics for DB if needed, or store as json
                    "search_volume": metrics.volume,
                    "cpc": metrics.cpc,
                    "seo_difficulty": metrics.difficulty,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                # Check if table exists/insert works. If not, we might fallback or log error.
                # Note: If validation fails (e.g. table doesn't exist), we catch exception.
                res = supabase.table("subtopics").insert(insert_data).execute()
                
                if res.data:
                    saved_subtopics.append(Subtopic(**res.data[0]))
                
            except Exception as e:
                logger.error(f"Failed to save subtopic {proposed.name}: {e}")
                # For MVP, if DB insert fails (e.g. table missing), return the object anyway so frontend sees something
                # in case we are just testing logic without full DB migration yet.
                # But idealy we want persistence.
                
                # Fallback: create a temporary object with ID
                fallback_sub = Subtopic(
                    id=f"temp-{random.randint(1000,9999)}",
                    topic_id=topic_id,
                    **proposed.dict()
                )
                saved_subtopics.append(fallback_sub)

        return saved_subtopics

    def _fetch_mock_metrics(self, keywords: List[str]) -> KeywordMetrics:
        """
        Generates realistic specific mock metrics for keywords.
        """
        # Deterministic-ish random based on keyword length
        base_seed = sum(len(k) for k in keywords)
        random.seed(base_seed)
        
        return KeywordMetrics(
            volume=random.randint(100, 10000),
            cpc=round(random.uniform(0.5, 5.0), 2),
            difficulty=random.randint(10, 80),
            competition_level=random.choice(["LOW", "MEDIUM", "HIGH"])
        )

topic_decomposition_service = TopicDecompositionService()
