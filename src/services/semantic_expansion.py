"""
Service for expanding topics into sub-niches using semantic analysis (Mocked for MVP).
"""

import logging
from typing import List
from ..core.models.topic_analysis import SubtopicCreate

logger = logging.getLogger(__name__)

class SemanticExpansionService:
    """
    Expands a broad topic into specific, profitable sub-niches.
    Currently mocks LLM behavior for speed and stability.
    """

    def generate_sub_niches(self, topic: str) -> List[SubtopicCreate]:
        """
        Generates sub-niches for a given topic.
        
        Args:
            topic: The main topic (e.g., "Golf")
            
        Returns:
            List of SubtopicCreate objects with proposed names and keywords.
        """
        logger.info(f"Generating sub-niches for topic: {topic}")
        
        # Mock logic: Generate deterministic but realistic subtopics based on input
        # In production, this would call an LLM (OpenAI/Anthropic)
        
        subtopics = [
            SubtopicCreate(
                name=f"Beginner {topic} Guides",
                description=f"Comprehensive guides for people just starting with {topic}.",
                keywords=[f"how to start {topic}", f"{topic} for beginners", f"{topic} basics"],
                trend_direction="up"
            ),
            SubtopicCreate(
                name=f"Best {topic} Equipment",
                description=f"Reviews and comparisons of top-rated {topic} gear.",
                keywords=[f"best {topic} gear", f"{topic} equipment reviews", f"cheap {topic} tools"],
                trend_direction="stable"
            ),
             SubtopicCreate(
                name=f"Advanced {topic} Strategies",
                description=f"In-depth techniques for experienced {topic} enthusiasts.",
                keywords=[f"advanced {topic} tips", f"mastering {topic}", f"{topic} pro guide"],
                trend_direction="up"
            ),
            SubtopicCreate(
                name=f"{topic} for Seniors",
                description=f"Specialized advice for seniors interested in {topic}.",
                keywords=[f"{topic} for seniors", f"safe {topic} for elderly", f"{topic} benefits for seniors"],
                trend_direction="up"
            ),
             SubtopicCreate(
                name=f"Budget {topic}",
                description=f"How to enjoy {topic} without breaking the bank.",
                keywords=[f"cheap {topic}", f"free {topic} resources", f"diy {topic}"],
                trend_direction="down"
            )
        ]
        
        return subtopics

    def cluster_keywords(self, keywords: List[str]) -> List[str]:
        """
        Groups raw keywords into semantic clusters.
        """
        # Mock: just return them as is or simple grouping
        return keywords

semantic_expansion_service = SemanticExpansionService()
