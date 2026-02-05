"""
Topic Decomposition Service
Handles LLM-powered topic decomposition for search queries
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog
from sqlalchemy.orm import Session
from ..core.llm_config import LLMConfigManager
from ..core.redis import cache
from ..models.topic_decomposition import TopicDecomposition

logger = structlog.get_logger()

class TopicDecompositionService:
    """Service for LLM-powered topic decomposition"""
    
    def __init__(self, db: Session):
        self.db = db
        self.llm_manager = LLMConfigManager()
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    async def decompose_topic(
        self, 
        search_query: str, 
        user_id: str,
        max_subtopics: int = 10
    ) -> Dict[str, Any]:
        """
        Decompose a search query into related subtopics using LLM analysis
        
        Args:
            search_query: The search query to decompose
            user_id: User ID for the request
            max_subtopics: Maximum number of subtopics to generate
            
        Returns:
            Dict containing the decomposition results
        """
        try:
            logger.info("Starting topic decomposition", 
                       search_query=search_query, user_id=user_id, max_subtopics=max_subtopics)
            
            # Check cache first
            cache_key = f"topic_decomp:{user_id}:{hash(search_query)}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info("Cache hit for topic decomposition", cache_key=cache_key)
                return cached_result
            
            # Get LLM provider for analysis
            llm_config = self.llm_manager.get_config()
            if not llm_config:
                raise Exception("No LLM provider available")
            
            # Generate subtopics using LLM
            subtopics = await self._generate_subtopics(
                search_query, max_subtopics, llm_config
            )
            
            # Save to database
            decomposition = self._save_decomposition(
                search_query, subtopics, user_id
            )
            
            result = {
                "id": str(decomposition.id),
                "search_query": search_query,
                "subtopics": subtopics,
                "created_at": decomposition.created_at.isoformat()
            }
            
            # Cache the result
            await self._set_cached_result(cache_key, result)
            
            logger.info("Topic decomposition completed successfully", 
                       search_query=search_query, subtopics_count=len(subtopics))
            
            return result
            
        except Exception as e:
            logger.error("Topic decomposition failed", 
                        search_query=search_query, error=str(e))
            raise
    
    async def _generate_subtopics(
        self, 
        search_query: str, 
        max_subtopics: int,
        llm_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate subtopics using LLM with structured prompt
        """
        prompt = self._build_decomposition_prompt(search_query, max_subtopics)
        
        try:
            # Use the LLM to generate subtopics
            response = await self._call_llm(prompt, llm_config)
            
            # Parse the response
            subtopics = self._parse_llm_response(response)
            
            # Validate and score subtopics
            validated_subtopics = self._validate_and_score_subtopics(
                subtopics, search_query
            )
            
            return validated_subtopics[:max_subtopics]
            
        except Exception as e:
            logger.error("LLM subtopic generation failed", error=str(e))
            # Fallback to predefined subtopics
            return self._get_fallback_subtopics(search_query, max_subtopics)
    
    def _build_decomposition_prompt(self, search_query: str, max_subtopics: int) -> str:
        """Build structured prompt for topic decomposition"""
        return f"""
You are an expert content strategist. Decompose the search query "{search_query}" into {max_subtopics} related subtopics.

For each subtopic, provide:
- name: A specific, focused subtopic name
- description: A brief description of what this subtopic covers
- relevance_score: A score from 0.0 to 1.0 indicating relevance to the main query
- category: A category that best fits this subtopic

Return the response as a JSON array of objects. Focus on creating diverse, actionable subtopics that would be useful for content creation.

Example format:
[
  {{
    "name": "Electric cars in California",
    "description": "EV market trends and opportunities in California",
    "relevance_score": 0.95,
    "category": "automotive"
  }},
  {{
    "name": "Car dealers",
    "description": "Automotive dealership opportunities and trends", 
    "relevance_score": 0.88,
    "category": "automotive"
  }}
]

Search query: "{search_query}"
"""
    
    async def _call_llm(self, prompt: str, llm_config: Dict[str, Any]) -> str:
        """Call the LLM with the given prompt"""
        # This would integrate with the existing LLM infrastructure
        # For now, we'll simulate the LLM call
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Mock response for testing
        return json.dumps([
            {
                "name": "Electric cars in California",
                "description": "EV market trends and opportunities in California",
                "relevance_score": 0.95,
                "category": "automotive"
            },
            {
                "name": "Car dealers",
                "description": "Automotive dealership opportunities and trends",
                "relevance_score": 0.88,
                "category": "automotive"
            },
            {
                "name": "Car parts",
                "description": "Automotive parts and accessories market",
                "relevance_score": 0.82,
                "category": "automotive"
            },
            {
                "name": "Car repair",
                "description": "Automotive repair services and trends",
                "relevance_score": 0.79,
                "category": "automotive"
            },
            {
                "name": "Car hacks",
                "description": "DIY car maintenance and modification tips",
                "relevance_score": 0.75,
                "category": "automotive"
            }
        ])
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response and extract subtopics"""
        try:
            subtopics = json.loads(response)
            if not isinstance(subtopics, list):
                raise ValueError("Response is not a list")
            return subtopics
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse LLM response", error=str(e), response=response)
            raise
    
    def _validate_and_score_subtopics(
        self, 
        subtopics: List[Dict[str, Any]], 
        search_query: str
    ) -> List[Dict[str, Any]]:
        """Validate and score subtopics for relevance"""
        validated = []
        
        for subtopic in subtopics:
            if self._is_valid_subtopic(subtopic):
                # Calculate additional relevance score based on keyword overlap
                enhanced_score = self._calculate_enhanced_relevance_score(
                    subtopic, search_query
                )
                subtopic["relevance_score"] = enhanced_score
                validated.append(subtopic)
        
        # Sort by relevance score
        validated.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return validated
    
    def _is_valid_subtopic(self, subtopic: Dict[str, Any]) -> bool:
        """Validate subtopic structure and content"""
        required_fields = ["name", "description", "relevance_score", "category"]
        
        if not all(field in subtopic for field in required_fields):
            return False
        
        if not isinstance(subtopic["relevance_score"], (int, float)):
            return False
        
        if not 0 <= subtopic["relevance_score"] <= 1:
            return False
        
        if not subtopic["name"].strip() or not subtopic["description"].strip():
            return False
        
        return True
    
    def _calculate_enhanced_relevance_score(
        self, 
        subtopic: Dict[str, Any], 
        search_query: str
    ) -> float:
        """Calculate enhanced relevance score based on keyword overlap"""
        base_score = subtopic["relevance_score"]
        
        # Simple keyword overlap calculation
        query_words = set(search_query.lower().split())
        subtopic_words = set(subtopic["name"].lower().split())
        
        overlap = len(query_words.intersection(subtopic_words))
        overlap_bonus = min(overlap * 0.1, 0.2)  # Max 0.2 bonus
        
        enhanced_score = min(base_score + overlap_bonus, 1.0)
        
        return round(enhanced_score, 2)
    
    def _get_fallback_subtopics(
        self, 
        search_query: str, 
        max_subtopics: int
    ) -> List[Dict[str, Any]]:
        """Get fallback subtopics when LLM fails"""
        logger.warning("Using fallback subtopics", search_query=search_query)
        
        # Generic fallback subtopics
        fallback_subtopics = [
            {
                "name": f"{search_query} basics",
                "description": f"Basic information about {search_query}",
                "relevance_score": 0.8,
                "category": "general"
            },
            {
                "name": f"{search_query} trends",
                "description": f"Current trends in {search_query}",
                "relevance_score": 0.7,
                "category": "trends"
            },
            {
                "name": f"{search_query} tips",
                "description": f"Tips and advice for {search_query}",
                "relevance_score": 0.6,
                "category": "tips"
            }
        ]
        
        return fallback_subtopics[:max_subtopics]
    
    def _save_decomposition(
        self, 
        search_query: str, 
        subtopics: List[Dict[str, Any]], 
        user_id: str
    ) -> TopicDecomposition:
        """Save topic decomposition to database"""
        decomposition = TopicDecomposition(
            user_id=user_id,
            search_query=search_query,
            subtopics=subtopics
        )
        
        self.db.add(decomposition)
        self.db.commit()
        self.db.refresh(decomposition)
        
        return decomposition
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result from Redis"""
        try:
            cached_data = cache.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning("Failed to get cached result", cache_key=cache_key, error=str(e))
            return None
    
    async def _set_cached_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache result in Redis"""
        try:
            cache.setex(cache_key, self.cache_ttl, json.dumps(result))
        except Exception as e:
            logger.warning("Failed to cache result", cache_key=cache_key, error=str(e))
    
    async def get_decomposition(self, decomposition_id: str) -> Optional[Dict[str, Any]]:
        """Get topic decomposition by ID"""
        try:
            decomposition = self.db.query(TopicDecomposition).filter(
                TopicDecomposition.id == decomposition_id
            ).first()
            
            if not decomposition:
                return None
            
            return {
                "id": str(decomposition.id),
                "search_query": decomposition.search_query,
                "subtopics": decomposition.subtopics,
                "created_at": decomposition.created_at.isoformat(),
                "updated_at": decomposition.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get decomposition", 
                        decomposition_id=decomposition_id, error=str(e))
            raise
    
    async def delete_decomposition(self, decomposition_id: str) -> bool:
        """Delete topic decomposition"""
        try:
            decomposition = self.db.query(TopicDecomposition).filter(
                TopicDecomposition.id == decomposition_id
            ).first()
            
            if not decomposition:
                return False
            
            self.db.delete(decomposition)
            self.db.commit()
            
            logger.info("Topic decomposition deleted", decomposition_id=decomposition_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete decomposition", 
                        decomposition_id=decomposition_id, error=str(e))
            raise
