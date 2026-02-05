"""
Scoring Utility

Calculates SEO optimization scores and traffic potential scores for keywords and content ideas.
"""

from typing import List, Dict, Any
import numpy as np
import logging
from dataclasses import dataclass

from ..models.keyword import Keyword

logger = logging.getLogger(__name__)

@dataclass
class ScoreWeights:
    """Weights for different scoring factors"""
    volume_weight: float = 0.4
    difficulty_weight: float = 0.3
    cpc_weight: float = 0.2
    intent_weight: float = 0.1

class ScoringUtility:
    """Utility for calculating various scores for keywords and content ideas"""
    
    def __init__(self, weights: ScoreWeights = None):
        self.weights = weights or ScoreWeights()
    
    def calculate_seo_score(self, keywords: List[Keyword]) -> float:
        """
        Calculate SEO optimization score (0-100) for a set of keywords
        
        Args:
            keywords: List of keywords to score
            
        Returns:
            SEO optimization score (0-100)
        """
        try:
            if not keywords:
                return 0.0
            
            # Calculate individual keyword scores
            keyword_scores = []
            for keyword in keywords:
                score = self._calculate_keyword_seo_score(keyword)
                keyword_scores.append(score)
            
            # Calculate weighted average
            avg_score = np.mean(keyword_scores)
            
            # Normalize to 0-100 scale
            normalized_score = min(100, max(0, avg_score))
            
            return round(normalized_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating SEO score: {str(e)}")
            return 0.0
    
    def calculate_traffic_score(self, keywords: List[Keyword]) -> float:
        """
        Calculate traffic potential score (0-100) for a set of keywords
        
        Args:
            keywords: List of keywords to score
            
        Returns:
            Traffic potential score (0-100)
        """
        try:
            if not keywords:
                return 0.0
            
            # Calculate individual keyword scores
            keyword_scores = []
            for keyword in keywords:
                score = self._calculate_keyword_traffic_score(keyword)
                keyword_scores.append(score)
            
            # Calculate weighted average
            avg_score = np.mean(keyword_scores)
            
            # Normalize to 0-100 scale
            normalized_score = min(100, max(0, avg_score))
            
            return round(normalized_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating traffic score: {str(e)}")
            return 0.0
    
    def calculate_opportunity_score(self, keyword: Keyword) -> float:
        """
        Calculate opportunity score for a single keyword
        
        Args:
            keyword: Keyword to score
            
        Returns:
            Opportunity score (0-100)
        """
        try:
            # Volume score (0-100)
            volume_score = self._normalize_volume(keyword.volume)
            
            # Difficulty score (inverted - lower difficulty = higher score)
            difficulty_score = 100 - keyword.difficulty
            
            # CPC score (0-100)
            cpc_score = self._normalize_cpc(keyword.cpc)
            
            # Intent score (0-100)
            intent_score = self._calculate_intent_score(keyword.intents)
            
            # Calculate weighted score
            opportunity_score = (
                volume_score * self.weights.volume_weight +
                difficulty_score * self.weights.difficulty_weight +
                cpc_score * self.weights.cpc_weight +
                intent_score * self.weights.intent_weight
            )
            
            return round(opportunity_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {str(e)}")
            return 0.0
    
    def _calculate_keyword_seo_score(self, keyword: Keyword) -> float:
        """Calculate SEO score for a single keyword"""
        # Factors that affect SEO score:
        # 1. Keyword difficulty (lower is better)
        # 2. Search volume (higher is better)
        # 3. Intent relevance (commercial/intent keywords score higher)
        # 4. Keyword length (shorter keywords often better)
        
        # Difficulty score (inverted)
        difficulty_score = 100 - keyword.difficulty
        
        # Volume score (logarithmic scale)
        volume_score = self._normalize_volume(keyword.volume)
        
        # Intent score
        intent_score = self._calculate_intent_score(keyword.intents)
        
        # Length score (shorter keywords get higher scores)
        length_score = max(0, 100 - len(keyword.keyword) * 2)
        
        # Weighted combination
        seo_score = (
            difficulty_score * 0.3 +
            volume_score * 0.3 +
            intent_score * 0.2 +
            length_score * 0.2
        )
        
        return min(100, max(0, seo_score))
    
    def _calculate_keyword_traffic_score(self, keyword: Keyword) -> float:
        """Calculate traffic potential score for a single keyword"""
        # Factors that affect traffic potential:
        # 1. Search volume (primary factor)
        # 2. CPC (higher CPC often indicates commercial value)
        # 3. Competition level (lower difficulty = more traffic potential)
        
        # Volume score (primary factor)
        volume_score = self._normalize_volume(keyword.volume)
        
        # CPC score
        cpc_score = self._normalize_cpc(keyword.cpc)
        
        # Competition score (inverted difficulty)
        competition_score = 100 - keyword.difficulty
        
        # Weighted combination
        traffic_score = (
            volume_score * 0.5 +
            cpc_score * 0.3 +
            competition_score * 0.2
        )
        
        return min(100, max(0, traffic_score))
    
    def _normalize_volume(self, volume: int) -> float:
        """Normalize search volume to 0-100 scale"""
        if volume <= 0:
            return 0
        
        # Use logarithmic scaling for volume
        # Typical range: 0-100,000+ monthly searches
        log_volume = np.log10(volume + 1)
        max_log_volume = 5  # 100,000 searches
        
        normalized = (log_volume / max_log_volume) * 100
        return min(100, max(0, normalized))
    
    def _normalize_cpc(self, cpc: float) -> float:
        """Normalize CPC to 0-100 scale"""
        if cpc <= 0:
            return 0
        
        # Use logarithmic scaling for CPC
        # Typical range: $0.01 - $10+
        log_cpc = np.log10(cpc + 0.01)
        max_log_cpc = 1  # $10 CPC
        
        normalized = (log_cpc / max_log_cpc) * 100
        return min(100, max(0, normalized))
    
    def _calculate_intent_score(self, intents: List[str]) -> float:
        """Calculate intent-based score"""
        if not intents:
            return 50  # Neutral score for no intent data
        
        # Intent scoring weights
        intent_weights = {
            'Commercial': 90,
            'Transactional': 85,
            'Informational': 70,
            'Navigational': 60
        }
        
        # Calculate weighted average
        total_score = 0
        total_weight = 0
        
        for intent in intents:
            weight = intent_weights.get(intent, 50)
            total_score += weight
            total_weight += 1
        
        if total_weight == 0:
            return 50
        
        return total_score / total_weight
    
    def calculate_content_quality_score(
        self,
        primary_keywords: List[Keyword],
        secondary_keywords: List[Keyword],
        content_type: str
    ) -> float:
        """
        Calculate content quality score based on keyword selection
        
        Args:
            primary_keywords: Primary keywords for the content
            secondary_keywords: Secondary keywords for the content
            content_type: Type of content (article, comparison, etc.)
            
        Returns:
            Content quality score (0-100)
        """
        try:
            if not primary_keywords:
                return 0.0
            
            # Base score from primary keywords
            primary_scores = [self.calculate_opportunity_score(k) for k in primary_keywords]
            base_score = np.mean(primary_scores)
            
            # Bonus for good secondary keyword selection
            if secondary_keywords:
                secondary_scores = [self.calculate_opportunity_score(k) for k in secondary_keywords]
                secondary_bonus = np.mean(secondary_scores) * 0.2
                base_score += secondary_bonus
            
            # Content type bonus
            type_bonus = self._get_content_type_bonus(content_type)
            base_score += type_bonus
            
            # Keyword diversity bonus
            diversity_bonus = self._calculate_diversity_bonus(primary_keywords + secondary_keywords)
            base_score += diversity_bonus
            
            return min(100, max(0, round(base_score, 2)))
            
        except Exception as e:
            logger.error(f"Error calculating content quality score: {str(e)}")
            return 0.0
    
    def _get_content_type_bonus(self, content_type: str) -> float:
        """Get bonus points based on content type"""
        type_bonuses = {
            'article': 5,
            'comparison': 10,
            'guide': 8,
            'tutorial': 7,
            'review': 6
        }
        
        return type_bonuses.get(content_type, 0)
    
    def _calculate_diversity_bonus(self, keywords: List[Keyword]) -> float:
        """Calculate bonus for keyword diversity"""
        if len(keywords) < 2:
            return 0
        
        # Check for different intent types
        intents = set()
        for keyword in keywords:
            intents.update(keyword.intents)
        
        # Bonus for having multiple intent types
        intent_diversity = len(intents)
        diversity_bonus = min(10, intent_diversity * 2)
        
        return diversity_bonus
