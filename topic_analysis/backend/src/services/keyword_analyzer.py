"""
Keyword analyzer service for calculating opportunity scores and categorizing keywords
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from ..models.keyword import Keyword
from ..core.config import settings
from .database import DatabaseService

logger = logging.getLogger(__name__)

class KeywordAnalyzerService:
    """Service for analyzing keywords and calculating opportunity scores"""
    
    def __init__(self, db_service: DatabaseService = None):
        self.db_service = db_service or DatabaseService()
        self.default_weights = {
            'search_volume': 0.4,
            'keyword_difficulty': 0.3,
            'cpc': 0.2,
            'search_intent': 0.1
        }
    
    def analyze_keywords(
        self, 
        keywords: List[Dict[str, Any]], 
        scoring_weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze keywords and calculate opportunity scores
        
        Args:
            keywords: List of keyword dictionaries
            scoring_weights: Custom scoring weights
            
        Returns:
            List of analyzed keyword dictionaries
        """
        if not keywords:
            return []
        
        # Use default weights if none provided
        if scoring_weights is None:
            scoring_weights = self.default_weights
        
        # Validate weights
        self._validate_weights(scoring_weights)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(keywords)
        
        # Calculate opportunity scores
        df = self._calculate_opportunity_scores(df, scoring_weights)
        
        # Categorize keywords
        df = self._categorize_keywords(df)
        
        # Set primary intents
        df = self._set_primary_intents(df)
        
        # Convert back to list of dictionaries
        analyzed_keywords = df.to_dict('records')
        
        logger.info(f"Analyzed {len(analyzed_keywords)} keywords")
        return analyzed_keywords
    
    def _calculate_opportunity_scores(
        self, 
        df: pd.DataFrame, 
        weights: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate opportunity scores for keywords"""
        
        # Normalize search volume (0-100 scale)
        df['volume_score'] = np.clip(df['Volume'] / 1000, 0, 100)
        
        # Normalize difficulty (invert so lower difficulty = higher score)
        df['difficulty_score'] = np.clip(100 - df['Difficulty'], 0, 100)
        
        # Normalize CPC (0-100 scale)
        df['cpc_score'] = np.clip(df['CPC'] * 20, 0, 100)
        
        # Calculate intent scores
        df['intent_score'] = df['Intents'].apply(self._calculate_intent_score)
        
        # Calculate weighted opportunity score
        df['opportunity_score'] = (
            df['volume_score'] * weights['search_volume'] +
            df['difficulty_score'] * weights['keyword_difficulty'] +
            df['cpc_score'] * weights['cpc'] +
            df['intent_score'] * weights['search_intent']
        )
        
        # Round to 2 decimal places
        df['opportunity_score'] = df['opportunity_score'].round(2)
        
        return df
    
    def _calculate_intent_score(self, intents: str) -> float:
        """Calculate intent score based on search intents"""
        if not intents or pd.isna(intents):
            return 50  # Default score for unknown intent
        
        intent_list = [intent.strip() for intent in str(intents).split(',')]
        
        # Intent priority scores
        intent_scores = {
            'Informational': 90,  # Highest priority for blog content
            'Commercial': 80,     # High priority for monetization
            'Navigational': 60,   # Medium priority
            'Transactional': 70   # Medium-high priority
        }
        
        # Find the highest priority intent
        max_score = 0
        for intent in intent_list:
            score = intent_scores.get(intent, 50)
            max_score = max(max_score, score)
        
        return max_score
    
    def _categorize_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize keywords based on opportunity scores"""
        
        # Define category thresholds
        high_threshold = 80
        medium_threshold = 60
        
        # Categorize keywords
        df['category'] = pd.cut(
            df['opportunity_score'],
            bins=[0, medium_threshold, high_threshold, 100],
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )
        
        return df
    
    def _set_primary_intents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set primary intent for each keyword"""
        
        def get_primary_intent(intents: str) -> str:
            if not intents or pd.isna(intents):
                return None
            
            intent_list = [intent.strip() for intent in str(intents).split(',')]
            
            # Priority order for primary intent
            intent_priority = ['Informational', 'Commercial', 'Transactional', 'Navigational']
            
            for intent in intent_priority:
                if intent in intent_list:
                    return intent
            
            # If no priority intent found, use first one
            return intent_list[0] if intent_list else None
        
        df['primary_intent'] = df['Intents'].apply(get_primary_intent)
        return df
    
    def _validate_weights(self, weights: Dict[str, float]) -> None:
        """Validate scoring weights"""
        required_keys = ['search_volume', 'keyword_difficulty', 'cpc', 'search_intent']
        
        # Check all required keys are present
        for key in required_keys:
            if key not in weights:
                raise ValueError(f"Missing required weight: {key}")
        
        # Check weights are non-negative
        for key, value in weights.items():
            if value < 0:
                raise ValueError(f"Weight {key} must be non-negative, got {value}")
        
        # Check weights sum to approximately 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def get_top_opportunities(self, keywords: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Get top opportunities from analyzed keywords"""
        
        df = pd.DataFrame(keywords)
        
        # High opportunity keywords (top 10)
        high_opportunity = df[df['category'] == 'high'].nlargest(10, 'opportunity_score')
        
        # Quick wins (low difficulty, decent volume)
        quick_wins = df[
            (df['Difficulty'] <= 25) & 
            (df['Volume'] >= 200)
        ].nlargest(10, 'opportunity_score')
        
        # High volume targets (top 5 by search volume)
        high_volume = df.nlargest(5, 'Volume')
        
        return {
            'high_opportunity_keywords': high_opportunity.to_dict('records'),
            'quick_wins': quick_wins.to_dict('records'),
            'high_volume_targets': high_volume.to_dict('records')
        }
    
    def get_content_recommendations(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get content format recommendations based on keywords"""
        
        recommendations = []
        
        for keyword in keywords:
            keyword_text = keyword['Keyword'].lower()
            content_format = self._detect_content_format(keyword_text)
            
            if content_format:
                recommendations.append({
                    'keyword': keyword['Keyword'],
                    'content_format': content_format,
                    'seo_score': keyword.get('opportunity_score', 0)
                })
        
        # Sort by SEO score
        recommendations.sort(key=lambda x: x['seo_score'], reverse=True)
        
        return recommendations
    
    def _detect_content_format(self, keyword: str) -> str:
        """Detect content format based on keyword patterns"""
        
        # List article patterns
        if any(pattern in keyword for pattern in ['best', 'top', 'list', 'review']):
            return 'list-article'
        
        # How-to guide patterns
        if any(pattern in keyword for pattern in ['how to', 'how do', 'guide', 'tutorial']):
            return 'how-to-guide'
        
        # Comparison patterns
        if any(pattern in keyword for pattern in ['vs', 'versus', 'compare', 'comparison']):
            return 'comparison'
        
        # Review patterns
        if any(pattern in keyword for pattern in ['review', 'rating', 'opinion']):
            return 'review'
        
        # Tutorial patterns
        if any(pattern in keyword for pattern in ['tutorial', 'learn', 'course']):
            return 'tutorial'
        
        # Case study patterns
        if any(pattern in keyword for pattern in ['case study', 'example', 'success story']):
            return 'case-study'
        
        # Default to informational
        return 'informational'
    
    def generate_insights(self, keywords: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from keyword analysis"""
        
        insights = []
        df = pd.DataFrame(keywords)
        
        # High opportunity keywords
        high_opportunity_count = len(df[df['category'] == 'high'])
        if high_opportunity_count > 0:
            insights.append(f"{high_opportunity_count} high-opportunity keywords identified")
        
        # Quick wins
        quick_wins_count = len(df[(df['Difficulty'] <= 25) & (df['Volume'] >= 200)])
        if quick_wins_count > 0:
            insights.append(f"{quick_wins_count} quick-win keywords available (low difficulty, decent volume)")
        
        # High volume targets
        high_volume_count = len(df[df['Volume'] >= 5000])
        if high_volume_count > 0:
            insights.append(f"{high_volume_count} high-volume keywords for pillar content")
        
        # Commercial opportunities
        commercial_count = len(df[df['CPC'] >= 2.0])
        if commercial_count > 0:
            insights.append(f"{commercial_count} high-CPC keywords for monetization")
        
        # Intent distribution
        intent_dist = df['primary_intent'].value_counts()
        if 'Informational' in intent_dist:
            insights.append(f"{intent_dist['Informational']} informational keywords ideal for blog content")
        
        return insights
    
    def generate_next_steps(self, keywords: List[Dict[str, Any]]) -> List[str]:
        """Generate next steps based on keyword analysis"""
        
        next_steps = []
        df = pd.DataFrame(keywords)
        
        # High opportunity keywords
        high_opportunity = df[df['category'] == 'high']
        if len(high_opportunity) > 0:
            next_steps.append("Prioritize high-opportunity keywords for immediate content creation")
        
        # Quick wins
        quick_wins = df[(df['Difficulty'] <= 25) & (df['Volume'] >= 200)]
        if len(quick_wins) > 0:
            next_steps.append("Create quick-win content for low-difficulty keywords")
        
        # High volume targets
        high_volume = df[df['Volume'] >= 5000]
        if len(high_volume) > 0:
            next_steps.append("Develop pillar content around high-volume keywords")
        
        # Commercial opportunities
        commercial = df[df['CPC'] >= 2.0]
        if len(commercial) > 0:
            next_steps.append("Focus on high-CPC keywords for monetization")
        
        # Content calendar
        next_steps.append("Create content calendar based on keyword priorities")
        
        return next_steps
    
    def cluster_keywords(self, keywords: List[Dict[str, Any]], n_clusters: int = 5) -> List[Dict[str, Any]]:
        """Cluster keywords for content grouping"""
        
        if len(keywords) < n_clusters:
            return keywords
        
        df = pd.DataFrame(keywords)
        
        # Prepare features for clustering
        features = ['Volume', 'Difficulty', 'CPC', 'opportunity_score']
        X = df[features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Add cluster information
        df['cluster_id'] = 'cluster_' + df['cluster'].astype(str)
        df['cluster_score'] = df.groupby('cluster')['opportunity_score'].transform('mean')
        
        return df.to_dict('records')
    
    async def save_keywords(self, keywords: List[Dict[str, Any]], file_id: str) -> bool:
        """Save keywords to database"""
        try:
            # Convert dictionaries to Keyword objects
            keyword_objects = []
            for kw_data in keywords:
                keyword = Keyword(
                    keyword=kw_data.get('Keyword', ''),
                    volume=kw_data.get('Volume', 0),
                    difficulty=kw_data.get('Difficulty', 0),
                    cpc=kw_data.get('CPC', 0),
                    intents=kw_data.get('Intents', '').split(',') if kw_data.get('Intents') else [],
                    opportunity_score=kw_data.get('opportunity_score', 0),
                    file_id=file_id
                )
                keyword_objects.append(keyword)
            
            # Save to database
            success = await self.db_service.save_keywords(keyword_objects)
            if success:
                logger.info(f"Saved {len(keywords)} keywords for file {file_id}")
            else:
                logger.error(f"Failed to save keywords for file {file_id}")
            return success
        except Exception as e:
            logger.error(f"Error saving keywords: {str(e)}")
            return False
    
    async def analyze_keywords_in_background(self, file_id: str, keywords_df: pd.DataFrame) -> bool:
        """Analyze keywords in background and save results"""
        try:
            # Update file status to processing
            await self.db_service.update_file_processing_status(file_id, "processing", 10, "Analyzing keywords...")
            
            # Convert DataFrame to list of dictionaries
            keywords = keywords_df.to_dict('records')
            
            # Analyze keywords
            analyzed_keywords = self.analyze_keywords(keywords)
            
            # Update progress
            await self.db_service.update_file_processing_status(file_id, "processing", 50, "Calculating opportunity scores...")
            
            # Save keywords to database
            success = await self.save_keywords(analyzed_keywords, file_id)
            
            if success:
                # Update progress
                await self.db_service.update_file_processing_status(file_id, "processing", 80, "Generating insights...")
                
                # Generate insights and recommendations
                insights = self.generate_insights(analyzed_keywords)
                next_steps = self.generate_next_steps(analyzed_keywords)
                top_opportunities = self.get_top_opportunities(analyzed_keywords)
                
                # Save analysis results
                analysis_results = {
                    'keywords': analyzed_keywords,
                    'insights': insights,
                    'next_steps': next_steps,
                    'top_opportunities': top_opportunities,
                    'total_keywords': len(analyzed_keywords),
                    'high_opportunity_count': len([k for k in analyzed_keywords if k.get('category') == 'high']),
                    'quick_wins_count': len([k for k in analyzed_keywords if k.get('Difficulty', 100) <= 25 and k.get('Volume', 0) >= 200])
                }
                
                await self.db_service.save_analysis_results(file_id, analysis_results)
                
                # Update status to completed
                await self.db_service.update_file_processing_status(file_id, "completed", 100, "Analysis completed successfully")
                
                logger.info(f"Successfully analyzed {len(analyzed_keywords)} keywords for file {file_id}")
                return True
            else:
                await self.db_service.update_file_processing_status(file_id, "error", 0, "Failed to save keywords")
                return False
                
        except Exception as e:
            logger.error(f"Error in background keyword analysis: {str(e)}")
            await self.db_service.update_file_processing_status(file_id, "error", 0, f"Analysis failed: {str(e)}")
            return False

# Global instance
keyword_analyzer_service = KeywordAnalyzerService()

