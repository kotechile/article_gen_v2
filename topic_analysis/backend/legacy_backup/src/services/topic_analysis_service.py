from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from typing import List, Optional, Dict, Any
import json
import openai
import time
from datetime import datetime, timedelta
import logging

from ..models.topic_analysis import TopicAnalysis, RelatedArea, AffiliateProgram, TopicSearchLog
from ..core.config import settings

logger = logging.getLogger(__name__)

class TopicAnalysisService:
    def __init__(self, db: Session):
        self.db = db

    async def analyze_topic(
        self, 
        topic: str, 
        user_id: Optional[str] = None,
        use_llm: bool = True,
        max_related_areas: int = 10,
        max_affiliate_programs: int = 8
    ) -> Dict[str, Any]:
        """
        Analyze a topic and return related areas and affiliate programs
        """
        start_time = time.time()
        topic_lower = topic.lower().strip()
        
        try:
            # 1. Check database first
            existing_analysis = self._get_existing_analysis(topic_lower)
            if existing_analysis:
                self._log_search(topic, user_id, 'database', start_time, True)
                return self._format_analysis_response(existing_analysis)

            # 2. Try keyword pattern matching
            pattern_match = self._find_pattern_match(topic_lower)
            if pattern_match:
                # Save to database for future use
                self._save_analysis(topic_lower, pattern_match)
                self._log_search(topic, user_id, 'pattern_match', start_time, True)
                return pattern_match

            # 3. Use LLM if enabled and API key available
            if use_llm and settings.OPENAI_API_KEY:
                llm_analysis = await self._analyze_with_llm(
                    topic, max_related_areas, max_affiliate_programs
                )
                if llm_analysis:
                    # Save to database
                    self._save_analysis(topic_lower, llm_analysis)
                    self._log_search(topic, user_id, 'llm', start_time, True)
                    return llm_analysis

            # 4. Fallback to generic analysis
            fallback_analysis = self._generate_fallback_analysis(topic)
            self._save_analysis(topic_lower, fallback_analysis)
            self._log_search(topic, user_id, 'fallback', start_time, True)
            return fallback_analysis

        except Exception as e:
            logger.error(f"Error analyzing topic '{topic}': {str(e)}")
            self._log_search(topic, user_id, 'error', start_time, False, str(e))
            raise

    def _get_existing_analysis(self, topic: str) -> Optional[TopicAnalysis]:
        """Get existing analysis from database"""
        return self.db.query(TopicAnalysis).filter(
            TopicAnalysis.topic == topic
        ).first()

    def _find_pattern_match(self, topic: str) -> Optional[Dict[str, Any]]:
        """Find analysis using keyword patterns"""
        # Define keyword patterns and their corresponding analyses
        patterns = {
            'car': {
                'related_areas': [
                    'Family SUVs', 'Minivans', 'Safety Features', 'Car Seats & Accessories',
                    'Auto Insurance', 'Car Financing', 'Vehicle Maintenance', 'Road Trip Planning'
                ],
                'affiliate_programs': [
                    {'name': 'Cars.com', 'commission': '3-8%', 'category': 'Car Listings', 'difficulty': 'Easy'},
                    {'name': 'AutoTrader', 'commission': '2-6%', 'category': 'Car Marketplace', 'difficulty': 'Easy'},
                    {'name': 'Edmunds', 'commission': '4-10%', 'category': 'Car Reviews', 'difficulty': 'Medium'}
                ]
            },
            'eco': {
                'related_areas': [
                    'Solar Power Systems', 'Green Home Architecture', 'Eco-Friendly HVAC',
                    'Water Heating Solutions', 'Energy Efficient Windows', 'Sustainable Building Materials'
                ],
                'affiliate_programs': [
                    {'name': 'Tesla Solar', 'commission': '8-15%', 'category': 'Solar Power', 'difficulty': 'Medium'},
                    {'name': 'Lennox HVAC', 'commission': '5-12%', 'category': 'HVAC', 'difficulty': 'Easy'}
                ]
            }
        }

        # Check for pattern matches
        for pattern, analysis in patterns.items():
            if pattern in topic:
                return analysis

        return None

    async def _analyze_with_llm(
        self, 
        topic: str, 
        max_related_areas: int, 
        max_affiliate_programs: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze topic using LLM"""
        try:
            openai.api_key = settings.OPENAI_API_KEY

            prompt = f"""
            Analyze the topic "{topic}" for affiliate marketing opportunities.
            
            Provide:
            1. {max_related_areas} related areas/subtopics
            2. {max_affiliate_programs} relevant affiliate programs
            
            Return as JSON with this structure:
            {{
                "related_areas": [
                    {{"area": "Area Name", "description": "Brief description", "relevance_score": 0.9}}
                ],
                "affiliate_programs": [
                    {{"name": "Program Name", "commission": "5-15%", "category": "Category", "difficulty": "Easy", "description": "Description"}}
                ]
            }}
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert affiliate marketing analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )

            analysis_text = response.choices[0].message.content
            # Parse JSON response (in real implementation, add proper error handling)
            analysis = json.loads(analysis_text)
            
            return analysis

        except Exception as e:
            logger.error(f"LLM analysis failed for topic '{topic}': {str(e)}")
            return None

    def _generate_fallback_analysis(self, topic: str) -> Dict[str, Any]:
        """Generate fallback analysis when no other method works"""
        words = topic.split()
        related_areas = [
            f"{word.title()} Solutions" for word in words[:5]
        ] + [
            "Product Reviews", "Buying Guides", "Expert Recommendations", "Comparison Tools"
        ]

        return {
            "related_areas": related_areas[:8],
            "affiliate_programs": [
                {
                    "name": "Amazon Associates",
                    "commission": "1-10%",
                    "category": "General",
                    "difficulty": "Easy",
                    "description": "Wide range of products"
                },
                {
                    "name": "ShareASale",
                    "commission": "5-15%",
                    "category": "Various",
                    "difficulty": "Medium",
                    "description": "Diverse merchant network"
                }
            ]
        }

    def _save_analysis(self, topic: str, analysis: Dict[str, Any]):
        """Save analysis to database"""
        try:
            topic_analysis = TopicAnalysis(
                topic=topic,
                related_areas=analysis.get('related_areas', []),
                affiliate_programs=analysis.get('affiliate_programs', []),
                analysis_metadata={
                    "generated_at": datetime.utcnow().isoformat(),
                    "source": "hybrid_analysis"
                }
            )
            
            self.db.add(topic_analysis)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to save analysis for topic '{topic}': {str(e)}")
            self.db.rollback()

    def _format_analysis_response(self, analysis: TopicAnalysis) -> Dict[str, Any]:
        """Format database analysis for API response"""
        return {
            "related_areas": analysis.related_areas,
            "affiliate_programs": analysis.affiliate_programs,
            "metadata": analysis.analysis_metadata
        }

    def _log_search(
        self, 
        topic: str, 
        user_id: Optional[str], 
        source: str, 
        start_time: float, 
        success: bool, 
        error_message: Optional[str] = None
    ):
        """Log search for analytics"""
        try:
            response_time = int((time.time() - start_time) * 1000)
            
            search_log = TopicSearchLog(
                topic=topic,
                user_id=user_id,
                analysis_source=source,
                response_time_ms=response_time,
                success='true' if success else 'false',
                error_message=error_message
            )
            
            self.db.add(search_log)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to log search: {str(e)}")

    def get_popular_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most searched topics"""
        return self.db.query(
            TopicSearchLog.topic,
            func.count(TopicSearchLog.id).label('search_count')
        ).group_by(TopicSearchLog.topic).order_by(
            func.count(TopicSearchLog.id).desc()
        ).limit(limit).all()

    def search_topics(self, query: str, limit: int = 10) -> List[str]:
        """Search topics by keyword"""
        return self.db.query(TopicAnalysis.topic).filter(
            TopicAnalysis.topic.ilike(f"%{query}%")
        ).limit(limit).all()

    def add_manual_topic(self, topic: str, analysis: Dict[str, Any]) -> bool:
        """Manually add a topic analysis"""
        try:
            topic_analysis = TopicAnalysis(
                topic=topic.lower(),
                related_areas=analysis.get('related_areas', []),
                affiliate_programs=analysis.get('affiliate_programs', []),
                analysis_metadata={
                    "source": "manual",
                    "added_at": datetime.utcnow().isoformat()
                }
            )
            
            self.db.add(topic_analysis)
            self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add manual topic '{topic}': {str(e)}")
            self.db.rollback()
            return False


