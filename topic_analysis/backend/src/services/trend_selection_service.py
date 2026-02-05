"""
Trend Selection Service
Handles trend selection, CSV processing, and trend management
"""

import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional
import structlog
from datetime import datetime
import json

from ..core.supabase_database import get_supabase_db
from ..core.llm_config import LLMConfigManager

logger = structlog.get_logger()

class TrendSelectionService:
    def __init__(self):
        self.db = get_supabase_db()
        self.llm_manager = LLMConfigManager()

    async def process_trend_csv(
        self,
        df: pd.DataFrame,
        search_term: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process uploaded CSV with trend data
        """
        try:
            logger.info("Processing trend CSV", 
                       search_term=search_term, 
                       rows=len(df),
                       user_id=user_id)
            
            # Clean and validate data
            df_clean = self._clean_trend_data(df)
            
            # Generate additional trend analysis using LLM
            enhanced_trends = await self._enhance_trends_with_llm(
                df_clean, search_term
            )
            
            # Save to database
            trends_saved = await self._save_trends_to_db(
                enhanced_trends, search_term, user_id
            )
            
            return {
                "trends_imported": len(enhanced_trends),
                "trends_saved": trends_saved,
                "search_term": search_term,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to process trend CSV", error=str(e))
            raise

    def _clean_trend_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate trend data"""
        # Remove rows with missing essential data
        df_clean = df.dropna(subset=['topic', 'search_volume'])
        
        # Convert search_volume to numeric
        df_clean['search_volume'] = pd.to_numeric(df_clean['search_volume'], errors='coerce')
        
        # Fill missing values
        df_clean['trend_direction'] = df_clean['trend_direction'].fillna('stable')
        df_clean['competition'] = df_clean['competition'].fillna('Medium')
        df_clean['opportunity_score'] = pd.to_numeric(
            df_clean['opportunity_score'], errors='coerce'
        ).fillna(50)
        
        # Generate trend IDs
        df_clean['trend_id'] = df_clean['topic'].apply(
            lambda x: f"trend_{hash(x) % 1000000}"
        )
        
        return df_clean

    async def _enhance_trends_with_llm(
        self, 
        df: pd.DataFrame, 
        search_term: str
    ) -> List[Dict[str, Any]]:
        """Enhance trend data with LLM analysis"""
        try:
            llm_config = self.llm_manager.get_config()
            if not llm_config:
                logger.warning("No LLM config available, returning basic trends")
                return df.to_dict('records')
            
            # Prepare trends for LLM analysis
            trends_text = "\n".join([
                f"- {row['topic']} (Volume: {row['search_volume']}, Direction: {row['trend_direction']})"
                for _, row in df.iterrows()
            ])
            
            prompt = f"""
            Analyze these trending topics related to "{search_term}" and enhance each with:
            1. Content angle suggestions
            2. Target audience insights
            3. Seasonal patterns
            4. Related keywords
            5. Content difficulty assessment
            
            Trends to analyze:
            {trends_text}
            
            Return as JSON array with enhanced data for each trend.
            """
            
            # Call LLM (simplified for now)
            enhanced_trends = []
            for _, row in df.iterrows():
                enhanced_trend = {
                    "trend_id": row['trend_id'],
                    "topic": row['topic'],
                    "search_volume": int(row['search_volume']),
                    "trend_direction": row['trend_direction'],
                    "competition": row['competition'],
                    "opportunity_score": int(row['opportunity_score']),
                    "content_angles": [
                        f"Best {row['topic']} for beginners",
                        f"Advanced {row['topic']} techniques",
                        f"{row['topic']} cost analysis"
                    ],
                    "target_audience": "General audience",
                    "seasonality": "Year-round",
                    "related_keywords": [row['topic'], f"{row['topic']} guide", f"{row['topic']} tips"],
                    "difficulty": "Medium",
                    "source": "csv_upload",
                    "created_at": datetime.utcnow().isoformat()
                }
                enhanced_trends.append(enhanced_trend)
            
            return enhanced_trends
            
        except Exception as e:
            logger.error("Failed to enhance trends with LLM", error=str(e))
            return df.to_dict('records')

    async def _save_trends_to_db(
        self,
        trends: List[Dict[str, Any]],
        search_term: str,
        user_id: Optional[str] = None
    ) -> int:
        """Save trends to database"""
        try:
            # This would save to a trends table
            # For now, return the count
            logger.info("Saving trends to database", count=len(trends))
            return len(trends)
            
        except Exception as e:
            logger.error("Failed to save trends to database", error=str(e))
            return 0

    async def select_trends_for_generation(
        self,
        selected_trend_ids: List[str],
        search_term: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select specific trends for content/software generation
        """
        try:
            logger.info("Selecting trends for generation",
                       selected_count=len(selected_trend_ids),
                       search_term=search_term,
                       user_id=user_id)
            
            # Get trend details
            selected_trends = await self._get_trends_by_ids(selected_trend_ids)
            
            # Save selection to database
            selection_id = await self._save_trend_selection(
                selected_trend_ids, search_term, user_id
            )
            
            return {
                "selection_id": selection_id,
                "selected_trends": selected_trends,
                "search_term": search_term,
                "selected_count": len(selected_trends),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to select trends", error=str(e))
            raise

    async def _get_trends_by_ids(self, trend_ids: List[str]) -> List[Dict[str, Any]]:
        """Get trend details by IDs"""
        # This would query the database
        # For now, return mock data
        return [
            {
                "trend_id": tid,
                "topic": f"Selected trend {tid}",
                "search_volume": 10000,
                "trend_direction": "rising",
                "competition": "Medium",
                "opportunity_score": 75
            }
            for tid in trend_ids
        ]

    async def _save_trend_selection(
        self,
        selected_trend_ids: List[str],
        search_term: str,
        user_id: Optional[str] = None
    ) -> str:
        """Save trend selection to database"""
        selection_id = f"selection_{datetime.utcnow().timestamp()}"
        logger.info("Saving trend selection", selection_id=selection_id)
        return selection_id

    async def get_available_trends(
        self,
        search_term: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get all available trends for a search term"""
        try:
            # This would query both analysis results and CSV uploads
            # For now, return mock data
            return {
                "analysis_trends": [
                    {
                        "trend_id": "analysis_1",
                        "topic": f"{search_term} sustainability trends",
                        "search_volume": 50000,
                        "trend_direction": "rising",
                        "competition": "Medium",
                        "opportunity_score": 80,
                        "source": "llm_analysis"
                    }
                ],
                "csv_trends": [
                    {
                        "trend_id": "csv_1",
                        "topic": f"{search_term} technology innovations",
                        "search_volume": 30000,
                        "trend_direction": "stable",
                        "competition": "High",
                        "opportunity_score": 70,
                        "source": "csv_upload"
                    }
                ],
                "total_trends": 2
            }
            
        except Exception as e:
            logger.error("Failed to get available trends", error=str(e))
            raise

    async def get_selected_trends(
        self,
        search_term: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get currently selected trends"""
        try:
            # This would query the database for selected trends
            # For now, return mock data
            return {
                "selected_trends": [],
                "selection_id": None,
                "search_term": search_term
            }
            
        except Exception as e:
            logger.error("Failed to get selected trends", error=str(e))
            raise

