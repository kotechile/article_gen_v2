"""
Subtopics service for dataflow persistence
This service handles all subtopic operations using Supabase Client/SDK
interaction with the new normalized 'subtopics' table
"""

import logging
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
from datetime import datetime
import json

from .supabase_service import get_supabase_service

logger = logging.getLogger(__name__)

class SubtopicsService:
    """Service for managing subtopics"""
    
    def __init__(self):
        self.supabase = get_supabase_service()
        self.table_name = "subtopics"
    
    async def create(self, research_topic_id: UUID, name: str, user_id: UUID, 
                     trend_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Create a new subtopic"""
        try:
            data = {
                "project_id": str(research_topic_id), # Mapped to project_id
                "user_id": str(user_id),
                "name": name,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if trend_data:
                # Calculate initial scores if trend data is provided
                data["trend_direction"] = trend_data.get("trend_direction")
                data["trend_score"] = trend_data.get("trend_score")
                data["interest_over_time"] = trend_data.get("interest_over_time", [])
                
                # Check for SEO data 
                if "seo_difficulty" in trend_data:
                    data["seo_difficulty"] = trend_data["seo_difficulty"]
                    
                if "search_volume" in trend_data:
                    data["search_volume"] = trend_data["search_volume"]
                    
                if "cpc" in trend_data:
                    # Fix: CPC should remain float. Validating schema type shortly.
                    try:
                        data["cpc"] = float(trend_data["cpc"])
                    except (ValueError, TypeError):
                        data["cpc"] = 0.0
                    
                if "keywords" in trend_data:
                    data["keywords"] = trend_data["keywords"]
                    
                if "rationale" in trend_data:
                    data["rationale"] = trend_data["rationale"]
                    
                if "target_audience" in trend_data:
                    data["target_audience"] = trend_data["target_audience"]

                # NEW: Rich Data Persistence
                if "trend_analysis" in trend_data:
                    data["trend_analysis"] = trend_data["trend_analysis"]
                    
                if "monetization" in trend_data:
                    data["monetization_data"] = trend_data["monetization"]
                
                # Calculate viability
                data["viability_score"] = self._calculate_viability_score(data)
                
            # Use execute_query directly to get error details
            # Always include user_id for security
            data["user_id"] = str(user_id)
            
            result = await self.supabase.execute_query(
                table=self.table_name,
                operation="insert",
                data=data
            )
            
            # Extract data if successful
            return_value = result["data"][0] if result.get("data") else None
            
            if result.get("error"):
                error_msg = result.get("error")
                with open("debug_db_insert_fail.txt", "w") as f:
                    f.write(f"Insert failed for {name}.\nError: {error_msg}\nData: {json.dumps(data, default=str)}")
                return None
            
            return return_value
            

            
        except Exception as e:
            logger.error(f"Error creating subtopic: {e}")
            with open("debug_db_exception.txt", "w") as f:
                f.write(f"DB Exception: {str(e)}")
            raise
    
    async def get_by_research_topic(self, research_topic_id: UUID, user_id: UUID) -> List[Dict[str, Any]]:
        """Get all subtopics for a research topic"""
        try:
            subtopics = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={"project_id": str(research_topic_id)}, # Mapped to project_id
                user_id=user_id,
                order_by={"viability_score": "desc"}
            )
            
            return subtopics
            
        except Exception as e:
            logger.error(f"Error getting subtopics for topic {research_topic_id}: {e}")
            raise

    async def get_by_id(self, subtopic_id: UUID, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get a subtopic by ID"""
        try:
            result = await self.supabase.get_by_id(
                table=self.table_name,
                id=subtopic_id,
                user_id=user_id
            )
            return result
        except Exception as e:
            logger.error(f"Error getting subtopic {subtopic_id}: {e}")
            raise

    async def update(self, subtopic_id: UUID, update_data: Dict[str, Any], user_id: UUID) -> Optional[Dict[str, Any]]:
        """Update a subtopic"""
        try:
            # Ensure updated_at is set
            update_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Recalculate viability score if relevant fields are updated
            if any(key in update_data for key in ["trend_score", "seo_difficulty", "search_volume", "affiliate_offer_count"]):
                # Need current data to calculate full score if partially updating
                current = await self.get_by_id(subtopic_id, user_id)
                if current:
                    merged_data = {**current, **update_data}
                    update_data["viability_score"] = self._calculate_viability_score(merged_data)
            
            # Filter out columns that don't exist in the remote schema
            # TODO: Apply migration 20250113000001_create_subtopics_table.sql to remote (specifically CPC)
            safe_update_data = {
                k: v for k, v in update_data.items() 
                if k not in [] # All columns should now exist
            }
            
            result = await self.supabase.update(
                table=self.table_name,
                id=subtopic_id,
                data=safe_update_data,
                user_id=user_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating subtopic {subtopic_id}: {e}")
            raise

    async def get_top_keyword(self, subtopic_id: UUID) -> Optional[str]:
        """Get the highest search volume keyword for a subtopic"""
        try:
            # Check if keyword table exists and query it
            result = self.supabase.get_client().table("keywords") \
                .select("keyword") \
                .eq("subtopic_id", str(subtopic_id)) \
                .order("search_volume", desc=True) \
                .limit(1) \
                .execute()
                
            if result.data and len(result.data) > 0:
                return result.data[0]["keyword"]
            return None
        except Exception as e:
            logger.warning(f"Failed to get top keyword for subtopic {subtopic_id}: {e}")
            return None

    async def delete(self, subtopic_id: UUID, user_id: UUID) -> bool:
        """Delete a subtopic"""
        try:
            return await self.supabase.delete(
                table=self.table_name,
                id=subtopic_id,
                user_id=user_id
            )
        except Exception as e:
            logger.error(f"Error deleting subtopic {subtopic_id}: {e}")
            raise

    def _calculate_viability_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate viability score (0-100) based on weighted metrics:
        - Trend Score (30%)
        - SEO Difficulty (40%) - Inverted (lower is better)
        - Monetization (30%) - Affiliate offer count
        """
        score = 0.0
        
        # 1. Trend Score (30%)
        trend_val = data.get("trend_score")
        trend_score = float(trend_val) if trend_val is not None else 0.0
        score += (trend_score / 100.0) * 30.0
        
        # 2. SEO Difficulty (40%) - Lower difficulty is better
        # If difficulty is None or 0, we can't really judge, but we use a baseline if None
        seo_val = data.get("seo_difficulty")
        seo_difficulty = float(seo_val) if seo_val is not None else 50.0 # default to medium if unknown
        score += ((100.0 - seo_difficulty) / 100.0) * 40.0
        
        # 3. Monetization (30%)
        # Cap affiliate offers at 10 for max score
        affiliate_val = data.get("affiliate_offer_count")
        affiliate_count = int(affiliate_val) if affiliate_val is not None else 0
        monetization_score = min(affiliate_count, 10) / 10.0
        score += monetization_score * 30.0
        
        return round(score, 2)
