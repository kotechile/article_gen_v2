
import logging
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
from datetime import datetime
import json

from .supabase_service import get_supabase_service

logger = logging.getLogger(__name__)

class KeywordsService:
    """Service for managing keywords persistence"""
    
    def __init__(self):
        self.supabase = get_supabase_service()
        self.table_name = "keywords"
    
    async def create(self, research_topic_id: UUID, user_id: UUID, keyword_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new keyword"""
        try:
            data = keyword_data.copy()
            data["research_topic_id"] = str(research_topic_id)
            data["user_id"] = str(user_id)
            
            # Ensure timestamps
            if "created_at" not in data:
                data["created_at"] = datetime.utcnow().isoformat()
            if "updated_at" not in data:
                data["updated_at"] = datetime.utcnow().isoformat()
            
            result = await self.supabase.execute_query(
                table=self.table_name,
                operation="insert",
                data=data
            )
            
            return result["data"][0] if result.get("data") else None
            
        except Exception as e:
            logger.error(f"Error creating keyword: {e}")
            raise
    
    async def create_batch(self, keywords_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create multiple keywords in batch"""
        if not keywords_data:
            return []
            
        try:
            # Prepare batch data
            batch = []
            for item in keywords_data:
                item_copy = item.copy()
                if "created_at" not in item_copy:
                    item_copy["created_at"] = datetime.utcnow().isoformat()
                if "updated_at" not in item_copy:
                    item_copy["updated_at"] = datetime.utcnow().isoformat()
                batch.append(item_copy)
            
            result = await self.supabase.execute_query(
                table=self.table_name,
                operation="upsert",
                data=batch,
                on_conflict="research_topic_id, keyword"
            )
            
            return result.get("data", [])
            
        except Exception as e:
            logger.error(f"Error creating batch keywords: {e}")
            # Try falling back to individual inserts if batch fails? 
            # Or just raise. Supabase client usually handles batch inserts fine.
            raise

    async def get_by_topic(self, topic_id: UUID, user_id: UUID) -> List[Dict[str, Any]]:
        """Get all keywords for a topic"""
        try:
            return await self.supabase.get_by_filters(
                table=self.table_name,
                filters={"research_topic_id": str(topic_id)},
                user_id=user_id,
                order_by={"profitability_score": "desc"}
            )
        except Exception as e:
            logger.error(f"Error getting keywords for topic {topic_id}: {e}")
            raise

    async def get_by_subtopic(self, subtopic_id: UUID, user_id: UUID) -> List[Dict[str, Any]]:
        """Get keywords for a specific subtopic"""
        try:
            return await self.supabase.get_by_filters(
                table=self.table_name,
                filters={"subtopic_id": str(subtopic_id)},
                user_id=user_id,
                order_by={"profitability_score": "desc"}
            )
        except Exception as e:
            logger.error(f"Error getting keywords for subtopic {subtopic_id}: {e}")
            raise
