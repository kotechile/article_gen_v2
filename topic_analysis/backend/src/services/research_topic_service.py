"""
Research Topic service for dataflow persistence
This service handles all research topic operations using Supabase Client/SDK
"""

import logging
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from ..models.research_topic import (
    ResearchTopic, ResearchTopicCreate, ResearchTopicUpdate, 
    ResearchTopicResponse, ResearchTopicListResponse, ResearchTopicComplete,
    ResearchTopicStats, ResearchTopicStatus
)
from .supabase_service import get_supabase_service

logger = logging.getLogger(__name__)

class ResearchTopicService:
    """Service for managing research topics"""
    
    def __init__(self):
        self.supabase = get_supabase_service()
        self.table_name = "research_topics"
    
    async def create(self, topic_data: ResearchTopicCreate, user_id: UUID) -> Optional[ResearchTopicResponse]:
        """Create a new research topic"""
        try:
            # Check if user already has a topic with the same title
            existing = await self.supabase.exists(
                table=self.table_name,
                filters={"title": topic_data.title},
                user_id=user_id
            )
            
            if existing:
                raise ValueError("A research topic with this title already exists")
            
            # Prepare data for insertion
            data = {
                "title": topic_data.title,
                "description": topic_data.description,
                "status": topic_data.status.value,
                "version": 1
            }
            
            result = await self.supabase.create(
                table=self.table_name,
                data=data,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return ResearchTopicResponse(**result)
            
        except Exception as e:
            logger.error(f"Error creating research topic: {e}")
            raise
    
    async def get_by_id(self, topic_id: UUID, user_id: UUID) -> Optional[ResearchTopicResponse]:
        """Get a research topic by ID"""
        try:
            result = await self.supabase.get_by_id(
                table=self.table_name,
                id=topic_id,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return ResearchTopicResponse(**result)
            
        except Exception as e:
            logger.error(f"Error getting research topic: {e}")
            raise
    
    async def get_all(self, user_id: UUID, status: Optional[ResearchTopicStatus] = None,
                     page: int = 1, size: int = 10, order_by: str = "created_at",
                     order_direction: str = "desc") -> ResearchTopicListResponse:
        """Get all research topics for a user with pagination"""
        try:
            filters = {}
            if status:
                filters["status"] = status.value
            
            offset = (page - 1) * size
            
            topics = await self.supabase.get_by_filters(
                table=self.table_name,
                filters=filters,
                user_id=user_id,
                order_by={order_by: order_direction},
                limit=size,
                offset=offset
            )
            
            total = await self.supabase.count(
                table=self.table_name,
                filters=filters,
                user_id=user_id
            )
            
            topic_responses = [ResearchTopicResponse(**topic) for topic in topics]
            
            return ResearchTopicListResponse(
                items=topic_responses,
                total=total,
                page=page,
                size=size,
                has_next=offset + size < total,
                has_prev=page > 1
            )
            
        except Exception as e:
            logger.error(f"Error getting research topics: {e}")
            raise
    
    async def update(self, topic_id: UUID, topic_data: ResearchTopicUpdate, user_id: UUID) -> Optional[ResearchTopicResponse]:
        """Update a research topic"""
        try:
            # Get current topic to check version
            current_topic = await self.get_by_id(topic_id, user_id)
            if not current_topic:
                return None
            
            # Check version for optimistic concurrency control
            if current_topic.version != topic_data.version:
                raise ValueError("Topic has been modified by another user. Please refresh and try again.")
            
            # Prepare update data
            update_data = {}
            if topic_data.title is not None:
                update_data["title"] = topic_data.title
            if topic_data.description is not None:
                update_data["description"] = topic_data.description
            if topic_data.status is not None:
                update_data["status"] = topic_data.status.value
            
            # Increment version
            update_data["version"] = current_topic.version + 1
            update_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = await self.supabase.update(
                table=self.table_name,
                id=topic_id,
                data=update_data,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return ResearchTopicResponse(**result)
            
        except Exception as e:
            logger.error(f"Error updating research topic: {e}")
            raise
    
    async def delete(self, topic_id: UUID, user_id: UUID) -> bool:
        """Delete a research topic"""
        try:
            # Check if topic exists
            topic = await self.get_by_id(topic_id, user_id)
            if not topic:
                return False
            
            # Delete the topic (cascade will handle related data)
            return await self.supabase.delete(
                table=self.table_name,
                id=topic_id,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error deleting research topic: {e}")
            raise
    
    async def search(self, query: str, user_id: UUID, page: int = 1, size: int = 10) -> ResearchTopicListResponse:
        """Search research topics by title or description"""
        try:
            search_fields = ["title", "description"]
            
            topics = await self.supabase.search(
                table=self.table_name,
                search_term=query,
                search_fields=search_fields,
                user_id=user_id,
                order_by={"created_at": "desc"},
                limit=size,
                offset=(page - 1) * size
            )
            
            # For search, we'll return all matching results without pagination
            # In a real implementation, you might want to implement proper search pagination
            topic_responses = [ResearchTopicResponse(**topic) for topic in topics]
            
            return ResearchTopicListResponse(
                items=topic_responses,
                total=len(topic_responses),
                page=page,
                size=size,
                has_next=False,  # Simplified for search
                has_prev=page > 1
            )
            
        except Exception as e:
            logger.error(f"Error searching research topics: {e}")
            raise
    
    async def get_stats(self, user_id: UUID) -> ResearchTopicStats:
        """Get research topic statistics for a user"""
        try:
            # Get total counts by status
            total_topics = await self.supabase.count(
                table=self.table_name,
                filters={},
                user_id=user_id
            )
            
            active_topics = await self.supabase.count(
                table=self.table_name,
                filters={"status": "active"},
                user_id=user_id
            )
            
            completed_topics = await self.supabase.count(
                table=self.table_name,
                filters={"status": "completed"},
                user_id=user_id
            )
            
            archived_topics = await self.supabase.count(
                table=self.table_name,
                filters={"status": "archived"},
                user_id=user_id
            )
            
            # Get related data counts
            subtopics_count = await self.supabase.count(
                table="topic_decompositions",
                filters={},
                user_id=user_id
            )
            
            analyses_count = await self.supabase.count(
                table="trend_analyses",
                filters={},
                user_id=user_id
            )
            
            content_ideas_count = await self.supabase.count(
                table="content_ideas",
                filters={},
                user_id=user_id
            )
            
            # Get last activity
            recent_topics = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={},
                user_id=user_id,
                order_by={"updated_at": "desc"},
                limit=1
            )
            
            last_activity = None
            if recent_topics:
                last_activity = datetime.fromisoformat(recent_topics[0]["updated_at"].replace('Z', '+00:00'))
            
            return ResearchTopicStats(
                total_topics=total_topics,
                active_topics=active_topics,
                completed_topics=completed_topics,
                archived_topics=archived_topics,
                total_subtopics=subtopics_count,
                total_analyses=analyses_count,
                total_content_ideas=content_ideas_count,
                last_activity=last_activity
            )
            
        except Exception as e:
            logger.error(f"Error getting research topic stats: {e}")
            raise
    
    async def get_complete_dataflow(self, topic_id: UUID, user_id: UUID) -> Optional[ResearchTopicComplete]:
        """Get complete dataflow for a research topic including all related data"""
        try:
            # Get the research topic
            topic = await self.get_by_id(topic_id, user_id)
            if not topic:
                return None
            
            # Get subtopics (topic decompositions)
            subtopics = await self.supabase.get_related_records(
                table="topic_decompositions",
                foreign_key="research_topic_id",
                foreign_id=topic_id,
                user_id=user_id,
                order_by={"created_at": "desc"}
            )
            
            # Get trend analyses for all subtopics
            trend_analyses = []
            for subtopic in subtopics:
                analyses = await self.supabase.get_related_records(
                    table="trend_analyses",
                    foreign_key="topic_decomposition_id",
                    foreign_id=UUID(subtopic["id"]),
                    user_id=user_id,
                    order_by={"created_at": "desc"}
                )
                trend_analyses.extend(analyses)
            
            # Get content ideas for all trend analyses
            content_ideas = []
            for analysis in trend_analyses:
                ideas = await self.supabase.get_related_records(
                    table="content_ideas",
                    foreign_key="trend_analysis_id",
                    foreign_id=UUID(analysis["id"]),
                    user_id=user_id,
                    order_by={"created_at": "desc"}
                )
                content_ideas.extend(ideas)
            
            # Convert to complete dataflow
            return ResearchTopicComplete(
                **topic.dict(),
                subtopics=subtopics,
                trend_analyses=trend_analyses,
                content_ideas=content_ideas
            )
            
        except Exception as e:
            logger.error(f"Error getting complete dataflow: {e}")
            raise
    
    async def archive(self, topic_id: UUID, user_id: UUID) -> Optional[ResearchTopicResponse]:
        """Archive a research topic"""
        try:
            topic = await self.get_by_id(topic_id, user_id)
            if not topic:
                return None
            
            update_data = ResearchTopicUpdate(
                status=ResearchTopicStatus.ARCHIVED,
                version=topic.version
            )
            
            return await self.update(topic_id, update_data, user_id)
            
        except Exception as e:
            logger.error(f"Error archiving research topic: {e}")
            raise
    
    async def restore(self, topic_id: UUID, user_id: UUID) -> Optional[ResearchTopicResponse]:
        """Restore an archived research topic"""
        try:
            topic = await self.get_by_id(topic_id, user_id)
            if not topic or topic.status != ResearchTopicStatus.ARCHIVED:
                return None
            
            update_data = ResearchTopicUpdate(
                status=ResearchTopicStatus.ACTIVE,
                version=topic.version
            )
            
            return await self.update(topic_id, update_data, user_id)
            
        except Exception as e:
            logger.error(f"Error restoring research topic: {e}")
            raise
    
    async def get_enriched_subtopics(self, topic_id: UUID, user_id: UUID) -> List[Dict[str, Any]]:
        """
        Get enriched subtopics with trend data, viability scores, and SEO ease
        
        Args:
            topic_id: Research topic UUID
            user_id: User UUID
            
        Returns:
            List of enriched subtopic dictionaries
        """
        try:
            # Query the subtopics table directly
            subtopics = await self.supabase.get_related_records(
                table="subtopics",
                foreign_key="research_topic_id",
                foreign_id=topic_id,
                user_id=user_id,
                order_by={"viability_score": "desc"} # Order by viability
            )
            
            if not subtopics:
                logger.info(f"No subtopics found for topic {topic_id}")
                return []
            
            logger.info(f"Retrieved {len(subtopics)} subtopics from subtopics table")
            
            enriched_subtopics = []
            
            for subtopic in subtopics:
                # Map subtopic row to expected format
                # Ensure fields are present and correctly typed
                
                # Calculate SEO ease if not present (though it might not be stored as string)
                # If we want to use the helper helper, we need to adapt it or re-implement logic
                # But assume 'seo_difficulty' is stored, we can map it to 'seo_ease' label if needed
                
                seo_difficulty = subtopic.get('seo_difficulty', 0)
                seo_ease = 'Moderate'
                if seo_difficulty:
                     if seo_difficulty < 30: seo_ease = 'Easy'
                     elif seo_difficulty < 70: seo_ease = 'Moderate'
                     else: seo_ease = 'Hard'
                
                enriched_subtopics.append({
                    'id': subtopic.get('id'),
                    'name': subtopic.get('name'),
                    'keywords': subtopic.get('keywords', []),
                    'keyword_count': len(subtopic.get('keywords', []) or []),
                    'trend_direction': subtopic.get('trend_direction', 'neutral'),
                    'interest_over_time': subtopic.get('interest_over_time', []),
                    'search_volume': subtopic.get('search_volume', 0),
                    'cpc': subtopic.get('cpc', 0),
                    'seo_ease': seo_ease,
                    'viability_score': subtopic.get('viability_score', 0)
                })
            
            logger.info(f"Returning {len(enriched_subtopics)} enriched subtopics")
            return enriched_subtopics
            
        except Exception as e:
            logger.error(f"Error getting enriched subtopics: {e}")
            raise
