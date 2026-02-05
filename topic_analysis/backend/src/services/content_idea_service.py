"""
Content Idea service for dataflow persistence
This service handles all content idea operations using Supabase Client/SDK
"""

import logging
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from ..models.content_idea import (
    ContentIdea, ContentIdeaCreate, ContentIdeaUpdate,
    ContentIdeaResponse, ContentIdeaListResponse, ContentIdeaWithTrendAnalysis,
    ContentIdeaStats, ContentIdeaFilter, ContentIdeaSearch,
    ContentType, IdeaType, ContentStatus
)
from .supabase_service import get_supabase_service

logger = logging.getLogger(__name__)

class ContentIdeaService:
    """Service for managing content ideas"""
    
    def __init__(self):
        self.supabase = get_supabase_service()
        self.table_name = "content_ideas"
    
    async def create(self, idea_data: ContentIdeaCreate, user_id: UUID) -> Optional[ContentIdeaResponse]:
        """Create a new content idea"""
        try:
            # Verify trend analysis exists and belongs to user
            trend_analysis = await self.supabase.get_by_id(
                table="trend_analyses",
                id=idea_data.trend_analysis_id,
                user_id=user_id
            )
            
            if not trend_analysis:
                raise ValueError("Trend analysis not found")
            
            # Verify research topic exists and belongs to user
            research_topic = await self.supabase.get_by_id(
                table="research_topics",
                id=idea_data.research_topic_id,
                user_id=user_id
            )
            
            if not research_topic:
                raise ValueError("Research topic not found")
            
            # Prepare data for insertion
            data = {
                "trend_analysis_id": str(idea_data.trend_analysis_id),
                "research_topic_id": str(idea_data.research_topic_id),
                "title": idea_data.title,
                "description": idea_data.description,
                "content_type": idea_data.content_type.value,
                "idea_type": idea_data.idea_type.value,
                "primary_keyword": idea_data.primary_keyword,
                "secondary_keywords": idea_data.secondary_keywords,
                "target_audience": idea_data.target_audience,
                "key_points": idea_data.key_points,
                "tags": idea_data.tags,
                "estimated_read_time": idea_data.estimated_read_time,
                "difficulty_level": idea_data.difficulty_level,
                "status": idea_data.status.value
            }
            
            result = await self.supabase.create(
                table=self.table_name,
                data=data,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return ContentIdeaResponse(**result)
            
        except Exception as e:
            logger.error(f"Error creating content idea: {e}")
            raise
    
    async def get_by_id(self, idea_id: UUID, user_id: UUID) -> Optional[ContentIdeaResponse]:
        """Get a content idea by ID"""
        try:
            result = await self.supabase.get_by_id(
                table=self.table_name,
                id=idea_id,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return ContentIdeaResponse(**result)
            
        except Exception as e:
            logger.error(f"Error getting content idea: {e}")
            raise
    
    async def get_all(self, user_id: UUID, filters: Optional[ContentIdeaFilter] = None,
                     page: int = 1, size: int = 10, order_by: str = "created_at",
                     order_direction: str = "desc") -> ContentIdeaListResponse:
        """Get all content ideas for a user with pagination and filtering"""
        try:
            filter_dict = {}
            if filters:
                if filters.content_type:
                    filter_dict["content_type"] = filters.content_type.value
                if filters.idea_type:
                    filter_dict["idea_type"] = filters.idea_type.value
                if filters.status:
                    filter_dict["status"] = filters.status.value
                if filters.primary_keyword:
                    filter_dict["primary_keyword"] = filters.primary_keyword
                if filters.difficulty_level:
                    filter_dict["difficulty_level"] = filters.difficulty_level
                if filters.research_topic_id:
                    filter_dict["research_topic_id"] = str(filters.research_topic_id)
                if filters.trend_analysis_id:
                    filter_dict["trend_analysis_id"] = str(filters.trend_analysis_id)
            
            offset = (page - 1) * size
            
            ideas = await self.supabase.get_by_filters(
                table=self.table_name,
                filters=filter_dict,
                user_id=user_id,
                order_by={order_by: order_direction},
                limit=size,
                offset=offset
            )
            
            # Apply additional filters that can't be done at database level
            if filters and filters.tags:
                ideas = [idea for idea in ideas if any(tag in idea.get("tags", []) for tag in filters.tags)]
            
            if filters and filters.created_after:
                ideas = [idea for idea in ideas if 
                        datetime.fromisoformat(idea["created_at"].replace('Z', '+00:00')) >= filters.created_after]
            
            if filters and filters.created_before:
                ideas = [idea for idea in ideas if 
                        datetime.fromisoformat(idea["created_at"].replace('Z', '+00:00')) <= filters.created_before]
            
            total = await self.supabase.count(
                table=self.table_name,
                filters=filter_dict,
                user_id=user_id
            )
            
            idea_responses = [ContentIdeaResponse(**idea) for idea in ideas]
            
            return ContentIdeaListResponse(
                items=idea_responses,
                total=total,
                page=page,
                size=size,
                has_next=offset + size < total,
                has_prev=page > 1
            )
            
        except Exception as e:
            logger.error(f"Error getting content ideas: {e}")
            raise
    
    async def update(self, idea_id: UUID, idea_data: ContentIdeaUpdate, 
                    user_id: UUID) -> Optional[ContentIdeaResponse]:
        """Update a content idea"""
        try:
            # Get current idea
            current_idea = await self.get_by_id(idea_id, user_id)
            if not current_idea:
                return None
            
            # Prepare update data
            update_data = {}
            if idea_data.title is not None:
                update_data["title"] = idea_data.title
            if idea_data.description is not None:
                update_data["description"] = idea_data.description
            if idea_data.content_type is not None:
                update_data["content_type"] = idea_data.content_type.value
            if idea_data.idea_type is not None:
                update_data["idea_type"] = idea_data.idea_type.value
            if idea_data.primary_keyword is not None:
                update_data["primary_keyword"] = idea_data.primary_keyword
            if idea_data.secondary_keywords is not None:
                update_data["secondary_keywords"] = idea_data.secondary_keywords
            if idea_data.target_audience is not None:
                update_data["target_audience"] = idea_data.target_audience
            if idea_data.key_points is not None:
                update_data["key_points"] = idea_data.key_points
            if idea_data.tags is not None:
                update_data["tags"] = idea_data.tags
            if idea_data.estimated_read_time is not None:
                update_data["estimated_read_time"] = idea_data.estimated_read_time
            if idea_data.difficulty_level is not None:
                update_data["difficulty_level"] = idea_data.difficulty_level
            if idea_data.status is not None:
                update_data["status"] = idea_data.status.value
            
            update_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = await self.supabase.update(
                table=self.table_name,
                id=idea_id,
                data=update_data,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return ContentIdeaResponse(**result)
            
        except Exception as e:
            logger.error(f"Error updating content idea: {e}")
            raise
    
    async def delete(self, idea_id: UUID, user_id: UUID) -> bool:
        """Delete a content idea"""
        try:
            # Check if idea exists
            idea = await self.get_by_id(idea_id, user_id)
            if not idea:
                return False
            
            # Delete the idea
            return await self.supabase.delete(
                table=self.table_name,
                id=idea_id,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error deleting content idea: {e}")
            raise
    
    async def get_with_trend_analysis(self, idea_id: UUID, user_id: UUID) -> Optional[ContentIdeaWithTrendAnalysis]:
        """Get content idea with associated trend analysis"""
        try:
            idea = await self.get_by_id(idea_id, user_id)
            if not idea:
                return None
            
            # Get trend analysis for this idea
            trend_analysis = await self.supabase.get_by_id(
                table="trend_analyses",
                id=idea.trend_analysis_id,
                user_id=user_id
            )
            
            return ContentIdeaWithTrendAnalysis(
                **idea.dict(),
                trend_analysis=trend_analysis
            )
            
        except Exception as e:
            logger.error(f"Error getting content idea with trend analysis: {e}")
            raise
    
    async def search(self, search_data: ContentIdeaSearch, user_id: UUID) -> ContentIdeaListResponse:
        """Search content ideas with advanced filtering"""
        try:
            # Build search filters
            filter_dict = {}
            if search_data.filters:
                if search_data.filters.content_type:
                    filter_dict["content_type"] = search_data.filters.content_type.value
                if search_data.filters.idea_type:
                    filter_dict["idea_type"] = search_data.filters.idea_type.value
                if search_data.filters.status:
                    filter_dict["status"] = search_data.filters.status.value
                if search_data.filters.research_topic_id:
                    filter_dict["research_topic_id"] = str(search_data.filters.research_topic_id)
                if search_data.filters.trend_analysis_id:
                    filter_dict["trend_analysis_id"] = str(search_data.filters.trend_analysis_id)
            
            # Perform search
            ideas = await self.supabase.search(
                table=self.table_name,
                search_term=search_data.query,
                search_fields=search_data.search_fields,
                user_id=user_id,
                filters=filter_dict,
                order_by={search_data.sort_by: search_data.sort_order}
            )
            
            # Apply additional filters that can't be done at database level
            if search_data.filters:
                if search_data.filters.tags:
                    ideas = [idea for idea in ideas if any(tag in idea.get("tags", []) for tag in search_data.filters.tags)]
                
                if search_data.filters.created_after:
                    ideas = [idea for idea in ideas if 
                            datetime.fromisoformat(idea["created_at"].replace('Z', '+00:00')) >= search_data.filters.created_after]
                
                if search_data.filters.created_before:
                    ideas = [idea for idea in ideas if 
                            datetime.fromisoformat(idea["created_at"].replace('Z', '+00:00')) <= search_data.filters.created_before]
            
            # Apply pagination
            page = 1  # Simplified pagination for search
            size = 50  # Default search result size
            start_idx = (page - 1) * size
            end_idx = start_idx + size
            paginated_ideas = ideas[start_idx:end_idx]
            
            idea_responses = [ContentIdeaResponse(**idea) for idea in paginated_ideas]
            
            return ContentIdeaListResponse(
                items=idea_responses,
                total=len(ideas),
                page=page,
                size=size,
                has_next=end_idx < len(ideas),
                has_prev=page > 1
            )
            
        except Exception as e:
            logger.error(f"Error searching content ideas: {e}")
            raise
    
    async def get_stats(self, user_id: UUID) -> ContentIdeaStats:
        """Get content idea statistics for a user"""
        try:
            # Get total ideas
            total_ideas = await self.supabase.count(
                table=self.table_name,
                filters={},
                user_id=user_id
            )
            
            # Get counts by status
            draft_ideas = await self.supabase.count(
                table=self.table_name,
                filters={"status": "draft"},
                user_id=user_id
            )
            
            published_ideas = await self.supabase.count(
                table=self.table_name,
                filters={"status": "published"},
                user_id=user_id
            )
            
            # Get all ideas for detailed analysis
            all_ideas = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={},
                user_id=user_id
            )
            
            # Analyze by content type
            ideas_by_type = {}
            ideas_by_idea_type = {}
            all_keywords = []
            all_tags = []
            read_times = []
            
            for idea in all_ideas:
                # Count by content type
                content_type = idea.get("content_type", "other")
                ideas_by_type[content_type] = ideas_by_type.get(content_type, 0) + 1
                
                # Count by idea type
                idea_type = idea.get("idea_type", "other")
                ideas_by_idea_type[idea_type] = ideas_by_idea_type.get(idea_type, 0) + 1
                
                # Collect keywords
                primary_keyword = idea.get("primary_keyword", "")
                if primary_keyword:
                    all_keywords.append(primary_keyword)
                
                secondary_keywords = idea.get("secondary_keywords", [])
                all_keywords.extend(secondary_keywords)
                
                # Collect tags
                tags = idea.get("tags", [])
                all_tags.extend(tags)
                
                # Collect read times
                read_time = idea.get("estimated_read_time")
                if read_time:
                    read_times.append(read_time)
            
            # Calculate average read time
            average_read_time = sum(read_times) / len(read_times) if read_times else None
            
            # Get most used keywords
            keyword_counts = {}
            for keyword in all_keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            most_used_keywords = [
                {"keyword": keyword, "count": count}
                for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
            
            # Get most used tags
            tag_counts = {}
            for tag in all_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            most_used_tags = [
                {"tag": tag, "count": count}
                for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
            
            # Get last idea created
            recent_ideas = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={},
                user_id=user_id,
                order_by={"created_at": "desc"},
                limit=1
            )
            
            last_idea_created = None
            if recent_ideas:
                last_idea_created = datetime.fromisoformat(
                    recent_ideas[0]["created_at"].replace('Z', '+00:00')
                )
            
            return ContentIdeaStats(
                total_ideas=total_ideas,
                draft_ideas=draft_ideas,
                published_ideas=published_ideas,
                ideas_by_type=ideas_by_type,
                ideas_by_idea_type=ideas_by_idea_type,
                average_read_time=average_read_time,
                most_used_keywords=most_used_keywords,
                most_used_tags=most_used_tags,
                last_idea_created=last_idea_created
            )
            
        except Exception as e:
            logger.error(f"Error getting content idea stats: {e}")
            raise
    
    async def get_by_research_topic(self, research_topic_id: UUID, user_id: UUID) -> List[ContentIdeaResponse]:
        """Get all content ideas for a research topic"""
        try:
            ideas = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={"research_topic_id": str(research_topic_id)},
                user_id=user_id,
                order_by={"created_at": "desc"}
            )
            
            return [ContentIdeaResponse(**idea) for idea in ideas]
            
        except Exception as e:
            logger.error(f"Error getting content ideas by research topic: {e}")
            raise
    
    async def get_by_trend_analysis(self, trend_analysis_id: UUID, user_id: UUID) -> List[ContentIdeaResponse]:
        """Get all content ideas for a trend analysis"""
        try:
            ideas = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={"trend_analysis_id": str(trend_analysis_id)},
                user_id=user_id,
                order_by={"created_at": "desc"}
            )
            
            return [ContentIdeaResponse(**idea) for idea in ideas]
            
        except Exception as e:
            logger.error(f"Error getting content ideas by trend analysis: {e}")
            raise
    
    async def bulk_create(self, ideas_data: List[ContentIdeaCreate], user_id: UUID) -> List[ContentIdeaResponse]:
        """Create multiple content ideas at once"""
        try:
            # Prepare data for bulk insertion
            data_list = []
            for idea_data in ideas_data:
                data = {
                    "trend_analysis_id": str(idea_data.trend_analysis_id),
                    "research_topic_id": str(idea_data.research_topic_id),
                    "title": idea_data.title,
                    "description": idea_data.description,
                    "content_type": idea_data.content_type.value,
                    "idea_type": idea_data.idea_type.value,
                    "primary_keyword": idea_data.primary_keyword,
                    "secondary_keywords": idea_data.secondary_keywords,
                    "target_audience": idea_data.target_audience,
                    "key_points": idea_data.key_points,
                    "tags": idea_data.tags,
                    "estimated_read_time": idea_data.estimated_read_time,
                    "difficulty_level": idea_data.difficulty_level,
                    "status": idea_data.status.value
                }
                data_list.append(data)
            
            result = await self.supabase.bulk_create(
                table=self.table_name,
                data_list=data_list,
                user_id=user_id
            )
            
            return [ContentIdeaResponse(**idea) for idea in result]
            
        except Exception as e:
            logger.error(f"Error bulk creating content ideas: {e}")
            raise
