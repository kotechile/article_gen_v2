"""
Trend Analysis service for dataflow persistence
This service handles all trend analysis operations using Supabase Client/SDK
"""

import logging
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from ..models.trend_analysis import (
    TrendAnalysis, TrendAnalysisCreate, TrendAnalysisUpdate,
    TrendAnalysisResponse, TrendAnalysisListResponse, TrendAnalysisWithContentIdeas,
    TrendAnalysisStats, TrendAnalysisStatus, TrendData, TrendAnalysisResult
)
from .supabase_service import get_supabase_service

logger = logging.getLogger(__name__)

class TrendAnalysisService:
    """Service for managing trend analyses"""
    
    def __init__(self):
        self.supabase = get_supabase_service()
        self.table_name = "trend_analyses"
    
    async def create(self, analysis_data: TrendAnalysisCreate, user_id: UUID) -> Optional[TrendAnalysisResponse]:
        """Create a new trend analysis"""
        try:
            # Verify topic decomposition exists and belongs to user
            decomposition = await self.supabase.get_by_id(
                table="topic_decompositions",
                id=analysis_data.topic_decomposition_id,
                user_id=user_id
            )
            
            if not decomposition:
                raise ValueError("Topic decomposition not found")
            
            # Verify subtopic exists in the decomposition
            subtopics = decomposition.get("subtopics", [])
            subtopic_names = [subtopic.get("name", "") for subtopic in subtopics]
            
            if analysis_data.subtopic_name not in subtopic_names:
                raise ValueError("Subtopic not found in the topic decomposition")
            
            # Check if analysis already exists for this subtopic
            existing = await self.supabase.exists(
                table=self.table_name,
                filters={
                    "topic_decomposition_id": str(analysis_data.topic_decomposition_id),
                    "subtopic_name": analysis_data.subtopic_name
                },
                user_id=user_id
            )
            
            if existing:
                raise ValueError("Trend analysis already exists for this subtopic")
            
            # Prepare data for insertion
            data = {
                "topic_decomposition_id": str(analysis_data.topic_decomposition_id),
                "analysis_name": analysis_data.analysis_name,
                "description": analysis_data.description,
                "subtopic_name": analysis_data.subtopic_name,
                "keywords": analysis_data.keywords,
                "timeframe": analysis_data.timeframe,
                "geo": analysis_data.geo,
                "status": analysis_data.status.value
            }
            
            result = await self.supabase.create(
                table=self.table_name,
                data=data,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return TrendAnalysisResponse(**result)
            
        except Exception as e:
            logger.error(f"Error creating trend analysis: {e}")
            raise
    
    async def get_by_id(self, analysis_id: UUID, user_id: UUID) -> Optional[TrendAnalysisResponse]:
        """Get a trend analysis by ID"""
        try:
            result = await self.supabase.get_by_id(
                table=self.table_name,
                id=analysis_id,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return TrendAnalysisResponse(**result)
            
        except Exception as e:
            logger.error(f"Error getting trend analysis: {e}")
            raise
    
    async def get_all(self, user_id: UUID, topic_decomposition_id: Optional[UUID] = None,
                     subtopic_name: Optional[str] = None, status: Optional[TrendAnalysisStatus] = None,
                     page: int = 1, size: int = 10, order_by: str = "created_at",
                     order_direction: str = "desc") -> TrendAnalysisListResponse:
        """Get all trend analyses for a user with pagination"""
        try:
            filters = {}
            if topic_decomposition_id:
                filters["topic_decomposition_id"] = str(topic_decomposition_id)
            if subtopic_name:
                filters["subtopic_name"] = subtopic_name
            if status:
                filters["status"] = status.value
            
            offset = (page - 1) * size
            
            analyses = await self.supabase.get_by_filters(
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
            
            analysis_responses = [TrendAnalysisResponse(**analysis) for analysis in analyses]
            
            return TrendAnalysisListResponse(
                items=analysis_responses,
                total=total,
                page=page,
                size=size,
                has_next=offset + size < total,
                has_prev=page > 1
            )
            
        except Exception as e:
            logger.error(f"Error getting trend analyses: {e}")
            raise
    
    async def update(self, analysis_id: UUID, analysis_data: TrendAnalysisUpdate, 
                    user_id: UUID) -> Optional[TrendAnalysisResponse]:
        """Update a trend analysis"""
        try:
            # Get current analysis
            current_analysis = await self.get_by_id(analysis_id, user_id)
            if not current_analysis:
                return None
            
            # Prepare update data
            update_data = {}
            if analysis_data.analysis_name is not None:
                update_data["analysis_name"] = analysis_data.analysis_name
            if analysis_data.description is not None:
                update_data["description"] = analysis_data.description
            if analysis_data.subtopic_name is not None:
                update_data["subtopic_name"] = analysis_data.subtopic_name
            if analysis_data.keywords is not None:
                update_data["keywords"] = analysis_data.keywords
            if analysis_data.timeframe is not None:
                update_data["timeframe"] = analysis_data.timeframe
            if analysis_data.geo is not None:
                update_data["geo"] = analysis_data.geo
            if analysis_data.status is not None:
                update_data["status"] = analysis_data.status.value
            if analysis_data.trend_data is not None:
                update_data["trend_data"] = analysis_data.trend_data
            
            update_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = await self.supabase.update(
                table=self.table_name,
                id=analysis_id,
                data=update_data,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return TrendAnalysisResponse(**result)
            
        except Exception as e:
            logger.error(f"Error updating trend analysis: {e}")
            raise
    
    async def delete(self, analysis_id: UUID, user_id: UUID) -> bool:
        """Delete a trend analysis"""
        try:
            # Check if analysis exists
            analysis = await self.get_by_id(analysis_id, user_id)
            if not analysis:
                return False
            
            # Delete the analysis (cascade will handle related data)
            return await self.supabase.delete(
                table=self.table_name,
                id=analysis_id,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error deleting trend analysis: {e}")
            raise
    
    async def get_with_content_ideas(self, analysis_id: UUID, user_id: UUID) -> Optional[TrendAnalysisWithContentIdeas]:
        """Get trend analysis with associated content ideas"""
        try:
            analysis = await self.get_by_id(analysis_id, user_id)
            if not analysis:
                return None
            
            # Get content ideas for this analysis
            content_ideas = await self.supabase.get_related_records(
                table="content_ideas",
                foreign_key="trend_analysis_id",
                foreign_id=analysis_id,
                user_id=user_id,
                order_by={"created_at": "desc"}
            )
            
            return TrendAnalysisWithContentIdeas(
                **analysis.dict(),
                content_ideas=content_ideas
            )
            
        except Exception as e:
            logger.error(f"Error getting trend analysis with content ideas: {e}")
            raise
    
    async def update_status(self, analysis_id: UUID, status: TrendAnalysisStatus, 
                           user_id: UUID, error_message: Optional[str] = None) -> Optional[TrendAnalysisResponse]:
        """Update trend analysis status"""
        try:
            update_data = {
                "status": status.value,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if error_message:
                update_data["error_message"] = error_message
            
            result = await self.supabase.update(
                table=self.table_name,
                id=analysis_id,
                data=update_data,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return TrendAnalysisResponse(**result)
            
        except Exception as e:
            logger.error(f"Error updating trend analysis status: {e}")
            raise
    
    async def update_trend_data(self, analysis_id: UUID, trend_data: TrendData, 
                               user_id: UUID) -> Optional[TrendAnalysisResponse]:
        """Update trend analysis with trend data"""
        try:
            update_data = {
                "trend_data": trend_data.dict(),
                "status": TrendAnalysisStatus.COMPLETED.value,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = await self.supabase.update(
                table=self.table_name,
                id=analysis_id,
                data=update_data,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return TrendAnalysisResponse(**result)
            
        except Exception as e:
            logger.error(f"Error updating trend data: {e}")
            raise
    
    async def get_stats(self, user_id: UUID) -> TrendAnalysisStats:
        """Get trend analysis statistics for a user"""
        try:
            # Get total analyses
            total_analyses = await self.supabase.count(
                table=self.table_name,
                filters={},
                user_id=user_id
            )
            
            # Get counts by status
            completed_analyses = await self.supabase.count(
                table=self.table_name,
                filters={"status": "completed"},
                user_id=user_id
            )
            
            failed_analyses = await self.supabase.count(
                table=self.table_name,
                filters={"status": "failed"},
                user_id=user_id
            )
            
            pending_analyses = await self.supabase.count(
                table=self.table_name,
                filters={"status": "pending"},
                user_id=user_id
            )
            
            in_progress_analyses = await self.supabase.count(
                table=self.table_name,
                filters={"status": "in_progress"},
                user_id=user_id
            )
            
            # Get most analyzed subtopics
            all_analyses = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={},
                user_id=user_id
            )
            
            subtopic_counts = {}
            for analysis in all_analyses:
                subtopic_name = analysis.get("subtopic_name", "")
                if subtopic_name:
                    subtopic_counts[subtopic_name] = subtopic_counts.get(subtopic_name, 0) + 1
            
            most_analyzed_subtopics = [
                {"subtopic_name": name, "count": count}
                for name, count in sorted(subtopic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
            
            # Calculate average completion time (simplified)
            completed_analyses_data = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={"status": "completed"},
                user_id=user_id
            )
            
            average_completion_time = None
            if completed_analyses_data:
                # This is a simplified calculation
                # In a real implementation, you'd calculate the actual time difference
                average_completion_time = 30.0  # Placeholder
            
            # Get last analysis
            recent_analyses = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={},
                user_id=user_id,
                order_by={"created_at": "desc"},
                limit=1
            )
            
            last_analysis = None
            if recent_analyses:
                last_analysis = datetime.fromisoformat(
                    recent_analyses[0]["created_at"].replace('Z', '+00:00')
                )
            
            return TrendAnalysisStats(
                total_analyses=total_analyses,
                completed_analyses=completed_analyses,
                failed_analyses=failed_analyses,
                pending_analyses=pending_analyses,
                in_progress_analyses=in_progress_analyses,
                average_completion_time=average_completion_time,
                most_analyzed_subtopics=most_analyzed_subtopics,
                last_analysis=last_analysis
            )
            
        except Exception as e:
            logger.error(f"Error getting trend analysis stats: {e}")
            raise
    
    async def search(self, query: str, user_id: UUID, page: int = 1, size: int = 10) -> TrendAnalysisListResponse:
        """Search trend analyses by name, description, or subtopic"""
        try:
            search_fields = ["analysis_name", "description", "subtopic_name"]
            
            analyses = await self.supabase.search(
                table=self.table_name,
                search_term=query,
                search_fields=search_fields,
                user_id=user_id,
                order_by={"created_at": "desc"},
                limit=size,
                offset=(page - 1) * size
            )
            
            analysis_responses = [TrendAnalysisResponse(**analysis) for analysis in analyses]
            
            return TrendAnalysisListResponse(
                items=analysis_responses,
                total=len(analysis_responses),
                page=page,
                size=size,
                has_next=False,  # Simplified for search
                has_prev=page > 1
            )
            
        except Exception as e:
            logger.error(f"Error searching trend analyses: {e}")
            raise
    
    async def get_by_subtopic(self, subtopic_name: str, user_id: UUID) -> List[TrendAnalysisResponse]:
        """Get all trend analyses for a specific subtopic"""
        try:
            analyses = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={"subtopic_name": subtopic_name},
                user_id=user_id,
                order_by={"created_at": "desc"}
            )
            
            return [TrendAnalysisResponse(**analysis) for analysis in analyses]
            
        except Exception as e:
            logger.error(f"Error getting trend analyses by subtopic: {e}")
            raise
    
    async def get_by_research_topic(self, research_topic_id: UUID, user_id: UUID) -> List[TrendAnalysisResponse]:
        """Get all trend analyses for a research topic"""
        try:
            # Get all decompositions for the research topic
            decompositions = await self.supabase.get_by_filters(
                table="topic_decompositions",
                filters={"research_topic_id": str(research_topic_id)},
                user_id=user_id
            )
            
            all_analyses = []
            for decomposition in decompositions:
                analyses = await self.supabase.get_related_records(
                    table=self.table_name,
                    foreign_key="topic_decomposition_id",
                    foreign_id=UUID(decomposition["id"]),
                    user_id=user_id,
                    order_by={"created_at": "desc"}
                )
                all_analyses.extend(analyses)
            
            return [TrendAnalysisResponse(**analysis) for analysis in all_analyses]
            
        except Exception as e:
            logger.error(f"Error getting trend analyses by research topic: {e}")
            raise
