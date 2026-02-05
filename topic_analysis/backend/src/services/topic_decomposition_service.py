"""
Topic Decomposition service for dataflow persistence
This service handles all topic decomposition operations using Supabase Client/SDK
"""

import logging
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from ..models.topic_decomposition import (
    TopicDecomposition, TopicDecompositionCreate, TopicDecompositionUpdate,
    TopicDecompositionResponse, TopicDecompositionListResponse, 
    TopicDecompositionWithAnalyses, TopicDecompositionStats, SubtopicAnalysis
)
from .supabase_service import get_supabase_service

logger = logging.getLogger(__name__)

class TopicDecompositionService:
    """Service for managing topic decompositions"""
    
    def __init__(self):
        self.supabase = get_supabase_service()
        self.table_name = "topic_decompositions"
    
    async def create_subtopics(self, research_topic_id: UUID, decomposition_data: TopicDecompositionCreate, 
                              user_id: UUID) -> Optional[TopicDecompositionResponse]:
        """Create subtopics for a research topic"""
        try:
            # Verify research topic exists and belongs to user
            research_topic = await self.supabase.get_by_id(
                table="research_topics",
                id=research_topic_id,
                user_id=user_id
            )
            
            if not research_topic:
                raise ValueError("Research topic not found")
            
            # Check if decomposition already exists for this research topic
            existing = await self.supabase.exists(
                table=self.table_name,
                filters={"research_topic_id": str(research_topic_id)},
                user_id=user_id
            )
            
            if existing:
                raise ValueError("Topic decomposition already exists for this research topic")
            
            # Prepare data for insertion
            data = {
                "research_topic_id": str(research_topic_id),
                "search_query": decomposition_data.search_query,
                "subtopics": [subtopic.dict() for subtopic in decomposition_data.subtopics],
                "original_topic_included": decomposition_data.original_topic_included
            }
            
            result = await self.supabase.create(
                table=self.table_name,
                data=data,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return TopicDecompositionResponse(**result)
            
        except Exception as e:
            logger.error(f"Error creating topic decomposition: {e}")
            raise
    
    async def get_by_id(self, decomposition_id: UUID, user_id: UUID) -> Optional[TopicDecompositionResponse]:
        """Get a topic decomposition by ID"""
        try:
            result = await self.supabase.get_by_id(
                table=self.table_name,
                id=decomposition_id,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return TopicDecompositionResponse(**result)
            
        except Exception as e:
            logger.error(f"Error getting topic decomposition: {e}")
            raise
    
    async def get_by_research_topic(self, research_topic_id: UUID, user_id: UUID) -> Optional[TopicDecompositionResponse]:
        """Get topic decomposition for a research topic"""
        try:
            decompositions = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={"research_topic_id": str(research_topic_id)},
                user_id=user_id,
                limit=1
            )
            
            if not decompositions:
                return None
            
            return TopicDecompositionResponse(**decompositions[0])
            
        except Exception as e:
            logger.error(f"Error getting topic decomposition by research topic: {e}")
            raise
    
    async def get_all(self, user_id: UUID, research_topic_id: Optional[UUID] = None,
                     page: int = 1, size: int = 10, order_by: str = "created_at",
                     order_direction: str = "desc") -> TopicDecompositionListResponse:
        """Get all topic decompositions for a user with pagination"""
        try:
            filters = {}
            if research_topic_id:
                filters["research_topic_id"] = str(research_topic_id)
            
            offset = (page - 1) * size
            
            decompositions = await self.supabase.get_by_filters(
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
            
            decomposition_responses = [TopicDecompositionResponse(**decomp) for decomp in decompositions]
            
            return TopicDecompositionListResponse(
                items=decomposition_responses,
                total=total,
                page=page,
                size=size,
                has_next=offset + size < total,
                has_prev=page > 1
            )
            
        except Exception as e:
            logger.error(f"Error getting topic decompositions: {e}")
            raise
    
    async def update(self, decomposition_id: UUID, decomposition_data: TopicDecompositionUpdate, 
                    user_id: UUID) -> Optional[TopicDecompositionResponse]:
        """Update a topic decomposition"""
        try:
            # Get current decomposition
            current_decomposition = await self.get_by_id(decomposition_id, user_id)
            if not current_decomposition:
                return None
            
            # Prepare update data
            update_data = {}
            if decomposition_data.search_query is not None:
                update_data["search_query"] = decomposition_data.search_query
            if decomposition_data.subtopics is not None:
                update_data["subtopics"] = [subtopic.dict() for subtopic in decomposition_data.subtopics]
            if decomposition_data.original_topic_included is not None:
                update_data["original_topic_included"] = decomposition_data.original_topic_included
            
            update_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = await self.supabase.update(
                table=self.table_name,
                id=decomposition_id,
                data=update_data,
                user_id=user_id
            )
            
            if not result:
                return None
            
            return TopicDecompositionResponse(**result)
            
        except Exception as e:
            logger.error(f"Error updating topic decomposition: {e}")
            raise
    
    async def delete(self, decomposition_id: UUID, user_id: UUID) -> bool:
        """Delete a topic decomposition"""
        try:
            # Check if decomposition exists
            decomposition = await self.get_by_id(decomposition_id, user_id)
            if not decomposition:
                return False
            
            # Delete the decomposition (cascade will handle related data)
            return await self.supabase.delete(
                table=self.table_name,
                id=decomposition_id,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error deleting topic decomposition: {e}")
            raise
    
    async def get_with_analyses(self, decomposition_id: UUID, user_id: UUID) -> Optional[TopicDecompositionWithAnalyses]:
        """Get topic decomposition with associated trend analyses"""
        try:
            decomposition = await self.get_by_id(decomposition_id, user_id)
            if not decomposition:
                return None
            
            # Get trend analyses for this decomposition
            analyses = await self.supabase.get_related_records(
                table="trend_analyses",
                foreign_key="topic_decomposition_id",
                foreign_id=decomposition_id,
                user_id=user_id,
                order_by={"created_at": "desc"}
            )
            
            return TopicDecompositionWithAnalyses(
                **decomposition.dict(),
                trend_analyses=analyses
            )
            
        except Exception as e:
            logger.error(f"Error getting topic decomposition with analyses: {e}")
            raise
    
    async def get_stats(self, user_id: UUID) -> TopicDecompositionStats:
        """Get topic decomposition statistics for a user"""
        try:
            # Get total decompositions
            total_decompositions = await self.supabase.count(
                table=self.table_name,
                filters={},
                user_id=user_id
            )
            
            # Get all decompositions to calculate subtopic statistics
            all_decompositions = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={},
                user_id=user_id
            )
            
            total_subtopics = 0
            subtopic_name_counts = {}
            
            for decomposition in all_decompositions:
                subtopics = decomposition.get("subtopics", [])
                total_subtopics += len(subtopics)
                
                for subtopic in subtopics:
                    name = subtopic.get("name", "")
                    if name:
                        subtopic_name_counts[name] = subtopic_name_counts.get(name, 0) + 1
            
            # Calculate average subtopics per decomposition
            average_subtopics = total_subtopics / total_decompositions if total_decompositions > 0 else 0
            
            # Get most common subtopic names
            most_common_subtopics = [
                {"name": name, "count": count}
                for name, count in sorted(subtopic_name_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
            
            # Get last decomposition
            recent_decompositions = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={},
                user_id=user_id,
                order_by={"created_at": "desc"},
                limit=1
            )
            
            last_decomposition = None
            if recent_decompositions:
                last_decomposition = datetime.fromisoformat(
                    recent_decompositions[0]["created_at"].replace('Z', '+00:00')
                )
            
            return TopicDecompositionStats(
                total_decompositions=total_decompositions,
                total_subtopics=total_subtopics,
                average_subtopics_per_decomposition=average_subtopics,
                most_common_subtopic_names=most_common_subtopics,
                last_decomposition=last_decomposition
            )
            
        except Exception as e:
            logger.error(f"Error getting topic decomposition stats: {e}")
            raise
    
    async def analyze_subtopics(self, user_id: UUID) -> List[SubtopicAnalysis]:
        """Analyze subtopic patterns across all decompositions"""
        try:
            # Get all decompositions
            all_decompositions = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={},
                user_id=user_id
            )
            
            subtopic_analysis = {}
            
            for decomposition in all_decompositions:
                research_topic_id = decomposition.get("research_topic_id")
                subtopics = decomposition.get("subtopics", [])
                
                for subtopic in subtopics:
                    name = subtopic.get("name", "")
                    if not name:
                        continue
                    
                    if name not in subtopic_analysis:
                        subtopic_analysis[name] = {
                            "frequency": 0,
                            "research_topics": set(),
                            "trend_analyses_count": 0,
                            "content_ideas_count": 0
                        }
                    
                    subtopic_analysis[name]["frequency"] += 1
                    subtopic_analysis[name]["research_topics"].add(research_topic_id)
            
            # Get trend analyses and content ideas counts for each subtopic
            for subtopic_name in subtopic_analysis:
                # Count trend analyses for this subtopic
                analyses = await self.supabase.get_by_filters(
                    table="trend_analyses",
                    filters={"subtopic_name": subtopic_name},
                    user_id=user_id
                )
                subtopic_analysis[subtopic_name]["trend_analyses_count"] = len(analyses)
                
                # Count content ideas for this subtopic (through trend analyses)
                content_ideas_count = 0
                for analysis in analyses:
                    ideas = await self.supabase.get_related_records(
                        table="content_ideas",
                        foreign_key="trend_analysis_id",
                        foreign_id=UUID(analysis["id"]),
                        user_id=user_id
                    )
                    content_ideas_count += len(ideas)
                
                subtopic_analysis[subtopic_name]["content_ideas_count"] = content_ideas_count
            
            # Convert to response format
            analysis_results = []
            for name, data in subtopic_analysis.items():
                analysis_results.append(SubtopicAnalysis(
                    subtopic_name=name,
                    frequency=data["frequency"],
                    research_topics=list(data["research_topics"]),
                    trend_analyses_count=data["trend_analyses_count"],
                    content_ideas_count=data["content_ideas_count"]
                ))
            
            # Sort by frequency descending
            analysis_results.sort(key=lambda x: x.frequency, reverse=True)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing subtopics: {e}")
            raise
    
    async def search_subtopics(self, query: str, user_id: UUID, page: int = 1, size: int = 10) -> List[Dict[str, Any]]:
        """Search subtopics across all decompositions"""
        try:
            # Get all decompositions
            all_decompositions = await self.supabase.get_by_filters(
                table=self.table_name,
                filters={},
                user_id=user_id
            )
            
            matching_subtopics = []
            query_lower = query.lower()
            
            for decomposition in all_decompositions:
                subtopics = decomposition.get("subtopics", [])
                for subtopic in subtopics:
                    name = subtopic.get("name", "")
                    description = subtopic.get("description", "")
                    
                    if (query_lower in name.lower() or 
                        query_lower in description.lower()):
                        matching_subtopics.append({
                            "subtopic": subtopic,
                            "decomposition_id": decomposition["id"],
                            "research_topic_id": decomposition["research_topic_id"],
                            "search_query": decomposition["search_query"]
                        })
            
            # Apply pagination
            start_idx = (page - 1) * size
            end_idx = start_idx + size
            paginated_results = matching_subtopics[start_idx:end_idx]
            
            return paginated_results
            
        except Exception as e:
            logger.error(f"Error searching subtopics: {e}")
            raise
