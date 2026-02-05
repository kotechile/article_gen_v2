"""
WorkflowSessionService for enhanced research workflow management
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog
from sqlalchemy.orm import Session
from ..models.workflow_sessions import WorkflowSessions, WorkflowStep, WorkflowStatus
from ..core.redis import cache
from .topic_decomposition_service import TopicDecompositionService
from .affiliate_offer_service import AffiliateOfferService
from .trend_analysis_service import TrendAnalysisService

logger = structlog.get_logger()


class WorkflowSessionService:
    """Service for managing workflow sessions"""
    
    def __init__(self, db: Session):
        self.db = db
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    async def create_session(
        self, 
        user_id: str, 
        session_name: str, 
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new workflow session"""
        try:
            logger.info("Creating workflow session", user_id=user_id, session_name=session_name)
            
            session_id = str(uuid.uuid4())
            workflow_session = WorkflowSessions(
                id=session_id,
                user_id=user_id,
                session_name=session_name,
                description=description,
                current_step=WorkflowStep.UPLOAD_CSV,
                progress_percentage=0,
                status=WorkflowStatus.ACTIVE
            )
            
            self.db.add(workflow_session)
            self.db.commit()
            self.db.refresh(workflow_session)
            
            # Cache the session
            await cache.set(f"workflow_session:{session_id}", workflow_session.to_dict(), ttl=self.cache_ttl)
            
            logger.info("Workflow session created successfully", session_id=session_id)
            
            return {
                "id": workflow_session.id,
                "user_id": workflow_session.user_id,
                "session_name": workflow_session.session_name,
                "description": workflow_session.description,
                "current_step": workflow_session.current_step.value,
                "progress_percentage": workflow_session.progress_percentage,
                "status": workflow_session.status.value,
                "created_at": workflow_session.created_at.isoformat(),
                "updated_at": workflow_session.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to create workflow session", error=str(e), user_id=user_id)
            self.db.rollback()
            raise
    
    async def get_session(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a workflow session by ID"""
        try:
            # Try cache first
            cached_session = await cache.get(f"workflow_session:{session_id}")
            if cached_session:
                return cached_session
            
            workflow_session = self.db.query(WorkflowSessions).filter(
                WorkflowSessions.id == session_id,
                WorkflowSessions.user_id == user_id
            ).first()
            
            if not workflow_session:
                return None
            
            session_data = {
                "id": workflow_session.id,
                "user_id": workflow_session.user_id,
                "session_name": workflow_session.session_name,
                "description": workflow_session.description,
                "current_step": workflow_session.current_step.value,
                "progress_percentage": workflow_session.progress_percentage,
                "workflow_data": workflow_session.workflow_data,
                "completed_steps": workflow_session.completed_steps,
                "error_message": workflow_session.error_message,
                "status": workflow_session.status.value,
                "created_at": workflow_session.created_at.isoformat(),
                "updated_at": workflow_session.updated_at.isoformat(),
                "completed_at": workflow_session.completed_at.isoformat() if workflow_session.completed_at else None
            }
            
            # Cache the session
            await cache.set(f"workflow_session:{session_id}", session_data, ttl=self.cache_ttl)
            
            return session_data
            
        except Exception as e:
            logger.error("Failed to get workflow session", error=str(e), session_id=session_id)
            raise
    
    async def list_sessions(
        self, 
        user_id: str, 
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List workflow sessions for a user"""
        try:
            query = self.db.query(WorkflowSessions).filter(WorkflowSessions.user_id == user_id)
            
            if status:
                query = query.filter(WorkflowSessions.status == WorkflowStatus(status))
            
            total = query.count()
            sessions = query.order_by(WorkflowSessions.created_at.desc()).offset(offset).limit(limit).all()
            
            sessions_data = []
            for session in sessions:
                sessions_data.append({
                    "id": session.id,
                    "session_name": session.session_name,
                    "description": session.description,
                    "current_step": session.current_step.value,
                    "progress_percentage": session.progress_percentage,
                    "status": session.status.value,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "completed_at": session.completed_at.isoformat() if session.completed_at else None
                })
            
            return {
                "sessions": sessions_data,
                "total": total,
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            logger.error("Failed to list workflow sessions", error=str(e), user_id=user_id)
            raise
    
    async def update_session(
        self, 
        session_id: str, 
        user_id: str, 
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a workflow session"""
        try:
            workflow_session = self.db.query(WorkflowSessions).filter(
                WorkflowSessions.id == session_id,
                WorkflowSessions.user_id == user_id
            ).first()
            
            if not workflow_session:
                return None
            
            # Update fields
            if "current_step" in updates:
                workflow_session.current_step = WorkflowStep(updates["current_step"])
            
            if "progress_percentage" in updates:
                workflow_session.progress_percentage = updates["progress_percentage"]
            
            if "workflow_data" in updates:
                workflow_session.workflow_data = updates["workflow_data"]
            
            if "completed_steps" in updates:
                workflow_session.completed_steps = updates["completed_steps"]
            
            if "error_message" in updates:
                workflow_session.error_message = updates["error_message"]
            
            if "status" in updates:
                workflow_session.status = WorkflowStatus(updates["status"])
                if updates["status"] == "completed":
                    workflow_session.completed_at = datetime.utcnow()
            
            workflow_session.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(workflow_session)
            
            # Update cache
            session_data = {
                "id": workflow_session.id,
                "user_id": workflow_session.user_id,
                "session_name": workflow_session.session_name,
                "description": workflow_session.description,
                "current_step": workflow_session.current_step.value,
                "progress_percentage": workflow_session.progress_percentage,
                "workflow_data": workflow_session.workflow_data,
                "completed_steps": workflow_session.completed_steps,
                "error_message": workflow_session.error_message,
                "status": workflow_session.status.value,
                "created_at": workflow_session.created_at.isoformat(),
                "updated_at": workflow_session.updated_at.isoformat(),
                "completed_at": workflow_session.completed_at.isoformat() if workflow_session.completed_at else None
            }
            
            await cache.set(f"workflow_session:{session_id}", session_data, ttl=self.cache_ttl)
            
            logger.info("Workflow session updated successfully", session_id=session_id)
            
            return session_data
            
        except Exception as e:
            logger.error("Failed to update workflow session", error=str(e), session_id=session_id)
            self.db.rollback()
            raise
    
    async def delete_session(self, session_id: str, user_id: str) -> bool:
        """Delete a workflow session"""
        try:
            workflow_session = self.db.query(WorkflowSessions).filter(
                WorkflowSessions.id == session_id,
                WorkflowSessions.user_id == user_id
            ).first()
            
            if not workflow_session:
                return False
            
            self.db.delete(workflow_session)
            self.db.commit()
            
            # Remove from cache
            await cache.delete(f"workflow_session:{session_id}")
            
            logger.info("Workflow session deleted successfully", session_id=session_id)
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete workflow session", error=str(e), session_id=session_id)
            self.db.rollback()
            raise
    
    async def get_session_progress(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Get detailed progress information for a session"""
        try:
            session = await self.get_session(session_id, user_id)
            if not session:
                return None
            
            # Calculate step progress
            total_steps = len(WorkflowStep)
            current_step_index = list(WorkflowStep).index(WorkflowStep(session["current_step"]))
            step_progress = (current_step_index / total_steps) * 100
            
            return {
                "session_id": session_id,
                "current_step": session["current_step"],
                "progress_percentage": session["progress_percentage"],
                "step_progress": step_progress,
                "completed_steps": session["completed_steps"],
                "status": session["status"],
                "workflow_data": session["workflow_data"]
            }
            
        except Exception as e:
            logger.error("Failed to get session progress", error=str(e), session_id=session_id)
            raise
    
    async def start_topic_decomposition(
        self, 
        session_id: str, 
        user_id: str, 
        search_query: str,
        max_subtopics: int = 10
    ) -> Dict[str, Any]:
        """Start topic decomposition for a workflow session"""
        try:
            # Get the session
            session = await self.get_session(session_id, user_id)
            if not session:
                raise ValueError("Workflow session not found")
            
            # Create topic decomposition service
            topic_service = TopicDecompositionService(self.db)
            
            # Decompose the topic
            decomposition_result = await topic_service.decompose_topic(
                search_query=search_query,
                user_id=user_id,
                max_subtopics=max_subtopics
            )
            
            # Update workflow session with topic decomposition
            workflow_data = session.get("workflow_data", {})
            workflow_data["topic_decomposition"] = decomposition_result
            
            await self.update_session(session_id, user_id, {
                "workflow_data": workflow_data,
                "current_step": "topic_decomposition",
                "progress_percentage": 20
            })
            
            logger.info("Topic decomposition started", 
                       session_id=session_id, 
                       search_query=search_query,
                       subtopics_count=len(decomposition_result.get("subtopics", [])))
            
            return decomposition_result
            
        except Exception as e:
            logger.error("Failed to start topic decomposition", 
                        error=str(e), 
                        session_id=session_id)
            raise
    
    async def get_topic_decomposition(
        self, 
        session_id: str, 
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get topic decomposition for a workflow session"""
        try:
            session = await self.get_session(session_id, user_id)
            if not session:
                return None
            
            workflow_data = session.get("workflow_data", {})
            return workflow_data.get("topic_decomposition")
            
        except Exception as e:
            logger.error("Failed to get topic decomposition", 
                        error=str(e), 
                        session_id=session_id)
            raise
    
    async def start_affiliate_research(
        self, 
        session_id: str, 
        user_id: str, 
        selected_subtopics: List[str]
    ) -> Dict[str, Any]:
        """Start affiliate research for selected subtopics"""
        try:
            # Get the session
            session = await self.get_session(session_id, user_id)
            if not session:
                raise ValueError("Workflow session not found")
            
            # Get topic decomposition
            topic_decomposition = await self.get_topic_decomposition(session_id, user_id)
            if not topic_decomposition:
                raise ValueError("Topic decomposition not found")
            
            # Create affiliate offer service
            affiliate_service = AffiliateOfferService(self.db)
            
            # Find subtopics by name
            subtopics = topic_decomposition.get("subtopics", [])
            selected_subtopic_objects = []
            
            for subtopic in subtopics:
                if subtopic["name"] in selected_subtopics:
                    selected_subtopic_objects.append(subtopic)
            
            # Create affiliate offers for selected subtopics
            affiliate_offers = []
            for subtopic in selected_subtopic_objects:
                # This would integrate with LinkUp API in a real implementation
                offer_data = {
                    "offer_name": f"Affiliate Program for {subtopic['name']}",
                    "offer_description": f"Affiliate opportunities related to {subtopic['name']}",
                    "commission_rate": 5.0,  # Mock commission rate
                    "access_instructions": "Apply at affiliate.example.com",
                    "subtopic_id": topic_decomposition["id"],
                    "linkup_data": {
                        "source": "linkup",
                        "approval_rate": 0.8,
                        "competition_level": "medium"
                    }
                }
                
                offer = affiliate_service.create_offer(
                    user_id=user_id,
                    workflow_session_id=session_id,
                    **offer_data
                )
                affiliate_offers.append(offer)
            
            # Update workflow session with affiliate research
            workflow_data = session.get("workflow_data", {})
            workflow_data["affiliate_research"] = {
                "selected_subtopics": selected_subtopics,
                "affiliate_offers": affiliate_offers
            }
            
            await self.update_session(session_id, user_id, {
                "workflow_data": workflow_data,
                "current_step": "affiliate_research",
                "progress_percentage": 40
            })
            
            logger.info("Affiliate research started", 
                       session_id=session_id, 
                       selected_subtopics=selected_subtopics,
                       offers_count=len(affiliate_offers))
            
            return {
                "selected_subtopics": selected_subtopics,
                "affiliate_offers": affiliate_offers
            }
            
        except Exception as e:
            logger.error("Failed to start affiliate research", 
                        error=str(e), 
                        session_id=session_id)
            raise
    
    async def start_trend_analysis(
        self, 
        session_id: str, 
        user_id: str, 
        analysis_name: str,
        keywords: List[str],
        timeframe: str = "12m",
        geo: str = "US",
        category: Optional[int] = None,
        source: str = "google_trends"
    ) -> Dict[str, Any]:
        """Start trend analysis for a workflow session"""
        try:
            # Get the session
            session = await self.get_session(session_id, user_id)
            if not session:
                raise ValueError("Workflow session not found")
            
            # Create trend analysis service
            trend_service = TrendAnalysisService(self.db)
            
            # Create trend analysis
            analysis_data = await trend_service.create_trend_analysis(
                user_id=user_id,
                workflow_session_id=session_id,
                analysis_name=analysis_name,
                keywords=keywords,
                timeframe=timeframe,
                geo=geo,
                category=category,
                description=f"Trend analysis for {analysis_name}",
                topic_decomposition_id=session.get("workflow_data", {}).get("topic_decomposition", {}).get("id")
            )
            
            # Start the analysis
            from ..models.trend_analysis import TrendAnalysisSource
            analysis_result = await trend_service.start_trend_analysis(
                analysis_id=analysis_data["id"],
                user_id=user_id,
                source=TrendAnalysisSource(source)
            )
            
            # Update workflow session with trend analysis
            workflow_data = session.get("workflow_data", {})
            workflow_data["trend_analysis"] = analysis_result
            
            await self.update_session(session_id, user_id, {
                "workflow_data": workflow_data,
                "current_step": "trend_analysis",
                "progress_percentage": 60
            })
            
            logger.info("Trend analysis started", 
                       session_id=session_id, 
                       analysis_name=analysis_name,
                       keywords_count=len(keywords))
            
            return analysis_result
            
        except Exception as e:
            logger.error("Failed to start trend analysis", 
                        error=str(e), 
                        session_id=session_id)
            raise
    
    async def get_trend_analysis(
        self, 
        session_id: str, 
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get trend analysis for a workflow session"""
        try:
            session = await self.get_session(session_id, user_id)
            if not session:
                return None
            
            workflow_data = session.get("workflow_data", {})
            return workflow_data.get("trend_analysis")
            
        except Exception as e:
            logger.error("Failed to get trend analysis", 
                        error=str(e), 
                        session_id=session_id)
            raise
    
    async def get_workflow_complete_data(
        self, 
        session_id: str, 
        user_id: str
    ) -> Dict[str, Any]:
        """Get complete workflow data including all steps"""
        try:
            session = await self.get_session(session_id, user_id)
            if not session:
                raise ValueError("Workflow session not found")
            
            workflow_data = session.get("workflow_data", {})
            
            # Get affiliate offers for this session
            affiliate_service = AffiliateOfferService(self.db)
            affiliate_offers = affiliate_service.list_offers_by_workflow(
                workflow_session_id=session_id,
                user_id=user_id
            )
            
            return {
                "session": session,
                "topic_decomposition": workflow_data.get("topic_decomposition"),
                "affiliate_research": workflow_data.get("affiliate_research"),
                "affiliate_offers": affiliate_offers,
                "trend_analysis": workflow_data.get("trend_analysis"),
                "content_generation": workflow_data.get("content_generation")
            }
            
        except Exception as e:
            logger.error("Failed to get complete workflow data", 
                        error=str(e), 
                        session_id=session_id)
            raise
