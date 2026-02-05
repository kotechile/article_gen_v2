"""
Database Service
Service for managing database connections and operations with Supabase.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
import pandas as pd
from supabase import Client
from ..core.supabase_singleton import get_supabase_client
from ..models.keyword import Keyword
from ..models.analysis_report import KeywordAnalysisReport
from ..models.content_opportunity import ContentOpportunity
from ..models.seo_content_idea import SEOContentIdea
from ..models.ahrefs_export_file import AhrefsExportFile
from ..models.idea_burst_session import IdeaBurstSession
from ..models.selection_indicator import SelectionIndicator

class DatabaseService:
    def __init__(self):
        """Initialize database connection to Supabase using singleton."""
        self.client: Client = get_supabase_client()
    
    # Keyword operations
    async def save_keywords(self, keywords: List[Keyword]) -> bool:
        """Save keywords to the database."""
        try:
            keywords_data = [keyword.model_dump() for keyword in keywords]
            result = self.client.table("keywords").insert(keywords_data).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error saving keywords: {e}")
            return False
    
    async def get_keywords(self, file_id: str) -> List[Keyword]:
        """Get keywords by file ID."""
        try:
            result = self.client.table("keywords").select("*").eq("file_id", file_id).execute()
            return [Keyword(**keyword) for keyword in result.data]
        except Exception as e:
            print(f"Error getting keywords: {e}")
            return []
    
    async def update_keyword(self, keyword_id: str, updates: Dict[str, Any]) -> bool:
        """Update a keyword."""
        try:
            result = self.client.table("keywords").update(updates).eq("id", keyword_id).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error updating keyword: {e}")
            return False
    
    async def delete_keyword(self, keyword_id: str) -> bool:
        """Delete a keyword."""
        try:
            result = self.client.table("keywords").delete().eq("id", keyword_id).execute()
            return True
        except Exception as e:
            print(f"Error deleting keyword: {e}")
            return False
    
    # Analysis Report operations
    async def save_analysis_report(self, report: KeywordAnalysisReport) -> bool:
        """Save analysis report to the database."""
        try:
            report_data = report.model_dump()
            result = self.client.table("analysis_reports").insert(report_data).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error saving analysis report: {e}")
            return False
    
    async def get_analysis_report(self, report_id: str) -> Optional[KeywordAnalysisReport]:
        """Get analysis report by ID."""
        try:
            result = self.client.table("analysis_reports").select("*").eq("id", report_id).execute()
            if result.data:
                return KeywordAnalysisReport(**result.data[0])
            return None
        except Exception as e:
            print(f"Error getting analysis report: {e}")
            return None
    
    async def get_analysis_reports_by_user(self, user_id: str) -> List[KeywordAnalysisReport]:
        """Get analysis reports by user ID."""
        try:
            result = self.client.table("analysis_reports").select("*").eq("user_id", user_id).execute()
            return [KeywordAnalysisReport(**report) for report in result.data]
        except Exception as e:
            print(f"Error getting analysis reports: {e}")
            return []
    
    # Content Opportunity operations
    async def save_content_opportunities(self, opportunities: List[ContentOpportunity]) -> bool:
        """Save content opportunities to the database."""
        try:
            opportunities_data = [opportunity.model_dump() for opportunity in opportunities]
            result = self.client.table("content_opportunities").insert(opportunities_data).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error saving content opportunities: {e}")
            return False
    
    async def get_content_opportunities(self, file_id: str) -> List[ContentOpportunity]:
        """Get content opportunities by file ID."""
        try:
            result = self.client.table("content_opportunities").select("*").eq("file_id", file_id).execute()
            return [ContentOpportunity(**opportunity) for opportunity in result.data]
        except Exception as e:
            print(f"Error getting content opportunities: {e}")
            return []
    
    # SEO Content Idea operations
    async def save_seo_content_ideas(self, ideas: List[SEOContentIdea]) -> bool:
        """Save SEO content ideas to the database."""
        try:
            ideas_data = [idea.model_dump() for idea in ideas]
            result = self.client.table("seo_content_ideas").insert(ideas_data).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error saving SEO content ideas: {e}")
            return False
    
    async def get_seo_content_ideas(self, file_id: str) -> List[SEOContentIdea]:
        """Get SEO content ideas by file ID."""
        try:
            result = self.client.table("seo_content_ideas").select("*").eq("file_id", file_id).execute()
            return [SEOContentIdea(**idea) for idea in result.data]
        except Exception as e:
            print(f"Error getting SEO content ideas: {e}")
            return []
    
    async def update_seo_content_idea(self, idea_id: str, updates: Dict[str, Any]) -> bool:
        """Update a SEO content idea."""
        try:
            result = self.client.table("seo_content_ideas").update(updates).eq("id", idea_id).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error updating SEO content idea: {e}")
            return False
    
    # Ahrefs Export File operations
    async def save_ahrefs_file(self, file: AhrefsExportFile) -> bool:
        """Save Ahrefs export file metadata."""
        try:
            file_data = file.model_dump()
            result = self.client.table("ahrefs_export_files").insert(file_data).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error saving Ahrefs file: {e}")
            return False
    
    async def get_ahrefs_file(self, file_id: str) -> Optional[AhrefsExportFile]:
        """Get Ahrefs export file by ID."""
        try:
            result = self.client.table("ahrefs_export_files").select("*").eq("id", file_id).execute()
            if result.data:
                return AhrefsExportFile(**result.data[0])
            return None
        except Exception as e:
            print(f"Error getting Ahrefs file: {e}")
            return None
    
    async def update_ahrefs_file_status(self, file_id: str, status: str, error_message: Optional[str] = None) -> bool:
        """Update Ahrefs file processing status."""
        try:
            updates = {"status": status, "updated_at": datetime.utcnow().isoformat()}
            if error_message:
                updates["error_message"] = error_message
            result = self.client.table("ahrefs_export_files").update(updates).eq("id", file_id).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error updating Ahrefs file status: {e}")
            return False
    
    # Idea Burst Session operations
    async def save_idea_burst_session(self, session: IdeaBurstSession) -> bool:
        """Save idea burst session to the database."""
        try:
            session_data = session.model_dump()
            result = self.client.table("idea_burst_sessions").insert(session_data).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error saving idea burst session: {e}")
            return False
    
    async def get_idea_burst_session(self, session_id: str) -> Optional[IdeaBurstSession]:
        """Get idea burst session by ID."""
        try:
            result = self.client.table("idea_burst_sessions").select("*").eq("id", session_id).execute()
            if result.data:
                return IdeaBurstSession(**result.data[0])
            return None
        except Exception as e:
            print(f"Error getting idea burst session: {e}")
            return None
    
    async def update_idea_burst_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update idea burst session."""
        try:
            result = self.client.table("idea_burst_sessions").update(updates).eq("id", session_id).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error updating idea burst session: {e}")
            return False
    
    async def get_user_idea_burst_sessions(self, user_id: str) -> List[IdeaBurstSession]:
        """Get idea burst sessions by user ID."""
        try:
            result = self.client.table("idea_burst_sessions").select("*").eq("user_id", user_id).execute()
            return [IdeaBurstSession(**session) for session in result.data]
        except Exception as e:
            print(f"Error getting user idea burst sessions: {e}")
            return []
    
    # Selection Indicator operations
    async def save_selection_indicators(self, indicators: List[SelectionIndicator]) -> bool:
        """Save selection indicators to the database."""
        try:
            indicators_data = [indicator.model_dump() for indicator in indicators]
            result = self.client.table("selection_indicators").insert(indicators_data).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error saving selection indicators: {e}")
            return False
    
    async def get_selection_indicators(self, session_id: str) -> List[SelectionIndicator]:
        """Get selection indicators by session ID."""
        try:
            result = self.client.table("selection_indicators").select("*").eq("session_id", session_id).execute()
            return [SelectionIndicator(**indicator) for indicator in result.data]
        except Exception as e:
            print(f"Error getting selection indicators: {e}")
            return []
    
    # File processing operations
    async def save_uploaded_file_metadata(self, file_id: str, filename: str, data: str) -> bool:
        """Save uploaded file metadata."""
        try:
            file_data = {
                "id": file_id,
                "filename": filename,
                "data": data,
                "status": "uploaded",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            result = self.client.table("uploaded_files").insert(file_data).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error saving file metadata: {e}")
            return False
    
    async def get_file_processing_status(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file processing status."""
        try:
            result = self.client.table("uploaded_files").select("*").eq("id", file_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            print(f"Error getting file status: {e}")
            return None
    
    async def update_file_processing_status(self, file_id: str, status: str, progress: int = 0, message: str = "") -> bool:
        """Update file processing status."""
        try:
            updates = {
                "status": status,
                "progress": progress,
                "message": message,
                "updated_at": datetime.utcnow().isoformat()
            }
            result = self.client.table("uploaded_files").update(updates).eq("id", file_id).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error updating file status: {e}")
            return False
    
    # Analysis operations
    async def save_analysis_results(self, file_id: str, results: Dict[str, Any]) -> bool:
        """Save analysis results."""
        try:
            analysis_data = {
                "file_id": file_id,
                "results": json.dumps(results),
                "status": "completed",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            result = self.client.table("analysis_results").insert(analysis_data).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error saving analysis results: {e}")
            return False
    
    async def get_analysis_results(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis results by file ID."""
        try:
            result = self.client.table("analysis_results").select("*").eq("file_id", file_id).execute()
            if result.data:
                return json.loads(result.data[0]["results"])
            return None
        except Exception as e:
            print(f"Error getting analysis results: {e}")
            return None
    
    async def get_analysis_status(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis status by file ID."""
        try:
            result = self.client.table("analysis_results").select("*").eq("file_id", file_id).execute()
            if result.data:
                return {
                    "status": result.data[0]["status"],
                    "progress": result.data[0].get("progress", 0),
                    "message": result.data[0].get("message", ""),
                    "created_at": result.data[0]["created_at"],
                    "updated_at": result.data[0]["updated_at"]
                }
            return None
        except Exception as e:
            print(f"Error getting analysis status: {e}")
            return None
    
    # Report operations
    async def save_report(self, report: Dict[str, Any]) -> bool:
        """Save report to the database."""
        try:
            result = self.client.table("reports").insert(report).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Error saving report: {e}")
            return False
    
    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report by ID."""
        try:
            result = self.client.table("reports").select("*").eq("id", report_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            print(f"Error getting report: {e}")
            return None
    
    async def get_user_reports(self, user_id: str) -> List[Dict[str, Any]]:
        """Get reports by user ID."""
        try:
            result = self.client.table("reports").select("*").eq("user_id", user_id).execute()
            return result.data
        except Exception as e:
            print(f"Error getting user reports: {e}")
            return []
    
    # Utility methods
    async def cleanup_old_data(self, days: int = 90) -> int:
        """Clean up data older than specified days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            cutoff_str = cutoff_date.isoformat()
            
            # Clean up old uploaded files
            result1 = self.client.table("uploaded_files").delete().lt("created_at", cutoff_str).execute()
            
            # Clean up old analysis results
            result2 = self.client.table("analysis_results").delete().lt("created_at", cutoff_str).execute()
            
            # Clean up old idea burst sessions
            result3 = self.client.table("idea_burst_sessions").delete().lt("created_at", cutoff_str).execute()
            
            total_deleted = len(result1.data) + len(result2.data) + len(result3.data)
            return total_deleted
        except Exception as e:
            print(f"Error cleaning up old data: {e}")
            return 0
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}
            
            # Count records in each table
            tables = [
                "keywords", "analysis_reports", "content_opportunities", 
                "seo_content_ideas", "ahrefs_export_files", "idea_burst_sessions",
                "selection_indicators", "uploaded_files", "analysis_results", "reports"
            ]
            
            for table in tables:
                try:
                    result = self.client.table(table).select("id", count="exact").execute()
                    stats[table] = result.count
                except Exception as e:
                    stats[table] = f"Error: {e}"
            
            return stats
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check database connection health."""
        try:
            # Simple query to test connection
            result = self.client.table("keywords").select("id").limit(1).execute()
            return True
        except Exception as e:
            print(f"Database health check failed: {e}")
            return False

