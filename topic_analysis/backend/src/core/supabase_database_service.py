"""
Supabase-only database service for TrendTap
This replaces all PostgreSQL/SQLAlchemy dependencies with Supabase SDK
"""

from typing import Dict, List, Any, Optional, Union, Generator
from supabase import Client
import structlog
import json
from datetime import datetime
import uuid
from contextlib import contextmanager
from src.core.supabase_singleton import get_supabase_client

logger = structlog.get_logger()

class SupabaseDatabaseService:
    """Main database service using Supabase API only"""
    
    def __init__(self):
        self.client = get_supabase_client()
        logger.info("Supabase database service initialized")
    
    # User operations
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            result = self.client.table("users").select("*").eq("id", user_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error getting user by ID", user_id=user_id, error=str(e))
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            result = self.client.table("users").select("*").eq("email", email).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error getting user by email", email=email, error=str(e))
            return None
    
    def create_user(self, user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new user"""
        try:
            user_data["id"] = str(uuid.uuid4())
            user_data["created_at"] = datetime.utcnow().isoformat()
            user_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.client.table("users").insert(user_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error creating user", error=str(e))
            return None
    
    def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update user data"""
        try:
            user_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.client.table("users").update(user_data).eq("id", user_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error updating user", user_id=user_id, error=str(e))
            return None
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        try:
            result = self.client.table("users").delete().eq("id", user_id).execute()
            return True
        except Exception as e:
            logger.error("Error deleting user", user_id=user_id, error=str(e))
            return False
    
    # Affiliate Research operations
    def get_affiliate_research_by_id(self, research_id: str) -> Optional[Dict[str, Any]]:
        """Get affiliate research by ID"""
        try:
            result = self.client.table("affiliate_researches").select("*").eq("id", research_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error getting affiliate research by ID", research_id=research_id, error=str(e))
            return None
    
    def get_affiliate_researches_by_user(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get affiliate researches by user ID"""
        try:
            result = self.client.table("affiliate_researches")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .range(offset, offset + limit - 1)\
                .execute()
            return result.data or []
        except Exception as e:
            logger.error("Error getting affiliate researches by user", user_id=user_id, error=str(e))
            return []
    
    def create_affiliate_research(self, research_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create new affiliate research"""
        try:
            research_data["id"] = str(uuid.uuid4())
            research_data["created_at"] = datetime.utcnow().isoformat()
            research_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.client.table("affiliate_researches").insert(research_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error creating affiliate research", error=str(e))
            return None
    
    def update_affiliate_research(self, research_id: str, research_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update affiliate research"""
        try:
            research_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.client.table("affiliate_researches").update(research_data).eq("id", research_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error updating affiliate research", research_id=research_id, error=str(e))
            return None
    
    # Trend Analysis operations
    def get_trend_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get trend analysis by ID"""
        try:
            result = self.client.table("trend_analyses").select("*").eq("id", analysis_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error getting trend analysis by ID", analysis_id=analysis_id, error=str(e))
            return None
    
    def get_trend_analyses_by_user(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get trend analyses by user ID"""
        try:
            result = self.client.table("trend_analyses")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .range(offset, offset + limit - 1)\
                .execute()
            return result.data or []
        except Exception as e:
            logger.error("Error getting trend analyses by user", user_id=user_id, error=str(e))
            return []
    
    def create_trend_analysis(self, analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create new trend analysis"""
        try:
            analysis_data["id"] = str(uuid.uuid4())
            analysis_data["created_at"] = datetime.utcnow().isoformat()
            analysis_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.client.table("trend_analyses").insert(analysis_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error creating trend analysis", error=str(e))
            return None
    
    def update_trend_analysis(self, analysis_id: str, analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update trend analysis"""
        try:
            analysis_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.client.table("trend_analyses").update(analysis_data).eq("id", analysis_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error updating trend analysis", analysis_id=analysis_id, error=str(e))
            return None
    
    # Content Ideas operations
    def get_content_ideas_by_user(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get content ideas by user ID"""
        try:
            result = self.client.table("content_ideas")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .range(offset, offset + limit - 1)\
                .execute()
            return result.data or []
        except Exception as e:
            logger.error("Error getting content ideas by user", user_id=user_id, error=str(e))
            return []
    
    def create_content_idea(self, idea_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create new content idea"""
        try:
            idea_data["id"] = str(uuid.uuid4())
            idea_data["created_at"] = datetime.utcnow().isoformat()
            idea_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.client.table("content_ideas").insert(idea_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error creating content idea", error=str(e))
            return None
    
    def update_content_idea(self, idea_id: str, idea_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update content idea"""
        try:
            idea_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.client.table("content_ideas").update(idea_data).eq("id", idea_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error updating content idea", idea_id=idea_id, error=str(e))
            return None
    
    # Generic operations
    def execute_query(self, table: str, operation: str, **kwargs) -> Any:
        """Execute a generic query on any table"""
        try:
            query = self.client.table(table)
            
            if operation == "select":
                query = query.select(kwargs.get("columns", "*"))
                if "filters" in kwargs:
                    for filter_key, filter_value in kwargs["filters"].items():
                        query = query.eq(filter_key, filter_value)
                if "order_by" in kwargs:
                    query = query.order(kwargs["order_by"], desc=kwargs.get("desc", False))
                if "limit" in kwargs:
                    query = query.limit(kwargs["limit"])
                if "offset" in kwargs:
                    query = query.range(kwargs["offset"], kwargs["offset"] + kwargs.get("limit", 1) - 1)
                
                result = query.execute()
                return result.data
            
            elif operation == "insert":
                data = kwargs.get("data", {})
                if "id" not in data:
                    data["id"] = str(uuid.uuid4())
                data["created_at"] = datetime.utcnow().isoformat()
                data["updated_at"] = datetime.utcnow().isoformat()
                
                result = query.insert(data).execute()
                return result.data[0] if result.data else None
            
            elif operation == "update":
                data = kwargs.get("data", {})
                data["updated_at"] = datetime.utcnow().isoformat()
                
                filters = kwargs.get("filters", {})
                for filter_key, filter_value in filters.items():
                    query = query.eq(filter_key, filter_value)
                
                result = query.update(data).execute()
                return result.data[0] if result.data else None
            
            elif operation == "delete":
                filters = kwargs.get("filters", {})
                for filter_key, filter_value in filters.items():
                    query = query.eq(filter_key, filter_value)
                
                result = query.delete().execute()
                return True
            
            else:
                raise ValueError(f"Unsupported operation: {operation}")
                
        except Exception as e:
            logger.error("Error executing query", table=table, operation=operation, error=str(e))
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            # Test connection with a simple query
            result = self.client.table("users").select("id").limit(1).execute()
            
            from src.core.config import get_settings
            settings = get_settings()
            return {
                "healthy": True,
                "status": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "supabase_url": settings.supabase_url
            }
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return {
                "healthy": False,
                "status": "disconnected",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Global instance
_database_service: Optional[SupabaseDatabaseService] = None

def get_database_service() -> SupabaseDatabaseService:
    """Get the global database service instance"""
    global _database_service
    if _database_service is None:
        _database_service = SupabaseDatabaseService()
    return _database_service

# Dependency for FastAPI
def get_db() -> Generator[SupabaseDatabaseService, None, None]:
    """FastAPI dependency to get database service"""
    yield get_database_service()

# Context manager for database operations
@contextmanager
def get_db_session() -> Generator[SupabaseDatabaseService, None, None]:
    """Context manager for database service"""
    service = get_database_service()
    try:
        yield service
    except Exception as e:
        logger.error("Database session error", error=str(e))
        raise
    finally:
        # Supabase client doesn't need explicit cleanup
        pass