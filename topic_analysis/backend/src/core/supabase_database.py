"""
Supabase-based database operations for TrendTap
This replaces direct PostgreSQL connections with Supabase API calls
"""
from typing import Dict, List, Any, Optional, Union
from supabase import Client
import structlog
import json
from datetime import datetime
import uuid
from src.core.supabase_singleton import get_supabase_client

logger = structlog.get_logger()

class SupabaseDatabase:
    """Database operations using Supabase API"""
    
    def __init__(self):
        self.client = get_supabase_client()
        logger.info("Supabase database client initialized")
    
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
        """Create new user"""
        try:
            result = self.client.table("users").insert(user_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error creating user", error=str(e))
            return None
    
    def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update user"""
        try:
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
    
    def get_users_list(self, page: int = 1, per_page: int = 10, 
                      search: Optional[str] = None, role: Optional[str] = None,
                      is_active: Optional[bool] = None) -> Dict[str, Any]:
        """Get paginated list of users"""
        try:
            query = self.client.table("users").select("*")
            
            # Apply filters
            if search:
                query = query.or_(f"email.ilike.%{search}%,first_name.ilike.%{search}%,last_name.ilike.%{search}%")
            if role:
                query = query.eq("role", role)
            if is_active is not None:
                query = query.eq("is_active", is_active)
            
            # Apply pagination
            offset = (page - 1) * per_page
            query = query.range(offset, offset + per_page - 1)
            
            result = query.execute()
            
            return {
                "users": result.data,
                "total": len(result.data),
                "page": page,
                "per_page": per_page,
                "total_pages": (len(result.data) + per_page - 1) // per_page
            }
        except Exception as e:
            logger.error("Error getting users list", error=str(e))
            return {"users": [], "total": 0, "page": page, "per_page": per_page, "total_pages": 0}
    
    # Affiliate Research operations
    def create_affiliate_research(self, research_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create affiliate research"""
        try:
            result = self.client.table("affiliate_research").insert(research_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error creating affiliate research", error=str(e))
            return None
    
    def get_affiliate_research_by_user(self, user_id: str, page: int = 1, per_page: int = 10) -> Dict[str, Any]:
        """Get affiliate research by user"""
        try:
            offset = (page - 1) * per_page
            result = self.client.table("affiliate_research").select("*").eq("user_id", user_id).range(offset, offset + per_page - 1).execute()
            
            return {
                "research": result.data,
                "total": len(result.data),
                "page": page,
                "per_page": per_page
            }
        except Exception as e:
            logger.error("Error getting affiliate research", user_id=user_id, error=str(e))
            return {"research": [], "total": 0, "page": page, "per_page": per_page}
    
    # Trend Analysis operations
    def create_trend_analysis(self, analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create trend analysis"""
        try:
            result = self.client.table("trend_analysis").insert(analysis_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error creating trend analysis", error=str(e))
            return None
    
    def get_trend_analysis_by_user(self, user_id: str, page: int = 1, per_page: int = 10) -> Dict[str, Any]:
        """Get trend analysis by user"""
        try:
            offset = (page - 1) * per_page
            result = self.client.table("trend_analysis").select("*").eq("user_id", user_id).range(offset, offset + per_page - 1).execute()
            
            return {
                "analysis": result.data,
                "total": len(result.data),
                "page": page,
                "per_page": per_page
            }
        except Exception as e:
            logger.error("Error getting trend analysis", user_id=user_id, error=str(e))
            return {"analysis": [], "total": 0, "page": page, "per_page": per_page}
    
    # LLM Provider operations
    def get_llm_providers(self) -> List[Dict[str, Any]]:
        """Get all LLM providers"""
        try:
            result = self.client.table("llm_providers").select("*").execute()
            return result.data
        except Exception as e:
            logger.error("Error getting LLM providers", error=str(e))
            return []
    
    def create_llm_provider(self, provider_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create LLM provider"""
        try:
            result = self.client.table("llm_providers").insert(provider_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error creating LLM provider", error=str(e))
            return None
    
    def update_llm_provider(self, provider_id: str, provider_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update LLM provider"""
        try:
            result = self.client.table("llm_providers").update(provider_data).eq("id", provider_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Error updating LLM provider", provider_id=provider_id, error=str(e))
            return None
    
    def delete_llm_provider(self, provider_id: str) -> bool:
        """Delete LLM provider"""
        try:
            result = self.client.table("llm_providers").delete().eq("id", provider_id).execute()
            return True
        except Exception as e:
            logger.error("Error deleting LLM provider", provider_id=provider_id, error=str(e))
            return False
    
    # Health check
    def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            # Try to query an existing table (llm_providers)
            result = self.client.table("llm_providers").select("id").limit(1).execute()
            
            # Also try to query users table if it exists
            try:
                users_result = self.client.table("users").select("id").limit(1).execute()
                tables_available = ["llm_providers", "llm_configurations", "users"]
            except:
                tables_available = ["llm_providers", "llm_configurations"]
            
            return {
                "status": "healthy",
                "message": "Database connection successful",
                "timestamp": datetime.utcnow().isoformat(),
                "tables_available": tables_available
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Database connection failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }

    # Affiliate program operations
    def search_programs(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for affiliate programs by name, description, or category"""
        try:
            result = self.client.table("affiliate_programs").select("*").or_(
                f"name.ilike.%{search_term}%,description.ilike.%{search_term}%,category.ilike.%{search_term}%"
            ).eq("is_active", True).order("usage_count", desc=True).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error("Error searching programs", search_term=search_term, error=str(e))
            return []
    
    def save_programs(self, programs_data: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
        """Save affiliate programs to the database"""
        saved_programs = []
        for program_data in programs_data:
            program_name = program_data.get("name")
            if not program_name:
                logger.warning("Skipping program with no name", program_data=program_data)
                continue
            
            try:
                # Check if program already exists
                existing = self.client.table("affiliate_programs").select("*").eq("name", program_name).execute()
                
                if existing.data:
                    # Update existing program
                    program_id = existing.data[0]["id"]
                    search_terms = existing.data[0].get("search_terms", [])
                    if search_term not in search_terms:
                        search_terms.append(search_term)
                    
                    update_data = {
                        "search_terms": search_terms,
                        "usage_count": existing.data[0].get("usage_count", 0) + 1,
                        "last_used": datetime.utcnow().isoformat()
                    }
                    
                    result = self.client.table("affiliate_programs").update(update_data).eq("id", program_id).execute()
                    if result.data:
                        saved_programs.append(result.data[0])
                        logger.info("Updated existing program", program_id=program_id, name=program_name)
                else:
                    # Create new program
                    new_program = {
                        "name": program_name,
                        "description": program_data.get("description"),
                        "commission": program_data.get("commission"),
                        "cookie_duration": program_data.get("cookie_duration"),
                        "payment_terms": program_data.get("payment_terms"),
                        "min_payout": program_data.get("min_payout"),
                        "category": program_data.get("category"),
                        "rating": program_data.get("rating", 0.0),
                        "estimated_earnings": program_data.get("estimated_earnings"),
                        "difficulty": program_data.get("difficulty"),
                        "affiliate_network": program_data.get("affiliate_network"),
                        "tracking_method": program_data.get("tracking_method"),
                        "payment_methods": program_data.get("payment_methods"),
                        "support_level": program_data.get("support_level"),
                        "promotional_materials": program_data.get("promotional_materials"),
                        "restrictions": program_data.get("restrictions"),
                        "source": program_data.get("source", "web_search"),
                        "search_terms": [search_term],
                        "is_verified": program_data.get("is_verified", False)
                    }
                    
                    result = self.client.table("affiliate_programs").insert(new_program).execute()
                    if result.data:
                        saved_programs.append(result.data[0])
                        logger.info("Saved new program", program_id=result.data[0]["id"], name=program_name)
                        
            except Exception as e:
                logger.error("Failed to save program", program_name=program_name, error=str(e))
                continue
        
        return saved_programs
    
    def update_program_usage(self, program_id: int):
        """Update program usage count and last_used timestamp"""
        try:
            # Get current program data
            result = self.client.table("affiliate_programs").select("usage_count").eq("id", program_id).execute()
            if result.data:
                current_count = result.data[0].get("usage_count", 0)
                
                # Update usage count and last_used
                update_data = {
                    "usage_count": current_count + 1,
                    "last_used": datetime.utcnow().isoformat()
                }
                
                self.client.table("affiliate_programs").update(update_data).eq("id", program_id).execute()
                logger.info("Updated program usage", program_id=program_id, usage_count=current_count + 1)
        except Exception as e:
            logger.error("Failed to update program usage", program_id=program_id, error=str(e))

# Global database instance
db = SupabaseDatabase()

def get_supabase_db() -> SupabaseDatabase:
    """Get the Supabase database instance"""
    return db
