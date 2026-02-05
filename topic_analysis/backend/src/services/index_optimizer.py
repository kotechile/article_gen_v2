"""
Database index optimization service
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

from ..core.config import get_settings
from src.core.supabase_database_service import SupabaseDatabaseService

logger = logging.getLogger(__name__)
settings = get_settings()

class IndexOptimizer:
    """Service for database index optimization"""
    
    def __init__(self, db: SupabaseDatabaseService):
        self.db = db
        self.inspector = inspect(self.db.bind)
    
    def analyze_indexes(self) -> Dict[str, Any]:
        """Analyze current database indexes"""
        try:
            # Get all tables
            tables = self.inspector.get_table_names()
            
            index_analysis = {
                "tables": {},
                "total_indexes": 0,
                "recommendations": [],
                "performance_impact": "unknown"
            }
            
            for table_name in tables:
                table_info = self._analyze_table_indexes(table_name)
                index_analysis["tables"][table_name] = table_info
                index_analysis["total_indexes"] += len(table_info.get("indexes", []))
            
            # Generate recommendations
            index_analysis["recommendations"] = self._generate_index_recommendations(index_analysis)
            
            return index_analysis
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in analyze_indexes: {str(e)}")
            return {"error": "Failed to analyze indexes"}
        except Exception as e:
            logger.error(f"Error in analyze_indexes: {str(e)}")
            return {"error": "Index analysis failed"}
    
    def _analyze_table_indexes(self, table_name: str) -> Dict[str, Any]:
        """Analyze indexes for a specific table"""
        try:
            # Get table indexes
            indexes = self.inspector.get_indexes(table_name)
            
            # Get table columns
            columns = self.inspector.get_columns(table_name)
            
            # Get foreign keys
            foreign_keys = self.inspector.get_foreign_keys(table_name)
            
            table_analysis = {
                "table_name": table_name,
                "columns": len(columns),
                "indexes": [],
                "foreign_keys": len(foreign_keys),
                "primary_key": None,
                "unique_indexes": 0,
                "composite_indexes": 0,
                "single_column_indexes": 0
            }
            
            # Analyze primary key
            pk_constraint = self.inspector.get_pk_constraint(table_name)
            if pk_constraint and pk_constraint.get('constrained_columns'):
                table_analysis["primary_key"] = pk_constraint['constrained_columns']
            
            # Analyze indexes
            for index in indexes:
                index_info = {
                    "name": index['name'],
                    "columns": index['column_names'],
                    "unique": index['unique'],
                    "type": "composite" if len(index['column_names']) > 1 else "single_column"
                }
                
                table_analysis["indexes"].append(index_info)
                
                if index['unique']:
                    table_analysis["unique_indexes"] += 1
                
                if len(index['column_names']) > 1:
                    table_analysis["composite_indexes"] += 1
                else:
                    table_analysis["single_column_indexes"] += 1
            
            return table_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing table {table_name}: {str(e)}")
            return {"error": f"Failed to analyze table {table_name}"}
    
    def _generate_index_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate index optimization recommendations"""
        recommendations = []
        
        for table_name, table_info in analysis.get("tables", {}).items():
            if "error" in table_info:
                continue
            
            # Check for missing indexes on foreign keys
            if table_info.get("foreign_keys", 0) > 0:
                fk_columns = self._get_foreign_key_columns(table_name)
                indexed_columns = set()
                
                for index in table_info.get("indexes", []):
                    indexed_columns.update(index["columns"])
                
                missing_fk_indexes = fk_columns - indexed_columns
                if missing_fk_indexes:
                    recommendations.append(
                        f"Table {table_name}: Consider adding indexes on foreign key columns: {list(missing_fk_indexes)}"
                    )
            
            # Check for missing indexes on commonly queried columns
            common_columns = self._get_commonly_queried_columns(table_name)
            if common_columns:
                indexed_columns = set()
                for index in table_info.get("indexes", []):
                    indexed_columns.update(index["columns"])
                
                missing_common_indexes = common_columns - indexed_columns
                if missing_common_indexes:
                    recommendations.append(
                        f"Table {table_name}: Consider adding indexes on commonly queried columns: {list(missing_common_indexes)}"
                    )
            
            # Check for too many indexes
            total_indexes = len(table_info.get("indexes", []))
            if total_indexes > 10:
                recommendations.append(
                    f"Table {table_name}: Has {total_indexes} indexes, consider removing unused ones"
                )
            
            # Check for missing composite indexes
            if table_info.get("composite_indexes", 0) == 0 and table_info.get("single_column_indexes", 0) > 3:
                recommendations.append(
                    f"Table {table_name}: Consider creating composite indexes for frequently used column combinations"
                )
        
        return recommendations
    
    def _get_foreign_key_columns(self, table_name: str) -> set:
        """Get foreign key columns for a table"""
        try:
            foreign_keys = self.inspector.get_foreign_keys(table_name)
            fk_columns = set()
            
            for fk in foreign_keys:
                fk_columns.update(fk.get('constrained_columns', []))
            
            return fk_columns
            
        except Exception as e:
            logger.error(f"Error getting foreign key columns for {table_name}: {str(e)}")
            return set()
    
    def _get_commonly_queried_columns(self, table_name: str) -> set:
        """Get commonly queried columns for a table"""
        # This is a simplified implementation
        # In a real system, this would analyze query logs or use database statistics
        
        common_columns = set()
        
        # Common patterns for user tables
        if "user" in table_name.lower():
            common_columns.update(["email", "username", "is_active", "created_at", "last_login"])
        
        # Common patterns for session tables
        if "session" in table_name.lower():
            common_columns.update(["user_id", "is_active", "expires_at", "created_at"])
        
        # Common patterns for log tables
        if "log" in table_name.lower():
            common_columns.update(["user_id", "created_at", "action", "success"])
        
        # Common patterns for token tables
        if "token" in table_name.lower():
            common_columns.update(["user_id", "token", "expires_at", "status"])
        
        return common_columns
    
    def create_recommended_indexes(self) -> Dict[str, Any]:
        """Create recommended indexes for optimization"""
        try:
            analysis = self.analyze_indexes()
            if "error" in analysis:
                return analysis
            
            created_indexes = []
            failed_indexes = []
            
            # Create indexes for users table
            user_indexes = self._create_user_indexes()
            created_indexes.extend(user_indexes.get("created", []))
            failed_indexes.extend(user_indexes.get("failed", []))
            
            # Create indexes for sessions table
            session_indexes = self._create_session_indexes()
            created_indexes.extend(session_indexes.get("created", []))
            failed_indexes.extend(session_indexes.get("failed", []))
            
            # Create indexes for auth logs table
            auth_log_indexes = self._create_auth_log_indexes()
            created_indexes.extend(auth_log_indexes.get("created", []))
            failed_indexes.extend(auth_log_indexes.get("failed", []))
            
            # Create indexes for CSRF tables
            csrf_indexes = self._create_csrf_indexes()
            created_indexes.extend(csrf_indexes.get("created", []))
            failed_indexes.extend(csrf_indexes.get("failed", []))
            
            return {
                "created_indexes": created_indexes,
                "failed_indexes": failed_indexes,
                "total_created": len(created_indexes),
                "total_failed": len(failed_indexes),
                "success_rate": len(created_indexes) / (len(created_indexes) + len(failed_indexes)) * 100 if (len(created_indexes) + len(failed_indexes)) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error creating recommended indexes: {str(e)}")
            return {"error": "Failed to create recommended indexes"}
    
    def _create_user_indexes(self) -> Dict[str, List[str]]:
        """Create indexes for users table"""
        created = []
        failed = []
        
        try:
            # Email index (unique)
            self.db.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)"))
            created.append("idx_users_email")
            
            # Username index (unique)
            self.db.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)"))
            created.append("idx_users_username")
            
            # Active users index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)"))
            created.append("idx_users_active")
            
            # Admin users index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_users_admin ON users(is_admin)"))
            created.append("idx_users_admin")
            
            # Created at index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)"))
            created.append("idx_users_created_at")
            
            # Last login index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_users_last_login ON users(last_login)"))
            created.append("idx_users_last_login")
            
            # Composite index for active admin users
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_users_active_admin ON users(is_active, is_admin)"))
            created.append("idx_users_active_admin")
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error creating user indexes: {str(e)}")
            failed.append(f"user_indexes: {str(e)}")
            self.db.rollback()
        
        return {"created": created, "failed": failed}
    
    def _create_session_indexes(self) -> Dict[str, List[str]]:
        """Create indexes for user_sessions table"""
        created = []
        failed = []
        
        try:
            # User ID index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id)"))
            created.append("idx_sessions_user_id")
            
            # Session ID index (unique)
            self.db.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_session_id ON user_sessions(session_id)"))
            created.append("idx_sessions_session_id")
            
            # Active sessions index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_active ON user_sessions(is_active)"))
            created.append("idx_sessions_active")
            
            # Expires at index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON user_sessions(expires_at)"))
            created.append("idx_sessions_expires_at")
            
            # Created at index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON user_sessions(created_at)"))
            created.append("idx_sessions_created_at")
            
            # Composite index for active user sessions
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_user_active ON user_sessions(user_id, is_active)"))
            created.append("idx_sessions_user_active")
            
            # Composite index for active sessions by expiration
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_active_expires ON user_sessions(is_active, expires_at)"))
            created.append("idx_sessions_active_expires")
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error creating session indexes: {str(e)}")
            failed.append(f"session_indexes: {str(e)}")
            self.db.rollback()
        
        return {"created": created, "failed": failed}
    
    def _create_auth_log_indexes(self) -> Dict[str, List[str]]:
        """Create indexes for authentication_logs table"""
        created = []
        failed = []
        
        try:
            # User ID index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_auth_logs_user_id ON authentication_logs(user_id)"))
            created.append("idx_auth_logs_user_id")
            
            # Action index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_auth_logs_action ON authentication_logs(action)"))
            created.append("idx_auth_logs_action")
            
            # Success index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_auth_logs_success ON authentication_logs(success)"))
            created.append("idx_auth_logs_success")
            
            # Created at index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_auth_logs_created_at ON authentication_logs(created_at)"))
            created.append("idx_auth_logs_created_at")
            
            # IP address index
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_auth_logs_ip_address ON authentication_logs(ip_address)"))
            created.append("idx_auth_logs_ip_address")
            
            # Composite index for user action success
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_auth_logs_user_action_success ON authentication_logs(user_id, action, success)"))
            created.append("idx_auth_logs_user_action_success")
            
            # Composite index for success and created at
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_auth_logs_success_created ON authentication_logs(success, created_at)"))
            created.append("idx_auth_logs_success_created")
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error creating auth log indexes: {str(e)}")
            failed.append(f"auth_log_indexes: {str(e)}")
            self.db.rollback()
        
        return {"created": created, "failed": failed}
    
    def _create_csrf_indexes(self) -> Dict[str, List[str]]:
        """Create indexes for CSRF protection tables"""
        created = []
        failed = []
        
        try:
            # CSRF protection table indexes
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_csrf_user_id ON csrf_protection(user_id)"))
            created.append("idx_csrf_user_id")
            
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_csrf_token ON csrf_protection(token)"))
            created.append("idx_csrf_token")
            
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_csrf_token_hash ON csrf_protection(token_hash)"))
            created.append("idx_csrf_token_hash")
            
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_csrf_status ON csrf_protection(status)"))
            created.append("idx_csrf_status")
            
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_csrf_expires_at ON csrf_protection(expires_at)"))
            created.append("idx_csrf_expires_at")
            
            # CSRF violations table indexes
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_csrf_violations_user_id ON csrf_violations(user_id)"))
            created.append("idx_csrf_violations_user_id")
            
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_csrf_violations_ip_address ON csrf_violations(ip_address)"))
            created.append("idx_csrf_violations_ip_address")
            
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_csrf_violations_type ON csrf_violations(violation_type)"))
            created.append("idx_csrf_violations_type")
            
            self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_csrf_violations_created_at ON csrf_violations(created_at)"))
            created.append("idx_csrf_violations_created_at")
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error creating CSRF indexes: {str(e)}")
            failed.append(f"csrf_indexes: {str(e)}")
            self.db.rollback()
        
        return {"created": created, "failed": failed}
    
    def get_index_usage_stats(self) -> Dict[str, Any]:
        """Get index usage statistics (simplified implementation)"""
        try:
            # This is a simplified implementation
            # In a real system, this would query database statistics
            
            return {
                "total_indexes": self._count_total_indexes(),
                "used_indexes": self._count_used_indexes(),
                "unused_indexes": self._count_unused_indexes(),
                "duplicate_indexes": self._count_duplicate_indexes(),
                "recommendations": [
                    "Monitor index usage regularly",
                    "Remove unused indexes to improve write performance",
                    "Consider composite indexes for frequently used column combinations",
                    "Use partial indexes for filtered queries"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting index usage stats: {str(e)}")
            return {"error": "Failed to get index usage statistics"}
    
    def _count_total_indexes(self) -> int:
        """Count total indexes in the database"""
        try:
            tables = self.inspector.get_table_names()
            total = 0
            
            for table in tables:
                indexes = self.inspector.get_indexes(table)
                total += len(indexes)
            
            return total
            
        except Exception as e:
            logger.error(f"Error counting total indexes: {str(e)}")
            return 0
    
    def _count_used_indexes(self) -> int:
        """Count used indexes (simplified implementation)"""
        # In a real system, this would query database statistics
        return self._count_total_indexes() // 2  # Simplified estimate
    
    def _count_unused_indexes(self) -> int:
        """Count unused indexes (simplified implementation)"""
        # In a real system, this would query database statistics
        return self._count_total_indexes() // 4  # Simplified estimate
    
    def _count_duplicate_indexes(self) -> int:
        """Count duplicate indexes (simplified implementation)"""
        # In a real system, this would analyze index definitions
        return 0  # Simplified estimate
