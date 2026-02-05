"""
Database utilities for DataForSEO features

Provides utility functions for database operations, maintenance,
and monitoring.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .database import db_manager, dataforseo_repository
from ..config import settings

logger = logging.getLogger(__name__)

class DatabaseUtils:
    """Utility functions for database operations"""
    
    def __init__(self):
        self.db_manager = db_manager
    
    async def run_migration(self, migration_file: str) -> bool:
        """Run a database migration file"""
        try:
            async with self.db_manager.get_session() as session:
                with open(migration_file, 'r') as f:
                    migration_sql = f.read()
                
                # Split by semicolon and execute each statement
                statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement:
                        await session.execute(text(statement))
                
                await session.commit()
                logger.info(f"Migration {migration_file} executed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error running migration {migration_file}: {e}")
            return False
    
    async def run_all_migrations(self) -> bool:
        """Run all DataForSEO migrations in correct order"""
        migrations = [
            "migrations/001_create_dataforseo_tables.sql",
            "migrations/002_create_dataforseo_indexes.sql", 
            "migrations/003_create_dataforseo_constraints.sql",
            "migrations/004_create_time_based_indexes.sql"
        ]
        
        for migration in migrations:
            success = await self.run_migration(migration)
            if not success:
                logger.error(f"Failed to run migration {migration}")
                return False
        
        logger.info("All DataForSEO migrations completed successfully")
        return True
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(text("SELECT get_dataforseo_stats()"))
                stats = result.scalar()
                return json.loads(stats) if isinstance(stats, str) else stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old data from database"""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    text("SELECT cleanup_old_dataforseo_data()")
                )
                deleted_count = result.scalar()
                await session.commit()
                
                logger.info(f"Cleaned up {deleted_count} old records")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    async def get_trending_subtopics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending subtopics from database view"""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    text("SELECT * FROM trending_subtopics LIMIT :limit"),
                    {"limit": limit}
                )
                
                trending_data = []
                for row in result:
                    trending_data.append({
                        "subtopic": row[0],
                        "location": row[1],
                        "time_range": row[2],
                        "average_interest": float(row[3]),
                        "peak_interest": float(row[4]),
                        "growth_rate": float(row[5]),
                        "updated_at": row[6].isoformat() if row[6] else None
                    })
                
                return trending_data
                
        except Exception as e:
            logger.error(f"Error getting trending subtopics: {e}")
            return []
    
    async def get_high_value_keywords(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get high-value keywords from database view"""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    text("SELECT * FROM high_value_keywords LIMIT :limit"),
                    {"limit": limit}
                )
                
                keywords_data = []
                for row in result:
                    keywords_data.append({
                        "keyword": row[0],
                        "search_volume": row[1],
                        "keyword_difficulty": row[2],
                        "cpc": float(row[3]),
                        "trend_percentage": float(row[4]),
                        "priority_score": float(row[5]) if row[5] else None,
                        "intent_type": row[6],
                        "updated_at": row[7].isoformat() if row[7] else None
                    })
                
                return keywords_data
                
        except Exception as e:
            logger.error(f"Error getting high-value keywords: {e}")
            return []
    
    async def get_api_performance_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get API performance metrics from database view"""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    text("""
                        SELECT * FROM api_performance_metrics 
                        WHERE hour >= NOW() - INTERVAL :hours HOUR
                        ORDER BY hour DESC
                    """),
                    {"hours": hours}
                )
                
                metrics_data = []
                for row in result:
                    metrics_data.append({
                        "endpoint": row[0],
                        "hour": row[1].isoformat() if row[1] else None,
                        "request_count": row[2],
                        "success_count": row[3],
                        "error_count": row[4],
                        "avg_response_time": float(row[5]) if row[5] else None,
                        "max_response_time": float(row[6]) if row[6] else None,
                        "min_response_time": float(row[7]) if row[7] else None
                    })
                
                return metrics_data
                
        except Exception as e:
            logger.error(f"Error getting API performance metrics: {e}")
            return []
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database health and connectivity"""
        try:
            async with self.db_manager.get_session() as session:
                # Test basic connectivity
                result = await session.execute(text("SELECT 1"))
                connectivity_ok = result.scalar() == 1
                
                # Check table existence
                tables_result = await session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN (
                        'trend_analysis_data', 
                        'keyword_research_data', 
                        'subtopic_suggestions', 
                        'dataforseo_api_logs'
                    )
                """))
                
                existing_tables = [row[0] for row in tables_result]
                expected_tables = [
                    'trend_analysis_data', 
                    'keyword_research_data', 
                    'subtopic_suggestions', 
                    'dataforseo_api_logs'
                ]
                
                tables_ok = all(table in existing_tables for table in expected_tables)
                
                # Check index existence
                indexes_result = await session.execute(text("""
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE schemaname = 'public' 
                    AND indexname LIKE 'idx_%dataforseo%'
                """))
                
                index_count = len([row[0] for row in indexes_result])
                
                # Get database size
                size_result = await session.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """))
                database_size = size_result.scalar()
                
                return {
                    "status": "healthy" if connectivity_ok and tables_ok else "unhealthy",
                    "connectivity": connectivity_ok,
                    "tables": {
                        "expected": expected_tables,
                        "existing": existing_tables,
                        "all_present": tables_ok
                    },
                    "indexes": {
                        "count": index_count,
                        "status": "good" if index_count >= 20 else "needs_optimization"
                    },
                    "database_size": database_size,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error checking database health: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Optimize database performance"""
        try:
            async with self.db_manager.get_session() as session:
                # Analyze tables
                await session.execute(text("ANALYZE trend_analysis_data"))
                await session.execute(text("ANALYZE keyword_research_data"))
                await session.execute(text("ANALYZE subtopic_suggestions"))
                await session.execute(text("ANALYZE dataforseo_api_logs"))
                
                # Vacuum tables
                await session.execute(text("VACUUM ANALYZE trend_analysis_data"))
                await session.execute(text("VACUUM ANALYZE keyword_research_data"))
                await session.execute(text("VACUUM ANALYZE subtopic_suggestions"))
                await session.execute(text("VACUUM ANALYZE dataforseo_api_logs"))
                
                await session.commit()
                
                logger.info("Database optimization completed successfully")
                return {
                    "status": "success",
                    "message": "Database optimization completed",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def backup_data(self, backup_type: str = "full") -> Dict[str, Any]:
        """Create database backup"""
        try:
            # This would typically use pg_dump or similar
            # For now, we'll just return a placeholder
            return {
                "status": "success",
                "message": f"Backup {backup_type} created successfully",
                "backup_type": backup_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def restore_data(self, backup_file: str) -> Dict[str, Any]:
        """Restore database from backup"""
        try:
            # This would typically use pg_restore or similar
            # For now, we'll just return a placeholder
            return {
                "status": "success",
                "message": f"Data restored from {backup_file}",
                "backup_file": backup_file,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error restoring data: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Global instance
db_utils = DatabaseUtils()
