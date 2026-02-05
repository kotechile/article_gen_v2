"""
Database migration utilities for Supabase integration
"""
import os
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import structlog
import psycopg2
from psycopg2.extras import RealDictCursor
from supabase import create_client, Client

logger = structlog.get_logger()

class MigrationType(Enum):
    """Migration type enumeration"""
    FULL = "full"
    INCREMENTAL = "incremental"
    ROLLBACK = "rollback"

class MigrationStatus(Enum):
    """Migration status enumeration"""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DatabaseMigration:
    """Database migration manager for PostgreSQL to Supabase"""
    
    def __init__(self):
        self.postgres_url = os.getenv("DATABASE_URL")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not all([self.postgres_url, self.supabase_url, self.supabase_key]):
            raise ValueError("Missing required environment variables for migration")
        
        self.supabase_client = create_client(self.supabase_url, self.supabase_key)
        self.migrations = {}
    
    def start_migration(self, migration_type: MigrationType, tables: Optional[List[str]] = None, dry_run: bool = False) -> Dict[str, Any]:
        """
        Start a database migration
        
        Args:
            migration_type: Type of migration to perform
            tables: Specific tables to migrate (None for all)
            dry_run: Whether to perform a dry run without making changes
            
        Returns:
            Dict containing migration details
        """
        migration_id = f"migration_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        migration_info = {
            "migration_id": migration_id,
            "migration_type": migration_type.value,
            "tables": tables or [],
            "dry_run": dry_run,
            "status": MigrationStatus.STARTED.value,
            "started_at": datetime.utcnow().isoformat(),
            "progress_percentage": 0,
            "current_table": None,
            "error_message": None
        }
        
        self.migrations[migration_id] = migration_info
        logger.info("Migration started", migration_id=migration_id, type=migration_type.value)
        
        return migration_info
    
    def get_migration_status(self, migration_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a migration
        
        Args:
            migration_id: Migration identifier
            
        Returns:
            Dict containing migration status or None if not found
        """
        return self.migrations.get(migration_id)
    
    def execute_migration(self, migration_id: str) -> Dict[str, Any]:
        """
        Execute the migration (synchronous version)
        
        Args:
            migration_id: Migration identifier
            
        Returns:
            Dict containing migration results
        """
        if migration_id not in self.migrations:
            raise ValueError(f"Migration {migration_id} not found")
        
        migration = self.migrations[migration_id]
        migration["status"] = MigrationStatus.IN_PROGRESS.value
        
        try:
            # Get tables to migrate
            tables_to_migrate = migration["tables"]
            if not tables_to_migrate:
                tables_to_migrate = self._get_all_tables()
            
            total_tables = len(tables_to_migrate)
            
            for i, table_name in enumerate(tables_to_migrate):
                migration["current_table"] = table_name
                migration["progress_percentage"] = int((i / total_tables) * 100)
                
                logger.info("Migrating table", table=table_name, progress=migration["progress_percentage"])
                
                if not migration["dry_run"]:
                    self._migrate_table(table_name)
                
                # Update progress
                migration["progress_percentage"] = int(((i + 1) / total_tables) * 100)
            
            migration["status"] = MigrationStatus.COMPLETED.value
            migration["completed_at"] = datetime.utcnow().isoformat()
            
            logger.info("Migration completed", migration_id=migration_id)
            return migration
            
        except Exception as e:
            migration["status"] = MigrationStatus.FAILED.value
            migration["error_message"] = str(e)
            migration["completed_at"] = datetime.utcnow().isoformat()
            
            logger.error("Migration failed", migration_id=migration_id, error=str(e))
            return migration
    
    def _get_all_tables(self) -> List[str]:
        """Get all table names from PostgreSQL database"""
        try:
            with psycopg2.connect(self.postgres_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """)
                    return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Failed to get table list", error=str(e))
            raise
    
    def _migrate_table(self, table_name: str):
        """Migrate a single table from PostgreSQL to Supabase"""
        try:
            # Get table schema
            schema = self._get_table_schema(table_name)
            
            # Get table data
            data = self._get_table_data(table_name)
            
            # Create table in Supabase (if not exists)
            self._create_supabase_table(table_name, schema)
            
            # Insert data into Supabase
            if data:
                self._insert_supabase_data(table_name, data)
            
            logger.info("Table migrated successfully", table=table_name)
            
        except Exception as e:
            logger.error("Failed to migrate table", table=table_name, error=str(e))
            raise
    
    def _get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema from PostgreSQL"""
        try:
            with psycopg2.connect(self.postgres_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_name = %s
                        ORDER BY ordinal_position
                    """, (table_name,))
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Failed to get table schema", table=table_name, error=str(e))
            raise
    
    def _get_table_data(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table data from PostgreSQL"""
        try:
            with psycopg2.connect(self.postgres_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(f"SELECT * FROM {table_name}")
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Failed to get table data", table=table_name, error=str(e))
            raise
    
    def _create_supabase_table(self, table_name: str, schema: List[Dict[str, Any]]):
        """Create table in Supabase (placeholder - would need actual SQL execution)"""
        # This is a placeholder - in a real implementation, you would:
        # 1. Generate CREATE TABLE SQL from schema
        # 2. Execute it via Supabase SQL editor or migration system
        logger.info("Creating Supabase table", table=table_name, columns=len(schema))
    
    def _insert_supabase_data(self, table_name: str, data: List[Dict[str, Any]]):
        """Insert data into Supabase table"""
        try:
            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                response = self.supabase_client.table(table_name).insert(batch).execute()
                logger.info("Inserted batch", table=table_name, batch_size=len(batch))
        except Exception as e:
            logger.error("Failed to insert data", table=table_name, error=str(e))
            raise
    
    def rollback_migration(self, migration_id: str) -> Dict[str, Any]:
        """
        Rollback a migration
        
        Args:
            migration_id: Migration identifier
            
        Returns:
            Dict containing rollback results
        """
        if migration_id not in self.migrations:
            raise ValueError(f"Migration {migration_id} not found")
        
        migration = self.migrations[migration_id]
        migration["status"] = MigrationStatus.IN_PROGRESS.value
        
        try:
            # Rollback logic would go here
            # This is a placeholder implementation
            logger.info("Rolling back migration", migration_id=migration_id)
            
            migration["status"] = MigrationStatus.COMPLETED.value
            migration["completed_at"] = datetime.utcnow().isoformat()
            
            return migration
            
        except Exception as e:
            migration["status"] = MigrationStatus.FAILED.value
            migration["error_message"] = str(e)
            migration["completed_at"] = datetime.utcnow().isoformat()
            
            logger.error("Migration rollback failed", migration_id=migration_id, error=str(e))
            return migration

# Global migration manager instance
migration_manager = DatabaseMigration()

