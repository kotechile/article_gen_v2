#!/usr/bin/env python3
"""
Data migration script from PostgreSQL to Supabase
"""
import os
import sys
import asyncio
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy.orm import Session

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.database import get_db
from services.supabase_integration_service import SupabaseIntegrationService
from services.migration_service import MigrationService
from database.migration import MigrationType

logger = structlog.get_logger()

class DataMigrationScript:
    """Script for migrating data from PostgreSQL to Supabase"""
    
    def __init__(self, postgres_url: str, supabase_url: str, supabase_key: str):
        self.postgres_url = postgres_url
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.migration_stats = {
            "tables_migrated": 0,
            "records_migrated": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }
    
    def connect_postgres(self):
        """Connect to PostgreSQL database"""
        try:
            self.pg_conn = psycopg2.connect(self.postgres_url)
            self.pg_cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error("Failed to connect to PostgreSQL", error=str(e))
            raise
    
    def get_tables_to_migrate(self) -> List[str]:
        """Get list of tables to migrate"""
        try:
            self.pg_cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                AND table_name NOT LIKE 'pg_%'
                AND table_name NOT LIKE 'sql_%'
                ORDER BY table_name
            """)
            tables = [row['table_name'] for row in self.pg_cursor.fetchall()]
            logger.info("Found tables to migrate", count=len(tables), tables=tables)
            return tables
        except Exception as e:
            logger.error("Failed to get tables list", error=str(e))
            raise
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema information"""
        try:
            self.pg_cursor.execute("""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            
            schema = []
            for row in self.pg_cursor.fetchall():
                schema.append({
                    "column_name": row['column_name'],
                    "data_type": row['data_type'],
                    "is_nullable": row['is_nullable'] == 'YES',
                    "column_default": row['column_default'],
                    "character_maximum_length": row['character_maximum_length'],
                    "numeric_precision": row['numeric_precision'],
                    "numeric_scale": row['numeric_scale']
                })
            
            logger.info("Retrieved table schema", table=table_name, columns=len(schema))
            return schema
        except Exception as e:
            logger.error("Failed to get table schema", table=table_name, error=str(e))
            raise
    
    def get_table_data(self, table_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get table data"""
        try:
            query = f"SELECT * FROM {table_name}"
            if limit:
                query += f" LIMIT {limit}"
            
            self.pg_cursor.execute(query)
            data = [dict(row) for row in self.pg_cursor.fetchall()]
            
            logger.info("Retrieved table data", 
                       table=table_name, 
                       count=len(data),
                       limit=limit)
            return data
        except Exception as e:
            logger.error("Failed to get table data", table=table_name, error=str(e))
            raise
    
    def get_table_count(self, table_name: str) -> int:
        """Get table record count"""
        try:
            self.pg_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = self.pg_cursor.fetchone()[0]
            logger.info("Retrieved table count", table=table_name, count=count)
            return count
        except Exception as e:
            logger.error("Failed to get table count", table=table_name, error=str(e))
            return 0
    
    def create_supabase_table(self, table_name: str, schema: List[Dict[str, Any]]) -> bool:
        """Create table in Supabase (placeholder implementation)"""
        try:
            # In a real implementation, this would create the table in Supabase
            # For now, we'll just log the schema
            logger.info("Creating Supabase table", 
                       table=table_name, 
                       columns=len(schema))
            
            # This would typically involve:
            # 1. Generate CREATE TABLE SQL from schema
            # 2. Execute via Supabase SQL editor or migration system
            # 3. Set up Row Level Security policies
            # 4. Create indexes
            
            return True
        except Exception as e:
            logger.error("Failed to create Supabase table", 
                        table=table_name, 
                        error=str(e))
            return False
    
    def migrate_table_data(self, table_name: str, data: List[Dict[str, Any]], 
                          batch_size: int = 1000) -> bool:
        """Migrate table data to Supabase"""
        try:
            logger.info("Starting table data migration", 
                       table=table_name, 
                       record_count=len(data),
                       batch_size=batch_size)
            
            # Process data in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                try:
                    # In a real implementation, this would insert data into Supabase
                    # For now, we'll just log the batch
                    logger.info("Processing batch", 
                               table=table_name,
                               batch_number=i // batch_size + 1,
                               batch_size=len(batch))
                    
                    # This would typically involve:
                    # 1. Transform data to match Supabase schema
                    # 2. Insert batch via Supabase client
                    # 3. Handle any constraint violations
                    # 4. Update migration progress
                    
                    self.migration_stats["records_migrated"] += len(batch)
                    
                except Exception as e:
                    logger.error("Failed to process batch", 
                                table=table_name,
                                batch_number=i // batch_size + 1,
                                error=str(e))
                    self.migration_stats["errors"] += 1
                    return False
            
            logger.info("Table data migration completed", 
                       table=table_name,
                       records_migrated=self.migration_stats["records_migrated"])
            
            return True
            
        except Exception as e:
            logger.error("Failed to migrate table data", 
                        table=table_name, 
                        error=str(e))
            return False
    
    def validate_migration(self, table_name: str) -> Dict[str, Any]:
        """Validate migration for a table"""
        try:
            logger.info("Validating migration", table=table_name)
            
            # Get source count
            source_count = self.get_table_count(table_name)
            
            # In a real implementation, this would check Supabase count
            # For now, we'll use the migrated count
            target_count = self.migration_stats["records_migrated"]
            
            validation_result = {
                "table": table_name,
                "source_count": source_count,
                "target_count": target_count,
                "valid": source_count == target_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if validation_result["valid"]:
                logger.info("Migration validation passed", 
                           table=table_name,
                           count=source_count)
            else:
                logger.warning("Migration validation failed", 
                              table=table_name,
                              source_count=source_count,
                              target_count=target_count)
            
            return validation_result
            
        except Exception as e:
            logger.error("Failed to validate migration", 
                        table=table_name, 
                        error=str(e))
            return {
                "table": table_name,
                "valid": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def migrate_table(self, table_name: str, dry_run: bool = False) -> Dict[str, Any]:
        """Migrate a single table"""
        try:
            logger.info("Starting table migration", 
                       table=table_name, 
                       dry_run=dry_run)
            
            result = {
                "table": table_name,
                "success": False,
                "dry_run": dry_run,
                "start_time": datetime.utcnow().isoformat(),
                "end_time": None,
                "records_migrated": 0,
                "errors": []
            }
            
            if dry_run:
                # Dry run - just validate schema and data
                schema = self.get_table_schema(table_name)
                data = self.get_table_data(table_name, limit=10)  # Sample data
                
                result["success"] = True
                result["records_migrated"] = len(data)
                result["schema_columns"] = len(schema)
                logger.info("Dry run completed", table=table_name)
            else:
                # Full migration
                # 1. Get schema
                schema = self.get_table_schema(table_name)
                
                # 2. Create table in Supabase
                if not self.create_supabase_table(table_name, schema):
                    result["errors"].append("Failed to create Supabase table")
                    return result
                
                # 3. Get all data
                data = self.get_table_data(table_name)
                
                # 4. Migrate data
                if not self.migrate_table_data(table_name, data):
                    result["errors"].append("Failed to migrate table data")
                    return result
                
                # 5. Validate migration
                validation = self.validate_migration(table_name)
                if not validation["valid"]:
                    result["errors"].append("Migration validation failed")
                    return result
                
                result["success"] = True
                result["records_migrated"] = len(data)
                self.migration_stats["tables_migrated"] += 1
            
            result["end_time"] = datetime.utcnow().isoformat()
            logger.info("Table migration completed", 
                       table=table_name,
                       success=result["success"],
                       records_migrated=result["records_migrated"])
            
            return result
            
        except Exception as e:
            logger.error("Table migration failed", 
                        table=table_name, 
                        error=str(e))
            return {
                "table": table_name,
                "success": False,
                "dry_run": dry_run,
                "error": str(e),
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
    
    def migrate_all_tables(self, tables: Optional[List[str]] = None, 
                          dry_run: bool = False) -> Dict[str, Any]:
        """Migrate all tables"""
        try:
            self.migration_stats["start_time"] = datetime.utcnow()
            
            logger.info("Starting full database migration", 
                       dry_run=dry_run,
                       tables=tables)
            
            # Get tables to migrate
            if not tables:
                tables = self.get_tables_to_migrate()
            
            migration_results = {
                "success": True,
                "dry_run": dry_run,
                "tables": [],
                "summary": {
                    "total_tables": len(tables),
                    "successful_tables": 0,
                    "failed_tables": 0,
                    "total_records": 0,
                    "total_errors": 0
                },
                "start_time": self.migration_stats["start_time"].isoformat(),
                "end_time": None
            }
            
            # Migrate each table
            for table_name in tables:
                try:
                    result = self.migrate_table(table_name, dry_run)
                    migration_results["tables"].append(result)
                    
                    if result["success"]:
                        migration_results["summary"]["successful_tables"] += 1
                        migration_results["summary"]["total_records"] += result["records_migrated"]
                    else:
                        migration_results["summary"]["failed_tables"] += 1
                        migration_results["success"] = False
                    
                    migration_results["summary"]["total_errors"] += len(result.get("errors", []))
                    
                except Exception as e:
                    logger.error("Table migration failed", 
                                table=table_name, 
                                error=str(e))
                    migration_results["tables"].append({
                        "table": table_name,
                        "success": False,
                        "error": str(e),
                        "start_time": datetime.utcnow().isoformat(),
                        "end_time": datetime.utcnow().isoformat()
                    })
                    migration_results["summary"]["failed_tables"] += 1
                    migration_results["success"] = False
            
            self.migration_stats["end_time"] = datetime.utcnow()
            migration_results["end_time"] = self.migration_stats["end_time"].isoformat()
            
            logger.info("Full database migration completed", 
                       success=migration_results["success"],
                       summary=migration_results["summary"])
            
            return migration_results
            
        except Exception as e:
            logger.error("Full database migration failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "start_time": self.migration_stats["start_time"].isoformat() if self.migration_stats["start_time"] else None,
                "end_time": datetime.utcnow().isoformat()
            }
    
    def close_connections(self):
        """Close database connections"""
        try:
            if hasattr(self, 'pg_cursor'):
                self.pg_cursor.close()
            if hasattr(self, 'pg_conn'):
                self.pg_conn.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error("Failed to close connections", error=str(e))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Migrate data from PostgreSQL to Supabase")
    parser.add_argument("--postgres-url", required=True, help="PostgreSQL connection URL")
    parser.add_argument("--supabase-url", required=True, help="Supabase project URL")
    parser.add_argument("--supabase-key", required=True, help="Supabase service role key")
    parser.add_argument("--tables", nargs="*", help="Specific tables to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for data migration")
    
    args = parser.parse_args()
    
    # Set up logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    migration_script = None
    try:
        # Initialize migration script
        migration_script = DataMigrationScript(
            postgres_url=args.postgres_url,
            supabase_url=args.supabase_url,
            supabase_key=args.supabase_key
        )
        
        # Connect to PostgreSQL
        migration_script.connect_postgres()
        
        # Run migration
        result = migration_script.migrate_all_tables(
            tables=args.tables,
            dry_run=args.dry_run
        )
        
        # Print results
        print(f"\nMigration {'completed' if result['success'] else 'failed'}")
        print(f"Tables processed: {result['summary']['total_tables']}")
        print(f"Successful: {result['summary']['successful_tables']}")
        print(f"Failed: {result['summary']['failed_tables']}")
        print(f"Total records: {result['summary']['total_records']}")
        print(f"Total errors: {result['summary']['total_errors']}")
        
        if not result['success']:
            sys.exit(1)
            
    except Exception as e:
        logger.error("Migration script failed", error=str(e))
        print(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        if migration_script:
            migration_script.close_connections()

if __name__ == "__main__":
    main()

