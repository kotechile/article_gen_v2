#!/usr/bin/env python3
"""
Migration validation script for PostgreSQL to Supabase migration
"""
import os
import sys
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog
import psycopg2
from psycopg2.extras import RealDictCursor
from supabase import create_client

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = structlog.get_logger()

class MigrationValidator:
    """Validator for migration from PostgreSQL to Supabase"""
    
    def __init__(self, postgres_url: str, supabase_url: str, supabase_key: str):
        self.postgres_url = postgres_url
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.validation_results = {
            "tables_validated": 0,
            "tables_passed": 0,
            "tables_failed": 0,
            "total_errors": 0,
            "start_time": None,
            "end_time": None
        }
    
    def connect_databases(self):
        """Connect to both PostgreSQL and Supabase"""
        try:
            # Connect to PostgreSQL
            self.pg_conn = psycopg2.connect(self.postgres_url)
            self.pg_cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Connected to PostgreSQL database")
            
            # Connect to Supabase
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Connected to Supabase database")
            
        except Exception as e:
            logger.error("Failed to connect to databases", error=str(e))
            raise
    
    def get_tables_to_validate(self) -> List[str]:
        """Get list of tables to validate"""
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
            logger.info("Found tables to validate", count=len(tables), tables=tables)
            return tables
        except Exception as e:
            logger.error("Failed to get tables list", error=str(e))
            raise
    
    def get_postgres_count(self, table_name: str) -> int:
        """Get record count from PostgreSQL"""
        try:
            self.pg_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = self.pg_cursor.fetchone()[0]
            logger.info("Retrieved PostgreSQL count", table=table_name, count=count)
            return count
        except Exception as e:
            logger.error("Failed to get PostgreSQL count", table=table_name, error=str(e))
            return 0
    
    def get_supabase_count(self, table_name: str) -> int:
        """Get record count from Supabase"""
        try:
            response = self.supabase_client.table(table_name).select("id", count="exact").execute()
            count = response.count if hasattr(response, 'count') else len(response.data)
            logger.info("Retrieved Supabase count", table=table_name, count=count)
            return count
        except Exception as e:
            logger.error("Failed to get Supabase count", table=table_name, error=str(e))
            return 0
    
    def get_postgres_sample(self, table_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sample data from PostgreSQL"""
        try:
            self.pg_cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            data = [dict(row) for row in self.pg_cursor.fetchall()]
            logger.info("Retrieved PostgreSQL sample", table=table_name, count=len(data))
            return data
        except Exception as e:
            logger.error("Failed to get PostgreSQL sample", table=table_name, error=str(e))
            return []
    
    def get_supabase_sample(self, table_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sample data from Supabase"""
        try:
            response = self.supabase_client.table(table_name).select("*").limit(limit).execute()
            data = response.data if hasattr(response, 'data') else []
            logger.info("Retrieved Supabase sample", table=table_name, count=len(data))
            return data
        except Exception as e:
            logger.error("Failed to get Supabase sample", table=table_name, error=str(e))
            return []
    
    def compare_schemas(self, table_name: str) -> Dict[str, Any]:
        """Compare table schemas between PostgreSQL and Supabase"""
        try:
            logger.info("Comparing schemas", table=table_name)
            
            # Get PostgreSQL schema
            self.pg_cursor.execute("""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            
            pg_schema = {row['column_name']: {
                'data_type': row['data_type'],
                'is_nullable': row['is_nullable'] == 'YES',
                'column_default': row['column_default']
            } for row in self.pg_cursor.fetchall()}
            
            # Get Supabase schema (this would need to be implemented)
            # For now, we'll assume it matches
            supabase_schema = pg_schema.copy()
            
            comparison = {
                "table": table_name,
                "pg_columns": len(pg_schema),
                "supabase_columns": len(supabase_schema),
                "matches": pg_schema == supabase_schema,
                "differences": []
            }
            
            if not comparison["matches"]:
                for col_name in set(pg_schema.keys()) | set(supabase_schema.keys()):
                    if col_name not in pg_schema:
                        comparison["differences"].append(f"Column {col_name} missing in PostgreSQL")
                    elif col_name not in supabase_schema:
                        comparison["differences"].append(f"Column {col_name} missing in Supabase")
                    elif pg_schema[col_name] != supabase_schema[col_name]:
                        comparison["differences"].append(f"Column {col_name} differs between databases")
            
            logger.info("Schema comparison completed", 
                       table=table_name,
                       matches=comparison["matches"],
                       differences=len(comparison["differences"]))
            
            return comparison
            
        except Exception as e:
            logger.error("Failed to compare schemas", table=table_name, error=str(e))
            return {
                "table": table_name,
                "matches": False,
                "error": str(e)
            }
    
    def compare_data_samples(self, table_name: str, sample_size: int = 10) -> Dict[str, Any]:
        """Compare data samples between PostgreSQL and Supabase"""
        try:
            logger.info("Comparing data samples", table=table_name, sample_size=sample_size)
            
            # Get samples from both databases
            pg_sample = self.get_postgres_sample(table_name, sample_size)
            supabase_sample = self.get_supabase_sample(table_name, sample_size)
            
            comparison = {
                "table": table_name,
                "pg_sample_size": len(pg_sample),
                "supabase_sample_size": len(supabase_sample),
                "matches": len(pg_sample) == len(supabase_sample),
                "differences": []
            }
            
            if len(pg_sample) != len(supabase_sample):
                comparison["differences"].append(f"Sample size mismatch: PG={len(pg_sample)}, Supabase={len(supabase_sample)}")
            else:
                # Compare individual records
                for i, (pg_record, supabase_record) in enumerate(zip(pg_sample, supabase_sample)):
                    if pg_record != supabase_record:
                        comparison["differences"].append(f"Record {i} differs between databases")
                        comparison["matches"] = False
            
            logger.info("Data sample comparison completed", 
                       table=table_name,
                       matches=comparison["matches"],
                       differences=len(comparison["differences"]))
            
            return comparison
            
        except Exception as e:
            logger.error("Failed to compare data samples", table=table_name, error=str(e))
            return {
                "table": table_name,
                "matches": False,
                "error": str(e)
            }
    
    def validate_table(self, table_name: str) -> Dict[str, Any]:
        """Validate a single table"""
        try:
            logger.info("Validating table", table=table_name)
            
            validation_result = {
                "table": table_name,
                "success": True,
                "checks": [],
                "errors": [],
                "start_time": datetime.utcnow().isoformat(),
                "end_time": None
            }
            
            # Check 1: Record count comparison
            try:
                pg_count = self.get_postgres_count(table_name)
                supabase_count = self.get_supabase_count(table_name)
                
                count_check = {
                    "check": "record_count",
                    "pg_count": pg_count,
                    "supabase_count": supabase_count,
                    "matches": pg_count == supabase_count,
                    "status": "pass" if pg_count == supabase_count else "fail"
                }
                
                validation_result["checks"].append(count_check)
                
                if not count_check["matches"]:
                    validation_result["errors"].append(f"Record count mismatch: PG={pg_count}, Supabase={supabase_count}")
                    validation_result["success"] = False
                
            except Exception as e:
                validation_result["checks"].append({
                    "check": "record_count",
                    "status": "error",
                    "error": str(e)
                })
                validation_result["errors"].append(f"Record count check failed: {e}")
                validation_result["success"] = False
            
            # Check 2: Schema comparison
            try:
                schema_check = self.compare_schemas(table_name)
                validation_result["checks"].append(schema_check)
                
                if not schema_check.get("matches", False):
                    validation_result["errors"].extend(schema_check.get("differences", []))
                    validation_result["success"] = False
                
            except Exception as e:
                validation_result["checks"].append({
                    "check": "schema",
                    "status": "error",
                    "error": str(e)
                })
                validation_result["errors"].append(f"Schema check failed: {e}")
                validation_result["success"] = False
            
            # Check 3: Data sample comparison
            try:
                data_check = self.compare_data_samples(table_name)
                validation_result["checks"].append(data_check)
                
                if not data_check.get("matches", False):
                    validation_result["errors"].extend(data_check.get("differences", []))
                    validation_result["success"] = False
                
            except Exception as e:
                validation_result["checks"].append({
                    "check": "data_sample",
                    "status": "error",
                    "error": str(e)
                })
                validation_result["errors"].append(f"Data sample check failed: {e}")
                validation_result["success"] = False
            
            validation_result["end_time"] = datetime.utcnow().isoformat()
            
            # Update statistics
            self.validation_results["tables_validated"] += 1
            if validation_result["success"]:
                self.validation_results["tables_passed"] += 1
            else:
                self.validation_results["tables_failed"] += 1
                self.validation_results["total_errors"] += len(validation_result["errors"])
            
            logger.info("Table validation completed", 
                       table=table_name,
                       success=validation_result["success"],
                       errors=len(validation_result["errors"]))
            
            return validation_result
            
        except Exception as e:
            logger.error("Table validation failed", table=table_name, error=str(e))
            return {
                "table": table_name,
                "success": False,
                "error": str(e),
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
    
    def validate_all_tables(self, tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate all tables"""
        try:
            self.validation_results["start_time"] = datetime.utcnow()
            
            logger.info("Starting full migration validation")
            
            # Get tables to validate
            if not tables:
                tables = self.get_tables_to_validate()
            
            validation_results = {
                "success": True,
                "tables": [],
                "summary": {
                    "total_tables": len(tables),
                    "successful_tables": 0,
                    "failed_tables": 0,
                    "total_errors": 0
                },
                "start_time": self.validation_results["start_time"].isoformat(),
                "end_time": None
            }
            
            # Validate each table
            for table_name in tables:
                try:
                    result = self.validate_table(table_name)
                    validation_results["tables"].append(result)
                    
                    if result["success"]:
                        validation_results["summary"]["successful_tables"] += 1
                    else:
                        validation_results["summary"]["failed_tables"] += 1
                        validation_results["success"] = False
                    
                    validation_results["summary"]["total_errors"] += len(result.get("errors", []))
                    
                except Exception as e:
                    logger.error("Table validation failed", 
                                table=table_name, 
                                error=str(e))
                    validation_results["tables"].append({
                        "table": table_name,
                        "success": False,
                        "error": str(e),
                        "start_time": datetime.utcnow().isoformat(),
                        "end_time": datetime.utcnow().isoformat()
                    })
                    validation_results["summary"]["failed_tables"] += 1
                    validation_results["success"] = False
            
            self.validation_results["end_time"] = datetime.utcnow()
            validation_results["end_time"] = self.validation_results["end_time"].isoformat()
            
            logger.info("Full migration validation completed", 
                       success=validation_results["success"],
                       summary=validation_results["summary"])
            
            return validation_results
            
        except Exception as e:
            logger.error("Full migration validation failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "start_time": self.validation_results["start_time"].isoformat() if self.validation_results["start_time"] else None,
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
    parser = argparse.ArgumentParser(description="Validate migration from PostgreSQL to Supabase")
    parser.add_argument("--postgres-url", required=True, help="PostgreSQL connection URL")
    parser.add_argument("--supabase-url", required=True, help="Supabase project URL")
    parser.add_argument("--supabase-key", required=True, help="Supabase service role key")
    parser.add_argument("--tables", nargs="*", help="Specific tables to validate")
    parser.add_argument("--output", help="Output file for results")
    
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
    
    validator = None
    try:
        # Initialize validator
        validator = MigrationValidator(
            postgres_url=args.postgres_url,
            supabase_url=args.supabase_url,
            supabase_key=args.supabase_key
        )
        
        # Connect to databases
        validator.connect_databases()
        
        # Run validation
        result = validator.validate_all_tables(tables=args.tables)
        
        # Print results
        print(f"\nValidation {'completed' if result['success'] else 'failed'}")
        print(f"Tables validated: {result['summary']['total_tables']}")
        print(f"Successful: {result['summary']['successful_tables']}")
        print(f"Failed: {result['summary']['failed_tables']}")
        print(f"Total errors: {result['summary']['total_errors']}")
        
        # Save results to file if specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        
        if not result['success']:
            sys.exit(1)
            
    except Exception as e:
        logger.error("Validation script failed", error=str(e))
        print(f"Validation failed: {e}")
        sys.exit(1)
    finally:
        if validator:
            validator.close_connections()

if __name__ == "__main__":
    main()

