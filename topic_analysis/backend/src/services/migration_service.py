"""
Database migration service for managing schema changes.
"""
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.core.database import get_engine, get_db_context
from src.core.config import settings

logger = logging.getLogger(__name__)

class MigrationService:
    """Service for managing database migrations and schema changes."""
    
    def __init__(self):
        self.engine = get_engine()
        self.migrations_table = "schema_migrations"
    
    def create_migrations_table(self) -> bool:
        """Create the migrations tracking table."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.migrations_table} (
                        id SERIAL PRIMARY KEY,
                        version VARCHAR(255) NOT NULL UNIQUE,
                        description TEXT,
                        applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        checksum VARCHAR(255),
                        execution_time_ms INTEGER
                    )
                """))
                connection.commit()
                logger.info("Migrations table created successfully")
                return True
        except SQLAlchemyError as e:
            logger.error(f"Error creating migrations table: {e}")
            return False
    
    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """Get list of applied migrations."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(f"""
                    SELECT version, description, applied_at, checksum, execution_time_ms
                    FROM {self.migrations_table}
                    ORDER BY applied_at
                """))
                
                return [
                    {
                        "version": row[0],
                        "description": row[1],
                        "applied_at": row[2],
                        "checksum": row[3],
                        "execution_time_ms": row[4]
                    }
                    for row in result.fetchall()
                ]
        except SQLAlchemyError as e:
            logger.error(f"Error getting applied migrations: {e}")
            return []
    
    def is_migration_applied(self, version: str) -> bool:
        """Check if a migration has been applied."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(f"""
                    SELECT COUNT(*) FROM {self.migrations_table} WHERE version = :version
                """), {"version": version})
                
                return result.fetchone()[0] > 0
        except SQLAlchemyError as e:
            logger.error(f"Error checking migration status: {e}")
            return False
    
    def record_migration(
        self, 
        version: str, 
        description: str, 
        checksum: str, 
        execution_time_ms: int
    ) -> bool:
        """Record a migration as applied."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text(f"""
                    INSERT INTO {self.migrations_table} (version, description, checksum, execution_time_ms)
                    VALUES (:version, :description, :checksum, :execution_time_ms)
                """), {
                    "version": version,
                    "description": description,
                    "checksum": checksum,
                    "execution_time_ms": execution_time_ms
                })
                connection.commit()
                return True
        except SQLAlchemyError as e:
            logger.error(f"Error recording migration: {e}")
            return False
    
    def get_database_schema(self) -> Dict[str, Any]:
        """Get current database schema information."""
        try:
            inspector = inspect(self.engine)
            
            schema_info = {
                "tables": {},
                "indexes": {},
                "constraints": {}
            }
            
            # Get table information
            for table_name in inspector.get_table_names():
                table_info = {
                    "columns": [],
                    "primary_keys": [],
                    "foreign_keys": [],
                    "indexes": []
                }
                
                # Get column information
                for column in inspector.get_columns(table_name):
                    table_info["columns"].append({
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column["nullable"],
                        "default": str(column["default"]) if column["default"] is not None else None,
                        "autoincrement": column.get("autoincrement", False)
                    })
                
                # Get primary key information
                pk_constraint = inspector.get_pk_constraint(table_name)
                if pk_constraint:
                    table_info["primary_keys"] = pk_constraint["constrained_columns"]
                
                # Get foreign key information
                for fk in inspector.get_foreign_keys(table_name):
                    table_info["foreign_keys"].append({
                        "columns": fk["constrained_columns"],
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"]
                    })
                
                # Get index information
                for index in inspector.get_indexes(table_name):
                    table_info["indexes"].append({
                        "name": index["name"],
                        "columns": index["column_names"],
                        "unique": index["unique"]
                    })
                
                schema_info["tables"][table_name] = table_info
            
            return schema_info
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting database schema: {e}")
            return {"error": str(e)}
    
    def create_initial_schema(self) -> bool:
        """Create the initial database schema."""
        try:
            start_time = datetime.now()
            
            # Create migrations table first
            if not self.create_migrations_table():
                return False
            
            # Create initial schema migration
            migration_version = "001_initial_schema"
            migration_description = "Create initial database schema with users, sessions, and auth tables"
            
            if self.is_migration_applied(migration_version):
                logger.info("Initial schema migration already applied")
                return True
            
            with self.engine.connect() as connection:
                # Create users table
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS users (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        first_name VARCHAR(50) NOT NULL,
                        last_name VARCHAR(50) NOT NULL,
                        role VARCHAR(20) NOT NULL DEFAULT 'user',
                        is_active BOOLEAN NOT NULL DEFAULT TRUE,
                        is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                        email_verification_token VARCHAR(255),
                        email_verification_expires TIMESTAMP WITH TIME ZONE,
                        last_login TIMESTAMP WITH TIME ZONE,
                        failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                        locked_until TIMESTAMP WITH TIME ZONE,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                    )
                """))
                
                # Create user_sessions table
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        token_jti VARCHAR(255) UNIQUE NOT NULL,
                        refresh_token VARCHAR(255) UNIQUE NOT NULL,
                        device_info JSONB,
                        ip_address VARCHAR(45),
                        user_agent VARCHAR(500),
                        is_active BOOLEAN NOT NULL DEFAULT TRUE,
                        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                        last_accessed TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                    )
                """))
                
                # Create password_resets table
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS password_resets (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        token VARCHAR(255) UNIQUE NOT NULL,
                        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        used_at TIMESTAMP WITH TIME ZONE,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                    )
                """))
                
                # Create authentication_logs table
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS authentication_logs (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                        event_type VARCHAR(50) NOT NULL,
                        ip_address VARCHAR(45),
                        user_agent VARCHAR(500),
                        success BOOLEAN NOT NULL,
                        failure_reason VARCHAR(255),
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                    )
                """))
                
                # Create indexes
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_token_jti ON user_sessions(token_jti)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_refresh_token ON user_sessions(refresh_token)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_password_resets_user_id ON password_resets(user_id)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_password_resets_token ON password_resets(token)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_password_resets_expires_at ON password_resets(expires_at)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_auth_logs_user_id ON authentication_logs(user_id)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_auth_logs_event_type ON authentication_logs(event_type)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS idx_auth_logs_created_at ON authentication_logs(created_at)"))
                
                connection.commit()
            
            # Record migration
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            self.record_migration(
                migration_version,
                migration_description,
                "initial_schema_checksum",
                execution_time
            )
            
            logger.info("Initial database schema created successfully")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating initial schema: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating initial schema: {e}")
            return False
    
    def run_migrations(self) -> bool:
        """Run all pending migrations."""
        try:
            # Ensure migrations table exists
            if not self.create_migrations_table():
                return False
            
            # Create initial schema if not exists
            if not self.create_initial_schema():
                return False
            
            # Get list of migration files (if using file-based migrations)
            # For now, we'll just ensure the initial schema is created
            logger.info("All migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running migrations: {e}")
            return False
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status information."""
        try:
            applied_migrations = self.get_applied_migrations()
            schema_info = self.get_database_schema()
            
            return {
                "migrations_applied": len(applied_migrations),
                "latest_migration": applied_migrations[-1] if applied_migrations else None,
                "tables_created": len(schema_info.get("tables", {})),
                "schema_info": schema_info
            }
            
        except Exception as e:
            logger.error(f"Error getting migration status: {e}")
            return {"error": str(e)}
    
    def validate_schema(self) -> bool:
        """Validate that the current schema is correct."""
        try:
            schema_info = self.get_database_schema()
            
            # Check required tables exist
            required_tables = ["users", "user_sessions", "password_resets", "authentication_logs"]
            existing_tables = set(schema_info.get("tables", {}).keys())
            
            missing_tables = set(required_tables) - existing_tables
            if missing_tables:
                logger.error(f"Missing required tables: {missing_tables}")
                return False
            
            # Check required columns exist in users table
            users_table = schema_info.get("tables", {}).get("users", {})
            required_columns = ["id", "email", "password_hash", "first_name", "last_name", "role"]
            existing_columns = {col["name"] for col in users_table.get("columns", [])}
            
            missing_columns = set(required_columns) - existing_columns
            if missing_columns:
                logger.error(f"Missing required columns in users table: {missing_columns}")
                return False
            
            logger.info("Schema validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating schema: {e}")
            return False

# Global migration service instance
migration_service = MigrationService()
