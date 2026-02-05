"""
MigrationService for data migration from PostgreSQL to Supabase
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

from ..database.migration import DatabaseMigration, MigrationType, MigrationStatus
from ..models.database_operation import DatabaseOperation, OperationType, OperationStatus
from src.core.supabase_database_service import SupabaseDatabaseService

logger = structlog.get_logger()

class MigrationService:
    """Service for managing database migrations"""
    
    def __init__(self, db_session: SupabaseDatabaseService):
        self.db = db_session
        self.migration_manager = DatabaseMigration()
    
    def start_migration(self, migration_type: MigrationType, tables: Optional[List[str]] = None, 
                       dry_run: bool = False) -> Dict[str, Any]:
        """
        Start a database migration
        
        Args:
            migration_type: Type of migration
            tables: Specific tables to migrate
            dry_run: Whether to perform a dry run
            
        Returns:
            Dict containing migration details
        """
        try:
            logger.info("Starting migration", 
                       type=migration_type.value, 
                       tables=tables, 
                       dry_run=dry_run)
            
            # Start migration
            migration_info = self.migration_manager.start_migration(
                migration_type=migration_type,
                tables=tables,
                dry_run=dry_run
            )
            
            # Log migration start
            self._log_migration_operation(
                operation_type=OperationType.CREATE,
                table_name="migrations",
                query_data=migration_info,
                status=OperationStatus.SUCCESS
            )
            
            logger.info("Migration started successfully", 
                       migration_id=migration_info["migration_id"])
            
            return migration_info
            
        except Exception as e:
            logger.error("Failed to start migration", 
                        type=migration_type.value,
                        error=str(e))
            raise
    
    def get_migration_status(self, migration_id: str) -> Optional[Dict[str, Any]]:
        """
        Get migration status
        
        Args:
            migration_id: Migration ID
            
        Returns:
            Dict containing migration status or None
        """
        try:
            status = self.migration_manager.get_migration_status(migration_id)
            
            if status:
                logger.info("Retrieved migration status", 
                           migration_id=migration_id,
                           status=status.get("status"))
            
            return status
            
        except Exception as e:
            logger.error("Failed to get migration status", 
                        migration_id=migration_id,
                        error=str(e))
            return None
    
    def execute_migration(self, migration_id: str) -> Dict[str, Any]:
        """
        Execute a migration
        
        Args:
            migration_id: Migration ID
            
        Returns:
            Dict containing migration results
        """
        try:
            logger.info("Executing migration", migration_id=migration_id)
            
            # Execute migration
            result = self.migration_manager.execute_migration(migration_id)
            
            # Log migration execution
            self._log_migration_operation(
                operation_type=OperationType.UPDATE,
                table_name="migrations",
                query_data={"migration_id": migration_id, "result": result},
                status=OperationStatus.SUCCESS if result["status"] == "completed" else OperationStatus.FAILED
            )
            
            logger.info("Migration executed", 
                       migration_id=migration_id,
                       status=result.get("status"))
            
            return result
            
        except Exception as e:
            logger.error("Failed to execute migration", 
                        migration_id=migration_id,
                        error=str(e))
            
            # Log migration failure
            self._log_migration_operation(
                operation_type=OperationType.UPDATE,
                table_name="migrations",
                query_data={"migration_id": migration_id, "error": str(e)},
                status=OperationStatus.FAILED
            )
            
            raise
    
    def rollback_migration(self, migration_id: str) -> Dict[str, Any]:
        """
        Rollback a migration
        
        Args:
            migration_id: Migration ID
            
        Returns:
            Dict containing rollback results
        """
        try:
            logger.info("Rolling back migration", migration_id=migration_id)
            
            # Rollback migration
            result = self.migration_manager.rollback_migration(migration_id)
            
            # Log migration rollback
            self._log_migration_operation(
                operation_type=OperationType.DELETE,
                table_name="migrations",
                query_data={"migration_id": migration_id, "result": result},
                status=OperationStatus.SUCCESS if result["status"] == "completed" else OperationStatus.FAILED
            )
            
            logger.info("Migration rolled back", 
                       migration_id=migration_id,
                       status=result.get("status"))
            
            return result
            
        except Exception as e:
            logger.error("Failed to rollback migration", 
                        migration_id=migration_id,
                        error=str(e))
            
            # Log rollback failure
            self._log_migration_operation(
                operation_type=OperationType.DELETE,
                table_name="migrations",
                query_data={"migration_id": migration_id, "error": str(e)},
                status=OperationStatus.FAILED
            )
            
            raise
    
    def get_migration_history(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get migration history
        
        Args:
            limit: Maximum number of migrations
            offset: Offset for pagination
            
        Returns:
            List of migration records
        """
        try:
            # Get migration operations from database
            operations = self.db.query(DatabaseOperation).filter(
                DatabaseOperation.table_name == "migrations",
                DatabaseOperation.operation_type == OperationType.CREATE
            ).order_by(DatabaseOperation.created_at.desc()).offset(offset).limit(limit).all()
            
            migrations = []
            for operation in operations:
                if operation.query_data:
                    migration_data = operation.query_data.copy()
                    migration_data["created_at"] = operation.created_at.isoformat()
                    migration_data["status"] = operation.status.value
                    migrations.append(migration_data)
            
            logger.info("Retrieved migration history", count=len(migrations))
            
            return migrations
            
        except Exception as e:
            logger.error("Failed to get migration history", error=str(e))
            return []
    
    def validate_migration_data(self, migration_id: str) -> Dict[str, Any]:
        """
        Validate migration data integrity
        
        Args:
            migration_id: Migration ID
            
        Returns:
            Dict containing validation results
        """
        try:
            logger.info("Validating migration data", migration_id=migration_id)
            
            # Get migration status
            status = self.get_migration_status(migration_id)
            if not status or status["status"] != "completed":
                return {
                    "valid": False,
                    "error": "Migration not completed",
                    "migration_id": migration_id
                }
            
            # Perform data validation checks
            validation_results = {
                "migration_id": migration_id,
                "valid": True,
                "checks": [],
                "errors": []
            }
            
            # Check if migration was dry run
            if status.get("dry_run", False):
                validation_results["checks"].append({
                    "check": "dry_run",
                    "status": "skipped",
                    "message": "Dry run migration - no data to validate"
                })
                return validation_results
            
            # Get tables that were migrated
            tables = status.get("tables", [])
            if not tables:
                # Get all tables if no specific tables were migrated
                tables = self._get_all_tables()
            
            # Validate each table
            for table_name in tables:
                table_validation = self._validate_table_data(table_name)
                validation_results["checks"].append(table_validation)
                
                if not table_validation["valid"]:
                    validation_results["valid"] = False
                    validation_results["errors"].extend(table_validation["errors"])
            
            logger.info("Migration data validation completed", 
                       migration_id=migration_id,
                       valid=validation_results["valid"])
            
            return validation_results
            
        except Exception as e:
            logger.error("Failed to validate migration data", 
                        migration_id=migration_id,
                        error=str(e))
            return {
                "valid": False,
                "error": str(e),
                "migration_id": migration_id
            }
    
    def _get_all_tables(self) -> List[str]:
        """Get all table names from PostgreSQL database"""
        try:
            result = self.db.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """))
            return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.error("Failed to get table list", error=str(e))
            return []
    
    def _validate_table_data(self, table_name: str) -> Dict[str, Any]:
        """
        Validate data for a specific table
        
        Args:
            table_name: Table name to validate
            
        Returns:
            Dict containing validation results
        """
        try:
            # Get row count from PostgreSQL
            pg_count_result = self.db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            pg_count = pg_count_result.scalar()
            
            # Note: In a real implementation, you would also check Supabase
            # For now, we'll just validate the PostgreSQL side
            
            validation = {
                "table": table_name,
                "valid": True,
                "checks": [],
                "errors": []
            }
            
            # Check if table has data
            if pg_count == 0:
                validation["checks"].append({
                    "check": "row_count",
                    "status": "warning",
                    "message": f"Table {table_name} is empty"
                })
            else:
                validation["checks"].append({
                    "check": "row_count",
                    "status": "pass",
                    "message": f"Table {table_name} has {pg_count} rows"
                })
            
            # Check table structure
            structure_check = self._validate_table_structure(table_name)
            validation["checks"].append(structure_check)
            
            if not structure_check["valid"]:
                validation["valid"] = False
                validation["errors"].extend(structure_check["errors"])
            
            return validation
            
        except Exception as e:
            logger.error("Failed to validate table data", 
                        table=table_name,
                        error=str(e))
            return {
                "table": table_name,
                "valid": False,
                "checks": [],
                "errors": [str(e)]
            }
    
    def _validate_table_structure(self, table_name: str) -> Dict[str, Any]:
        """
        Validate table structure
        
        Args:
            table_name: Table name to validate
            
        Returns:
            Dict containing structure validation results
        """
        try:
            # Get table columns
            result = self.db.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """), {"table_name": table_name})
            
            columns = result.fetchall()
            
            if not columns:
                return {
                    "check": "table_structure",
                    "valid": False,
                    "status": "fail",
                    "message": f"Table {table_name} not found",
                    "errors": [f"Table {table_name} does not exist"]
                }
            
            return {
                "check": "table_structure",
                "valid": True,
                "status": "pass",
                "message": f"Table {table_name} has {len(columns)} columns",
                "errors": []
            }
            
        except Exception as e:
            return {
                "check": "table_structure",
                "valid": False,
                "status": "fail",
                "message": f"Failed to validate table structure for {table_name}",
                "errors": [str(e)]
            }
    
    def _log_migration_operation(self, operation_type: OperationType, table_name: str, 
                                query_data: Dict[str, Any], status: OperationStatus):
        """
        Log migration operation
        
        Args:
            operation_type: Type of operation
            table_name: Table name
            query_data: Query data
            status: Operation status
        """
        try:
            operation = DatabaseOperation(
                operation_type=operation_type,
                table_name=table_name,
                query_data=query_data,
                status=status
            )
            
            self.db.add(operation)
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to log migration operation", error=str(e))
    
    def get_migration_statistics(self) -> Dict[str, Any]:
        """
        Get migration statistics
        
        Returns:
            Dict containing migration statistics
        """
        try:
            # Get total migrations
            total_migrations = self.db.query(DatabaseOperation).filter(
                DatabaseOperation.table_name == "migrations",
                DatabaseOperation.operation_type == OperationType.CREATE
            ).count()
            
            # Get successful migrations
            successful_migrations = self.db.query(DatabaseOperation).filter(
                DatabaseOperation.table_name == "migrations",
                DatabaseOperation.operation_type == OperationType.CREATE,
                DatabaseOperation.status == OperationStatus.SUCCESS
            ).count()
            
            # Get failed migrations
            failed_migrations = self.db.query(DatabaseOperation).filter(
                DatabaseOperation.table_name == "migrations",
                DatabaseOperation.operation_type == OperationType.CREATE,
                DatabaseOperation.status == OperationStatus.FAILED
            ).count()
            
            return {
                "total_migrations": total_migrations,
                "successful_migrations": successful_migrations,
                "failed_migrations": failed_migrations,
                "success_rate": (successful_migrations / total_migrations * 100) if total_migrations > 0 else 0
            }
            
        except Exception as e:
            logger.error("Failed to get migration statistics", error=str(e))
            return {
                "total_migrations": 0,
                "successful_migrations": 0,
                "failed_migrations": 0,
                "success_rate": 0
            }