"""
SupabaseIntegrationService for connecting existing services to Supabase
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import structlog
from sqlalchemy.orm import Session

from ..database.supabase_client import get_supabase_client, test_supabase_connection
from ..services.supabase_service import SupabaseService
from ..services.database_operation_service import DatabaseOperationService
from ..models.database_operation import OperationType, OperationStatus

logger = structlog.get_logger()

class SupabaseIntegrationService:
    """Service for integrating existing services with Supabase"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.supabase_service = SupabaseService()
        self.operation_service = DatabaseOperationService(db_session)
        self.client = get_supabase_client()
    
    def migrate_user_to_supabase(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate user data to Supabase
        
        Args:
            user_data: User data to migrate
            
        Returns:
            Dict containing migration result
        """
        try:
            logger.info("Migrating user to Supabase", user_id=user_data.get("id"))
            
            # Create operation record
            operation = self.operation_service.create_operation(
                operation_type=OperationType.CREATE,
                table_name="users",
                query_data=user_data,
                user_id=str(user_data.get("id"))
            )
            
            start_time = datetime.utcnow()
            
            try:
                # Prepare data for Supabase
                supabase_data = {
                    "id": user_data["id"],
                    "email": user_data["email"],
                    "first_name": user_data.get("first_name"),
                    "last_name": user_data.get("last_name"),
                    "role": user_data.get("role", "user"),
                    "subscription_tier": user_data.get("subscription_tier", "free"),
                    "is_active": user_data.get("is_active", True),
                    "is_verified": user_data.get("is_verified", False),
                    "last_login": user_data.get("last_login"),
                    "created_at": user_data.get("created_at"),
                    "updated_at": user_data.get("updated_at"),
                    "timezone": user_data.get("timezone", "UTC"),
                    "language": user_data.get("language", "en"),
                    "subscription_status": user_data.get("subscription_status", "active"),
                    "api_calls_count": user_data.get("api_calls_count", 0),
                    "last_api_call": user_data.get("last_api_call"),
                    "supabase_user_id": user_data.get("supabase_user_id"),
                    "profile_data": user_data.get("profile_data")
                }
                
                # Create user in Supabase
                result = self.supabase_service.create_record("users", supabase_data)
                
                # Calculate execution time
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Mark operation as successful
                self.operation_service.mark_operation_success(
                    str(operation.operation_id), 
                    execution_time
                )
                
                logger.info("User migrated to Supabase successfully", 
                           user_id=user_data.get("id"),
                           execution_time_ms=execution_time)
                
                return {
                    "success": True,
                    "data": result["data"],
                    "execution_time_ms": execution_time
                }
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Mark operation as failed
                self.operation_service.mark_operation_failed(
                    str(operation.operation_id),
                    str(e),
                    execution_time
                )
                
                logger.error("Failed to migrate user to Supabase", 
                            user_id=user_data.get("id"),
                            error=str(e),
                            execution_time_ms=execution_time)
                raise
                
        except Exception as e:
            logger.error("User migration failed", 
                        user_id=user_data.get("id"),
                        error=str(e))
            raise
    
    def migrate_trend_analysis_to_supabase(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate trend analysis data to Supabase
        
        Args:
            trend_data: Trend analysis data to migrate
            
        Returns:
            Dict containing migration result
        """
        try:
            logger.info("Migrating trend analysis to Supabase", 
                       analysis_id=trend_data.get("id"))
            
            # Create operation record
            operation = self.operation_service.create_operation(
                operation_type=OperationType.CREATE,
                table_name="trend_analyses",
                query_data=trend_data,
                user_id=str(trend_data.get("user_id"))
            )
            
            start_time = datetime.utcnow()
            
            try:
                # Prepare data for Supabase
                supabase_data = {
                    "id": trend_data["id"],
                    "user_id": trend_data["user_id"],
                    "workflow_session_id": trend_data.get("workflow_session_id"),
                    "topic_decomposition_id": trend_data.get("topic_decomposition_id"),
                    "analysis_name": trend_data["analysis_name"],
                    "description": trend_data.get("description"),
                    "keywords": trend_data.get("keywords", []),
                    "timeframe": trend_data.get("timeframe", "12m"),
                    "geo": trend_data.get("geo", "US"),
                    "category": trend_data.get("category"),
                    "trend_data": trend_data.get("trend_data", {}),
                    "analysis_results": trend_data.get("analysis_results", {}),
                    "insights": trend_data.get("insights", {}),
                    "source": trend_data.get("source", "google_trends"),
                    "status": trend_data.get("status", "pending"),
                    "error_message": trend_data.get("error_message"),
                    "processing_time_ms": trend_data.get("processing_time_ms"),
                    "api_calls_made": trend_data.get("api_calls_made", 0),
                    "cache_hit": trend_data.get("cache_hit", False),
                    "created_at": trend_data.get("created_at"),
                    "updated_at": trend_data.get("updated_at"),
                    "completed_at": trend_data.get("completed_at")
                }
                
                # Create trend analysis in Supabase
                result = self.supabase_service.create_record("trend_analyses", supabase_data)
                
                # Calculate execution time
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Mark operation as successful
                self.operation_service.mark_operation_success(
                    str(operation.operation_id), 
                    execution_time
                )
                
                logger.info("Trend analysis migrated to Supabase successfully", 
                           analysis_id=trend_data.get("id"),
                           execution_time_ms=execution_time)
                
                return {
                    "success": True,
                    "data": result["data"],
                    "execution_time_ms": execution_time
                }
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Mark operation as failed
                self.operation_service.mark_operation_failed(
                    str(operation.operation_id),
                    str(e),
                    execution_time
                )
                
                logger.error("Failed to migrate trend analysis to Supabase", 
                            analysis_id=trend_data.get("id"),
                            error=str(e),
                            execution_time_ms=execution_time)
                raise
                
        except Exception as e:
            logger.error("Trend analysis migration failed", 
                        analysis_id=trend_data.get("id"),
                        error=str(e))
            raise
    
    def sync_data_to_supabase(self, table_name: str, data: List[Dict[str, Any]], 
                             batch_size: int = 100) -> Dict[str, Any]:
        """
        Sync data to Supabase in batches
        
        Args:
            table_name: Name of the table
            data: List of data records
            batch_size: Batch size for processing
            
        Returns:
            Dict containing sync results
        """
        try:
            logger.info("Syncing data to Supabase", 
                       table=table_name, 
                       record_count=len(data),
                       batch_size=batch_size)
            
            # Create operation record
            operation = self.operation_service.create_operation(
                operation_type=OperationType.CREATE,
                table_name=table_name,
                query_data={"record_count": len(data), "batch_size": batch_size}
            )
            
            start_time = datetime.utcnow()
            successful_batches = 0
            failed_batches = 0
            total_records = 0
            
            try:
                # Process data in batches
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    
                    try:
                        # Insert batch into Supabase
                        result = self.supabase_service.create_record(table_name, batch)
                        successful_batches += 1
                        total_records += len(batch)
                        
                        logger.info("Batch synced successfully", 
                                   table=table_name,
                                   batch_number=i // batch_size + 1,
                                   batch_size=len(batch))
                        
                    except Exception as e:
                        failed_batches += 1
                        logger.error("Batch sync failed", 
                                   table=table_name,
                                   batch_number=i // batch_size + 1,
                                   error=str(e))
                
                # Calculate execution time
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Mark operation as successful if all batches succeeded
                if failed_batches == 0:
                    self.operation_service.mark_operation_success(
                        str(operation.operation_id), 
                        execution_time
                    )
                else:
                    self.operation_service.mark_operation_failed(
                        str(operation.operation_id),
                        f"Failed to sync {failed_batches} batches",
                        execution_time
                    )
                
                logger.info("Data sync to Supabase completed", 
                           table=table_name,
                           successful_batches=successful_batches,
                           failed_batches=failed_batches,
                           total_records=total_records,
                           execution_time_ms=execution_time)
                
                return {
                    "success": failed_batches == 0,
                    "successful_batches": successful_batches,
                    "failed_batches": failed_batches,
                    "total_records": total_records,
                    "execution_time_ms": execution_time
                }
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Mark operation as failed
                self.operation_service.mark_operation_failed(
                    str(operation.operation_id),
                    str(e),
                    execution_time
                )
                
                logger.error("Data sync to Supabase failed", 
                            table=table_name,
                            error=str(e),
                            execution_time_ms=execution_time)
                raise
                
        except Exception as e:
            logger.error("Data sync failed", 
                        table=table_name,
                        error=str(e))
            raise
    
    def get_supabase_health_status(self) -> Dict[str, Any]:
        """
        Get Supabase health status
        
        Returns:
            Dict containing health status
        """
        try:
            return self.supabase_service.get_health_status()
        except Exception as e:
            logger.error("Failed to get Supabase health status", error=str(e))
            return {
                "status": "unhealthy",
                "supabase_status": "error",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    def test_supabase_connection(self) -> Dict[str, Any]:
        """
        Test Supabase connection
        
        Returns:
            Dict containing test results
        """
        try:
            return test_supabase_connection()
        except Exception as e:
            logger.error("Supabase connection test failed", error=str(e))
            return {
                "status": "unhealthy",
                "supabase_status": "error",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    def get_migration_statistics(self) -> Dict[str, Any]:
        """
        Get migration statistics
        
        Returns:
            Dict containing migration statistics
        """
        try:
            # Get operation statistics
            stats = self.operation_service.get_operation_stats(hours=24)
            
            # Get Supabase health
            health = self.get_supabase_health_status()
            
            return {
                "migration_stats": stats,
                "supabase_health": health,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get migration statistics", error=str(e))
            return {
                "migration_stats": {
                    "total_operations": 0,
                    "successful_operations": 0,
                    "failed_operations": 0,
                    "success_rate": 0
                },
                "supabase_health": {
                    "status": "unknown",
                    "error": str(e)
                },
                "timestamp": datetime.utcnow().isoformat()
            }

