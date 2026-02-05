#!/usr/bin/env python3
"""
Migration monitoring script for tracking migration progress
"""
import os
import sys
import argparse
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import structlog
import json

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.database import get_db
from services.migration_service import MigrationService
from services.database_operation_service import DatabaseOperationService

logger = structlog.get_logger()

class MigrationMonitor:
    """Monitor for tracking migration progress"""
    
    def __init__(self):
        self.db = next(get_db())
        self.migration_service = MigrationService(self.db)
        self.operation_service = DatabaseOperationService(self.db)
        self.monitoring_stats = {
            "start_time": None,
            "last_check": None,
            "total_checks": 0,
            "migrations_found": 0,
            "active_migrations": 0,
            "completed_migrations": 0,
            "failed_migrations": 0
        }
    
    def get_migration_status(self, migration_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific migration"""
        try:
            return self.migration_service.get_migration_status(migration_id)
        except Exception as e:
            logger.error("Failed to get migration status", 
                        migration_id=migration_id,
                        error=str(e))
            return None
    
    def get_all_migrations(self) -> List[Dict[str, Any]]:
        """Get all migrations"""
        try:
            return self.migration_service.get_migration_history(limit=100)
        except Exception as e:
            logger.error("Failed to get migration history", error=str(e))
            return []
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get database operation statistics"""
        try:
            return self.operation_service.get_operation_stats(hours=24)
        except Exception as e:
            logger.error("Failed to get operation statistics", error=str(e))
            return {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "success_rate": 0
            }
    
    def get_migration_statistics(self) -> Dict[str, Any]:
        """Get migration statistics"""
        try:
            return self.migration_service.get_migration_statistics()
        except Exception as e:
            logger.error("Failed to get migration statistics", error=str(e))
            return {
                "total_migrations": 0,
                "successful_migrations": 0,
                "failed_migrations": 0,
                "success_rate": 0
            }
    
    def check_migration_health(self) -> Dict[str, Any]:
        """Check overall migration health"""
        try:
            logger.info("Checking migration health")
            
            # Get all migrations
            migrations = self.get_all_migrations()
            
            # Count migrations by status
            status_counts = {
                "started": 0,
                "in_progress": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0
            }
            
            active_migrations = []
            recent_migrations = []
            
            for migration in migrations:
                status = migration.get("status", "unknown")
                if status in status_counts:
                    status_counts[status] += 1
                
                # Track active migrations
                if status in ["started", "in_progress"]:
                    active_migrations.append(migration)
                
                # Track recent migrations (last 24 hours)
                created_at = migration.get("created_at")
                if created_at:
                    try:
                        created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        if created_time > datetime.utcnow() - timedelta(hours=24):
                            recent_migrations.append(migration)
                    except:
                        pass
            
            # Get operation statistics
            operation_stats = self.get_operation_statistics()
            
            # Get migration statistics
            migration_stats = self.get_migration_statistics()
            
            health_status = {
                "overall_status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "migrations": {
                    "total": len(migrations),
                    "status_counts": status_counts,
                    "active_count": len(active_migrations),
                    "recent_count": len(recent_migrations)
                },
                "operations": operation_stats,
                "statistics": migration_stats,
                "active_migrations": active_migrations[:5],  # Limit to 5 most recent
                "recent_migrations": recent_migrations[:10]  # Limit to 10 most recent
            }
            
            # Determine overall health
            if status_counts["failed"] > 0:
                health_status["overall_status"] = "degraded"
            if len(active_migrations) > 5:  # Too many active migrations
                health_status["overall_status"] = "degraded"
            if operation_stats.get("success_rate", 100) < 90:  # Low success rate
                health_status["overall_status"] = "degraded"
            
            logger.info("Migration health check completed", 
                       status=health_status["overall_status"],
                       active_migrations=len(active_migrations),
                       total_migrations=len(migrations))
            
            return health_status
            
        except Exception as e:
            logger.error("Migration health check failed", error=str(e))
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def monitor_migration(self, migration_id: str, check_interval: int = 30) -> Dict[str, Any]:
        """Monitor a specific migration"""
        try:
            logger.info("Starting migration monitoring", 
                       migration_id=migration_id,
                       check_interval=check_interval)
            
            monitoring_result = {
                "migration_id": migration_id,
                "start_time": datetime.utcnow().isoformat(),
                "checks": [],
                "final_status": None,
                "end_time": None
            }
            
            check_count = 0
            max_checks = 120  # Maximum 1 hour of monitoring (30s * 120 = 3600s)
            
            while check_count < max_checks:
                try:
                    # Get migration status
                    status = self.get_migration_status(migration_id)
                    
                    if not status:
                        logger.warning("Migration not found", migration_id=migration_id)
                        break
                    
                    check_result = {
                        "check_number": check_count + 1,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": status.get("status"),
                        "progress_percentage": status.get("progress_percentage", 0),
                        "current_table": status.get("current_table"),
                        "error_message": status.get("error_message")
                    }
                    
                    monitoring_result["checks"].append(check_result)
                    
                    # Check if migration is complete
                    migration_status = status.get("status")
                    if migration_status in ["completed", "failed", "cancelled"]:
                        monitoring_result["final_status"] = migration_status
                        break
                    
                    # Wait before next check
                    time.sleep(check_interval)
                    check_count += 1
                    
                    logger.info("Migration check completed", 
                               migration_id=migration_id,
                               check_number=check_count,
                               status=migration_status,
                               progress=status.get("progress_percentage", 0))
                
                except Exception as e:
                    logger.error("Migration check failed", 
                                migration_id=migration_id,
                                check_number=check_count + 1,
                                error=str(e))
                    check_count += 1
                    time.sleep(check_interval)
            
            monitoring_result["end_time"] = datetime.utcnow().isoformat()
            
            if not monitoring_result["final_status"]:
                monitoring_result["final_status"] = "timeout"
            
            logger.info("Migration monitoring completed", 
                       migration_id=migration_id,
                       final_status=monitoring_result["final_status"],
                       checks_performed=len(monitoring_result["checks"]))
            
            return monitoring_result
            
        except Exception as e:
            logger.error("Migration monitoring failed", 
                        migration_id=migration_id,
                        error=str(e))
            return {
                "migration_id": migration_id,
                "error": str(e),
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate migration monitoring report"""
        try:
            logger.info("Generating migration report")
            
            # Get health status
            health = self.check_migration_health()
            
            # Get all migrations
            migrations = self.get_all_migrations()
            
            # Generate report
            report = {
                "report_generated_at": datetime.utcnow().isoformat(),
                "health_status": health,
                "migration_summary": {
                    "total_migrations": len(migrations),
                    "recent_migrations": len([m for m in migrations if 
                        m.get("created_at") and 
                        datetime.fromisoformat(m["created_at"].replace('Z', '+00:00')) > 
                        datetime.utcnow() - timedelta(hours=24)])
                },
                "recommendations": []
            }
            
            # Add recommendations based on health status
            if health["overall_status"] == "degraded":
                report["recommendations"].append("Migration system is degraded - check failed migrations")
            
            if health["migrations"]["active_count"] > 5:
                report["recommendations"].append("Too many active migrations - consider reducing concurrency")
            
            if health["operations"]["success_rate"] < 90:
                report["recommendations"].append("Low operation success rate - check error logs")
            
            if not report["recommendations"]:
                report["recommendations"].append("Migration system is healthy")
            
            # Save report to file if specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info("Report saved to file", file=output_file)
            
            logger.info("Migration report generated", 
                       recommendations=len(report["recommendations"]))
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate migration report", error=str(e))
            return {
                "error": str(e),
                "report_generated_at": datetime.utcnow().isoformat()
            }
    
    def continuous_monitoring(self, check_interval: int = 60, duration: int = 3600):
        """Run continuous monitoring"""
        try:
            logger.info("Starting continuous monitoring", 
                       check_interval=check_interval,
                       duration=duration)
            
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(seconds=duration)
            check_count = 0
            
            while datetime.utcnow() < end_time:
                try:
                    # Check migration health
                    health = self.check_migration_health()
                    
                    check_count += 1
                    self.monitoring_stats["total_checks"] += 1
                    self.monitoring_stats["last_check"] = datetime.utcnow().isoformat()
                    
                    logger.info("Continuous monitoring check", 
                               check_number=check_count,
                               status=health["overall_status"],
                               active_migrations=health["migrations"]["active_count"])
                    
                    # Wait before next check
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.error("Continuous monitoring check failed", 
                                check_number=check_count + 1,
                                error=str(e))
                    time.sleep(check_interval)
            
            logger.info("Continuous monitoring completed", 
                       checks_performed=check_count,
                       duration_seconds=duration)
            
        except Exception as e:
            logger.error("Continuous monitoring failed", error=str(e))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Monitor migration progress")
    parser.add_argument("--migration-id", help="Monitor specific migration")
    parser.add_argument("--health-check", action="store_true", help="Run health check")
    parser.add_argument("--report", help="Generate report and save to file")
    parser.add_argument("--continuous", action="store_true", help="Run continuous monitoring")
    parser.add_argument("--check-interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--duration", type=int, default=3600, help="Duration in seconds for continuous monitoring")
    
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
    
    monitor = None
    try:
        # Initialize monitor
        monitor = MigrationMonitor()
        
        if args.migration_id:
            # Monitor specific migration
            result = monitor.monitor_migration(args.migration_id, args.check_interval)
            print(f"Migration monitoring completed: {result['final_status']}")
            print(f"Checks performed: {len(result['checks'])}")
            
        elif args.health_check:
            # Run health check
            health = monitor.check_migration_health()
            print(f"Migration health: {health['overall_status']}")
            print(f"Active migrations: {health['migrations']['active_count']}")
            print(f"Total migrations: {health['migrations']['total']}")
            
        elif args.report:
            # Generate report
            report = monitor.generate_report(args.report)
            print(f"Report generated: {args.report}")
            print(f"Health status: {report['health_status']['overall_status']}")
            print(f"Recommendations: {len(report['recommendations'])}")
            
        elif args.continuous:
            # Run continuous monitoring
            monitor.continuous_monitoring(args.check_interval, args.duration)
            print("Continuous monitoring completed")
            
        else:
            # Default: run health check
            health = monitor.check_migration_health()
            print(f"Migration health: {health['overall_status']}")
            print(f"Active migrations: {health['migrations']['active_count']}")
            
    except Exception as e:
        logger.error("Monitoring script failed", error=str(e))
        print(f"Monitoring failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

