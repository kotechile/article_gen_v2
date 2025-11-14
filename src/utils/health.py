"""
Health check utilities for Content Generator V2.

This module provides health check functionality for
monitoring system components.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import psutil
import redis
from celery import Celery

from ..utils.config import get_config


logger = logging.getLogger(__name__)


class HealthChecker:
    """Health checker for system components."""
    
    def __init__(self):
        self.config = get_config()
        self._redis_client = None
        self._celery_app = None
    
    def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            if not self._redis_client:
                self._redis_client = redis.Redis.from_url(self.config.CELERY_BROKER_URL)
            
            # Test connection
            self._redis_client.ping()
            
            # Get Redis info
            info = self._redis_client.info()
            
            return {
                "status": "healthy",
                "host": info.get("redis_version", "unknown"),
                "uptime": info.get("uptime_in_seconds", 0),
                "memory_used": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0)
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_celery(self) -> Dict[str, Any]:
        """Check Celery worker status."""
        try:
            if not self._celery_app:
                from ..tasks.celery_app import celery_app
                self._celery_app = celery_app
            
            # Get worker stats
            stats = self._celery_app.control.stats()
            
            if not stats:
                return {
                    "status": "unhealthy",
                    "error": "No Celery workers found"
                }
            
            # Calculate total stats
            total_workers = len(stats)
            total_tasks = sum(worker.get('total', 0) for worker in stats.values())
            
            return {
                "status": "healthy",
                "workers": total_workers,
                "total_tasks": total_tasks,
                "active_tasks": sum(worker.get('active', 0) for worker in stats.values())
            }
            
        except Exception as e:
            logger.error(f"Celery health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_database(self) -> Dict[str, Any]:
        """Check database connectivity (if configured)."""
        try:
            # This would be implemented based on the database being used
            # For now, return a placeholder
            return {
                "status": "not_configured",
                "message": "Database not configured"
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_external_services(self) -> Dict[str, Any]:
        """Check external service connectivity."""
        services = {}
        
        # Check RAG service
        if self.config.RAG_API_URL:
            services["rag"] = self._check_rag_service()
        else:
            services["rag"] = {"status": "not_configured"}
        
        # Check LinkUp service
        if self.config.LINKUP_API_URL:
            services["linkup"] = self._check_linkup_service()
        else:
            services["linkup"] = {"status": "not_configured"}
        
        # Check LiteLLM service
        if self.config.LITELLM_API_URL:
            services["litellm"] = self._check_litellm_service()
        else:
            services["litellm"] = {"status": "not_configured"}
        
        return services
    
    def _check_rag_service(self) -> Dict[str, Any]:
        """Check RAG service connectivity."""
        try:
            import requests
            
            response = requests.get(
                f"{self.config.RAG_API_URL}/health",
                timeout=5,
                headers={"Authorization": f"Bearer {self.config.RAG_API_KEY}"}
            )
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _check_linkup_service(self) -> Dict[str, Any]:
        """Check LinkUp service connectivity."""
        try:
            import requests
            
            response = requests.get(
                f"{self.config.LINKUP_API_URL}/health",
                timeout=5,
                headers={"Authorization": f"Bearer {self.config.LINKUP_API_KEY}"}
            )
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _check_litellm_service(self) -> Dict[str, Any]:
        """Check LiteLLM service connectivity."""
        try:
            import requests
            
            response = requests.get(
                f"{self.config.LITELLM_API_URL}/health",
                timeout=5,
                headers={"Authorization": f"Bearer {self.config.LITELLM_API_KEY}"}
            )
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed health status for all components."""
        return {
            "redis": self.check_redis(),
            "celery": self.check_celery(),
            "database": self.check_database(),
            "external_services": self.check_external_services(),
            "system": self.get_system_metrics()
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            return {
                "status": "healthy",
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "disk_percent": disk.percent,
                "disk_free": disk.free,
                "uptime": time.time() - psutil.boot_time()
            }
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health": self.get_detailed_status(),
            "configuration": {
                "debug": self.config.DEBUG,
                "log_level": self.config.LOG_LEVEL,
                "max_parallel_requests": self.config.MAX_PARALLEL_REQUESTS,
                "task_timeout": self.config.RESEARCH_TASK_TIMEOUT
            }
        }
