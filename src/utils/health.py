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
from .broker import check_broker, mask_credentials


logger = logging.getLogger(__name__)


class HealthChecker:
    """Health checker for system components."""
    
    def __init__(self):
        self.config = get_config()
        self._redis_client = None
        self._celery_app = None
    
    def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity, address by address.

        A hostname can legitimately resolve to more than one container (e.g. a
        compose service named ``redis`` beside Coolify's own cache), in which case
        some connections work and others are rejected. Reporting per-address makes
        that visible from the outside instead of surfacing as a random 500.
        """
        try:
            report = check_broker(self.config.CELERY_BROKER_URL, timeout=1.5)
        except Exception as e:  # noqa: BLE001 - health checks never raise
            logger.error(f"Redis health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

        if report.get("status") == "healthy" and not report.get("suspect_alias_collision"):
            healthy = [a for a in report.get("addresses", []) if a.get("status") == "healthy"]
            return {
                "status": "healthy",
                "broker": report.get("url"),
                "host": report.get("host"),
                "version": healthy[0].get("version") if healthy else None,
                "addresses": report.get("addresses", []),
            }

        logger.error(
            "Redis health check failed for %s: %s",
            report.get("url"),
            report.get("hint") or report.get("error"),
        )
        return {
            "status": "unhealthy",
            "broker": report.get("url"),
            "host": report.get("host"),
            "error": report.get("error") or (report.get("hint") or "broker unreachable"),
            "hint": report.get("hint"),
            "addresses": report.get("addresses", []),
            "suspect_alias_collision": report.get("suspect_alias_collision", False),
        }

    def check_celery(self) -> Dict[str, Any]:
        """Check Celery worker status.

        ``Control`` has no ``stats()`` (that lives on ``Inspect``), so ask for an
        inspector first and use it for both the ping and the stats.
        """
        try:
            if not self._celery_app:
                from ..tasks.celery_app import celery_app
                self._celery_app = celery_app

            broker = mask_credentials(getattr(self._celery_app.conf, "broker_url", None))
            inspector = self._celery_app.control.inspect(timeout=1.0)
            pings = (inspector.ping() if inspector else None) or {}

            if not pings:
                return {
                    "status": "unhealthy",
                    "broker": broker,
                    "error": "No Celery workers responded on the broker (worker down, "
                             "or the worker cannot reach the broker - check its logs for "
                             "'Cannot connect to redis://...')",
                }

            stats = (inspector.stats() if inspector else None) or {}
            active_tasks = 0
            reserved_tasks = 0
            for worker_stats in stats.values():
                active_tasks += len(worker_stats.get("active") or [])
                reserved_tasks += len(worker_stats.get("reserved") or [])

            return {
                "status": "healthy",
                "broker": broker,
                "workers": len(pings),
                "worker_names": list(pings.keys()),
                "active_tasks": active_tasks,
                "reserved_tasks": reserved_tasks,
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
