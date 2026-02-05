"""
Celery Configuration
Background task processing with Redis broker
"""

from celery import Celery
from celery.schedules import crontab
import os

from .config import settings

# Celery configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

# Create Celery app
celery_app = Celery(
    "trendtap",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "src.tasks.affiliate_tasks",
        "src.tasks.trend_tasks",
        "src.tasks.keyword_tasks",
        "src.tasks.content_tasks",
        "src.tasks.software_tasks",
        "src.tasks.export_tasks",
        "src.tasks.calendar_tasks",
        "src.tasks.maintenance_tasks"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task execution
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "src.tasks.affiliate_tasks.*": {"queue": "affiliate"},
        "src.tasks.trend_tasks.*": {"queue": "trends"},
        "src.tasks.keyword_tasks.*": {"queue": "keywords"},
        "src.tasks.content_tasks.*": {"queue": "content"},
        "src.tasks.software_tasks.*": {"queue": "software"},
        "src.tasks.export_tasks.*": {"queue": "export"},
        "src.tasks.calendar_tasks.*": {"queue": "calendar"},
        "src.tasks.maintenance_tasks.*": {"queue": "maintenance"}
    },
    
    # Task execution settings
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Task retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Worker settings
    worker_hijack_root_logger=False,
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "cleanup-old-data": {
            "task": "src.tasks.maintenance_tasks.cleanup_old_data",
            "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM
        },
        "update-trend-data": {
            "task": "src.tasks.trend_tasks.update_trend_data",
            "schedule": crontab(minute=0),  # Every hour
        },
        "process-pending-exports": {
            "task": "src.tasks.export_tasks.process_pending_exports",
            "schedule": crontab(minute="*/5"),  # Every 5 minutes
        },
        "send-reminder-notifications": {
            "task": "src.tasks.calendar_tasks.send_reminder_notifications",
            "schedule": crontab(minute=0),  # Every hour
        },
        "update-affiliate-programs": {
            "task": "src.tasks.affiliate_tasks.update_affiliate_programs",
            "schedule": crontab(hour=6, minute=0),  # Daily at 6 AM
        },
        "generate-content-suggestions": {
            "task": "src.tasks.content_tasks.generate_content_suggestions",
            "schedule": crontab(hour=8, minute=0),  # Daily at 8 AM
        }
    }
)

# Task monitoring
celery_app.conf.update(
    # Task monitoring
    task_send_sent_event=True,
    worker_send_task_events=True,
    
    # Task compression
    task_compression="gzip",
    result_compression="gzip",
    
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,  # 10 minutes
    
    # Worker settings
    worker_max_tasks_per_child=1000,
    worker_max_memory_per_child=200000,  # 200MB
)

# Health check
@celery_app.task(bind=True)
def health_check(self):
    """Health check task"""
    return {
        "status": "healthy",
        "worker": self.request.hostname,
        "timestamp": self.request.utcnow().isoformat()
    }

# Get Celery app instance
def get_celery_app() -> Celery:
    """Get Celery app instance"""
    return celery_app
