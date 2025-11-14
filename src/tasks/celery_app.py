"""
Celery application configuration for Content Generator V2.

This module configures the Celery application for asynchronous
task processing.
"""

from celery import Celery
from ..utils.config import get_config

# Get configuration
config = get_config()

# Create Celery app
celery_app = Celery('content_generator_v2')

# Configure Celery
celery_app.conf.update(
    broker_url=config.CELERY_BROKER_URL,
    result_backend=config.CELERY_RESULT_BACKEND,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=config.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=config.CELERY_TASK_SOFT_TIME_LIMIT,
    worker_prefetch_multiplier=config.CELERY_WORKER_PREFETCH_MULTIPLIER,
    worker_max_tasks_per_child=config.CELERY_WORKER_MAX_TASKS_PER_CHILD,
    task_routes={
        'src.tasks.research.*': {'queue': 'research'},
        'src.tasks.monitoring.*': {'queue': 'monitoring'},
    }
)

# Tasks will be imported when needed to avoid circular imports
