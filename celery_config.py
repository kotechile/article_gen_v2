"""
Celery configuration for Content Generator V2.

This module configures Celery for asynchronous task processing.
"""

import os
import logging
from celery import Celery
from kombu import Queue

from src.utils.broker import generic_host_warning, mask_credentials, normalize_broker_url

logger = logging.getLogger(__name__)

# Get configuration from environment.
# REDIS_PASSWORD is folded into the URL: a password configured separately but never
# applied surfaces as "HELLO must be called with the client already authenticated".
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
CELERY_BROKER_URL = normalize_broker_url(
    os.environ.get('CELERY_BROKER_URL') or REDIS_URL,
    os.environ.get('REDIS_PASSWORD'),
)
CELERY_RESULT_BACKEND = normalize_broker_url(
    os.environ.get('CELERY_RESULT_BACKEND') or os.environ.get('CELERY_BROKER_URL') or REDIS_URL,
    os.environ.get('REDIS_PASSWORD'),
)

logger.info("Celery broker: %s", mask_credentials(CELERY_BROKER_URL))
_broker_warning = generic_host_warning(CELERY_BROKER_URL)
if _broker_warning:
    logger.warning("Celery broker configuration: %s", _broker_warning)

# Create Celery app
celery = Celery('content_generator_v2')

# Configure Celery
celery.conf.update(
    # Broker and backend
    broker_url=CELERY_BROKER_URL,
    result_backend=CELERY_RESULT_BACKEND,
    # Keep trying to reach the broker on boot instead of dying in a crash loop when
    # the broker container starts a few seconds after us.
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Timezone
    timezone='UTC',
    enable_utc=True,
    
    # Task configuration
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    
    # Task routing
    task_routes={
        'content_generator_v2.tasks.research.*': {'queue': 'research'},
        'content_generator_v2.tasks.monitoring.*': {'queue': 'monitoring'},
    },
    
    # Queue configuration
    task_default_queue='research',
    task_queues=(
        Queue('research', routing_key='research'),
        Queue('monitoring', routing_key='monitoring'),
    ),
    
    # Result backend configuration
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Worker configuration
    worker_hijack_root_logger=False,
    worker_log_color=False,
    worker_pool='threads',
    
    # Task execution
    task_always_eager=False,  # Set to True for testing
    task_eager_propagates=True,
    
    # Error handling
    task_acks_late=True,
    worker_disable_rate_limits=False,
    
    # Auto-discover tasks
    imports=('tasks',),
)

# Optional: Configure logging
celery.conf.update(
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
)

# Tasks are auto-discovered via the 'imports' configuration above.
# No need for manual import here which causes circular dependencies.

if __name__ == '__main__':
    celery.start()
