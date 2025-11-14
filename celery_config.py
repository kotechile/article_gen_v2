"""
Celery configuration for Content Generator V2.

This module configures Celery for asynchronous task processing.
"""

import os
from celery import Celery
from kombu import Queue

# Get configuration from environment
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', REDIS_URL)
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', REDIS_URL)

# Create Celery app
celery_app = Celery('content_generator_v2')

# Configure Celery
celery_app.conf.update(
    # Broker and backend
    broker_url=CELERY_BROKER_URL,
    result_backend=CELERY_RESULT_BACKEND,
    
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
celery_app.conf.update(
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
)

# Import tasks to register them with Celery
try:
    from tasks import process_research_task, get_task_status, cancel_task
except ImportError as e:
    import logging
    logging.warning(f"Could not import tasks: {e}. Tasks will be auto-discovered at worker startup.")

if __name__ == '__main__':
    celery_app.start()
