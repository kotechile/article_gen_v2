#!/usr/bin/env python3
"""
Celery worker runner for Content Generator V2.

This script starts a Celery worker to process research tasks.
"""

import os
import sys
import logging
from celery import Celery
from celery_config import celery_app

# Import tasks to register them
from tasks import process_research_task, get_task_status, cancel_task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the Celery worker."""
    try:
        logger.info("Starting Content Generator V2 Celery Worker...")
        logger.info("Worker will process tasks from the 'research' queue")
        
        # Start the worker
        worker = celery_app.Worker(
            queues=['research', 'monitoring'],
            concurrency=2,  # Adjust based on your system
            loglevel='info',
            hostname='content-generator-worker@%h'
        )
        
        worker.start()
        
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Worker failed to start: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
