"""
Logging configuration for Content Generator V2.

This module provides centralized logging setup and
configuration for the application.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Dict, Any


def setup_logging(config: Dict[str, Any]):
    """
    Setup logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    # Create logs directory if it doesn't exist
    log_file = config.get('LOG_FILE', 'logs/app.log')
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.get('LOG_LEVEL', 'INFO')))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.get('LOG_LEVEL', 'INFO')))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=config.get('LOG_MAX_BYTES', 10485760),  # 10MB
            backupCount=config.get('LOG_BACKUP_COUNT', 5)
        )
        file_handler.setLevel(getattr(logging, config.get('LOG_LEVEL', 'INFO')))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    configure_loggers()


def configure_loggers():
    """Configure specific loggers for different components."""
    
    # Flask logger
    flask_logger = logging.getLogger('werkzeug')
    flask_logger.setLevel(logging.WARNING)
    
    # Celery logger
    celery_logger = logging.getLogger('celery')
    celery_logger.setLevel(logging.INFO)
    
    # LiteLLM logger
    litellm_logger = logging.getLogger('litellm')
    litellm_logger.setLevel(logging.WARNING)
    
    # HTTP requests logger
    requests_logger = logging.getLogger('requests')
    requests_logger.setLevel(logging.WARNING)
    
    # urllib3 logger
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.WARNING)


class StructuredLogger:
    """Structured logger for better log formatting."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self._log(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Log message with structured data."""
        extra = {
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        
        self.logger.log(level, message, extra=extra)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    return StructuredLogger(name)


class RequestLogger:
    """Logger for HTTP requests."""
    
    def __init__(self):
        self.logger = get_logger('request')
    
    def log_request(self, method: str, path: str, remote_addr: str, **kwargs):
        """Log incoming request."""
        self.logger.info(
            f"Request: {method} {path}",
            method=method,
            path=path,
            remote_addr=remote_addr,
            **kwargs
        )
    
    def log_response(self, method: str, path: str, status_code: int, 
                    duration: float, **kwargs):
        """Log response."""
        self.logger.info(
            f"Response: {status_code} for {method} {path}",
            method=method,
            path=path,
            status_code=status_code,
            duration=duration,
            **kwargs
        )
    
    def log_error(self, method: str, path: str, error: str, **kwargs):
        """Log request error."""
        self.logger.error(
            f"Request error: {error}",
            method=method,
            path=path,
            error=error,
            **kwargs
        )


class TaskLogger:
    """Logger for Celery tasks."""
    
    def __init__(self):
        self.logger = get_logger('task')
    
    def log_task_start(self, task_id: str, task_name: str, **kwargs):
        """Log task start."""
        self.logger.info(
            f"Task started: {task_name}",
            task_id=task_id,
            task_name=task_name,
            **kwargs
        )
    
    def log_task_progress(self, task_id: str, progress: int, message: str, **kwargs):
        """Log task progress."""
        self.logger.info(
            f"Task progress: {progress}% - {message}",
            task_id=task_id,
            progress=progress,
            message=message,
            **kwargs
        )
    
    def log_task_complete(self, task_id: str, task_name: str, duration: float, **kwargs):
        """Log task completion."""
        self.logger.info(
            f"Task completed: {task_name}",
            task_id=task_id,
            task_name=task_name,
            duration=duration,
            **kwargs
        )
    
    def log_task_error(self, task_id: str, task_name: str, error: str, **kwargs):
        """Log task error."""
        self.logger.error(
            f"Task error: {error}",
            task_id=task_id,
            task_name=task_name,
            error=error,
            **kwargs
        )
