"""
Configuration management for Content Generator V2.

This module provides configuration loading and management
for the application.
"""

import os
from typing import Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY: str = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG: bool = os.environ.get('DEBUG', 'false').lower() == 'true'
    TESTING: bool = os.environ.get('TESTING', 'false').lower() == 'true'
    
    # API settings
    API_TITLE: str = 'Content Generator V2'
    API_VERSION: str = '2.0.0'
    
    # Authentication
    API_KEY_HEADER: str = 'X-API-Key'
    API_KEYS: frozenset = field(default_factory=lambda: frozenset([
        key.strip() for key in (os.environ.get('API_KEYS', '').split(',') if os.environ.get('API_KEYS') else [])
    ]))
    
    # Rate limiting
    RATELIMIT_STORAGE_URL: str = os.environ.get('RATELIMIT_STORAGE_URL', 'memory://')
    RATELIMIT_DEFAULT: str = os.environ.get('RATELIMIT_DEFAULT', '1000 per hour')
    
    # Logging
    LOG_LEVEL: str = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.environ.get('LOG_FILE', 'logs/app.log')
    LOG_MAX_BYTES: int = int(os.environ.get('LOG_MAX_BYTES', 10485760))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.environ.get('LOG_BACKUP_COUNT', 5))
    LOG_REQUESTS: bool = os.environ.get('LOG_REQUESTS', 'true').lower() == 'true'
    
    # External services
    RAG_API_URL: Optional[str] = os.environ.get('RAG_API_URL')
    RAG_API_KEY: Optional[str] = os.environ.get('RAG_API_KEY')
    LINKUP_API_URL: Optional[str] = os.environ.get('LINKUP_API_URL')
    LINKUP_API_KEY: Optional[str] = os.environ.get('LINKUP_API_KEY')
    LITELLM_API_URL: Optional[str] = os.environ.get('LITELLM_API_URL')
    LITELLM_API_KEY: Optional[str] = os.environ.get('LITELLM_API_KEY')
    
    # Celery configuration
    CELERY_BROKER_URL: str = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND: str = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    CELERY_TASK_TIME_LIMIT: int = int(os.environ.get('CELERY_TASK_TIME_LIMIT', '3600'))  # 1 hour
    CELERY_TASK_SOFT_TIME_LIMIT: int = int(os.environ.get('CELERY_TASK_SOFT_TIME_LIMIT', '3300'))  # 55 minutes
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = int(os.environ.get('CELERY_WORKER_PREFETCH_MULTIPLIER', '1'))
    CELERY_WORKER_MAX_TASKS_PER_CHILD: int = int(os.environ.get('CELERY_WORKER_MAX_TASKS_PER_CHILD', '1000'))
    
    # Research processing settings
    MAX_PARALLEL_REQUESTS: int = int(os.environ.get('MAX_PARALLEL_REQUESTS', '10'))
    RESEARCH_TASK_TIMEOUT: int = int(os.environ.get('RESEARCH_TASK_TIMEOUT', '3000'))  # 50 minutes
    RESEARCH_TASK_RETRY_COUNT: int = int(os.environ.get('RESEARCH_TASK_RETRY_COUNT', '3'))
    RESEARCH_TASK_RETRY_DELAY: int = int(os.environ.get('RESEARCH_TASK_RETRY_DELAY', '60'))  # 1 minute
    
    # Monitoring and health checks
    ENABLE_METRICS: bool = os.environ.get('ENABLE_METRICS', 'true').lower() == 'true'
    
    # Request settings
    MAX_CONTENT_LENGTH: int = int(os.environ.get('MAX_CONTENT_LENGTH', 1048576))  # 1MB
    REQUEST_TIMEOUT: int = int(os.environ.get('REQUEST_TIMEOUT', 30))
    
    # CORS settings
    CORS_ORIGINS: List[str] = field(default_factory=lambda: os.environ.get('CORS_ORIGINS', '*').split(','))


@dataclass
class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG: bool = True
    LOG_LEVEL: str = 'DEBUG'
    RATELIMIT_DEFAULT: str = '5000 per hour'  # Very lenient for development


@dataclass
class ProductionConfig(Config):
    """Production configuration."""
    DEBUG: bool = False
    LOG_LEVEL: str = 'WARNING'


@dataclass
class TestingConfig(Config):
    """Testing configuration."""
    TESTING: bool = True
    DEBUG: bool = True
    LOG_LEVEL: str = 'CRITICAL'
    API_KEYS: frozenset = field(default_factory=lambda: frozenset(['test-api-key']))


def get_config(config_name: str = None) -> Config:
    """
    Get configuration based on environment.
    
    Args:
        config_name: Configuration name (development, production, testing)
        
    Returns:
        Configuration object
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development').lower()
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    config_class = config_map.get(config_name, DevelopmentConfig)
    return config_class()


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check required settings
    if not config.API_KEYS:
        errors.append("API_KEYS must be configured")
    
    if config.SECRET_KEY == 'dev-secret-key-change-in-production' and config.DEBUG is False:
        errors.append("SECRET_KEY must be changed in production")
    
    # Check external service URLs
    if config.RAG_API_URL and not config.RAG_API_KEY:
        errors.append("RAG_API_KEY is required when RAG_API_URL is set")
    
    if config.LINKUP_API_URL and not config.LINKUP_API_KEY:
        errors.append("LINKUP_API_KEY is required when LINKUP_API_URL is set")
    
    # Check Celery configuration
    if not config.CELERY_BROKER_URL:
        errors.append("CELERY_BROKER_URL must be configured")
    
    if not config.CELERY_RESULT_BACKEND:
        errors.append("CELERY_RESULT_BACKEND must be configured")
    
    return errors
