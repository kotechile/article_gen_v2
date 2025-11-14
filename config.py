"""
Configuration module for Content Generator V2.

This module handles environment variables, settings validation,
and configuration management for the entire application.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class Environment(Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig:
    """Database configuration."""
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_db: int = 0

@dataclass
class LinkupConfig:
    """Linkup web search configuration."""
    api_key: Optional[str] = None
    endpoint: str = "https://api.linkup.com/v1/search"
    timeout: int = 30
    max_retries: int = 3
    max_results: int = 10

@dataclass
class LLMConfig:
    """LLM configuration."""
    default_provider: str = "openai"
    default_model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Provider-specific settings
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    kimi_api_key: Optional[str] = None
    moonshot_api_key: Optional[str] = None

@dataclass
class RAGConfig:
    """RAG configuration."""
    enabled: bool = True
    endpoint: str = "http://localhost:8080/query_hybrid_enhanced"
    collection: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    llm_provider: Optional[str] = None

@dataclass
class LinkupOptimizationConfig:
    """Linkup search optimization configuration."""
    rag_coverage_min_sources: int = 3
    rag_coverage_min_relevance: float = 0.6
    cache_enabled: bool = True
    cache_ttl_standard_seconds: int = 21600  # 6 hours
    cache_ttl_deep_seconds: int = 86400  # 24 hours
    auto_depth_downgrade: bool = True  # Downgrade comprehensive to standard if RAG sufficient
    # Deep search cost controls
    deep_trigger_min_sources: int = 2  # if RAG sources < this, deep may be considered
    deep_trigger_min_avg_relevance: float = 0.45  # if avg relevance below this, deep may be considered
    deep_trigger_min_keyword_coverage: float = 0.3  # if keyword coverage below this, deep may be considered
    deep_min_standard_results_threshold: int = 3  # require at least this many standard results to avoid deep

@dataclass
class LinkupConfig:
    """Linkup web search configuration."""
    enabled: bool = True
    api_key: Optional[str] = None
    endpoint: str = "https://api.linkup.com/v1/search"
    timeout: int = 30
    max_retries: int = 3
    max_results: int = 10

@dataclass
class CeleryConfig:
    """Celery configuration."""
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"
    task_serializer: str = "json"
    accept_content: List[str] = field(default_factory=lambda: ["json"])
    result_serializer: str = "json"
    timezone: str = "UTC"
    enable_utc: bool = True
    task_track_started: bool = True
    task_time_limit: int = 3600
    task_soft_time_limit: int = 3300
    worker_prefetch_multiplier: int = 1
    worker_max_tasks_per_child: int = 50
    result_expires: int = 3600
    result_persistent: bool = True

@dataclass
class FlaskConfig:
    """Flask application configuration."""
    host: str = "0.0.0.0"
    port: int = 5001
    debug: bool = False
    secret_key: str = "dev-secret-key-change-in-production"
    
    # CORS settings
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_storage: str = "memory://"
    rate_limit_default: str = "1000 per hour, 60 per minute"
    rate_limit_research: str = "10 per minute"
    
    # API settings
    api_key_header: str = "X-API-Key"
    api_keys: List[str] = field(default_factory=list)

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "[%(asctime)s: %(levelname)s/%(name)s] %(message)s"
    file_path: str = "logs/app.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True

@dataclass
class AppConfig:
    """Main application configuration."""
    environment: Environment = Environment.DEVELOPMENT
    app_name: str = "Content Generator V2"
    version: str = "2.0.0"
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    linkup: LinkupConfig = field(default_factory=LinkupConfig)
    linkup_optimization: LinkupOptimizationConfig = field(default_factory=LinkupOptimizationConfig)
    celery: CeleryConfig = field(default_factory=CeleryConfig)
    flask: FlaskConfig = field(default_factory=FlaskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._load_from_environment()
        self._validate_config()
        self._setup_logging()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Environment
        env_str = os.getenv("ENVIRONMENT", "development").lower()
        if env_str in [e.value for e in Environment]:
            self.environment = Environment(env_str)
        
        # Database
        self.database.redis_url = os.getenv("REDIS_URL", self.database.redis_url)
        self.database.redis_password = os.getenv("REDIS_PASSWORD")
        self.database.redis_db = int(os.getenv("REDIS_DB", str(self.database.redis_db)))
        
        # LLM
        self.llm.default_provider = os.getenv("LLM_PROVIDER", self.llm.default_provider)
        self.llm.default_model = os.getenv("LLM_MODEL", self.llm.default_model)
        self.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(self.llm.max_tokens)))
        self.llm.temperature = float(os.getenv("LLM_TEMPERATURE", str(self.llm.temperature)))
        self.llm.timeout = int(os.getenv("LLM_TIMEOUT", str(self.llm.timeout)))
        self.llm.max_retries = int(os.getenv("LLM_MAX_RETRIES", str(self.llm.max_retries)))
        self.llm.retry_delay = float(os.getenv("LLM_RETRY_DELAY", str(self.llm.retry_delay)))
        
        # API Keys
        self.llm.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.llm.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.llm.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.llm.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.llm.kimi_api_key = os.getenv("KIMI_API_KEY")
        self.llm.moonshot_api_key = os.getenv("MOONSHOT_API_KEY")
        
        # Linkup
        self.linkup.api_key = os.getenv("LINKUP_API_KEY")
        self.linkup.endpoint = os.getenv("LINKUP_ENDPOINT", self.linkup.endpoint)
        self.linkup.timeout = int(os.getenv("LINKUP_TIMEOUT", str(self.linkup.timeout)))
        self.linkup.max_retries = int(os.getenv("LINKUP_MAX_RETRIES", str(self.linkup.max_retries)))
        self.linkup.max_results = int(os.getenv("LINKUP_MAX_RESULTS", str(self.linkup.max_results)))
        
        # RAG
        self.rag.enabled = os.getenv("RAG_ENABLED", "true").lower() == "true"
        self.rag.endpoint = os.getenv("RAG_ENDPOINT", self.rag.endpoint)
        self.rag.collection = os.getenv("RAG_COLLECTION")
        self.rag.timeout = int(os.getenv("RAG_TIMEOUT", str(self.rag.timeout)))
        self.rag.max_retries = int(os.getenv("RAG_MAX_RETRIES", str(self.rag.max_retries)))
        self.rag.llm_provider = os.getenv("RAG_LLM_PROVIDER")
        
        # Linkup
        self.linkup.enabled = os.getenv("LINKUP_ENABLED", "true").lower() == "true"
        self.linkup.api_key = os.getenv("LINKUP_API_KEY")
        self.linkup.endpoint = os.getenv("LINKUP_ENDPOINT", self.linkup.endpoint)
        self.linkup.timeout = int(os.getenv("LINKUP_TIMEOUT", str(self.linkup.timeout)))
        self.linkup.max_retries = int(os.getenv("LINKUP_MAX_RETRIES", str(self.linkup.max_retries)))
        self.linkup.max_results = int(os.getenv("LINKUP_MAX_RESULTS", str(self.linkup.max_results)))
        
        # Linkup Optimization
        self.linkup_optimization.rag_coverage_min_sources = int(os.getenv("LINKUP_OPT_RAG_MIN_SOURCES", str(self.linkup_optimization.rag_coverage_min_sources)))
        self.linkup_optimization.rag_coverage_min_relevance = float(os.getenv("LINKUP_OPT_RAG_MIN_RELEVANCE", str(self.linkup_optimization.rag_coverage_min_relevance)))
        self.linkup_optimization.cache_enabled = os.getenv("LINKUP_OPT_CACHE_ENABLED", "true").lower() == "true"
        self.linkup_optimization.cache_ttl_standard_seconds = int(os.getenv("LINKUP_OPT_CACHE_TTL_STANDARD", str(self.linkup_optimization.cache_ttl_standard_seconds)))
        self.linkup_optimization.cache_ttl_deep_seconds = int(os.getenv("LINKUP_OPT_CACHE_TTL_DEEP", str(self.linkup_optimization.cache_ttl_deep_seconds)))
        self.linkup_optimization.auto_depth_downgrade = os.getenv("LINKUP_OPT_AUTO_DOWNGRADE", "true").lower() == "true"
        self.linkup_optimization.deep_trigger_min_sources = int(os.getenv("LINKUP_OPT_DEEP_TRIG_MIN_SOURCES", str(self.linkup_optimization.deep_trigger_min_sources)))
        self.linkup_optimization.deep_trigger_min_avg_relevance = float(os.getenv("LINKUP_OPT_DEEP_TRIG_MIN_AVG_REL", str(self.linkup_optimization.deep_trigger_min_avg_relevance)))
        self.linkup_optimization.deep_trigger_min_keyword_coverage = float(os.getenv("LINKUP_OPT_DEEP_TRIG_MIN_KW_COV", str(self.linkup_optimization.deep_trigger_min_keyword_coverage)))
        self.linkup_optimization.deep_min_standard_results_threshold = int(os.getenv("LINKUP_OPT_DEEP_MIN_STD_RESULTS", str(self.linkup_optimization.deep_min_standard_results_threshold)))
        
        # Celery
        self.celery.broker_url = os.getenv("CELERY_BROKER_URL", self.database.redis_url)
        self.celery.result_backend = os.getenv("CELERY_RESULT_BACKEND", self.database.redis_url)
        
        # Flask
        self.flask.host = os.getenv("FLASK_HOST", self.flask.host)
        self.flask.port = int(os.getenv("FLASK_PORT", str(self.flask.port)))
        self.flask.debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
        self.flask.secret_key = os.getenv("FLASK_SECRET_KEY", self.flask.secret_key)
        
        # API Keys for authentication
        api_keys_str = os.getenv("API_KEYS", "")
        if api_keys_str:
            self.flask.api_keys = [key.strip() for key in api_keys_str.split(",") if key.strip()]
        
        # Logging
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.file_path = os.getenv("LOG_FILE", self.logging.file_path)
        self.logging.enable_console = os.getenv("LOG_CONSOLE", "true").lower() == "true"
        self.logging.enable_file = os.getenv("LOG_FILE_ENABLED", "true").lower() == "true"
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate environment
        if self.environment == Environment.PRODUCTION:
            if self.flask.debug:
                logger.warning("Debug mode is enabled in production!")
            if not self.flask.secret_key or self.flask.secret_key == "dev-secret-key-change-in-production":
                raise ValueError("Secret key must be set in production!")
        
        # Validate API keys for enabled providers
        if self.llm.default_provider == "openai" and not self.llm.openai_api_key:
            logger.warning("OpenAI API key not set")
        if self.llm.default_provider == "gemini" and not self.llm.gemini_api_key:
            logger.warning("Gemini API key not set")
        if self.llm.default_provider == "anthropic" and not self.llm.anthropic_api_key:
            logger.warning("Anthropic API key not set")
        if self.llm.default_provider == "kimi" and not self.llm.kimi_api_key:
            logger.warning("Kimi API key not set")
        if self.llm.default_provider == "moonshot" and not self.llm.moonshot_api_key:
            logger.warning("Moonshot API key not set")
        
        # Validate RAG configuration
        if self.rag.enabled and not self.rag.endpoint:
            raise ValueError("RAG endpoint must be set when RAG is enabled")
        
        # Validate Linkup configuration
        if self.linkup.enabled and not self.linkup.api_key:
            logger.warning("Linkup API key not set")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.logging.file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format,
            handlers=[]
        )
        
        # Add console handler
        if self.logging.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.logging.level.upper()))
            console_handler.setFormatter(logging.Formatter(self.logging.format))
            logging.getLogger().addHandler(console_handler)
        
        # Add file handler
        if self.logging.enable_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.logging.file_path,
                maxBytes=self.logging.max_file_size,
                backupCount=self.logging.backup_count
            )
            file_handler.setLevel(getattr(logging, self.logging.level.upper()))
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            logging.getLogger().addHandler(file_handler)
    
    def get_llm_config(self, provider: str, model: str, api_key: str) -> Dict[str, Any]:
        """Get LLM configuration for a specific provider and model."""
        return {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "timeout": self.llm.timeout,
            "max_retries": self.llm.max_retries,
            "retry_delay": self.llm.retry_delay
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

# Global configuration instance
config = AppConfig()

# Export commonly used configurations
def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config

def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return config.database

def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return config.llm

def get_rag_config() -> RAGConfig:
    """Get RAG configuration."""
    return config.rag

def get_linkup_config() -> LinkupConfig:
    """Get Linkup configuration."""
    return config.linkup

def get_celery_config() -> CeleryConfig:
    """Get Celery configuration."""
    return config.celery

def get_flask_config() -> FlaskConfig:
    """Get Flask configuration."""
    return config.flask

def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return config.logging
