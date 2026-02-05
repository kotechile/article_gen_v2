"""
Configuration management for TrendTap
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
import os

class Settings(BaseSettings):
    """Application settings"""

    # Application
    app_name: str = "TrendTap API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Database - Supabase only
    # Note: Using Supabase SDK instead of direct database connections
    # database_url is no longer used as we use Supabase SDK
    
    # Supabase
    supabase_url: Optional[str] = Field(default=None, env="SUPABASE_URL")
    supabase_service_role_key: Optional[str] = Field(default=None, env="SUPABASE_SERVICE_ROLE_KEY")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    jwt_secret: str = Field(
        default="your-jwt-secret-change-in-production",
        env="JWT_SECRET"
    )
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # CORS
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "https://trendtap.com"],
        env="ALLOWED_ORIGINS"
    )
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    # External APIs
    # Google Trends (Future - not yet available)
    google_trends_api_key: Optional[str] = Field(default=None, env="GOOGLE_TRENDS_API_KEY")
    
    # Google Trends Fallback Configuration
    use_llm_trends_fallback: bool = Field(default=True, env="USE_LLM_TRENDS_FALLBACK")
    llm_trends_provider: str = Field(default="openai", env="LLM_TRENDS_PROVIDER")  # openai, anthropic, google
    
    # Affiliate Networks
    shareasale_api_key: Optional[str] = Field(default=None, env="SHAREASALE_API_KEY")
    impact_api_key: Optional[str] = Field(default=None, env="IMPACT_API_KEY")
    amazon_associates_tag: Optional[str] = Field(default=None, env="AMAZON_ASSOCIATES_TAG")
    cj_api_key: Optional[str] = Field(default=None, env="CJ_API_KEY")
    partnerize_api_key: Optional[str] = Field(default=None, env="PARTNERIZE_API_KEY")
    
    # LLM Providers - Using environment variables for now (Supabase integration ready)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_ai_api_key: Optional[str] = Field(default=None, env="GOOGLE_AI_API_KEY")
    deepseek_api_key: Optional[str] = Field(default=None, env="DEEPSEEK_API_KEY")
    linkup_api_key: Optional[str] = Field(default=None, env="LINKUP_API_KEY")
    
    # DataForSEO
    dataforseo_username: Optional[str] = Field(default=None, env="DATAFORSEO_USERNAME")
    dataforseo_password: Optional[str] = Field(default=None, env="DATAFORSEO_PASSWORD")
    
    # LLM Providers - Now loaded from Supabase (commented out to avoid Pydantic validation)
    # These are accessed via properties below
    
    # Social Media APIs
    reddit_client_id: Optional[str] = Field(default=None, env="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, env="REDDIT_CLIENT_SECRET")
    twitter_bearer_token: Optional[str] = Field(default=None, env="TWITTER_BEARER_TOKEN")
    tiktok_api_key: Optional[str] = Field(default=None, env="TIKTOK_API_KEY")
    
    # Content Optimization
    surfer_seo_api_key: Optional[str] = Field(default=None, env="SURFER_SEO_API_KEY")
    frase_api_key: Optional[str] = Field(default=None, env="FRASE_API_KEY")
    coschedule_api_key: Optional[str] = Field(default=None, env="COSCHEDULE_API_KEY")
    
    # Export Platforms
    google_docs_api_key: Optional[str] = Field(default=None, env="GOOGLE_DOCS_API_KEY")
    notion_api_key: Optional[str] = Field(default=None, env="NOTION_API_KEY")
    wordpress_api_url: Optional[str] = Field(default=None, env="WORDPRESS_API_URL")
    wordpress_api_key: Optional[str] = Field(default=None, env="WORDPRESS_API_KEY")
    wordpress_username: Optional[str] = Field(default=None, env="WORDPRESS_USERNAME")
    wordpress_password: Optional[str] = Field(default=None, env="WORDPRESS_PASSWORD")
    
    # File uploads
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_file_types: List[str] = Field(
        default=["text/csv", "application/csv"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # CSV Processing
    csv_max_rows: int = Field(default=10000, env="CSV_MAX_ROWS")
    csv_processing_timeout: int = Field(default=300, env="CSV_PROCESSING_TIMEOUT")  # 5 minutes
    
    # Keyword Clustering
    clustering_algorithm: str = Field(default="kmeans", env="CLUSTERING_ALGORITHM")  # kmeans, dbscan
    max_clusters: int = Field(default=20, env="MAX_CLUSTERS")
    min_cluster_size: int = Field(default=3, env="MIN_CLUSTER_SIZE")
    clustering_timeout: int = Field(default=600, env="CLUSTERING_TIMEOUT")  # 10 minutes
    
    # Performance
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env file

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings"""
    return settings

def validate_required_settings() -> List[str]:
    """Validate that all required settings are present"""
    missing_settings = []
    
    # Check required settings based on environment
    if settings.environment == "production":
        required_settings = [
            "secret_key",
            "database_url",
            "redis_url"
        ]
        
        for setting in required_settings:
            if not getattr(settings, setting) or getattr(settings, setting) == "your-secret-key-change-in-production":
                missing_settings.append(setting)
    
    return missing_settings

def get_database_url() -> str:
    """Get database URL with proper formatting"""
    return settings.database_url

def get_redis_url() -> str:
    """Get Redis URL with proper formatting"""
    return settings.redis_url

def is_development() -> bool:
    """Check if running in development mode"""
    return settings.environment == "development" or settings.debug

def is_production() -> bool:
    """Check if running in production mode"""
    return settings.environment == "production" and not settings.debug