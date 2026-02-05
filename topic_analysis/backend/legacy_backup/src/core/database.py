"""
Database configuration and connection management for TrendTap
"""

from sqlalchemy import create_engine, MetaData, Index, text, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import os
from typing import Generator, Optional
import structlog
import time
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = structlog.get_logger()

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
)


# Convert psycopg2 URL to psycopg URL for Python 3.13 compatibility
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://")

# Create engine with enhanced connection pooling
# SQLite doesn't support connection pooling, so we adjust for that
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
        connect_args={"check_same_thread": False}  # SQLite specific
    )
else:
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "30")),
        pool_pre_ping=True,
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
        pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
        connect_args={
            "options": "-c timezone=utc",
            "application_name": "trendtap_backend"
        }
    )

# Add connection event listeners for monitoring
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set connection-level settings"""
    if "postgresql" in DATABASE_URL:
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET timezone TO 'UTC'")
            cursor.execute("SET statement_timeout = '30s'")
            cursor.execute("SET lock_timeout = '10s'")

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log connection checkout"""
    logger.debug("Connection checked out from pool")

@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log connection checkin"""
    logger.debug("Connection checked in to pool")

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()


def get_db() -> Generator:
    """
    Dependency to get database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error("Database session error", error=str(e))
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database session with automatic cleanup
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error("Database session error", error=str(e))
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session_manual() -> Session:
    """
    Get a database session that must be manually closed
    """
    return SessionLocal()


def close_db_session(session: Session) -> None:
    """
    Close a database session
    """
    try:
        session.close()
    except Exception as e:
        logger.error("Error closing database session", error=str(e))


def execute_in_transaction(func, *args, **kwargs):
    """
    Execute a function within a database transaction
    """
    with get_db_session() as db:
        try:
            result = func(db, *args, **kwargs)
            return result
        except Exception as e:
            logger.error("Transaction error", error=str(e))
            raise


def init_db() -> None:
    """
    Initialize database tables
    """
    try:
        # Import all models to ensure they are registered
        from ..models import user, affiliate_research, trend_analysis, keyword_data, content_ideas, software_solutions, content_calendar
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Create indexes
        create_indexes()
        
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise


def create_indexes() -> None:
    """
    Create database indexes for performance optimization
    """
    try:
        with engine.connect() as connection:
            # User table indexes
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
                CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
                CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
                CREATE INDEX IF NOT EXISTS idx_users_subscription_tier ON users(subscription_tier);
                CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
            """))
            
            # AffiliateResearch table indexes
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_affiliate_researches_user_id ON affiliate_researches(user_id);
                CREATE INDEX IF NOT EXISTS idx_affiliate_researches_status ON affiliate_researches(status);
                CREATE INDEX IF NOT EXISTS idx_affiliate_researches_created_at ON affiliate_researches(created_at);
                CREATE INDEX IF NOT EXISTS idx_affiliate_researches_topic ON affiliate_researches(topic);
            """))
            
            # TrendAnalysis table indexes
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_trend_analyses_user_id ON trend_analyses(user_id);
                CREATE INDEX IF NOT EXISTS idx_trend_analyses_affiliate_research_id ON trend_analyses(affiliate_research_id);
                CREATE INDEX IF NOT EXISTS idx_trend_analyses_status ON trend_analyses(status);
                CREATE INDEX IF NOT EXISTS idx_trend_analyses_created_at ON trend_analyses(created_at);
            """))
            
            # KeywordData table indexes
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_keyword_data_user_id ON keyword_data(user_id);
                CREATE INDEX IF NOT EXISTS idx_keyword_data_trend_analysis_id ON keyword_data(trend_analysis_id);
                CREATE INDEX IF NOT EXISTS idx_keyword_data_status ON keyword_data(status);
                CREATE INDEX IF NOT EXISTS idx_keyword_data_source ON keyword_data(source);
                CREATE INDEX IF NOT EXISTS idx_keyword_data_created_at ON keyword_data(created_at);
            """))
            
            # ContentIdeas table indexes
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_content_ideas_user_id ON content_ideas(user_id);
                CREATE INDEX IF NOT EXISTS idx_content_ideas_keyword_data_id ON content_ideas(keyword_data_id);
                CREATE INDEX IF NOT EXISTS idx_content_ideas_content_type ON content_ideas(content_type);
                CREATE INDEX IF NOT EXISTS idx_content_ideas_status ON content_ideas(status);
                CREATE INDEX IF NOT EXISTS idx_content_ideas_priority_score ON content_ideas(priority_score);
                CREATE INDEX IF NOT EXISTS idx_content_ideas_created_at ON content_ideas(created_at);
            """))
            
            # SoftwareSolutions table indexes
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_software_solutions_user_id ON software_solutions(user_id);
                CREATE INDEX IF NOT EXISTS idx_software_solutions_keyword_data_id ON software_solutions(keyword_data_id);
                CREATE INDEX IF NOT EXISTS idx_software_solutions_software_type ON software_solutions(software_type);
                CREATE INDEX IF NOT EXISTS idx_software_solutions_development_status ON software_solutions(development_status);
                CREATE INDEX IF NOT EXISTS idx_software_solutions_complexity_score ON software_solutions(complexity_score);
                CREATE INDEX IF NOT EXISTS idx_software_solutions_priority_score ON software_solutions(priority_score);
                CREATE INDEX IF NOT EXISTS idx_software_solutions_created_at ON software_solutions(created_at);
            """))
            
            # ContentCalendar table indexes
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_content_calendar_user_id ON content_calendar(user_id);
                CREATE INDEX IF NOT EXISTS idx_content_calendar_content_idea_id ON content_calendar(content_idea_id);
                CREATE INDEX IF NOT EXISTS idx_content_calendar_software_solution_id ON content_calendar(software_solution_id);
                CREATE INDEX IF NOT EXISTS idx_content_calendar_entry_type ON content_calendar(entry_type);
                CREATE INDEX IF NOT EXISTS idx_content_calendar_status ON content_calendar(status);
                CREATE INDEX IF NOT EXISTS idx_content_calendar_scheduled_date ON content_calendar(scheduled_date);
                CREATE INDEX IF NOT EXISTS idx_content_calendar_priority ON content_calendar(priority);
                CREATE INDEX IF NOT EXISTS idx_content_calendar_created_at ON content_calendar(created_at);
            """))
            
            # Composite indexes for common queries
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_users_active_subscription ON users(is_active, subscription_tier);
                CREATE INDEX IF NOT EXISTS idx_affiliate_researches_user_status ON affiliate_researches(user_id, status);
                CREATE INDEX IF NOT EXISTS idx_trend_analyses_user_status ON trend_analyses(user_id, status);
                CREATE INDEX IF NOT EXISTS idx_keyword_data_user_status ON keyword_data(user_id, status);
                CREATE INDEX IF NOT EXISTS idx_content_ideas_user_type ON content_ideas(user_id, content_type);
                CREATE INDEX IF NOT EXISTS idx_software_solutions_user_type ON software_solutions(user_id, software_type);
                CREATE INDEX IF NOT EXISTS idx_content_calendar_user_date ON content_calendar(user_id, scheduled_date);
                CREATE INDEX IF NOT EXISTS idx_content_calendar_user_status ON content_calendar(user_id, status);
            """))
            
            logger.info("Database indexes created successfully")
            
    except Exception as e:
        logger.error("Failed to create database indexes", error=str(e))
        raise


def check_db_connection() -> bool:
    """
    Check if database connection is healthy
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database connection healthy")
        return True
    except Exception as e:
        logger.error("Database connection failed", error=str(e))
        return False


def get_connection_pool_status() -> dict:
    """
    Get connection pool status
    """
    pool = engine.pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.invalid()
    }


def optimize_database() -> None:
    """
    Optimize database performance
    """
    try:
        with engine.connect() as connection:
            # Update table statistics
            connection.execute(text("ANALYZE users;"))
            connection.execute(text("ANALYZE affiliate_researches;"))
            connection.execute(text("ANALYZE trend_analyses;"))
            connection.execute(text("ANALYZE keyword_data;"))
            connection.execute(text("ANALYZE content_ideas;"))
            connection.execute(text("ANALYZE software_solutions;"))
            connection.execute(text("ANALYZE content_calendar;"))
            
            logger.info("Database optimization completed")
            
    except Exception as e:
        logger.error("Failed to optimize database", error=str(e))
        raise


def get_database_stats() -> dict:
    """
    Get database statistics
    """
    try:
        with engine.connect() as connection:
            # Get table sizes
            result = connection.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    correlation
                FROM pg_stats 
                WHERE schemaname = 'public'
                ORDER BY tablename, attname;
            """))
            
            stats = {}
            for row in result:
                table = row.tablename
                if table not in stats:
                    stats[table] = {}
                stats[table][row.attname] = {
                    "n_distinct": row.n_distinct,
                    "correlation": row.correlation
                }
            
            return stats
            
    except Exception as e:
        logger.error("Failed to get database stats", error=str(e))
        return {}


def cleanup_old_data(days: int = 30) -> None:
    """
    Clean up old data (for maintenance)
    """
    try:
        with engine.connect() as connection:
            # Clean up old failed records
            connection.execute(text("""
                DELETE FROM affiliate_researches 
                WHERE status = 'failed' 
                AND created_at < NOW() - INTERVAL '%s days';
            """ % days))
            
            connection.execute(text("""
                DELETE FROM trend_analyses 
                WHERE status = 'failed' 
                AND created_at < NOW() - INTERVAL '%s days';
            """ % days))
            
            connection.execute(text("""
                DELETE FROM keyword_data 
                WHERE status = 'failed' 
                AND created_at < NOW() - INTERVAL '%s days';
            """ % days))
            
            logger.info(f"Cleaned up old data older than {days} days")
            
    except Exception as e:
        logger.error("Failed to cleanup old data", error=str(e))
        raise