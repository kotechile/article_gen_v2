"""
Database initialization and setup.
"""
import logging
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.core.database import Base, get_engine, test_connection, health_check, cleanup_engine
from src.models.user import User, UserRole
from src.models.user_session import UserSession
from src.models.password_reset import PasswordReset
from src.models.authentication_log import AuthenticationLog, AuthenticationEventType
from src.services.password_service import password_service
from src.services.migration_service import migration_service

logger = logging.getLogger(__name__)

class DatabaseInitializer:
    """Database initialization and setup manager."""
    
    def __init__(self):
        self.database_url = settings.get_database_url()
        self.engine = None
        self.SessionLocal = None
        self.initialized = False
    
    def create_engine(self):
        """Create database engine with proper configuration."""
        try:
            # Use the enhanced database engine from database.py
            self.engine = get_engine()
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database engine created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test database connection."""
        return test_connection()
    
    def create_tables(self) -> bool:
        """Create all database tables."""
        if not self.engine:
            logger.error("Database engine not created")
            return False
        
        try:
            # Import all models to ensure they are registered
            from src.models.user import User
            from src.models.user_session import UserSession
            from src.models.password_reset import PasswordReset
            from src.models.authentication_log import AuthenticationLog
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to create database tables: {e}")
            return False
    
    def drop_tables(self) -> bool:
        """Drop all database tables (use with caution)."""
        if not self.engine:
            logger.error("Database engine not created")
            return False
        
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop database tables: {e}")
            return False
    
    def create_admin_user(self) -> bool:
        """Create default admin user if it doesn't exist."""
        if not self.SessionLocal:
            logger.error("Database session not created")
            return False
        
        try:
            db = self.SessionLocal()
            
            # Check if admin user already exists
            admin_user = db.query(User).filter(User.email == "admin@trendanalysis.com").first()
            if admin_user:
                logger.info("Admin user already exists")
                db.close()
                return True
            
            # Create admin user
            admin_password = "AdminPassword123!"  # Should be changed in production
            hashed_password = password_service.hash_password(admin_password)
            
            admin_user = User(
                email="admin@trendanalysis.com",
                password_hash=hashed_password,
                first_name="Admin",
                last_name="User",
                role=UserRole.ADMIN,
                is_active=True,
                is_verified=True
            )
            
            db.add(admin_user)
            db.commit()
            
            logger.info("Admin user created successfully")
            logger.warning(f"Default admin password: {admin_password} - CHANGE IN PRODUCTION!")
            
            db.close()
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to create admin user: {e}")
            if 'db' in locals():
                db.rollback()
                db.close()
            return False
    
    def create_test_user(self) -> bool:
        """Create test user for development."""
        if not self.SessionLocal:
            logger.error("Database session not created")
            return False
        
        if not settings.is_development():
            logger.info("Skipping test user creation in non-development environment")
            return True
        
        try:
            db = self.SessionLocal()
            
            # Check if test user already exists
            test_user = db.query(User).filter(User.email == "test@example.com").first()
            if test_user:
                logger.info("Test user already exists")
                db.close()
                return True
            
            # Create test user
            test_password = "TestPassword123!"
            hashed_password = password_service.hash_password(test_password)
            
            test_user = User(
                email="test@example.com",
                password_hash=hashed_password,
                first_name="Test",
                last_name="User",
                role=UserRole.USER,
                is_active=True,
                is_verified=True
            )
            
            db.add(test_user)
            db.commit()
            
            logger.info("Test user created successfully")
            logger.info(f"Test user credentials: test@example.com / {test_password}")
            
            db.close()
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to create test user: {e}")
            if 'db' in locals():
                db.rollback()
                db.close()
            return False
    
    def create_sample_data(self) -> bool:
        """Create sample data for development and testing."""
        if not self.SessionLocal:
            logger.error("Database session not created")
            return False
        
        if not settings.is_development():
            logger.info("Skipping sample data creation in non-development environment")
            return True
        
        try:
            db = self.SessionLocal()
            
            # Create sample users
            sample_users = [
                {
                    "email": "john.doe@example.com",
                    "first_name": "John",
                    "last_name": "Doe",
                    "role": UserRole.USER,
                    "password": "JohnPassword123!"
                },
                {
                    "email": "jane.smith@example.com",
                    "first_name": "Jane",
                    "last_name": "Smith",
                    "role": UserRole.USER,
                    "password": "JanePassword123!"
                },
                {
                    "email": "moderator@example.com",
                    "first_name": "Moderator",
                    "last_name": "User",
                    "role": UserRole.USER,
                    "password": "ModPassword123!"
                }
            ]
            
            created_count = 0
            for user_data in sample_users:
                # Check if user already exists
                existing_user = db.query(User).filter(User.email == user_data["email"]).first()
                if existing_user:
                    continue
                
                # Create user
                hashed_password = password_service.hash_password(user_data["password"])
                user = User(
                    email=user_data["email"],
                    password_hash=hashed_password,
                    first_name=user_data["first_name"],
                    last_name=user_data["last_name"],
                    role=user_data["role"],
                    is_active=True,
                    is_verified=True
                )
                
                db.add(user)
                created_count += 1
            
            if created_count > 0:
                db.commit()
                logger.info(f"Created {created_count} sample users")
            else:
                logger.info("All sample users already exist")
            
            db.close()
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to create sample data: {e}")
            if 'db' in locals():
                db.rollback()
                db.close()
            return False
    
    def verify_tables(self) -> bool:
        """Verify that all required tables exist."""
        if not self.engine:
            logger.error("Database engine not created")
            return False
        
        try:
            with self.engine.connect() as connection:
                # Check if all required tables exist
                required_tables = [
                    "users",
                    "user_sessions", 
                    "password_resets",
                    "authentication_logs"
                ]
                
                # Check if we're using SQLite or PostgreSQL
                from src.core.config import settings
                database_url = settings.get_database_url()
                
                if "sqlite" in database_url:
                    # SQLite table verification
                    for table_name in required_tables:
                        result = connection.execute(text(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"
                        ), {"table_name": table_name})
                        
                        if not result.fetchone():
                            logger.error(f"Required table '{table_name}' does not exist")
                            return False
                else:
                    # PostgreSQL table verification
                    for table_name in required_tables:
                        result = connection.execute(text(
                            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :table_name)"
                        ), {"table_name": table_name})
                        
                        if not result.fetchone()[0]:
                            logger.error(f"Required table '{table_name}' does not exist")
                            return False
                
                logger.info("All required tables exist")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to verify tables: {e}")
            return False
    
    def get_database_info(self) -> dict:
        """Get database information and statistics."""
        if not self.engine:
            return {"error": "Database engine not created"}
        
        try:
            with self.engine.connect() as connection:
                # Get database version
                version_result = connection.execute(text("SELECT version()"))
                db_version = version_result.fetchone()[0]
                
                # Get table counts
                table_counts = {}
                tables = ["users", "user_sessions", "password_resets", "authentication_logs"]
                
                for table in tables:
                    try:
                        count_result = connection.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        table_counts[table] = count_result.fetchone()[0]
                    except:
                        table_counts[table] = 0
                
                return {
                    "database_url": self.database_url,
                    "version": db_version,
                    "table_counts": table_counts,
                    "initialized": self.initialized
                }
                
        except SQLAlchemyError as e:
            return {"error": f"Failed to get database info: {e}"}
    
    def initialize(self, create_admin: bool = True, create_test_data: bool = True) -> bool:
        """Initialize database with all setup steps."""
        logger.info("Starting database initialization...")
        
        # Step 1: Create engine
        if not self.create_engine():
            logger.error("Failed to create database engine")
            return False
        
        # Step 2: Test connection
        if not self.test_connection():
            logger.error("Failed to test database connection")
            return False
        
        # Step 3: Create tables
        if not self.create_tables():
            logger.error("Failed to create database tables")
            return False
        
        # Step 4: Verify tables
        if not self.verify_tables():
            logger.error("Failed to verify database tables")
            return False
        
        # Step 5: Create admin user
        if create_admin and not self.create_admin_user():
            logger.error("Failed to create admin user")
            return False
        
        # Step 6: Create test user (development only)
        if create_test_data and not self.create_test_user():
            logger.error("Failed to create test user")
            return False
        
        # Step 7: Create sample data (development only)
        if create_test_data and not self.create_sample_data():
            logger.error("Failed to create sample data")
            return False
        
        self.initialized = True
        logger.info("Database initialization completed successfully")
        return True
    
    def cleanup(self):
        """Clean up database resources."""
        cleanup_engine()
        self.initialized = False

# Global database initializer instance
db_initializer = DatabaseInitializer()

def init_database(create_admin: bool = True, create_test_data: bool = None) -> bool:
    """Initialize the database with default settings."""
    if create_test_data is None:
        create_test_data = settings.is_development()
    
    return db_initializer.initialize(create_admin=create_admin, create_test_data=create_test_data)

def get_database_info() -> dict:
    """Get database information."""
    return db_initializer.get_database_info()

def cleanup_database():
    """Clean up database resources."""
    db_initializer.cleanup()
