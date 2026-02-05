#!/usr/bin/env python3
"""
Environment Setup Script
Sets up environment variables and initializes the application
"""
import os
import sys
from pathlib import Path
import subprocess

def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found. Please run setup-database.py first")
        return False
    
    # Load .env file
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
    
    print("✅ Environment variables loaded from .env file")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    try:
        # Install backend dependencies
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✅ Backend dependencies installed")
        
        # Install development dependencies
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-dev.txt'], check=True)
        print("✅ Development dependencies installed")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_database_migrations():
    """Run database migrations"""
    print("🔄 Running database migrations...")
    
    try:
        # Run Alembic migrations
        subprocess.run(['alembic', 'upgrade', 'head'], check=True)
        print("✅ Database migrations completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run migrations: {e}")
        return False

def create_admin_user():
    """Create default admin user"""
    print("👤 Creating admin user...")
    
    try:
        # Import and run user creation
        sys.path.insert(0, str(Path('src')))
        from core.database_init import create_default_user
        
        create_default_user()
        print("✅ Admin user created")
        return True
    except Exception as e:
        print(f"❌ Failed to create admin user: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    print("🔍 Testing database connection...")
    
    try:
        sys.path.insert(0, str(Path('src')))
        from core.database import check_db_connection
        
        if check_db_connection():
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

def test_redis_connection():
    """Test Redis connection"""
    print("🔍 Testing Redis connection...")
    
    try:
        sys.path.insert(0, str(Path('src')))
        from core.redis import check_redis_connection
        
        if check_redis_connection():
            print("✅ Redis connection successful")
            return True
        else:
            print("❌ Redis connection failed")
            return False
    except Exception as e:
        print(f"❌ Redis connection test failed: {e}")
        return False

def start_development_server():
    """Start the development server"""
    print("🚀 Starting development server...")
    
    try:
        # Start FastAPI server
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'src.main:app', 
            '--reload', 
            '--host', '0.0.0.0', 
            '--port', '8000'
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up TrendTap Backend")
    print("=" * 50)
    
    # Load environment variables
    if not load_env_file():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Test database connection
    if not test_database_connection():
        print("❌ Database connection failed. Please check your database setup.")
        return False
    
    # Test Redis connection
    if not test_redis_connection():
        print("❌ Redis connection failed. Please check your Redis setup.")
        return False
    
    # Run migrations
    if not run_database_migrations():
        return False
    
    # Create admin user
    if not create_admin_user():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Backend setup complete!")
    print("=" * 50)
    print("✅ Dependencies installed")
    print("✅ Database connected and migrated")
    print("✅ Redis connected")
    print("✅ Admin user created")
    print("\n🚀 Starting development server...")
    print("📱 Backend will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    
    # Start server
    start_development_server()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)