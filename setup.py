#!/usr/bin/env python3
"""
Setup script for Content Generator V2.

This script sets up the development environment and installs dependencies.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up Content Generator V2...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create virtual environment if it doesn't exist
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        if not run_command("python -m venv venv", "Virtual environment creation"):
            sys.exit(1)
    
    # Determine activation script
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_command = "venv\\Scripts\\pip"
    else:  # Unix-like
        activate_script = "venv/bin/activate"
        pip_command = "venv/bin/pip"
    
    # Install dependencies
    print("Installing dependencies...")
    if not run_command(f"{pip_command} install --upgrade pip", "Pip upgrade"):
        sys.exit(1)
    
    if not run_command(f"{pip_command} install -r requirements.txt", "Dependencies installation"):
        sys.exit(1)
    
    # Create necessary directories
    directories = ["logs", "data", "cache"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file from template...")
        env_example = Path("env.example")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            print("✓ Created .env file from template")
        else:
            print("⚠ No env.example file found, please create .env manually")
    
    print("\n✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your configuration")
    print("2. Start Redis server: redis-server")
    print("3. Start Celery worker: python run_celery.py")
    print("4. Start Flask app: python run_app.py")
    print("5. Run tests: python -m pytest tests/")


if __name__ == '__main__':
    main()
