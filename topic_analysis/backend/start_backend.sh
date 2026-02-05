#!/bin/bash

# Backend Startup Script
# This script starts the Trend Analysis backend server

cd "$(dirname "$0")"

echo "🚀 Starting Trend Analysis Backend..."
echo "📍 Directory: $(pwd)"
echo "🐍 Python: $(which python3)"
echo ""

# Use the venv Python to ensure correct dependencies
./venv/bin/python main.py

