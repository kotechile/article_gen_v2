#!/bin/bash

# Change directory to the folder containing this script
cd "$(dirname "$0")"

# Set terminal title
echo -ne "\033]0;ArtiVids Local Studio\007"

echo "=================================================="
echo "🚀 Starting ArtiVids Content Studio locally..."
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed on this system."
    echo "Please download and install it from: https://www.python.org/downloads/"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed on this system."
    echo "Please download and install it from: https://nodejs.org/"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Set environment variables
export RENDER_ON_LAMBDA=false
export FLASK_ENV=development

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment (venv)..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade python dependencies
echo "🐍 Checking/installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Start backend in the background
echo "🔥 Starting Flask backend (port 5001)..."
export PYTHONPATH=.
python3 -m src.api.app > /tmp/artivids_backend.log 2>&1 &
BACKEND_PID=$!

# Navigate to frontend
cd frontend

# Install frontend dependencies if node_modules is missing
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend Node dependencies (npm install)..."
    npm install
fi

# Start frontend dev server
echo "⚡ Starting Vite frontend (port 5173)..."
npm run dev > /tmp/artivids_frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait 3 seconds for Vite server to boot, then open browser
sleep 3
open "http://localhost:5173/video-studio"

echo "=================================================="
echo "✔ Local environment is running successfully!"
echo "  - Backend running (PID: $BACKEND_PID)"
echo "  - Frontend running (PID: $FRONTEND_PID)"
echo "  - Logs saved to /tmp/artivids_backend.log & /tmp/artivids_frontend.log"
echo "  - Keep this terminal window open."
echo "  - Press Ctrl+C in this terminal to shut down both."
echo "=================================================="

# Handle graceful shutdown on Ctrl+C or terminal close
cleanup() {
    echo -e "\n🛑 Shutting down local services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Wait for background processes to keep terminal open
wait
