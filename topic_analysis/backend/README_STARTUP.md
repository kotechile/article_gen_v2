# Backend Server - Startup Guide

## ✅ Current Status
The backend server is **working** and running on `http://localhost:8000`

## 🚀 Quick Start

### Option 1: Using the startup script (recommended)
```bash
cd trend-analysis-platform/backend
./start_backend.sh
```

### Option 2: Using venv Python directly
```bash
cd trend-analysis-platform/backend
./venv/bin/python main.py
```

### Option 3: Activate venv and run
```bash
cd trend-analysis-platform/backend
source venv/bin/activate
python main.py
```

## 📡 Server Information

- **URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🔧 What Was Fixed

1. **Missing Dependencies**: Installed FastAPI, uvicorn, and other required packages compatible with Python 3.13
2. **Root Endpoint**: Added a root endpoint (`/`) that provides API information
3. **Python Environment**: Configured to use the virtual environment Python interpreter

## 🧪 Test the Server

```bash
# Check root endpoint
curl http://localhost:8000/

# Check health endpoint
curl http://localhost:8000/health

# View interactive API documentation
open http://localhost:8000/docs
```

## 📋 Available Endpoints

The backend provides the following main endpoints:

- `/` - Root endpoint with API information
- `/health` - Health check
- `/api/topic-decomposition` - Topic decomposition
- `/api/enhanced-topic-decomposition` - Enhanced topic decomposition
- `/api/affiliate-research` - Affiliate research
- `/api/content-ideas/generate` - Generate content ideas
- `/api/content-ideas/list` - List content ideas
- `/api/keywords/generate` - Generate keywords
- And more...

## 🐛 Troubleshooting

### Server won't start
- Make sure you're in the backend directory
- Ensure the venv has all dependencies: `./venv/bin/pip install fastapi uvicorn[standard]`
- Check if port 8000 is already in use: `lsof -i :8000`

### Module not found errors
- Activate the virtual environment: `source venv/bin/activate`
- Install missing dependencies: `pip install <package-name>`

## 📝 Notes

- The server logs to `backend.log` in the backend directory
- Supabase integration is configured and working
- CORS is enabled for all origins (development mode)

