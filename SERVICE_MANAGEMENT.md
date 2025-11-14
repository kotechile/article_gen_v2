# Service Management Guide

This guide explains how to manage the Content Generator V2 services using the provided restart scripts.

## 🚀 Quick Start

### Using Bash Script (macOS/Linux)
```bash
# Restart all services (recommended)
./restart_services.sh

# Or with specific command
./restart_services.sh restart
```

### Using Python Script (Cross-platform)
```bash
# Restart all services (recommended)
python restart_services.py

# Or with specific command
python restart_services.py restart
```

## 📋 Available Commands

### 1. **Restart Services** (Default)
```bash
# Bash
./restart_services.sh restart

# Python
python restart_services.py restart
```
- Stops all running services
- Starts Redis server
- Starts Celery worker
- Starts Flask app
- Performs health checks

### 2. **Stop Services**
```bash
# Bash
./restart_services.sh stop

# Python
python restart_services.py stop
```
- Stops all running services
- Kills processes on ports 5001 and 6379
- Cleans up PID files

### 3. **Start Services**
```bash
# Bash
./restart_services.sh start

# Python
python restart_services.py start
```
- Starts Redis server
- Starts Celery worker
- Starts Flask app
- Performs health checks

### 4. **Check Status**
```bash
# Bash
./restart_services.sh status

# Python
python restart_services.py status
```
- Shows status of all services
- Displays process IDs
- Shows port information

### 5. **Health Check**
```bash
# Bash
./restart_services.sh health

# Python
python restart_services.py health
```
- Performs comprehensive health checks
- Tests Redis connectivity
- Tests Flask API endpoints
- Verifies Celery worker status

## 🔧 Service Details

### Redis Server
- **Port**: 6379
- **Purpose**: Message broker for Celery
- **Logs**: `logs/redis.log`
- **PID**: `pids/redis.pid`

### Celery Worker
- **Purpose**: Background task processing
- **Logs**: `logs/celery.log`
- **PID**: `pids/celery.pid`
- **Queues**: research, monitoring

### Flask App
- **Port**: 5001 (avoiding macOS conflicts)
- **Purpose**: REST API server
- **Logs**: `logs/flask.log`
- **PID**: `pids/flask.pid`
- **Health Check**: `http://localhost:5001/api/v1/health`

## 🐛 Troubleshooting

### Services Won't Start
1. Check if ports are already in use:
   ```bash
   lsof -i :5001  # Flask app
   lsof -i :6379  # Redis
   ```

2. Check logs for errors:
   ```bash
   tail -f logs/flask.log
   tail -f logs/celery.log
   tail -f logs/redis.log
   ```

3. Ensure Redis is installed and running:
   ```bash
   redis-cli ping
   ```

### Services Keep Crashing
1. Check system resources:
   ```bash
   top
   df -h
   ```

2. Check for Python dependency issues:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify configuration:
   ```bash
   python -c "from src.utils.config import get_config; print(get_config())"
   ```

### Port Conflicts
If you get port conflicts, you can change the ports by setting environment variables:
```bash
export PORT=5002  # Change Flask port
export REDIS_PORT=6380  # Change Redis port
./restart_services.sh restart
```

## 📊 Monitoring

### Real-time Logs
```bash
# Watch all logs
tail -f logs/*.log

# Watch specific service
tail -f logs/flask.log
tail -f logs/celery.log
```

### Process Monitoring
```bash
# Check running processes
ps aux | grep -E "(redis|celery|flask|python)"

# Check port usage
netstat -tulpn | grep -E "(5001|6379)"
```

### API Testing
```bash
# Health check
curl http://localhost:5001/api/v1/health

# API documentation
curl http://localhost:5001/api/v1/docs
```

## 🔄 Automated Restart

### Using Cron (Linux/macOS)
Add to crontab for automatic restart on system boot:
```bash
@reboot cd /path/to/content_generator_v2 && ./restart_services.sh start
```

### Using systemd (Linux)
Create a systemd service file for automatic management.

### Using launchd (macOS)
Create a launchd plist file for automatic management.

## 🚨 Emergency Procedures

### Force Stop Everything
```bash
# Kill all Python processes
pkill -f "python.*run_"

# Kill Redis
pkill redis-server

# Kill processes on specific ports
lsof -ti:5001 | xargs kill -9
lsof -ti:6379 | xargs kill -9
```

### Clean Restart
```bash
# Remove PID files
rm -f pids/*.pid

# Clear logs
rm -f logs/*.log

# Restart services
./restart_services.sh restart
```

## 📝 Notes

- The scripts automatically create necessary directories (`logs/`, `pids/`)
- All services run in the background with proper PID tracking
- Logs are automatically rotated and stored in the `logs/` directory
- The Flask app runs on port 5001 to avoid conflicts with macOS AirPlay
- Redis must be installed and accessible for the system to work
- The Python script requires the `psutil` package for process management
