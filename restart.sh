#!/bin/bash

# Content Generator V2 - Service Restart Script
# This script manages Flask app, Celery worker, and Redis

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_PORT=5001
REDIS_PORT=6379
LOG_DIR="logs"
PID_DIR="pids"

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

echo -e "${BLUE}🔄 Content Generator V2 - Service Restart Script${NC}"
echo "=================================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Function to check if a process is running
is_process_running() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to kill process by PID
kill_process() {
    local pid=$1
    local name=$2
    
    if [ -n "$pid" ] && is_process_running "$pid"; then
        print_info "Stopping $name (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        if is_process_running "$pid"; then
            print_warning "Force killing $name..."
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi
        
        if is_process_running "$pid"; then
            print_error "Failed to stop $name"
            return 1
        else
            print_status "$name stopped successfully"
        fi
    else
        print_info "$name is not running"
    fi
}

# Function to kill all related processes
kill_all_related_processes() {
    print_info "Killing all related Python processes..."
    
    # Kill only our specific application processes (exclude RAG server and other services)
    local python_processes=$(ps aux | grep -E "(python.*app\.py|python.*run_celery_worker\.py)" | grep -v grep | grep -v "rag\|RAG\|8080" | awk '{print $2}' || true)
    
    if [ -n "$python_processes" ]; then
        print_warning "Found related Python processes: $python_processes"
        for pid in $python_processes; do
            if is_process_running "$pid"; then
                print_info "Killing Python process (PID: $pid)..."
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    # Kill all tail processes monitoring our logs
    local tail_processes=$(ps aux | grep "tail -f.*logs" | grep -v grep | awk '{print $2}' || true)
    if [ -n "$tail_processes" ]; then
        print_warning "Found tail processes: $tail_processes"
        for pid in $tail_processes; do
            if is_process_running "$pid"; then
                print_info "Killing tail process (PID: $pid)..."
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    # Kill any multiprocessing resource trackers
    local mp_processes=$(ps aux | grep "multiprocessing.resource_tracker" | grep -v grep | awk '{print $2}' || true)
    if [ -n "$mp_processes" ]; then
        print_warning "Found multiprocessing processes: $mp_processes"
        for pid in $mp_processes; do
            if is_process_running "$pid"; then
                print_info "Killing multiprocessing process (PID: $pid)..."
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi
}

# Function to stop services
stop_services() {
    echo -e "\n${YELLOW}🛑 Stopping Services...${NC}"
    echo "========================"
    
    # First, kill all related processes comprehensively
    kill_all_related_processes
    
    # Stop Celery worker by PID file
    if [ -f "$PID_DIR/celery.pid" ]; then
        CELERY_PID=$(cat "$PID_DIR/celery.pid")
        kill_process "$CELERY_PID" "Celery Worker"
        rm -f "$PID_DIR/celery.pid"
    fi
    
    # Stop Flask app by PID file
    if [ -f "$PID_DIR/flask.pid" ]; then
        FLASK_PID=$(cat "$PID_DIR/flask.pid")
        kill_process "$FLASK_PID" "Flask App"
        rm -f "$PID_DIR/flask.pid"
    fi
    
    # Kill any remaining processes on our ports
    print_info "Checking for processes on port $APP_PORT..."
    APP_PROCESS=$(lsof -ti:$APP_PORT 2>/dev/null || true)
    if [ -n "$APP_PROCESS" ]; then
        print_warning "Found process on port $APP_PORT, killing..."
        kill -9 $APP_PROCESS 2>/dev/null || true
    fi
    
    print_info "Checking for processes on port $REDIS_PORT..."
    REDIS_PROCESS=$(lsof -ti:$REDIS_PORT 2>/dev/null || true)
    if [ -n "$REDIS_PROCESS" ]; then
        print_warning "Found process on port $REDIS_PORT, killing..."
        kill -9 $REDIS_PROCESS 2>/dev/null || true
    fi
    
    # Final cleanup - kill any remaining processes that might have been missed
    print_info "Performing final cleanup..."
    kill_all_related_processes
    
    # Clear Redis cache to remove old task results
    print_info "Clearing Redis cache..."
    if command -v redis-cli >/dev/null 2>&1; then
        redis-cli FLUSHDB >/dev/null 2>&1 || true
        print_status "Redis cache cleared"
    else
        print_warning "Redis CLI not found, cache may not be cleared"
    fi
    
    # Wait a moment for processes to fully terminate
    sleep 2
    
    print_status "All services stopped"
}

# Function to start Redis
start_redis() {
    echo -e "\n${YELLOW}🔴 Starting Redis...${NC}"
    echo "==================="
    
    # Check if Redis is already running
    if redis-cli ping >/dev/null 2>&1; then
        print_status "Redis is already running"
        return 0
    fi
    
    # Try to start Redis
    print_info "Starting Redis server..."
    if command -v redis-server >/dev/null 2>&1; then
        redis-server --daemonize yes --port $REDIS_PORT
        sleep 2
        
        # Check if Redis started
        if redis-cli ping >/dev/null 2>&1; then
            print_status "Redis started successfully"
        else
            print_error "Failed to start Redis"
            return 1
        fi
    else
        print_warning "Redis not found. Please install Redis or start it manually."
        print_info "You can install Redis with: brew install redis"
        return 1
    fi
}

# Function to start Celery worker
start_celery() {
    echo -e "\n${YELLOW}🔄 Starting Celery Worker...${NC}"
    echo "============================="
    
    print_info "Starting Celery worker..."
    nohup ./venv1/bin/python run_celery_worker.py > "$LOG_DIR/celery.log" 2>&1 &
    CELERY_PID=$!
    echo $CELERY_PID > "$PID_DIR/celery.pid"
    
    # Wait for Celery to start
    sleep 3
    
    # Check if Celery is running
    if is_process_running "$CELERY_PID"; then
        print_status "Celery worker started successfully (PID: $CELERY_PID)"
    else
        print_error "Failed to start Celery worker"
        return 1
    fi
}

# Function to start Flask app
start_flask() {
    echo -e "\n${YELLOW}🌐 Starting Flask App...${NC}"
    echo "==============================="
    
    print_info "Starting FULL Flask app (src/api/app.py) on port $APP_PORT..."
    nohup ./venv1/bin/python -m src.api.app > "$LOG_DIR/flask.log" 2>&1 &
    FLASK_PID=$!
    echo $FLASK_PID > "$PID_DIR/flask.pid"
    
    # Wait for Flask to start
    sleep 5
    
    # Check if Flask is running
    if is_process_running "$FLASK_PID"; then
        print_status "Flask app started successfully (PID: $FLASK_PID)"
        print_info "API available at: http://localhost:$APP_PORT"
        print_info "Health check: http://localhost:$APP_PORT/api/v1/health"
    else
        print_error "Failed to start Flask app"
        return 1
    fi
}

# Function to check service health
check_health() {
    echo -e "\n${YELLOW}🏥 Checking Service Health...${NC}"
    echo "============================="
    
    # Check Redis
    if redis-cli ping >/dev/null 2>&1; then
        print_status "Redis is healthy"
    else
        print_error "Redis is not responding"
        return 1
    fi
    
    # Check Celery worker
    if [ -f "$PID_DIR/celery.pid" ]; then
        CELERY_PID=$(cat "$PID_DIR/celery.pid")
        if is_process_running "$CELERY_PID"; then
            print_status "Celery worker is healthy"
        else
            print_error "Celery worker is not running"
            return 1
        fi
    else
        print_error "Celery worker PID file not found"
        return 1
    fi
    
    # Check Flask app
    if curl -s "http://localhost:$APP_PORT/api/v1/health" >/dev/null 2>&1; then
        print_status "Flask app is healthy"
        
        # Test API response
        HEALTH_RESPONSE=$(curl -s "http://localhost:$APP_PORT/api/v1/health")
        print_info "Health response: $HEALTH_RESPONSE"
    else
        print_error "Flask app is not responding"
        return 1
    fi
    
    print_status "All services are healthy!"
}

# Function to show all related processes
show_processes() {
    echo -e "\n${YELLOW}🔍 All Related Processes${NC}"
    echo "========================="
    
    # Show Python processes
    print_info "Python processes:"
    ps aux | grep -E "(python.*app\.py|python.*run_celery_worker\.py|python.*main\.py)" | grep -v grep || print_info "No Python processes found"
    
    # Show tail processes
    print_info "Tail processes:"
    ps aux | grep "tail -f.*logs" | grep -v grep || print_info "No tail processes found"
    
    # Show multiprocessing processes
    print_info "Multiprocessing processes:"
    ps aux | grep "multiprocessing.resource_tracker" | grep -v grep || print_info "No multiprocessing processes found"
    
    # Show Redis processes
    print_info "Redis processes:"
    ps aux | grep redis | grep -v grep || print_info "No Redis processes found"
}

# Function to show service status
show_status() {
    echo -e "\n${YELLOW}📊 Service Status${NC}"
    echo "=================="
    
    # Redis status
    if redis-cli ping >/dev/null 2>&1; then
        print_status "Redis: Running on port $REDIS_PORT"
    else
        print_error "Redis: Not running"
    fi
    
    # Celery status
    if [ -f "$PID_DIR/celery.pid" ]; then
        CELERY_PID=$(cat "$PID_DIR/celery.pid")
        if is_process_running "$CELERY_PID"; then
            print_status "Celery Worker: Running (PID: $CELERY_PID)"
        else
            print_error "Celery Worker: Not running"
        fi
    else
        print_error "Celery Worker: Not started"
    fi
    
    # Flask status
    if curl -s "http://localhost:$APP_PORT/api/v1/health" >/dev/null 2>&1; then
        print_status "Flask App: Running on port $APP_PORT"
    else
        print_error "Flask App: Not running"
    fi
}

# Main execution
main() {
    # Parse command line arguments
    case "${1:-restart}" in
        "stop")
            stop_services
            ;;
        "start")
            start_redis
            start_celery
            start_flask
            check_health
            ;;
        "restart")
            stop_services
            sleep 2
            start_redis
            start_celery
            start_flask
            check_health
            ;;
        "status")
            show_status
            ;;
        "health")
            check_health
            ;;
        "processes")
            show_processes
            ;;
        "clean")
            stop_services
            ;;
        "clear-cache")
            print_info "Clearing Redis cache..."
            if command -v redis-cli >/dev/null 2>&1; then
                redis-cli FLUSHDB >/dev/null 2>&1 || true
                print_status "Redis cache cleared"
            else
                print_error "Redis CLI not found"
                exit 1
            fi
            ;;
        *)
            echo "Usage: $0 {stop|start|restart|status|health|processes|clean|clear-cache}"
            echo ""
            echo "Commands:"
            echo "  stop         - Stop all services (Redis, Celery, Flask)"
            echo "  start        - Start all services (Redis, Celery, Flask)"
            echo "  restart      - Stop and start all services (default)"
            echo "  status       - Show service status"
            echo "  health       - Check service health"
            echo "  processes    - Show all related processes"
            echo "  clean        - Stop all services and clean up processes"
            echo "  clear-cache  - Clear Redis cache (remove old task results)"
            exit 1
            ;;
    esac
    
    echo -e "\n${GREEN}🎉 Operation completed successfully!${NC}"
}

# Run main function
main "$@"
