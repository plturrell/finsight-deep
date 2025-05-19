#!/bin/bash
# Production Startup Script for Digital Human Unified Backend
# Handles all dependencies and services in the correct order

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/venv"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_DIR="${PROJECT_ROOT}/pids"
CONFIG_DIR="${PROJECT_ROOT}/config"

# Create necessary directories
mkdir -p "$LOG_DIR" "$PID_DIR"

# Logging function
log() {
    echo -e "${2:-$BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_DIR/startup.log"
}

# Error handling
error_exit() {
    log "ERROR: $1" "$RED"
    exit 1
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Wait for service to be ready
wait_for_service() {
    local service_name=$1
    local check_command=$2
    local max_attempts=${3:-30}
    local attempt=0
    
    log "Waiting for $service_name to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if eval "$check_command" >/dev/null 2>&1; then
            log "$service_name is ready!" "$GREEN"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    error_exit "$service_name failed to start after $max_attempts attempts"
}

# Check system dependencies
check_dependencies() {
    log "Checking system dependencies..." "$YELLOW"
    
    local deps=(
        "python3"
        "redis-cli"
        "psql"
        "nginx"
        "curl"
    )
    
    for dep in "${deps[@]}"; do
        if command_exists "$dep"; then
            log "✓ $dep found" "$GREEN"
        else
            error_exit "$dep not found. Please install it first."
        fi
    done
}

# Setup Python virtual environment
setup_python_env() {
    log "Setting up Python environment..." "$YELLOW"
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc) -ne 1 ]]; then
        error_exit "Python 3.9+ required, found $PYTHON_VERSION"
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip wheel setuptools
    
    # Install requirements
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        log "Installing Python dependencies..."
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi
    
    # Install additional production dependencies
    pip install gunicorn uvloop httptools psycopg2-binary redis hiredis
}

# Start Redis
start_redis() {
    log "Starting Redis..." "$YELLOW"
    
    # Check if Redis is already running
    if redis-cli ping >/dev/null 2>&1; then
        log "Redis is already running" "$GREEN"
        return 0
    fi
    
    # Start Redis based on system
    if command_exists systemctl; then
        sudo systemctl start redis
    elif command_exists service; then
        sudo service redis-server start
    else
        # Start Redis manually
        redis-server --daemonize yes --pidfile "$PID_DIR/redis.pid" --logfile "$LOG_DIR/redis.log"
    fi
    
    wait_for_service "Redis" "redis-cli ping" 10
}

# Start PostgreSQL
start_postgresql() {
    log "Starting PostgreSQL..." "$YELLOW"
    
    # Check if PostgreSQL is already running
    if pg_isready >/dev/null 2>&1; then
        log "PostgreSQL is already running" "$GREEN"
        return 0
    fi
    
    # Start PostgreSQL based on system
    if command_exists systemctl; then
        sudo systemctl start postgresql
    elif command_exists service; then
        sudo service postgresql start
    else
        error_exit "Cannot start PostgreSQL. Please start it manually."
    fi
    
    wait_for_service "PostgreSQL" "pg_isready" 10
}

# Setup database
setup_database() {
    log "Setting up database..." "$YELLOW"
    
    # Check if database exists
    if psql -U postgres -lqt | cut -d \| -f 1 | grep -qw digital_human; then
        log "Database already exists" "$GREEN"
        return 0
    fi
    
    # Create database and user
    sudo -u postgres psql <<EOF
CREATE USER aiqtoolkit WITH PASSWORD 'aiqtoolkit_dev';
CREATE DATABASE digital_human OWNER aiqtoolkit;
GRANT ALL PRIVILEGES ON DATABASE digital_human TO aiqtoolkit;
EOF
    
    log "Database created successfully" "$GREEN"
}

# Load environment variables
load_environment() {
    log "Loading environment variables..." "$YELLOW"
    
    # Default environment file
    ENV_FILE="$PROJECT_ROOT/.env"
    
    # Use production env if specified
    if [ "$1" == "production" ] && [ -f "$PROJECT_ROOT/.env.production" ]; then
        ENV_FILE="$PROJECT_ROOT/.env.production"
    fi
    
    if [ -f "$ENV_FILE" ]; then
        log "Loading environment from $ENV_FILE"
        export $(grep -v '^#' "$ENV_FILE" | xargs)
    else
        log "No environment file found, using defaults" "$YELLOW"
        
        # Set default environment variables
        export ENVIRONMENT=${ENVIRONMENT:-development}
        export REDIS_URL=${REDIS_URL:-redis://localhost:6379}
        export DATABASE_URL=${DATABASE_URL:-postgresql://aiqtoolkit:aiqtoolkit_dev@localhost/digital_human}
        export JWT_SECRET=${JWT_SECRET:-$(openssl rand -hex 32)}
        export LOG_LEVEL=${LOG_LEVEL:-INFO}
        export ENABLE_CONSENSUS=${ENABLE_CONSENSUS:-true}
        export ENABLE_NVIDIA=${ENABLE_NVIDIA:-false}
        export ENABLE_MCP=${ENABLE_MCP:-true}
    fi
}

# Start NGINX (development mode)
start_nginx_dev() {
    log "Starting NGINX (development)..." "$YELLOW"
    
    # Create NGINX config for development
    cat > /tmp/digital-human-nginx.conf <<EOF
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server 127.0.0.1:8000;
    }
    
    server {
        listen 8080;
        server_name localhost;
        
        location /ws {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 86400;
        }
        
        location /static {
            alias $PROJECT_ROOT/frontend;
        }
        
        location / {
            proxy_pass http://backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }
    }
}
EOF
    
    # Start NGINX with custom config
    nginx -c /tmp/digital-human-nginx.conf
    
    log "NGINX started on port 8080" "$GREEN"
}

# Start the unified backend server
start_backend() {
    log "Starting Digital Human Unified Backend..." "$YELLOW"
    
    cd "$PROJECT_ROOT"
    
    # Check if already running
    if [ -f "$PID_DIR/backend.pid" ]; then
        OLD_PID=$(cat "$PID_DIR/backend.pid")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            log "Backend is already running (PID: $OLD_PID)" "$YELLOW"
            return 0
        fi
    fi
    
    # Start the server
    if [ "$ENVIRONMENT" == "production" ]; then
        # Production mode with gunicorn
        gunicorn api.unified_production_server:create_app \
            --worker-class uvicorn.workers.UvicornWorker \
            --workers 4 \
            --bind 0.0.0.0:8000 \
            --log-level info \
            --access-logfile "$LOG_DIR/access.log" \
            --error-logfile "$LOG_DIR/error.log" \
            --pid "$PID_DIR/backend.pid" \
            --daemon
    else
        # Development mode
        python api/unified_production_server.py \
            --host 0.0.0.0 \
            --port 8000 \
            --workers 1 \
            --env development \
            > "$LOG_DIR/backend.log" 2>&1 &
        
        echo $! > "$PID_DIR/backend.pid"
    fi
    
    # Wait for backend to be ready
    wait_for_service "Backend API" "curl -f http://localhost:8000/health" 30
}

# Health check
health_check() {
    log "Running health check..." "$YELLOW"
    
    # Check Redis
    if redis-cli ping >/dev/null 2>&1; then
        log "✓ Redis: Healthy" "$GREEN"
    else
        log "✗ Redis: Unhealthy" "$RED"
    fi
    
    # Check PostgreSQL
    if pg_isready >/dev/null 2>&1; then
        log "✓ PostgreSQL: Healthy" "$GREEN"
    else
        log "✗ PostgreSQL: Unhealthy" "$RED"
    fi
    
    # Check Backend API
    if curl -sf http://localhost:8000/health >/dev/null; then
        log "✓ Backend API: Healthy" "$GREEN"
        
        # Show detailed health
        HEALTH_JSON=$(curl -s http://localhost:8000/health)
        log "Health Details: $HEALTH_JSON"
    else
        log "✗ Backend API: Unhealthy" "$RED"
    fi
    
    # Check WebSocket
    if curl -sf http://localhost:8000/ws >/dev/null; then
        log "✓ WebSocket: Available" "$GREEN"
    else
        log "✗ WebSocket: Unavailable" "$RED"
    fi
}

# Stop all services
stop_all() {
    log "Stopping all services..." "$YELLOW"
    
    # Stop backend
    if [ -f "$PID_DIR/backend.pid" ]; then
        PID=$(cat "$PID_DIR/backend.pid")
        if ps -p "$PID" > /dev/null 2>&1; then
            kill "$PID"
            log "Stopped backend (PID: $PID)" "$GREEN"
        fi
        rm "$PID_DIR/backend.pid"
    fi
    
    # Stop NGINX (development)
    if pgrep -f "nginx.*digital-human" > /dev/null; then
        nginx -s stop
        log "Stopped NGINX" "$GREEN"
    fi
    
    # Stop Redis (if we started it)
    if [ -f "$PID_DIR/redis.pid" ]; then
        PID=$(cat "$PID_DIR/redis.pid")
        if ps -p "$PID" > /dev/null 2>&1; then
            kill "$PID"
            log "Stopped Redis (PID: $PID)" "$GREEN"
        fi
        rm "$PID_DIR/redis.pid"
    fi
}

# Main startup sequence
main() {
    log "Digital Human Backend Startup Script" "$GREEN"
    log "===================================" "$GREEN"
    
    # Parse arguments
    COMMAND=${1:-start}
    ENVIRONMENT=${2:-development}
    
    case $COMMAND in
        start)
            log "Starting in $ENVIRONMENT mode..." "$YELLOW"
            
            # Check dependencies
            check_dependencies
            
            # Setup Python environment
            setup_python_env
            
            # Load environment variables
            load_environment "$ENVIRONMENT"
            
            # Start services in order
            start_redis
            start_postgresql
            setup_database
            
            # Start NGINX in development mode
            if [ "$ENVIRONMENT" == "development" ]; then
                start_nginx_dev
            fi
            
            # Start backend
            start_backend
            
            # Run health check
            sleep 3
            health_check
            
            log "All services started successfully!" "$GREEN"
            log "===================================" "$GREEN"
            log "Access URLs:" "$BLUE"
            log "API: http://localhost:8000" "$BLUE"
            log "Health: http://localhost:8000/health" "$BLUE"
            log "Metrics: http://localhost:8000/metrics" "$BLUE"
            log "WebSocket: ws://localhost:8000/ws" "$BLUE"
            if [ "$ENVIRONMENT" == "development" ]; then
                log "Frontend: http://localhost:8080" "$BLUE"
                log "API Docs: http://localhost:8000/docs" "$BLUE"
            fi
            log "===================================" "$GREEN"
            
            # Show logs
            log "View logs: tail -f $LOG_DIR/backend.log" "$YELLOW"
            log "Stop all: $0 stop" "$YELLOW"
            ;;
            
        stop)
            stop_all
            log "All services stopped" "$GREEN"
            ;;
            
        restart)
            log "Restarting services..." "$YELLOW"
            stop_all
            sleep 2
            main start "$ENVIRONMENT"
            ;;
            
        status)
            health_check
            ;;
            
        logs)
            tail -f "$LOG_DIR/backend.log"
            ;;
            
        *)
            log "Usage: $0 {start|stop|restart|status|logs} [development|production]" "$YELLOW"
            exit 1
            ;;
    esac
}

# Trap signals for cleanup
trap 'log "Received interrupt signal. Cleaning up..."; stop_all; exit 130' INT TERM

# Run main function
main "$@"