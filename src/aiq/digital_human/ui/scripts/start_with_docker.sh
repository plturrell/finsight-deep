#!/bin/bash
# Docker-based Startup Script for Digital Human Backend
# Manages all dependencies automatically

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

# Logging
log() {
    echo -e "${2:-$BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Error handling
error_exit() {
    log "ERROR: $1" "$RED"
    exit 1
}

# Check Docker installation
check_docker() {
    log "Checking Docker installation..." "$YELLOW"
    
    if ! command -v docker >/dev/null 2>&1; then
        error_exit "Docker not found. Please install Docker first."
    fi
    
    if ! command -v docker-compose >/dev/null 2>&1; then
        error_exit "Docker Compose not found. Please install Docker Compose first."
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker daemon is not running. Please start Docker."
    fi
    
    log "Docker is ready" "$GREEN"
}

# Create environment file
create_env_file() {
    log "Creating environment file..." "$YELLOW"
    
    ENV_FILE="$PROJECT_ROOT/.env"
    
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" <<EOF
# Digital Human Backend Environment Variables
ENVIRONMENT=production
JWT_SECRET=$(openssl rand -hex 32)
ENABLE_CONSENSUS=true
ENABLE_NVIDIA=false
ENABLE_MCP=true
LOG_LEVEL=INFO
WORKERS=4
GRAFANA_PASSWORD=admin

# API Keys (add your own)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
EOF
        log "Created .env file with defaults" "$GREEN"
    else
        log "Using existing .env file" "$GREEN"
    fi
}

# Build Docker images
build_images() {
    log "Building Docker images..." "$YELLOW"
    
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose.prod.yml build
    
    log "Docker images built successfully" "$GREEN"
}

# Start services
start_services() {
    log "Starting all services..." "$YELLOW"
    
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose.prod.yml up -d
    
    log "Services started" "$GREEN"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..." "$YELLOW"
    
    # Wait for backend health check
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf http://localhost:8000/health >/dev/null; then
            log "Backend is ready!" "$GREEN"
            break
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        error_exit "Backend failed to start. Check logs with: docker-compose logs backend"
    fi
}

# Show service status
show_status() {
    log "Service Status:" "$YELLOW"
    docker-compose -f docker-compose.prod.yml ps
    
    # Check health
    log "\nHealth Check:" "$YELLOW"
    curl -s http://localhost:8000/health | jq . || true
}

# Stop all services
stop_services() {
    log "Stopping all services..." "$YELLOW"
    
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose.prod.yml down
    
    log "All services stopped" "$GREEN"
}

# View logs
view_logs() {
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose.prod.yml logs -f "$1"
}

# Main function
main() {
    log "Digital Human Docker Startup Script" "$GREEN"
    log "===================================" "$GREEN"
    
    COMMAND=${1:-start}
    
    case $COMMAND in
        start)
            check_docker
            create_env_file
            build_images
            start_services
            wait_for_services
            show_status
            
            log "\n===================================" "$GREEN"
            log "All services started successfully!" "$GREEN"
            log "===================================" "$GREEN"
            log "Access URLs:" "$BLUE"
            log "API: http://localhost:8000" "$BLUE"
            log "Health: http://localhost:8000/health" "$BLUE"
            log "Metrics: http://localhost:8000/metrics" "$BLUE"
            log "Prometheus: http://localhost:9090" "$BLUE"
            log "Grafana: http://localhost:3000 (admin/admin)" "$BLUE"
            log "===================================" "$GREEN"
            log "Commands:" "$YELLOW"
            log "View logs: $0 logs [service]" "$YELLOW"
            log "Stop all: $0 stop" "$YELLOW"
            log "Restart: $0 restart" "$YELLOW"
            ;;
            
        stop)
            stop_services
            ;;
            
        restart)
            stop_services
            sleep 2
            main start
            ;;
            
        status)
            show_status
            ;;
            
        logs)
            view_logs "$2"
            ;;
            
        build)
            build_images
            ;;
            
        shell)
            # Open shell in backend container
            docker-compose -f docker-compose.prod.yml exec backend /bin/bash
            ;;
            
        *)
            log "Usage: $0 {start|stop|restart|status|logs|build|shell}" "$YELLOW"
            log "  start   - Start all services" "$YELLOW"
            log "  stop    - Stop all services" "$YELLOW"
            log "  restart - Restart all services" "$YELLOW"
            log "  status  - Show service status" "$YELLOW"
            log "  logs    - View logs (optionally specify service)" "$YELLOW"
            log "  build   - Build Docker images" "$YELLOW"
            log "  shell   - Open shell in backend container" "$YELLOW"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"