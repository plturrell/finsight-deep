#!/bin/bash

# Elite Digital Human UI Deployment Script

echo "🚀 Deploying Elite Digital Human UI"
echo "=================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
FRONTEND_PORT=8080
BACKEND_PORT=8000
LOG_DIR="./logs"
PID_DIR="./pids"

# Create directories
mkdir -p $LOG_DIR
mkdir -p $PID_DIR

# Function to check if port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}Port $1 is already in use${NC}"
        return 1
    fi
    return 0
}

# Function to start a service
start_service() {
    local name=$1
    local command=$2
    local port=$3
    local log_file="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo -e "${YELLOW}Starting $name on port $port...${NC}"
    
    if check_port $port; then
        nohup $command > "$log_file" 2>&1 &
        local pid=$!
        echo $pid > "$PID_DIR/${name}.pid"
        sleep 2
        
        if ps -p $pid > /dev/null; then
            echo -e "${GREEN}✓ $name started successfully (PID: $pid)${NC}"
            echo -e "  Log: $log_file"
            return 0
        else
            echo -e "${RED}✗ Failed to start $name${NC}"
            cat "$log_file"
            return 1
        fi
    else
        echo -e "${RED}Cannot start $name - port $port is busy${NC}"
        return 1
    fi
}

# Change to script directory
cd "$(dirname "$0")"

# Kill any existing processes
echo -e "${BLUE}Cleaning up existing processes...${NC}"
if [ -f "$PID_DIR/backend.pid" ]; then
    kill $(cat "$PID_DIR/backend.pid") 2>/dev/null || true
fi
if [ -f "$PID_DIR/frontend.pid" ]; then
    kill $(cat "$PID_DIR/frontend.pid") 2>/dev/null || true
fi
sleep 1

# Setup Python environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
if [ ! -f ".deps_installed" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install fastapi uvicorn websockets python-multipart
    touch .deps_installed
fi

# Start Backend
cd backend
start_service "backend" "python simple_websocket_server.py" $BACKEND_PORT
backend_status=$?
cd ..

# Start Frontend
cd frontend
start_service "frontend" "python -m http.server $FRONTEND_PORT" $FRONTEND_PORT
frontend_status=$?
cd ..

# Check deployment status
echo ""
echo -e "${BLUE}Deployment Status:${NC}"
echo "=================="

if [ $backend_status -eq 0 ] && [ $frontend_status -eq 0 ]; then
    echo -e "${GREEN}✓ All services deployed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Access the Elite Digital Human UI:${NC}"
    echo -e "${GREEN}http://localhost:$FRONTEND_PORT/elite_interface.html${NC}"
    echo ""
    echo -e "${BLUE}API Documentation:${NC}"
    echo -e "${GREEN}http://localhost:$BACKEND_PORT/docs${NC}"
    echo ""
    echo -e "${YELLOW}To monitor logs:${NC}"
    echo "tail -f $LOG_DIR/*.log"
    echo ""
    echo -e "${YELLOW}To stop services:${NC}"
    echo "./stop.sh"
    
    # Create stop script
    cat > stop.sh << 'EOF'
#!/bin/bash
echo "Stopping Elite Digital Human UI..."
[ -f pids/backend.pid ] && kill $(cat pids/backend.pid) 2>/dev/null
[ -f pids/frontend.pid ] && kill $(cat pids/frontend.pid) 2>/dev/null
echo "Services stopped."
EOF
    chmod +x stop.sh
    
else
    echo -e "${RED}✗ Deployment failed!${NC}"
    echo "Check logs in $LOG_DIR for details"
    exit 1
fi

# Create status check script
cat > status.sh << 'EOF'
#!/bin/bash
echo "Elite Digital Human UI Status"
echo "============================"
echo -n "Backend:  "
if [ -f pids/backend.pid ] && ps -p $(cat pids/backend.pid) > /dev/null 2>&1; then
    echo "✓ Running (PID: $(cat pids/backend.pid))"
else
    echo "✗ Not running"
fi
echo -n "Frontend: "
if [ -f pids/frontend.pid ] && ps -p $(cat pids/frontend.pid) > /dev/null 2>&1; then
    echo "✓ Running (PID: $(cat pids/frontend.pid))"
else
    echo "✗ Not running"
fi
EOF
chmod +x status.sh

# Open browser automatically (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    sleep 2
    open "http://localhost:$FRONTEND_PORT/elite_interface.html"
fi

echo -e "${GREEN}Deployment complete!${NC}"