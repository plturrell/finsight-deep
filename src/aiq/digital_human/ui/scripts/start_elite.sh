#!/bin/bash

# Elite Digital Human UI Launcher - Best in Class Edition

echo "🚀 Starting Elite Digital Human UI..."
echo "===================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Change to the UI directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install fastapi uvicorn websockets
    touch .deps_installed
fi

# Kill any existing processes
echo -e "${BLUE}Cleaning up existing processes...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

# Start backend server
echo -e "${GREEN}Starting backend server on port 8000...${NC}"
cd backend
python simple_websocket_server.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 2

# Start frontend server
echo -e "${GREEN}Starting frontend server on port 8080...${NC}"
cd frontend
python -m http.server 8080 &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}Elite Digital Human UI started successfully!${NC}"
echo ""
echo -e "${BLUE}Access the Elite interface at:${NC}"
echo -e "${GREEN}http://localhost:8080/elite_interface.html${NC}"
echo ""
echo -e "Backend PID: $BACKEND_PID"
echo -e "Frontend PID: $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop all servers"

# Trap for cleanup
cleanup() {
    echo -e "\n${YELLOW}Stopping servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}Servers stopped.${NC}"
    exit 0
}

trap cleanup INT TERM

# Keep running
while true; do
    sleep 1
done