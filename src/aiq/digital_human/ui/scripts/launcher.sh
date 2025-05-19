#!/bin/bash

# Digital Human UI Launcher - Starts both frontend and backend servers

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Digital Human UI Launcher${NC}"
echo "Starting servers..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found!${NC}"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Kill any existing processes on ports 8000 and 8080
echo "Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

# Start backend server in background
echo -e "${GREEN}Starting backend server on port 8000...${NC}"
cd backend
python simple_websocket_server.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 2

# Start frontend server in background
echo -e "${GREEN}Starting frontend server on port 8080...${NC}"
cd frontend
python -m http.server 8080 &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}Servers started successfully!${NC}"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo -e "${GREEN}Access the UI at: http://localhost:8080/digital_human_interface.html${NC}"
echo ""
echo "Press Ctrl+C to stop all servers"

# Function to handle shutdown
cleanup() {
    echo -e "\n${RED}Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Servers stopped."
    exit 0
}

# Set up trap to handle Ctrl+C
trap cleanup INT TERM

# Keep script running
while true; do
    sleep 1
done