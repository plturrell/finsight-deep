#!/bin/bash

# Script to start both frontend and backend services

echo "Starting AIQToolkit services..."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Kill any existing processes on ports 3002 and 8000
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
lsof -ti:3002 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 2

# Navigate to project root
cd "$(dirname "$0")/.."

# Start the frontend UI
echo -e "${GREEN}Starting Frontend UI on port 3002...${NC}"
cd external/aiqtoolkit-opensource-ui
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..

# Give frontend time to start
sleep 5

# Start the backend API
echo -e "${GREEN}Starting Backend API on port 8000...${NC}"
source .venv/bin/activate
python scripts/run_simple_api.py > api.log 2>&1 &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Check if services are running
echo -e "${YELLOW}Checking services...${NC}"

if kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${GREEN}✓ Frontend is running (PID: $FRONTEND_PID)${NC}"
else
    echo -e "${RED}✗ Frontend failed to start${NC}"
fi

if kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${GREEN}✓ Backend is running (PID: $BACKEND_PID)${NC}"
else
    echo -e "${RED}✗ Backend failed to start${NC}"
fi

echo -e "\n${GREEN}Services started successfully!${NC}"
echo -e "Frontend UI: ${YELLOW}http://localhost:3002${NC}"
echo -e "Backend API: ${YELLOW}http://localhost:8000${NC}"
echo -e "API Docs: ${YELLOW}http://localhost:8000/docs${NC}"
echo -e "\nLogs:"
echo -e "Frontend: external/aiqtoolkit-opensource-ui/frontend.log"
echo -e "Backend: api.log"
echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}"

# Save PIDs to file for later cleanup
echo "$FRONTEND_PID" > .frontend.pid
echo "$BACKEND_PID" > .backend.pid

# Function to cleanup on exit
cleanup() {
    echo -e "\n${RED}Stopping services...${NC}"
    kill $FRONTEND_PID 2>/dev/null
    kill $BACKEND_PID 2>/dev/null
    rm -f .frontend.pid .backend.pid
    echo -e "${GREEN}Services stopped${NC}"
}

# Trap Ctrl+C and clean up
trap cleanup EXIT INT TERM

# Keep script running
while true; do
    sleep 1
done