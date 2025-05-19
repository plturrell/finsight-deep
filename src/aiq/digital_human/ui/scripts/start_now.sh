#!/bin/bash

# Simple startup script for Digital Human UI
# Uses the correct paths for our current directory structure

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Digital Human UI System${NC}"

# Get the UI directory (parent of scripts/)
UI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$UI_DIR"

# Kill any existing processes on ports 8000 and 8080
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
sleep 1

# Start the WebSocket server
echo -e "${GREEN}Starting WebSocket server on port 8000...${NC}"
python websocket/simple_websocket_server.py > logs/websocket.log 2>&1 &
WEBSOCKET_PID=$!

# Wait for WebSocket server to start
sleep 2

# Start the frontend server
echo -e "${GREEN}Starting frontend server on port 8080...${NC}"
cd frontend
python -m http.server 8080 > ../logs/frontend_server.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend server to start
sleep 2

echo -e "${GREEN}✓ All servers started successfully!${NC}"
echo ""
echo -e "${GREEN}Service Information:${NC}"
echo -e "  WebSocket Server: ${YELLOW}ws://localhost:8000${NC} (PID: $WEBSOCKET_PID)"
echo -e "  Frontend Server: ${YELLOW}http://localhost:8080${NC} (PID: $FRONTEND_PID)"
echo ""
echo -e "${GREEN}Access Points:${NC}"
echo -e "  Main UI: ${YELLOW}http://localhost:8080/digital_human_interface.html${NC}"
echo -e "  Elite UI: ${YELLOW}http://localhost:8080/elite_interface.html${NC}"
echo -e "  Research Dashboard: ${YELLOW}http://localhost:8080/research_dashboard.html${NC}"
echo ""
echo -e "${GREEN}Opening main UI in browser...${NC}"

# Open browser (macOS specific)
open http://localhost:8080/digital_human_interface.html || echo "Please open http://localhost:8080/digital_human_interface.html manually"

echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"

# Function to handle shutdown
cleanup() {
    echo -e "\n${RED}Shutting down servers...${NC}"
    kill $WEBSOCKET_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}All servers stopped.${NC}"
    exit 0
}

# Set up trap to handle Ctrl+C
trap cleanup INT TERM

# Keep script running and monitor processes
while true; do
    if ! ps -p $WEBSOCKET_PID > /dev/null; then
        echo -e "${RED}WebSocket server stopped unexpectedly!${NC}"
        break
    fi
    if ! ps -p $FRONTEND_PID > /dev/null; then
        echo -e "${RED}Frontend server stopped unexpectedly!${NC}"
        break
    fi
    sleep 5
done

cleanup