#!/bin/bash

# Auto-start script for Digital Human UI
# This script ensures everything starts properly on system boot

# Add color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Digital Human UI Auto-Start${NC}"
echo "=============================="

# Change to the script directory
cd "$(dirname "$0")"

# Check if we're in the right directory
if [ ! -f "launcher.sh" ]; then
    echo -e "${RED}Error: launcher.sh not found!${NC}"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p logs

# Start the launcher in the background and redirect output to log
echo -e "${YELLOW}Starting services...${NC}"
nohup ./launcher.sh > logs/digital_human_$(date +%Y%m%d_%H%M%S).log 2>&1 &
LAUNCHER_PID=$!

# Wait a moment for services to start
sleep 3

# Check if processes are running
echo -e "${YELLOW}Checking services...${NC}"
if ps -p $LAUNCHER_PID > /dev/null; then
    echo -e "${GREEN}✓ Launcher is running (PID: $LAUNCHER_PID)${NC}"
    
    # Check individual services
    ./check_status.sh
    
    echo -e "\n${GREEN}Digital Human UI started successfully!${NC}"
    echo -e "Access the UI at: ${GREEN}http://localhost:8080/digital_human_interface.html${NC}"
    echo -e "Logs available at: logs/"
else
    echo -e "${RED}✗ Failed to start launcher${NC}"
    echo "Check logs for errors"
    exit 1
fi

# Save PID for later shutdown
echo $LAUNCHER_PID > .launcher.pid

echo -e "\n${YELLOW}To stop the services, run:${NC}"
echo "kill \$(cat .launcher.pid)"