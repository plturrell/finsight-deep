#!/bin/bash

echo "Checking Digital Human UI Status..."
echo "================================="

# Check if backend is running
echo -n "Backend server (port 8000): "
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Running"
else
    echo "✗ Not running"
fi

# Check if frontend is running
echo -n "Frontend server (port 8080): "
if curl -s http://localhost:8080 > /dev/null 2>&1; then
    echo "✓ Running"
else
    echo "✗ Not running"
fi

# Check WebSocket endpoint
echo -n "WebSocket endpoint: "
if curl -s http://localhost:8000/ws | grep -q "Not a websocket"; then
    echo "✓ Available (requires WebSocket upgrade)"
else
    echo "✗ Not available"
fi

echo "================================="
echo "To start the UI, run: ./start.sh"
echo "To access the UI: http://localhost:8080/digital_human_interface.html"