#!/bin/bash

echo "Starting Digital Human UI with 2D/3D Avatar Support..."

# Start the backend WebSocket server
echo "Starting backend server..."
cd /Users/apple/projects/AIQToolkit/src/aiq/digital_human/ui/backend
python3 websocket_server.py &
BACKEND_PID=$!

# Give backend a moment to start
sleep 2

# Start the frontend server
echo "Starting frontend server..."
cd /Users/apple/projects/AIQToolkit/src/aiq/digital_human/ui/frontend
python3 -m http.server 8080 &
FRONTEND_PID=$!

echo "Digital Human UI is running!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Open your browser to: http://localhost:8080/digital_human_interface.html"
echo ""
echo "To stop the servers, press Ctrl+C"

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait