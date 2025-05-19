#!/bin/bash

# Quick start for Digital Human UI
echo "🚀 Starting Digital Human UI..."

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start backend
echo "Starting backend..."
cd "$DIR/backend"
python3 websocket_server.py &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Start frontend
echo "Starting frontend..."
cd "$DIR/frontend"
python3 -m http.server 8080 &
FRONTEND_PID=$!

# Give frontend time to start
sleep 2

# Open browser
echo "Opening browser..."
open http://localhost:8080/digital_human_interface.html || xdg-open http://localhost:8080/digital_human_interface.html || echo "Please open http://localhost:8080/digital_human_interface.html in your browser"

echo ""
echo "✅ Digital Human UI is running!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop"

# Handle Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT

# Wait
wait