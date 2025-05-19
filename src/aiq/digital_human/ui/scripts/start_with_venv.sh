#!/bin/bash

# Digital Human UI Launcher with Virtual Environment
echo "🚀 Starting Digital Human UI with virtual environment..."

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$DIR/venv/bin/activate"

# Start backend
echo "Starting backend server..."
cd "$DIR/backend"
python websocket_server.py &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Start frontend
echo "Starting frontend server..."
cd "$DIR/frontend"
python -m http.server 8080 &
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
trap "kill $BACKEND_PID $FRONTEND_PID; deactivate; exit" INT

# Wait
wait