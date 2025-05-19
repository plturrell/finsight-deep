#!/bin/bash

# Digital Human UI Demo Launcher
echo "🚀 Starting Digital Human UI Demo..."

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment if exists
if [ -d "$DIR/venv" ]; then
    source "$DIR/venv/bin/activate"
fi

# Stop any existing processes on the ports
echo "Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8080 | xargs kill -9 2>/dev/null
sleep 1

# Start backend with simple server
echo "Starting backend server..."
cd "$DIR/backend"
python simple_websocket_server.py &
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
echo "✅ Digital Human UI Demo is running!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:8080"
echo "UI: http://localhost:8080/digital_human_interface.html"
echo ""
echo "Press Ctrl+C to stop"

# Handle Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT

# Wait
wait