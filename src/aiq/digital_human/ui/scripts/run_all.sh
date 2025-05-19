#!/bin/bash

# Digital Human UI Complete Startup Script
echo "🚀 Starting Digital Human UI with 2D Avatar..."

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Kill any existing processes
echo "Cleaning up existing processes..."
pkill -f "simple_websocket_server.py" 2>/dev/null
pkill -f "python3 -m http.server 8080" 2>/dev/null
sleep 2

# Activate virtual environment if exists
if [ -d "$DIR/venv" ]; then
    source "$DIR/venv/bin/activate"
    echo "✓ Virtual environment activated"
fi

# Start backend server
echo "Starting backend server..."
cd "$DIR/backend"
python simple_websocket_server.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 3

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    echo "✓ Backend server is running"
else
    echo "❌ Backend server failed to start"
    exit 1
fi

# Start frontend server
echo "Starting frontend server..."
cd "$DIR/frontend"
python3 -m http.server 8080 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend to start
sleep 2

# Check if frontend is running
if ps -p $FRONTEND_PID > /dev/null; then
    echo "✓ Frontend server is running"
else
    echo "❌ Frontend server failed to start"
    kill $BACKEND_PID
    exit 1
fi

# Open browser
echo "Opening browser..."
open http://localhost:8080/digital_human_interface.html

echo ""
echo "✅ Digital Human UI is running!"
echo "================================"
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:8080"
echo "UI:       http://localhost:8080/digital_human_interface.html"
echo ""
echo "Logs:"
echo "  Backend:  $DIR/backend/backend.log"
echo "  Frontend: $DIR/frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"

# Handle shutdown
trap "echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

# Keep script running
while true; do
    sleep 1
done