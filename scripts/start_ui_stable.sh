#!/bin/bash

# More stable startup script for AIQToolkit UI

echo "🚀 Starting AIQToolkit Standard UI..."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Cleanup function
cleanup() {
    echo -e "\n${RED}Stopping services...${NC}"
    # Kill any processes using our ports
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3001 | xargs kill -9 2>/dev/null || true
    lsof -ti:3002 | xargs kill -9 2>/dev/null || true
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    echo -e "${GREEN}Services stopped${NC}"
}

# Trap exit signals
trap cleanup EXIT INT TERM

# Clean up first
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
cleanup

# Start Backend API
echo -e "${GREEN}Starting Backend API...${NC}"
cd "$PROJECT_ROOT"
source .venv/bin/activate

# Create a simple API script that doesn't crash
cat > simple_api_stable.py << 'EOF'
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import asyncio
import json

app = FastAPI(title="AIQToolkit API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/v1/chat")
async def chat(request: Dict):
    # Simple echo response
    return {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"Echo: {request.get('message', 'No message')}"
            }
        }]
    }

@app.websocket("/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            await websocket.send_json({
                "type": "message",
                "content": f"Echo: {data.get('content', 'No content')}"
            })
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    print("Starting API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

python simple_api_stable.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✓ Backend API is running${NC}"
else
    echo -e "${RED}✗ Backend failed to start${NC}"
    exit 1
fi

# Start Frontend UI
echo -e "${GREEN}Starting Frontend UI...${NC}"
cd "$PROJECT_ROOT/external/aiqtoolkit-opensource-ui"

# Create a simple startup script for the frontend
cat > start_frontend.js << 'EOF'
const { spawn } = require('child_process');

// Start next dev
const next = spawn('npm', ['run', 'dev'], {
    stdio: 'inherit',
    env: { ...process.env }
});

// Handle exit
process.on('SIGINT', () => {
    next.kill('SIGINT');
    process.exit();
});

process.on('SIGTERM', () => {
    next.kill('SIGTERM');
    process.exit();
});
EOF

node start_frontend.js &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

echo -e "\n${GREEN}AIQToolkit UI is running!${NC}"
echo -e "Frontend: ${YELLOW}http://localhost:3000${NC} (or 3001/3002 if occupied)"
echo -e "Backend API: ${YELLOW}http://localhost:8000${NC}"
echo -e "API Health: ${YELLOW}http://localhost:8000/health${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop${NC}"

# Keep the script running
wait