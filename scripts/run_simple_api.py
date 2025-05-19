#!/usr/bin/env python3
"""
Simple API server to run AIQToolkit backend for the UI
"""

import os
import sys
import logging
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="AIQToolkit Backend API",
    description="Backend API for AIQToolkit UI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "aiqtoolkit-backend"}

# Simple chat endpoint
@app.post("/chat")
async def chat(message: dict):
    # Simple echo response for testing
    return {
        "response": f"Echo: {message.get('message', 'No message')}",
        "status": "success"
    }

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # Echo back the message for testing
            await websocket.send_json({
                "type": "response",
                "data": f"Echo: {data.get('message', 'No message')}"
            })
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting AIQToolkit Backend API on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    
    # Run the server (without reload for direct execution)
    uvicorn.run(app, host="0.0.0.0", port=8000)