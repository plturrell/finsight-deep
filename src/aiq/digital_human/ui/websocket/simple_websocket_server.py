"""
Simplified WebSocket server for Digital Human UI demo
Works standalone without full AIQ toolkit dependencies
"""

import asyncio
import json
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Digital Human Demo API",
    description="Simplified API for Digital Human UI testing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Store active connections
connections = {}

@app.get("/")
async def root():
    return {"message": "Digital Human WebSocket Server", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_id = f"user_{datetime.now().timestamp()}"
    connections[connection_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "startSession":
                await websocket.send_json({
                    "type": "session_started",
                    "session_id": connection_id,
                    "status": "connected"
                })
            
            elif data.get("type") == "message" or data.get("type") == "user_message":
                # Simulate AI response
                message = data.get("content", "")
                response_text = f"I understand you said: '{message}'. As a digital financial advisor, I'm here to help with your investment questions."
                
                await websocket.send_json({
                    "type": "response",
                    "content": response_text,
                    "emotion": "confident",
                    "emotion_intensity": 0.8,
                    "processing_time": 0.5,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Send animation update
                await websocket.send_json({
                    "type": "avatar_update",
                    "animation": {
                        "expression_weights": {"smile": 0.7, "eyebrows_up": 0.3},
                        "emotion_label": "Friendly",
                        "isSpeaking": True
                    }
                })
                
                # Send metrics update
                await websocket.send_json({
                    "type": "metrics_update",
                    "metrics": {
                        "portfolio_value": 125430.50,
                        "daily_change": 2.3,
                        "risk_level": "Moderate",
                        "confidence": 0.92,
                        "historical_values": [
                            120000, 121000, 122500, 124000, 125430.50
                        ]
                    }
                })
            
            elif data.get("type") == "command":
                command = data.get("command")
                if command == "status":
                    await websocket.send_json({
                        "type": "status",
                        "data": {
                            "state": "active",
                            "connected_users": len(connections),
                            "server_time": datetime.now().isoformat()
                        }
                    })
    
    except WebSocketDisconnect:
        logger.info(f"Client {connection_id} disconnected")
        del connections[connection_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if connection_id in connections:
            del connections[connection_id]

# Mock API endpoints for the UI
@app.get("/api/user/data")
async def get_user_data():
    return {
        "portfolio": {
            "value": 125430.50,
            "change": 2340.25,
            "change_percent": 1.90
        },
        "analysis": {
            "sentiment": "positive",
            "risk_level": "moderate",
            "recommendations": []
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)