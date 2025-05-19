"""WebSocket server for Digital Human interface"""

import asyncio
import json
import logging
from typing import Dict, Set, Any, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aiq.digital_human.orchestrator import DigitalHumanOrchestrator, OrchestratorConfig
from aiq.digital_human.nvidia_integration.audio2face_integration import (
    DigitalHumanNVIDIAIntegration,
    Audio2FaceConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Message(BaseModel):
    """WebSocket message format"""
    type: str
    data: Dict[str, Any]
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


class SessionManager:
    """Manage WebSocket sessions"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Connect a new WebSocket session"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.sessions[session_id] = {
            "connected_at": datetime.now().isoformat(),
            "websocket": websocket
        }
        logger.info(f"Client {session_id} connected")
    
    def disconnect(self, session_id: str):
        """Disconnect a WebSocket session"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.sessions:
            del self.sessions[session_id]
        logger.info(f"Client {session_id} disconnected")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific session"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_json(message)
    
    async def broadcast(self, message: Dict[str, Any], exclude_session: Optional[str] = None):
        """Broadcast message to all connected sessions"""
        disconnected_sessions = []
        
        for session_id, websocket in self.active_connections.items():
            if session_id != exclude_session:
                try:
                    await websocket.send_json(message)
                except WebSocketDisconnect:
                    disconnected_sessions.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            self.disconnect(session_id)


# Initialize FastAPI app
app = FastAPI(title="Digital Human WebSocket Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Initialize managers
session_manager = SessionManager()

# Initialize Digital Human components
orchestrator_config = OrchestratorConfig(
    enable_gpu=True,
    max_concurrent_sessions=50,
    session_timeout_minutes=30,
    enable_caching=True
)

audio2face_config = Audio2FaceConfig(
    api_key=os.getenv("NVIDIA_API_KEY", ""),
    endpoint=os.getenv("AUDIO2FACE_ENDPOINT", "https://api.nvidia.com/audio2face/v1"),
    enable_emotions=True
)

nvidia_integration = DigitalHumanNVIDIAIntegration(
    audio2face_config=audio2face_config,
    llm_config={
        "api_key": os.getenv("NVIDIA_API_KEY", ""),
        "endpoint": os.getenv("LLM_ENDPOINT", "https://integrate.api.nvidia.com/v1")
    },
    asr_config={
        "api_key": os.getenv("NVIDIA_API_KEY", ""),
        "endpoint": os.getenv("ASR_ENDPOINT", "https://api.nvidia.com/parakeet/v1")
    }
)

orchestrator = DigitalHumanOrchestrator(orchestrator_config)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for Digital Human communication"""
    
    await session_manager.connect(websocket, session_id)
    
    # Create digital human session
    await nvidia_integration.create_session(session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = Message(**data)
            
            # Handle different message types
            if message.type == "startSession":
                await handle_start_session(session_id, message.data)
            
            elif message.type == "message":
                await handle_user_message(session_id, message.data)
            
            elif message.type == "voice":
                await handle_voice_input(session_id, message.data)
            
            elif message.type == "emotion":
                await handle_emotion_update(session_id, message.data)
            
            elif message.type == "command":
                await handle_command(session_id, message.data)
            
            else:
                await session_manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message.type}"
                })
    
    except WebSocketDisconnect:
        session_manager.disconnect(session_id)
        await nvidia_integration.close_session(session_id)
    except Exception as e:
        logger.error(f"Error in WebSocket session {session_id}: {e}")
        await session_manager.send_message(session_id, {
            "type": "error",
            "message": str(e)
        })


async def handle_start_session(session_id: str, data: Dict[str, Any]):
    """Handle session start"""
    user_profile = data.get("userProfile", {})
    
    # Initialize session with user profile
    session_info = await nvidia_integration.get_session_info(session_id)
    
    # Send initial response
    await session_manager.send_message(session_id, {
        "type": "sessionStarted",
        "sessionInfo": session_info,
        "message": "Digital Human session started successfully"
    })
    
    # Send welcome message
    response = await nvidia_integration.process_user_input(
        session_id=session_id,
        text_input="Hello! I'm your AI financial advisor. How can I help you today?",
        emotion_context={"detected_emotion": "friendly"}
    )
    
    await session_manager.send_message(session_id, {
        "type": "response",
        "text": response["text"],
        "emotion": response["emotion"],
        "hasAudio": bool(response.get("audio")),
        "hasAnimation": bool(response.get("animation"))
    })


async def handle_user_message(session_id: str, data: Dict[str, Any]):
    """Handle text message from user"""
    user_text = data.get("content", "")
    
    # Send typing indicator
    await session_manager.send_message(session_id, {
        "type": "typing",
        "isTyping": True
    })
    
    try:
        # Process message with digital human
        response = await nvidia_integration.process_user_input(
            session_id=session_id,
            text_input=user_text
        )
        
        # Send response
        await session_manager.send_message(session_id, {
            "type": "response",
            "text": response["text"],
            "emotion": response["emotion"],
            "audio": response.get("audio"),  # Base64 encoded audio
            "animation": response.get("animation"),  # Animation data
            "avatar": {
                "expression": response["emotion"],
                "gesture": "present" if "explain" in response["text"].lower() else "nod"
            }
        })
        
        # Update portfolio data if relevant
        if any(keyword in user_text.lower() for keyword in ["portfolio", "performance", "value"]):
            portfolio_data = await get_portfolio_data(session_id)
            await session_manager.send_message(session_id, {
                "type": "portfolioUpdate",
                "data": portfolio_data
            })
        
        # Update analysis if relevant
        if any(keyword in user_text.lower() for keyword in ["analysis", "market", "risk"]):
            analysis_data = await get_analysis_data(session_id)
            await session_manager.send_message(session_id, {
                "type": "analysisUpdate",
                "data": analysis_data
            })
    
    finally:
        # Stop typing indicator
        await session_manager.send_message(session_id, {
            "type": "typing",
            "isTyping": False
        })


async def handle_voice_input(session_id: str, data: Dict[str, Any]):
    """Handle voice input from user"""
    audio_data = data.get("audio")  # Base64 encoded audio
    
    if not audio_data:
        await session_manager.send_message(session_id, {
            "type": "error",
            "message": "No audio data received"
        })
        return
    
    # Send processing indicator
    await session_manager.send_message(session_id, {
        "type": "processing",
        "message": "Processing voice input..."
    })
    
    try:
        # Decode audio
        import base64
        import numpy as np
        
        audio_bytes = base64.b64decode(audio_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Process with digital human
        response = await nvidia_integration.process_user_input(
            session_id=session_id,
            audio_input=audio_array
        )
        
        # Send response
        await session_manager.send_message(session_id, {
            "type": "response",
            "text": response["text"],
            "emotion": response["emotion"],
            "audio": response.get("audio"),
            "animation": response.get("animation"),
            "transcription": response.get("transcription", "")
        })
    
    except Exception as e:
        logger.error(f"Error processing voice input: {e}")
        await session_manager.send_message(session_id, {
            "type": "error",
            "message": "Failed to process voice input"
        })


async def handle_emotion_update(session_id: str, data: Dict[str, Any]):
    """Handle emotion update from client"""
    emotion = data.get("emotion", "neutral")
    intensity = data.get("intensity", 1.0)
    
    # Update avatar emotion
    await session_manager.send_message(session_id, {
        "type": "avatarUpdate",
        "avatar": {
            "expression": emotion,
            "intensity": intensity
        }
    })


async def handle_command(session_id: str, data: Dict[str, Any]):
    """Handle commands from client"""
    command = data.get("command")
    params = data.get("params", {})
    
    if command == "rebalance_portfolio":
        await handle_rebalance_portfolio(session_id, params)
    
    elif command == "run_analysis":
        await handle_run_analysis(session_id, params)
    
    elif command == "show_chart":
        await handle_show_chart(session_id, params)
    
    else:
        await session_manager.send_message(session_id, {
            "type": "error",
            "message": f"Unknown command: {command}"
        })


async def get_portfolio_data(session_id: str) -> Dict[str, Any]:
    """Get portfolio data for session"""
    # In production, this would fetch real portfolio data
    return {
        "totalValue": 125430.50,
        "change": {
            "amount": 2340.25,
            "percentage": 1.90
        },
        "metrics": {
            "totalReturn": 12.5,
            "sharpeRatio": 1.42,
            "riskLevel": 55
        },
        "allocation": {
            "Stocks": 60,
            "Bonds": 25,
            "Cash": 10,
            "Crypto": 5
        }
    }


async def get_analysis_data(session_id: str) -> Dict[str, Any]:
    """Get analysis data for session"""
    # In production, this would fetch real analysis data
    return {
        "sentiment": {
            "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "values": [65, 70, 68, 75, 80, 78]
        },
        "recommendations": [
            {
                "text": "Market conditions are favorable for growth stocks. Consider increasing tech allocation by 5%."
            }
        ],
        "riskMetrics": {
            "valueAtRisk": -5234,
            "maxDrawdown": -8.2
        }
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Digital Human WebSocket Server"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(session_manager.active_connections)
    }


if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)