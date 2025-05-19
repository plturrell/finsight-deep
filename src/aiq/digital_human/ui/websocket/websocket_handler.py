"""
WebSocket Handler for Digital Human UI

Handles real-time communication between the UI and the orchestrator.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from aiq.digital_human.orchestrator.digital_human_orchestrator import DigitalHumanOrchestrator
from aiq.digital_human.conversation.context_manager import ConversationContext

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handles WebSocket connections for real-time digital human interactions"""
    
    def __init__(self, orchestrator: DigitalHumanOrchestrator):
        self.orchestrator = orchestrator
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_map: Dict[str, str] = {}  # websocket_id -> session_id
        
    async def handle_connection(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await websocket.accept()
        connection_id = str(id(websocket))
        self.active_connections[connection_id] = websocket
        
        logger.info(f"New WebSocket connection: {connection_id}")
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Process message
                response = await self._process_message(
                    message, 
                    connection_id,
                    websocket
                )
                
                # Send response if available
                if response:
                    await websocket.send_json(response)
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
            await self._cleanup_connection(connection_id)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await self._cleanup_connection(connection_id)
    
    async def _process_message(
        self, 
        message: Dict[str, Any],
        connection_id: str,
        websocket: WebSocket
    ) -> Optional[Dict[str, Any]]:
        """Process incoming WebSocket message"""
        message_type = message.get("type")
        
        if message_type == "start_session":
            return await self._handle_start_session(message, connection_id)
            
        elif message_type == "user_message":
            return await self._handle_user_message(message, connection_id, websocket)
            
        elif message_type == "toggle_audio":
            return await self._handle_toggle_audio(message, connection_id)
            
        elif message_type == "toggle_video":
            return await self._handle_toggle_video(message, connection_id)
            
        elif message_type == "end_session":
            return await self._handle_end_session(connection_id)
            
        else:
            logger.warning(f"Unknown message type: {message_type}")
            return {
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }
    
    async def _handle_start_session(
        self,
        message: Dict[str, Any],
        connection_id: str
    ) -> Dict[str, Any]:
        """Handle session start request"""
        user_id = message.get("user_id", f"user_{connection_id}")
        
        # Start new session
        session_id = await self.orchestrator.start_session(
            user_id=user_id,
            initial_context=message.get("context", {})
        )
        
        # Map connection to session
        self.session_map[connection_id] = session_id
        
        return {
            "type": "session_started",
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_user_message(
        self,
        message: Dict[str, Any],
        connection_id: str,
        websocket: WebSocket
    ) -> Dict[str, Any]:
        """Handle user message and generate response"""
        content = message.get("content", "")
        
        # Get session ID
        session_id = self.session_map.get(connection_id)
        if not session_id:
            # Auto-create session if not exists
            user_id = f"user_{connection_id}"
            session_id = await self.orchestrator.start_session(user_id)
            self.session_map[connection_id] = session_id
        
        # Process message through orchestrator
        try:
            response = await self.orchestrator.process_user_input(
                user_input=content,
                audio_data=message.get("audio_data")
            )
            
            # Format response for WebSocket
            return {
                "type": "response",
                "session_id": response["session_id"],
                "text": response["text"],
                "emotion": response["emotion"],
                "emotion_intensity": response["emotion_intensity"],
                "animation": response["animation"],
                "reasoning": response.get("reasoning", {}),
                "financial_data": self._extract_financial_data(response),
                "processing_time": response["processing_time"],
                "timestamp": response["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            return {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _handle_toggle_audio(
        self,
        message: Dict[str, Any],
        connection_id: str
    ) -> Dict[str, Any]:
        """Handle audio toggle request"""
        enabled = message.get("enabled", True)
        
        # Update settings in orchestrator
        session_id = self.session_map.get(connection_id)
        if session_id:
            # TODO: Implement audio toggle in orchestrator
            pass
        
        return {
            "type": "audio_toggled",
            "enabled": enabled,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_toggle_video(
        self,
        message: Dict[str, Any],
        connection_id: str
    ) -> Dict[str, Any]:
        """Handle video toggle request"""
        enabled = message.get("enabled", True)
        
        # Update settings in orchestrator
        session_id = self.session_map.get(connection_id)
        if session_id:
            # TODO: Implement video toggle in orchestrator
            pass
        
        return {
            "type": "video_toggled",
            "enabled": enabled,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_end_session(self, connection_id: str) -> Dict[str, Any]:
        """Handle session end request"""
        session_id = self.session_map.get(connection_id)
        
        if session_id:
            session_data = await self.orchestrator.end_session()
            del self.session_map[connection_id]
            
            return {
                "type": "session_ended",
                "session_id": session_id,
                "session_data": session_data,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "type": "error",
            "message": "No active session found",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _cleanup_connection(self, connection_id: str):
        """Clean up disconnected connection"""
        # Remove from active connections
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # End session if exists
        if connection_id in self.session_map:
            session_id = self.session_map[connection_id]
            await self.orchestrator.end_session()
            del self.session_map[connection_id]
    
    def _extract_financial_data(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract financial data from response for UI metrics"""
        # Mock financial data for now
        return {
            "portfolio_value": 325000.0,
            "daily_change": 2.3,
            "risk_level": "Moderate",
            "confidence": 0.92,
            "historical_values": [
                320000, 322000, 319000, 324000, 326000, 325000
            ]
        }
    
    async def broadcast_update(self, update: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        disconnected = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(update)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            await self._cleanup_connection(connection_id)
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific session"""
        # Find connection ID for session
        connection_id = None
        for conn_id, sess_id in self.session_map.items():
            if sess_id == session_id:
                connection_id = conn_id
                break
        
        if connection_id and connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to session {session_id}: {e}")
                await self._cleanup_connection(connection_id)