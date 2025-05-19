"""FinSight Deep Production Deployment with NVIDIA Audio2Face-3D"""

import base64
import asyncio
import json
import os
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
from contextlib import asynccontextmanager
from datetime import datetime
import numpy as np

# Import the NVIDIA Audio2Face client we created
from nvidia_audio2face_client import Audio2Face3DClient, create_audio2face_handler

# Get credentials from environment
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


class FinSightNeuralOrchestrator:
    """Neural supercomputer orchestrator for FinSight Deep"""
    
    def __init__(self):
        self.together_client = httpx.AsyncClient(
            base_url="https://api.together.xyz/v1",
            headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"}
        )
        self.system_prompt = """You are FinSight Deep, a premier neural financial advisor powered by NVIDIA's neural supercomputing infrastructure. You provide expert financial analysis and investment recommendations using advanced AI models and Monte Carlo simulations. Be professional, knowledgeable, and empathetic."""
        
        # Initialize Audio2Face with James as default
        self.audio2face = Audio2Face3DClient(
            api_key=NVIDIA_API_KEY,
            model_name="james"  # The photorealistic male model
        )
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query through neural network"""
        response = await self.together_client.post(
            "/chat/completions",
            json={
                "model": "meta-llama/Llama-2-70b-chat-hf",
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.7,
                "stream": True
            },
            headers={"Accept": "text/event-stream"}
        )
        
        # Collect the full response for TTS processing
        full_text = ""
        animation_data = []
        
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            text_chunk = delta["content"]
                            full_text += text_chunk
                            
                            # For streaming, we could process each chunk through Audio2Face
                            # but for better quality, we'll process the full response
                            
                except json.JSONDecodeError:
                    continue
        
        # Generate audio and facial animations
        if full_text:
            # Convert text to speech (using a placeholder for now)
            # In production, you'd use NVIDIA TTS or another service
            audio_data = await self._text_to_speech(full_text)
            
            # Process audio through Audio2Face-3D
            animation_frames = await self.audio2face.process_audio_stream(audio_data)
            
            animation_data = {
                "frames": animation_frames,
                "audio_base64": base64.b64encode(audio_data).decode('utf-8'),
                "text": full_text,
                "model": "james"  # Specify which avatar model
            }
        
        return {
            "text": full_text,
            "animation": animation_data,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _text_to_speech(self, text: str) -> bytes:
        """Convert text to speech - placeholder for NVIDIA TTS"""
        # In production, use NVIDIA TTS or similar service
        # For now, return empty audio data
        return b""
    
    async def close(self):
        """Clean up resources"""
        await self.together_client.aclose()
        await self.audio2face.close()


# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.orchestrator = FinSightNeuralOrchestrator()
    yield
    await app.state.orchestrator.close()

app = FastAPI(title="FinSight Deep Neural Supercomputer", lifespan=lifespan)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    orchestrator = app.state.orchestrator
    
    try:
        while True:
            # Receive user query
            data = await websocket.receive_json()
            query = data.get("query", "")
            
            # Process through neural network and Audio2Face
            response = await orchestrator.process_query(query)
            
            # Send back with animation data
            await websocket.send_json({
                "type": "response",
                "data": response
            })
            
    except WebSocketDisconnect:
        print("Client disconnected")


@app.get("/")
async def index():
    """Serve the main interface with NVIDIA 3D avatar"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinSight Deep - Neural Supercomputer</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: #000;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
        }
        
        .container {
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            padding: 20px;
            background: linear-gradient(to right, #000, #1a1a1a);
            border-bottom: 2px solid #76b900;
        }
        
        .header h1 {
            margin: 0;
            font-size: 24px;
            color: #76b900;
            display: flex;
            align-items: center;
        }
        
        .nvidia-logo {
            width: 30px;
            height: 30px;
            margin-right: 10px;
            filter: brightness(2);
        }
        
        .main-content {
            flex: 1;
            display: flex;
            position: relative;
        }
        
        .avatar-container {
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #0a0a0a;
        }
        
        .avatar-3d {
            width: 600px;
            height: 600px;
            position: relative;
        }
        
        #avatar-video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 10px;
        }
        
        .avatar-info {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px 20px;
            border-radius: 8px;
            border: 1px solid #76b900;
        }
        
        .chat-panel {
            width: 400px;
            display: flex;
            flex-direction: column;
            background: #111;
            border-left: 1px solid #333;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 8px;
            animation: fadeIn 0.3s ease-in;
        }
        
        .user-message {
            background: #1e1e1e;
            margin-left: 50px;
        }
        
        .ai-message {
            background: #0d2b0d;
            border: 1px solid #76b900;
            margin-right: 50px;
        }
        
        .input-container {
            padding: 20px;
            border-top: 1px solid #333;
        }
        
        #input {
            width: 100%;
            padding: 12px;
            background: #1e1e1e;
            border: 1px solid #444;
            border-radius: 8px;
            color: #fff;
            font-size: 14px;
        }
        
        #input:focus {
            outline: none;
            border-color: #76b900;
        }
        
        .status-indicator {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            align-items: center;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px 20px;
            border-radius: 8px;
            border: 1px solid #333;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
            background: #76b900;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }
        
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 3px solid #333;
            border-top: 3px solid #76b900;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                <svg class="nvidia-logo" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8.94 3L3 8.95v6.1L8.94 21h6.12L21 15.05v-6.1L15.06 3H8.94zm3.06 4.5c3.59 0 6.5 2.91 6.5 6.5s-2.91 6.5-6.5 6.5-6.5-2.91-6.5-6.5 2.91-6.5 6.5-6.5z"/>
                </svg>
                FinSight Deep - Neural Supercomputer
            </h1>
        </div>
        
        <div class="main-content">
            <div class="avatar-container">
                <div class="avatar-3d" id="avatar-container">
                    <div class="loading">
                        <div class="loading-spinner"></div>
                        <p>Initializing NVIDIA Audio2Face-3D...</p>
                    </div>
                    <video id="avatar-video" style="display: none;" autoplay loop muted></video>
                </div>
                <div class="avatar-info">
                    <p>Model: NVIDIA James (Photorealistic)</p>
                    <p>Status: <span id="avatar-status">Initializing</span></p>
                </div>
            </div>
            
            <div class="chat-panel">
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    <span>Neural Network Active</span>
                </div>
                
                <div class="messages" id="messages">
                    <div class="message ai-message">
                        Welcome to FinSight Deep. I'm your neural-powered financial advisor, utilizing NVIDIA's advanced AI models for comprehensive financial analysis. How may I assist you today?
                    </div>
                </div>
                
                <div class="input-container">
                    <input 
                        type="text" 
                        id="input" 
                        placeholder="Ask about investments, portfolios, or financial strategies..."
                        onkeypress="if(event.key === 'Enter') sendMessage()"
                    >
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('input');
        const avatarVideo = document.getElementById('avatar-video');
        const avatarContainer = document.getElementById('avatar-container');
        const avatarStatus = document.getElementById('avatar-status');
        
        // For storing audio context
        let audioContext = null;
        let audioQueue = [];
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('Connected to neural network');
                avatarStatus.textContent = 'Connected';
                
                // Initialize avatar display
                initializeAvatar();
            };
            
            ws.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'response') {
                    const response = data.data;
                    
                    // Display text response
                    addMessage(response.text, 'ai');
                    
                    // Handle avatar animation
                    if (response.animation) {
                        await playAvatarAnimation(response.animation);
                    }
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                avatarStatus.textContent = 'Error';
            };
            
            ws.onclose = () => {
                console.log('Disconnected');
                avatarStatus.textContent = 'Disconnected';
                setTimeout(connectWebSocket, 3000);
            };
        }
        
        function initializeAvatar() {
            // Remove loading indicator
            const loading = avatarContainer.querySelector('.loading');
            if (loading) loading.remove();
            
            // Initialize audio context
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Show avatar video
            avatarVideo.style.display = 'block';
            
            // For production, this would connect to the NVIDIA Avatar Cloud stream
            // Here we'll use a placeholder for the James model
            avatarVideo.src = '/static/nvidia_james_idle.mp4';
            
            avatarStatus.textContent = 'Ready';
        }
        
        async function playAvatarAnimation(animationData) {
            avatarStatus.textContent = 'Speaking';
            
            // Decode and play audio
            if (animationData.audio_base64) {
                const audioData = atob(animationData.audio_base64);
                const audioBuffer = new ArrayBuffer(audioData.length);
                const view = new Uint8Array(audioBuffer);
                for (let i = 0; i < audioData.length; i++) {
                    view[i] = audioData.charCodeAt(i);
                }
                
                try {
                    const decodedAudio = await audioContext.decodeAudioData(audioBuffer);
                    const source = audioContext.createBufferSource();
                    source.buffer = decodedAudio;
                    source.connect(audioContext.destination);
                    source.start();
                    
                    // Play animation frames synchronized with audio
                    if (animationData.frames) {
                        playAnimationFrames(animationData.frames, decodedAudio.duration * 1000);
                    }
                    
                } catch (e) {
                    console.error('Audio playback error:', e);
                }
            }
            
            // Return to idle after speaking
            setTimeout(() => {
                avatarStatus.textContent = 'Ready';
            }, animationData.frames ? animationData.frames.length * 33 : 3000);
        }
        
        function playAnimationFrames(frames, duration) {
            const frameTime = duration / frames.length;
            let frameIndex = 0;
            
            const animate = () => {
                if (frameIndex < frames.length) {
                    // Apply blendshape values to avatar
                    const frame = frames[frameIndex];
                    
                    // In a real implementation, this would update the 3D model's blendshapes
                    // For now, we'll simulate the animation
                    updateAvatarBlendshapes(frame);
                    
                    frameIndex++;
                    setTimeout(animate, frameTime);
                }
            };
            
            animate();
        }
        
        function updateAvatarBlendshapes(blendshapes) {
            // This would apply the blendshape values to the 3D model
            // For NVIDIA Audio2Face, these control facial expressions
            console.log('Applying blendshapes:', Object.keys(blendshapes).length);
        }
        
        function sendMessage() {
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ query: message }));
            }
        }
        
        function addMessage(text, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Connect on load
        connectWebSocket();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)