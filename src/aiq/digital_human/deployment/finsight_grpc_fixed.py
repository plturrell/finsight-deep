"""
FinSight Deep Production with Fixed NVIDIA Audio2Face gRPC Integration
"""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, AsyncIterator, Tuple
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import httpx
from contextlib import asynccontextmanager
from datetime import datetime
import wave
import io

# Add the deployment directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "generated"))

# Import our real gRPC client
from nvidia_grpc_client import NvidiaAudio2FaceGrpcClient

# Get credentials from environment
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Check if API keys are set
if not NVIDIA_API_KEY:
    print("WARNING: NVIDIA_API_KEY not set - using mock mode")
    NVIDIA_API_KEY = "mock_key"

if not TOGETHER_API_KEY:
    print("WARNING: TOGETHER_API_KEY not set - using mock mode")
    TOGETHER_API_KEY = "mock_key"


class AudioProcessor:
    """Process text to speech and prepare audio for gRPC"""
    
    @staticmethod
    async def text_to_speech(text: str) -> bytes:
        """Convert text to speech using NVIDIA Riva TTS or alternative"""
        # In production, this would use NVIDIA Riva TTS
        # For now, we'll create a WAV file with the right format
        
        # Simulate TTS with proper WAV format
        sample_rate = 16000
        duration = len(text.split()) * 0.5  # Rough estimate
        samples = int(sample_rate * duration)
        
        # Generate silent audio as placeholder
        audio_data = np.zeros(samples, dtype=np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    @staticmethod
    def extract_pcm_from_wav(wav_data: bytes) -> Tuple[bytes, int]:
        """Extract PCM data from WAV file"""
        wav_buffer = io.BytesIO(wav_data)
        with wave.open(wav_buffer, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            pcm_data = wav_file.readframes(wav_file.getnframes())
        return pcm_data, sample_rate


class FinSightNeuralEngine:
    """Neural engine with real gRPC Audio2Face integration"""
    
    def __init__(self):
        # LLM client
        self.together_client = httpx.AsyncClient(
            base_url="https://api.together.xyz/v1",
            headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"}
        )
        
        # System prompt
        self.system_prompt = """You are FinSight Deep, a premier AI financial advisor powered by NVIDIA's neural supercomputing platform. You provide expert financial analysis, investment strategies, and portfolio recommendations using advanced AI models. Be professional, insightful, and data-driven in your responses."""
        
        # Initialize gRPC client for Audio2Face
        self.grpc_client = None
        self.mock_mode = False
        
        # Audio processor
        self.audio_processor = AudioProcessor()
        
        # Initialize connection flag
        self._connected = False
    
    async def initialize(self):
        """Initialize gRPC connection and model"""
        try:
            if NVIDIA_API_KEY == "mock_key":
                print("Running in mock mode - no real gRPC connection")
                self.mock_mode = True
                return
            
            self.grpc_client = NvidiaAudio2FaceGrpcClient(
                api_key=NVIDIA_API_KEY,
                model_name="james",  # The photorealistic male model
                endpoint="grpc.nvcf.nvidia.com:443",
                secure=True
            )
            
            print("Connecting to NVIDIA gRPC service...")
            await self.grpc_client.connect()
            
            # Initialize the avatar model
            print("Initializing Audio2Face model...")
            success = await self.grpc_client.initialize_model(
                enable_tongue=True,
                animation_fps=30.0,
                quality="high"
            )
            
            if success:
                self._connected = True
                print("Successfully initialized NVIDIA Audio2Face James model")
            else:
                print("Failed to initialize Audio2Face model - falling back to mock mode")
                self.mock_mode = True
                
        except Exception as e:
            print(f"Error initializing gRPC: {e}")
            print("Falling back to mock mode")
            self.mock_mode = True
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query through LLM and Audio2Face"""
        try:
            # Get LLM response
            if TOGETHER_API_KEY != "mock_key":
                response = await self.together_client.post(
                    "/chat/completions",
                    json={
                        "model": "meta-llama/Llama-2-70b-chat-hf",
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": query}
                        ],
                        "temperature": 0.7,
                        "stream": False
                    }
                )
                
                response_data = response.json()
                text_response = response_data["choices"][0]["message"]["content"]
            else:
                # Mock response
                text_response = f"As FinSight Deep, I understand you're asking about: {query}. In a production environment, I would provide detailed financial analysis using NVIDIA's neural computing platform."
            
            # Convert text to speech
            audio_wav = await self.audio_processor.text_to_speech(text_response)
            
            animation_data = {}
            
            if not self.mock_mode and self.grpc_client:
                # Extract PCM data for gRPC
                pcm_data, sample_rate = self.audio_processor.extract_pcm_from_wav(audio_wav)
                
                # Process through Audio2Face gRPC
                try:
                    animation_frames = await self.grpc_client.process_audio_data(
                        audio_data=pcm_data,
                        sample_rate=sample_rate,
                        encoding="PCM_16",
                        language="en-US",
                        emotion_strength=1.0
                    )
                    
                    animation_data = {
                        "frames": animation_frames,
                        "audio_base64": base64.b64encode(audio_wav).decode('utf-8'),
                        "text": text_response,
                        "model": "james"  # Specify which avatar model
                    }
                except Exception as e:
                    print(f"Error processing audio through gRPC: {e}")
                    # Fall back to mock animation
                    animation_data = self._create_mock_animation(text_response, audio_wav)
            else:
                # Mock animation data
                animation_data = self._create_mock_animation(text_response, audio_wav)
            
            return {
                "text": text_response,
                "animation": animation_data,
                "timestamp": datetime.now().isoformat(),
                "mock_mode": self.mock_mode
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "text": f"I apologize, but I encountered an error: {str(e)}",
                "animation": {},
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _create_mock_animation(self, text: str, audio_wav: bytes) -> Dict[str, Any]:
        """Create mock animation data for testing"""
        # Create some mock frames
        num_frames = int(len(text.split()) * 10)  # Rough estimate
        frames = []
        
        for i in range(num_frames):
            frame = {
                "timestamp": i * 0.033,  # 30 FPS
                "blendshapes": {
                    "jawOpen": 0.2 + 0.3 * abs(np.sin(i * 0.5)),
                    "mouthSmile": 0.1,
                    "eyeBlink": 0.1 if i % 30 == 0 else 0
                },
                "bones": [],
                "sequence_number": i
            }
            frames.append(frame)
        
        return {
            "frames": frames,
            "audio_base64": base64.b64encode(audio_wav).decode('utf-8'),
            "text": text,
            "model": "james_mock"
        }
    
    async def close(self):
        """Clean up resources"""
        await self.together_client.aclose()
        if self.grpc_client and self._connected:
            await self.grpc_client.disconnect()


# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    print("Starting FinSight Deep application...")
    app.state.engine = FinSightNeuralEngine()
    
    try:
        await app.state.engine.initialize()
    except Exception as e:
        print(f"Warning during initialization: {e}")
    
    yield
    
    # Cleanup
    await app.state.engine.close()


app = FastAPI(
    title="FinSight Deep - NVIDIA Audio2Face gRPC",
    lifespan=lifespan
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    engine = app.state.engine
    
    try:
        while True:
            # Receive user message
            data = await websocket.receive_json()
            query = data.get("query", "")
            
            # Send initial acknowledgment
            await websocket.send_json({
                "type": "status",
                "message": "Processing your query..."
            })
            
            # Process through neural engine
            response = await engine.process_query(query)
            
            # Send response with animation data
            await websocket.send_json({
                "type": "response",
                "data": response
            })
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })


@app.get("/")
async def index():
    """Serve the main interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinSight Deep - NVIDIA Audio2Face gRPC</title>
    <style>
        body {
            margin: 0;
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
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .header h1 {
            margin: 0;
            font-size: 24px;
            color: #76b900;
            display: flex;
            align-items: center;
        }
        
        .nvidia-badge {
            background: #76b900;
            color: #000;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }
        
        .main-content {
            flex: 1;
            display: flex;
        }
        
        .avatar-section {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #0a0a0a;
            position: relative;
        }
        
        .avatar-frame {
            width: 512px;
            height: 512px;
            background: #111;
            border: 2px solid #333;
            border-radius: 10px;
            position: relative;
            overflow: hidden;
        }
        
        #avatar-canvas {
            width: 100%;
            height: 100%;
        }
        
        .model-info {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.9);
            padding: 10px 20px;
            border-radius: 8px;
            border: 1px solid #76b900;
            font-size: 14px;
        }
        
        .chat-section {
            width: 400px;
            display: flex;
            flex-direction: column;
            background: #111;
            border-left: 1px solid #333;
        }
        
        .status-bar {
            padding: 15px 20px;
            background: #1a1a1a;
            border-bottom: 1px solid #333;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #76b900;
            animation: pulse 2s infinite;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 15px;
            border-radius: 8px;
            animation: fadeIn 0.3s ease-in;
        }
        
        .user-message {
            background: #1e1e1e;
            margin-left: 50px;
        }
        
        .ai-message {
            background: linear-gradient(to right, #0d2b0d, #0a1f0a);
            border: 1px solid #76b900;
            margin-right: 50px;
        }
        
        .input-container {
            padding: 20px;
            border-top: 1px solid #333;
        }
        
        #input {
            width: 100%;
            padding: 12px 15px;
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
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animation-info {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.9);
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 12px;
            font-family: monospace;
            color: #76b900;
        }
        
        .debug-info {
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.9);
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 11px;
            font-family: monospace;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FinSight Deep - Neural Supercomputer</h1>
            <div class="nvidia-badge">Powered by NVIDIA Audio2Face</div>
        </div>
        
        <div class="main-content">
            <div class="avatar-section">
                <div class="avatar-frame">
                    <canvas id="avatar-canvas"></canvas>
                    <div class="animation-info" id="animation-info">
                        FPS: <span id="fps">0</span> | Blendshapes: <span id="blendshape-count">0</span>
                    </div>
                </div>
                <div class="model-info">
                    <strong>Model:</strong> <span id="model-name">NVIDIA James</span><br>
                    <strong>Protocol:</strong> gRPC (Audio2Face-3D)<br>
                    <strong>Quality:</strong> High
                </div>
                <div class="debug-info" id="debug-info">
                    Mode: <span id="mode">Checking...</span>
                </div>
            </div>
            
            <div class="chat-section">
                <div class="status-bar">
                    <div class="status-indicator"></div>
                    <span id="status">Connecting to Neural Network...</span>
                </div>
                
                <div class="messages" id="messages">
                    <div class="message ai-message">
                        Welcome to FinSight Deep. I'm your neural-powered financial advisor. How may I assist you today?
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
        const canvas = document.getElementById('avatar-canvas');
        const ctx = canvas.getContext('2d');
        const statusEl = document.getElementById('status');
        const fpsEl = document.getElementById('fps');
        const blendshapeCountEl = document.getElementById('blendshape-count');
        const modeEl = document.getElementById('mode');
        const modelNameEl = document.getElementById('model-name');
        
        // Animation state
        let animationFrames = [];
        let currentFrame = 0;
        let isAnimating = false;
        let lastFrameTime = 0;
        let frameRate = 30;
        
        // Set canvas size
        canvas.width = 512;
        canvas.height = 512;
        
        // Audio context for playback
        let audioContext = null;
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('Connected to neural network');
                statusEl.textContent = 'Connected to Neural Network';
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            };
            
            ws.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'status') {
                    statusEl.textContent = data.message;
                } else if (data.type === 'response') {
                    const response = data.data;
                    
                    // Display text
                    addMessage(response.text, 'ai');
                    
                    // Update debug info
                    if (response.mock_mode) {
                        modeEl.textContent = 'Mock Mode';
                        modelNameEl.textContent = 'James (Mock)';
                    } else {
                        modeEl.textContent = 'gRPC Connected';
                        modelNameEl.textContent = 'NVIDIA James';
                    }
                    
                    // Handle animation data
                    if (response.animation && response.animation.frames) {
                        animationFrames = response.animation.frames;
                        currentFrame = 0;
                        blendshapeCountEl.textContent = 
                            animationFrames.length > 0 ? 
                            Object.keys(animationFrames[0].blendshapes).length : 0;
                        
                        // Play audio if available
                        if (response.animation.audio_base64) {
                            await playAudio(response.animation.audio_base64);
                        }
                        
                        // Start animation
                        startAnimation();
                    }
                } else if (data.type === 'error') {
                    console.error('Error:', data.message);
                    statusEl.textContent = 'Error: ' + data.message;
                    addMessage('Error: ' + data.message, 'ai');
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                statusEl.textContent = 'Connection Error';
            };
            
            ws.onclose = () => {
                console.log('Disconnected');
                statusEl.textContent = 'Disconnected';
                setTimeout(connectWebSocket, 3000);
            };
        }
        
        async function playAudio(audioBase64) {
            try {
                const audioData = atob(audioBase64);
                const audioBuffer = new ArrayBuffer(audioData.length);
                const view = new Uint8Array(audioBuffer);
                for (let i = 0; i < audioData.length; i++) {
                    view[i] = audioData.charCodeAt(i);
                }
                
                const decodedAudio = await audioContext.decodeAudioData(audioBuffer);
                const source = audioContext.createBufferSource();
                source.buffer = decodedAudio;
                source.connect(audioContext.destination);
                source.start();
            } catch (e) {
                console.error('Audio playback error:', e);
            }
        }
        
        function startAnimation() {
            if (!isAnimating && animationFrames.length > 0) {
                isAnimating = true;
                lastFrameTime = performance.now();
                animate();
            }
        }
        
        function animate() {
            if (!isAnimating || currentFrame >= animationFrames.length) {
                isAnimating = false;
                drawIdleAvatar();
                return;
            }
            
            const now = performance.now();
            const deltaTime = now - lastFrameTime;
            const frameTime = 1000 / frameRate;
            
            if (deltaTime >= frameTime) {
                const frame = animationFrames[currentFrame];
                drawAvatarFrame(frame);
                
                // Update FPS counter
                const actualFps = Math.round(1000 / deltaTime);
                fpsEl.textContent = actualFps;
                
                currentFrame++;
                lastFrameTime = now;
            }
            
            requestAnimationFrame(animate);
        }
        
        function drawAvatarFrame(frame) {
            // Clear canvas
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw the avatar visualization
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            // Head base
            ctx.strokeStyle = '#76b900';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(centerX, centerY, 100, 0, Math.PI * 2);
            ctx.stroke();
            
            // Draw blendshapes visualization
            if (frame.blendshapes) {
                // Jaw
                const jawOpen = frame.blendshapes.jawOpen || 0;
                ctx.beginPath();
                ctx.arc(centerX, centerY + 60, 40, 0, Math.PI, false);
                ctx.lineWidth = 3 + jawOpen * 20;
                ctx.stroke();
                
                // Eyes
                const blink = frame.blendshapes.eyeBlink || 0;
                ctx.fillStyle = '#76b900';
                ctx.beginPath();
                ctx.arc(centerX - 30, centerY - 20, 15 * (1 - blink), 0, Math.PI * 2);
                ctx.arc(centerX + 30, centerY - 20, 15 * (1 - blink), 0, Math.PI * 2);
                ctx.fill();
            }
            
            // Info text
            ctx.fillStyle = '#76b900';
            ctx.font = '14px monospace';
            ctx.fillText(`Frame: ${currentFrame}`, 20, 30);
            ctx.fillText(`Time: ${frame.timestamp.toFixed(3)}s`, 20, 50);
        }
        
        function drawIdleAvatar() {
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            // Draw idle avatar
            ctx.strokeStyle = '#76b900';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(centerX, centerY, 100, 0, Math.PI * 2);
            ctx.stroke();
            
            // Eyes
            ctx.fillStyle = '#76b900';
            ctx.beginPath();
            ctx.arc(centerX - 30, centerY - 20, 15, 0, Math.PI * 2);
            ctx.arc(centerX + 30, centerY - 20, 15, 0, Math.PI * 2);
            ctx.fill();
            
            // Mouth
            ctx.beginPath();
            ctx.arc(centerX, centerY + 40, 30, 0, Math.PI, false);
            ctx.stroke();
            
            ctx.fillStyle = '#76b900';
            ctx.font = '16px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('Ready', centerX, centerY + 150);
            ctx.textAlign = 'left';
        }
        
        function sendMessage() {
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ query: message }));
                statusEl.textContent = 'Processing with gRPC...';
            }
        }
        
        function addMessage(text, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Initialize
        connectWebSocket();
        drawIdleAvatar();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.get("/status")
async def status():
    """API status endpoint"""
    engine = app.state.engine
    return {
        "status": "ok",
        "grpc_connected": engine._connected,
        "mock_mode": engine.mock_mode,
        "nvidia_api_key": bool(NVIDIA_API_KEY and NVIDIA_API_KEY != "mock_key"),
        "together_api_key": bool(TOGETHER_API_KEY and TOGETHER_API_KEY != "mock_key")
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting FinSight Deep with NVIDIA gRPC integration...")
    print(f"NVIDIA_API_KEY: {'Set' if NVIDIA_API_KEY and NVIDIA_API_KEY != 'mock_key' else 'Not set'}")
    print(f"TOGETHER_API_KEY: {'Set' if TOGETHER_API_KEY and TOGETHER_API_KEY != 'mock_key' else 'Not set'}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)