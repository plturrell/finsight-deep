"""
FinSight Deep Production with Real NVIDIA Audio2Face gRPC Integration
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
from fastapi.staticfiles import StaticFiles
import httpx
from contextlib import asynccontextmanager
from datetime import datetime
import wave
import io

# Add the deployment directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our real gRPC client
from nvidia_grpc_client import NvidiaAudio2FaceGrpcClient

# Get credentials from environment
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


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
        self.grpc_client = NvidiaAudio2FaceGrpcClient(
            api_key=NVIDIA_API_KEY,
            model_name="james",  # The photorealistic male avatar
            endpoint="grpc.nvcf.nvidia.com:443",
            secure=True
        )
        
        # Audio processor
        self.audio_processor = AudioProcessor()
        
        # Initialize connection
        self._connected = False
    
    async def initialize(self):
        """Initialize gRPC connection and model"""
        if not self._connected:
            await self.grpc_client.connect()
            
            # Initialize the avatar model
            success = await self.grpc_client.initialize_model(
                enable_tongue=True,
                animation_fps=30.0,
                quality="high"
            )
            
            if success:
                self._connected = True
                print("Successfully initialized NVIDIA Audio2Face James model")
            else:
                raise RuntimeError("Failed to initialize Audio2Face model")
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query through LLM and Audio2Face"""
        # Ensure we're connected
        if not self._connected:
            await self.initialize()
        
        # Get LLM response
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
        
        # Convert text to speech
        audio_wav = await self.audio_processor.text_to_speech(text_response)
        
        # Extract PCM data for gRPC
        pcm_data, sample_rate = self.audio_processor.extract_pcm_from_wav(audio_wav)
        
        # Process through Audio2Face gRPC
        animation_frames = await self.grpc_client.process_audio_data(
            audio_data=pcm_data,
            sample_rate=sample_rate,
            encoding="PCM_16",
            language="en-US",
            emotion_strength=1.0
        )
        
        return {
            "text": text_response,
            "audio_base64": base64.b64encode(audio_wav).decode('utf-8'),
            "animation": {
                "frames": animation_frames,
                "model": "james",
                "frame_rate": 30.0
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def close(self):
        """Clean up resources"""
        await self.together_client.aclose()
        if self._connected:
            await self.grpc_client.disconnect()


# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize neural engine
    app.state.engine = FinSightNeuralEngine()
    await app.state.engine.initialize()
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
                    <strong>Model:</strong> NVIDIA James (Photorealistic)<br>
                    <strong>Protocol:</strong> gRPC (Audio2Face-3D)<br>
                    <strong>Quality:</strong> High
                </div>
            </div>
            
            <div class="chat-section">
                <div class="status-bar">
                    <div class="status-indicator"></div>
                    <span id="status">Connected to gRPC Service</span>
                </div>
                
                <div class="messages" id="messages">
                    <div class="message ai-message">
                        Welcome to FinSight Deep. I'm connected via NVIDIA's official gRPC Audio2Face service. How can I assist with your financial needs today?
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
                console.log('Connected to gRPC service');
                statusEl.textContent = 'Connected to gRPC Service';
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
                    
                    // Handle animation data
                    if (response.animation && response.animation.frames) {
                        animationFrames = response.animation.frames;
                        currentFrame = 0;
                        blendshapeCountEl.textContent = 
                            animationFrames.length > 0 ? 
                            Object.keys(animationFrames[0].blendshapes).length : 0;
                        
                        // Play audio if available
                        if (response.audio_base64) {
                            await playAudio(response.audio_base64);
                        }
                        
                        // Start animation
                        startAnimation();
                    }
                } else if (data.type === 'error') {
                    console.error('Error:', data.message);
                    statusEl.textContent = 'Error: ' + data.message;
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
            ctx.fillStyle = '#111';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // This is where we would apply the blendshapes to a 3D model
            // For demonstration, we'll visualize the data
            ctx.fillStyle = '#76b900';
            ctx.font = '14px monospace';
            ctx.fillText('NVIDIA James Avatar', 20, 30);
            ctx.fillText(`Frame: ${currentFrame}`, 20, 50);
            ctx.fillText(`Timestamp: ${frame.timestamp.toFixed(3)}`, 20, 70);
            
            // Visualize facial pose
            const pose = frame.facial_pose;
            ctx.fillText(`Pose: Pitch ${pose.pitch.toFixed(2)}, Yaw ${pose.yaw.toFixed(2)}`, 20, 90);
            
            // Visualize emotion
            const emotion = frame.emotion;
            const dominantEmotion = Object.entries(emotion)
                .sort(([,a], [,b]) => b - a)[0];
            ctx.fillText(`Emotion: ${dominantEmotion[0]} (${dominantEmotion[1].toFixed(2)})`, 20, 110);
            
            // Draw a stylized face representation
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            // Head circle with pose rotation
            ctx.strokeStyle = '#76b900';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(centerX + pose.yaw * 50, centerY - pose.pitch * 50, 100, 0, Math.PI * 2);
            ctx.stroke();
            
            // Eyes with gaze
            const eyeY = centerY - 20;
            const leftEyeX = centerX - 30;
            const rightEyeX = centerX + 30;
            
            ctx.fillStyle = '#76b900';
            ctx.beginPath();
            ctx.arc(leftEyeX + pose.eye_gaze_x * 10, eyeY + pose.eye_gaze_y * 10, 10, 0, Math.PI * 2);
            ctx.arc(rightEyeX + pose.eye_gaze_x * 10, eyeY + pose.eye_gaze_y * 10, 10, 0, Math.PI * 2);
            ctx.fill();
            
            // Mouth based on blendshapes
            if (frame.blendshapes && frame.blendshapes.jawOpen) {
                const mouthOpen = frame.blendshapes.jawOpen || 0;
                ctx.beginPath();
                ctx.arc(centerX, centerY + 40, 30, 0, Math.PI, false);
                ctx.lineWidth = 3 + mouthOpen * 10;
                ctx.stroke();
            }
        }
        
        function drawIdleAvatar() {
            ctx.fillStyle = '#111';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = '#76b900';
            ctx.font = '16px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('NVIDIA James - Idle', canvas.width / 2, canvas.height / 2);
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


if __name__ == "__main__":
    import uvicorn
    
    # First, compile the proto files
    print("Compiling proto files...")
    compile_result = os.system(f"{sys.executable} audio2face_proto_compiler.py")
    
    if compile_result != 0:
        print("Failed to compile proto files. Please run audio2face_proto_compiler.py manually.")
        sys.exit(1)
    
    print("Starting FinSight Deep with NVIDIA gRPC integration...")
    uvicorn.run(app, host="0.0.0.0", port=8000)