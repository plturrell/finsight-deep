"""
Launch Digital Human UI Application

Entry point for the Digital Human Financial Advisor system with
comprehensive features and real-time avatar rendering.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from aiq.digital_human.orchestrator.digital_human_orchestrator import DigitalHumanOrchestrator
from aiq.digital_human.ui.api_server import create_api_server
from aiq.digital_human.ui.websocket_handler import WebSocketHandler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DigitalHumanUI:
    """Main UI application for Digital Human system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_config()
        
        # Initialize components
        self.orchestrator = None
        self.app = None
        self.websocket_handler = None
        
        # Setup paths
        self.static_path = Path(__file__).parent / "ui" / "static"
        self.template_path = Path(__file__).parent / "ui" / "templates"
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment or defaults"""
        return {
            "model_name": os.getenv("DIGITAL_HUMAN_MODEL", "meta-llama/Llama-3.1-70B-Instruct"),
            "device": os.getenv("DEVICE", "cuda"),
            "api_port": int(os.getenv("API_PORT", "8000")),
            "enable_gpu": os.getenv("ENABLE_GPU", "true").lower() == "true",
            "enable_observability": True,
            "enable_mcp": True,
            "resolution": (1920, 1080),
            "target_fps": 60
        }
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Digital Human UI...")
        
        # Initialize orchestrator
        self.orchestrator = DigitalHumanOrchestrator(
            config=self.config,
            enable_profiling=True,
            enable_gpu_acceleration=self.config.get("enable_gpu", True)
        )
        
        # Create FastAPI app
        self.app = FastAPI(
            title="AIQ Digital Human",
            description="Best-in-class Digital Human Financial Advisor",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Setup routes
        self._setup_routes()
        
        # Initialize WebSocket handler
        self.websocket_handler = WebSocketHandler(self.orchestrator)
        
        # Mount static files
        if self.static_path.exists():
            self.app.mount("/static", StaticFiles(directory=str(self.static_path)), name="static")
        
        # Create API server routes
        api_server = create_api_server(self.orchestrator)
        self.app.mount("/api", api_server)
        
        logger.info("Digital Human UI initialized successfully")
    
    def _setup_routes(self):
        """Setup application routes"""
        
        @self.app.get("/")
        async def index():
            """Serve main UI page"""
            return HTMLResponse(self._generate_comprehensive_ui())
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            status = self.orchestrator.get_system_status()
            return {
                "status": "healthy",
                "system": status,
                "timestamp": asyncio.get_event_loop().time()
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication"""
            await self.websocket_handler.handle_connection(websocket)
    
    def _generate_comprehensive_ui(self) -> str:
        """Generate comprehensive UI HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIQ Digital Human - Financial Advisor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #ffffff;
            height: 100vh;
            overflow: hidden;
        }
        
        .main-container {
            display: grid;
            grid-template-columns: 1fr 400px;
            height: 100vh;
            gap: 20px;
            padding: 20px;
        }
        
        .avatar-section {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .header h1 {
            font-size: 2rem;
            background: linear-gradient(45deg, #00a8ff, #0066ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .avatar-container {
            flex: 1;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 20px;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        #avatar-canvas {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .emotion-indicator {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px 20px;
            border-radius: 30px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .emotion-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff00;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .controls {
            display: flex;
            gap: 15px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .control-btn {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(135deg, #0066ff, #004499);
            color: white;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
        }
        
        .control-btn:hover {
            transform: translateY(-2px);
        }
        
        .control-btn:active {
            transform: translateY(0);
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .chat-container {
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px 0;
            scroll-behavior: smooth;
        }
        
        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            text-align: right;
        }
        
        .message-content {
            display: inline-block;
            max-width: 80%;
            padding: 15px 20px;
            border-radius: 20px;
            position: relative;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #0066ff, #004499);
            color: white;
        }
        
        .message.assistant .message-content {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .message-time {
            font-size: 12px;
            color: rgba(255, 255, 255, 0.5);
            margin-top: 5px;
        }
        
        .chat-input-container {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        #user-input {
            flex: 1;
            padding: 15px 20px;
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.05);
            color: white;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        #user-input:focus {
            border-color: #0066ff;
        }
        
        #send-button {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #0066ff, #004499);
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s;
        }
        
        #send-button:hover {
            transform: scale(1.1);
        }
        
        .metrics-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .metric-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .metric-item:last-child {
            border-bottom: none;
        }
        
        .metric-value {
            font-weight: bold;
            color: #00a8ff;
        }
        
        .portfolio-view {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .portfolio-chart {
            height: 200px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            margin-top: 15px;
            position: relative;
            overflow: hidden;
        }
        
        #portfolio-canvas {
            width: 100%;
            height: 100%;
        }
        
        .loading-indicator {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 24px;
            display: none;
        }
        
        .typing-indicator {
            display: flex;
            gap: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #0066ff;
            animation: typing 1.4s infinite ease-in-out both;
        }
        
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes typing {
            0%, 80%, 100% {
                transform: scale(1);
                opacity: 0.5;
            }
            40% {
                transform: scale(1.3);
                opacity: 1;
            }
        }
        
        .status-bar {
            position: absolute;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px 20px;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 14px;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #00ff00;
        }
        
        @media (max-width: 768px) {
            .main-container {
                grid-template-columns: 1fr;
                grid-template-rows: 1fr 1fr;
            }
            
            .sidebar {
                grid-row: 2;
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="avatar-section">
            <div class="header">
                <h1>AIQ Digital Human</h1>
                <p>Your AI-Powered Financial Advisor with Emotional Intelligence</p>
            </div>
            
            <div class="avatar-container">
                <canvas id="avatar-canvas"></canvas>
                <div class="emotion-indicator">
                    <div class="emotion-dot"></div>
                    <span id="emotion-text">Ready</span>
                </div>
                <div class="status-bar">
                    <div class="status-item">
                        <div class="status-indicator"></div>
                        <span>Connected</span>
                    </div>
                    <div class="status-item">
                        <span id="fps-counter">60 FPS</span>
                    </div>
                    <div class="status-item">
                        <span id="latency">0ms</span>
                    </div>
                </div>
                <div class="loading-indicator" id="loading">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                <button class="control-btn" onclick="toggleAudio()">🔊 Audio</button>
                <button class="control-btn" onclick="toggleVideo()">📹 Video</button>
                <button class="control-btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
                <button class="control-btn" onclick="showSettings()">⚙️ Settings</button>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="chat-container">
                <h3>Conversation</h3>
                <div class="chat-messages" id="chat-messages"></div>
                <div class="chat-input-container">
                    <input type="text" id="user-input" placeholder="Ask about your finances..." />
                    <button id="send-button" onclick="sendMessage()">➤</button>
                </div>
            </div>
            
            <div class="metrics-container">
                <h3>Portfolio Metrics</h3>
                <div class="metric-item">
                    <span>Total Value</span>
                    <span class="metric-value" id="portfolio-value">$0</span>
                </div>
                <div class="metric-item">
                    <span>Daily Change</span>
                    <span class="metric-value" id="daily-change">+0%</span>
                </div>
                <div class="metric-item">
                    <span>Risk Level</span>
                    <span class="metric-value" id="risk-level">Low</span>
                </div>
                <div class="metric-item">
                    <span>Confidence</span>
                    <span class="metric-value" id="confidence">95%</span>
                </div>
            </div>
            
            <div class="portfolio-view">
                <h3>Performance</h3>
                <div class="portfolio-chart">
                    <canvas id="portfolio-canvas"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        let ws = null;
        let reconnectInterval = null;
        let isAudioEnabled = true;
        let isVideoEnabled = true;
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://localhost:${window.location.port || 8000}/ws`);
            
            ws.onopen = () => {
                console.log('Connected to Digital Human');
                document.getElementById('emotion-text').textContent = 'Connected';
                clearInterval(reconnectInterval);
                
                // Send initial session request
                ws.send(JSON.stringify({
                    type: 'start_session',
                    user_id: 'user_' + Date.now()
                }));
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleResponse(data);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                document.getElementById('emotion-text').textContent = 'Error';
            };
            
            ws.onclose = () => {
                document.getElementById('emotion-text').textContent = 'Disconnected';
                // Try to reconnect
                if (!reconnectInterval) {
                    reconnectInterval = setInterval(connectWebSocket, 5000);
                }
            };
        }
        
        // Connect on load
        connectWebSocket();
        
        // UI handlers
        const userInput = document.getElementById('user-input');
        const chatMessages = document.getElementById('chat-messages');
        
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        function sendMessage() {
            const message = userInput.value.trim();
            if (!message || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            // Add user message to chat
            addChatMessage('user', message);
            
            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            
            // Send to backend
            ws.send(JSON.stringify({
                type: 'user_message',
                content: message,
                timestamp: new Date().toISOString()
            }));
            
            userInput.value = '';
        }
        
        function addChatMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            timeDiv.textContent = new Date().toLocaleTimeString();
            
            messageDiv.appendChild(contentDiv);
            messageDiv.appendChild(timeDiv);
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function handleResponse(data) {
            // Hide loading indicator
            document.getElementById('loading').style.display = 'none';
            
            if (data.type === 'response') {
                addChatMessage('assistant', data.text);
                updateAvatar(data.animation);
                updateEmotion(data.emotion, data.emotion_intensity);
                updateMetrics(data.financial_data);
                
                // Update latency
                if (data.processing_time) {
                    document.getElementById('latency').textContent = 
                        Math.round(data.processing_time * 1000) + 'ms';
                }
            } else if (data.type === 'avatar_update') {
                updateAvatar(data.animation);
            } else if (data.type === 'metrics_update') {
                updateMetrics(data.metrics);
            } else if (data.type === 'session_started') {
                console.log('Session started:', data.session_id);
            }
        }
        
        function updateAvatar(animationData) {
            const canvas = document.getElementById('avatar-canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw avatar (placeholder for now)
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = Math.min(canvas.width, canvas.height) * 0.3;
            
            // Draw face
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
            ctx.fillStyle = '#ffffff20';
            ctx.fill();
            ctx.strokeStyle = '#0066ff';
            ctx.lineWidth = 3;
            ctx.stroke();
            
            // Draw eyes based on emotion
            if (animationData && animationData.expression_weights) {
                const eyeY = centerY - radius * 0.3;
                const eyeSpacing = radius * 0.3;
                
                // Left eye
                ctx.beginPath();
                ctx.arc(centerX - eyeSpacing, eyeY, radius * 0.1, 0, 2 * Math.PI);
                ctx.fillStyle = '#0066ff';
                ctx.fill();
                
                // Right eye
                ctx.beginPath();
                ctx.arc(centerX + eyeSpacing, eyeY, radius * 0.1, 0, 2 * Math.PI);
                ctx.fill();
                
                // Mouth based on emotion
                ctx.beginPath();
                ctx.arc(centerX, centerY + radius * 0.2, radius * 0.5, 0, Math.PI);
                ctx.strokeStyle = '#0066ff';
                ctx.lineWidth = 5;
                ctx.stroke();
            }
            
            // Add expression text
            if (animationData && animationData.emotion_label) {
                ctx.fillStyle = '#ffffff';
                ctx.font = '20px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(animationData.emotion_label, centerX, canvas.height - 40);
            }
        }
        
        function updateEmotion(emotion, intensity) {
            const emotionText = document.getElementById('emotion-text');
            const emotionDot = document.querySelector('.emotion-dot');
            
            emotionText.textContent = emotion || 'Neutral';
            
            // Update color based on emotion
            const emotionColors = {
                happy: '#00ff00',
                sad: '#0066ff',
                angry: '#ff0000',
                surprised: '#ffff00',
                neutral: '#cccccc',
                thinking: '#ff00ff',
                confident: '#00ffff'
            };
            
            emotionDot.style.background = emotionColors[emotion] || '#00ff00';
            
            // Update intensity with opacity
            if (intensity) {
                emotionDot.style.opacity = intensity;
            }
        }
        
        function updateMetrics(data) {
            if (!data) return;
            
            if (data.portfolio_value !== undefined) {
                document.getElementById('portfolio-value').textContent = 
                    '$' + data.portfolio_value.toLocaleString();
            }
            
            if (data.daily_change !== undefined) {
                const change = document.getElementById('daily-change');
                change.textContent = (data.daily_change > 0 ? '+' : '') + 
                    data.daily_change.toFixed(2) + '%';
                change.style.color = data.daily_change > 0 ? '#00ff00' : '#ff0000';
            }
            
            if (data.risk_level !== undefined) {
                document.getElementById('risk-level').textContent = data.risk_level;
            }
            
            if (data.confidence !== undefined) {
                document.getElementById('confidence').textContent = 
                    Math.round(data.confidence * 100) + '%';
            }
            
            // Update portfolio chart
            updatePortfolioChart(data);
        }
        
        function updatePortfolioChart(data) {
            const canvas = document.getElementById('portfolio-canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw simple line chart
            if (data.historical_values && data.historical_values.length > 0) {
                const values = data.historical_values;
                const maxValue = Math.max(...values);
                const minValue = Math.min(...values);
                const range = maxValue - minValue || 1;
                
                ctx.beginPath();
                ctx.strokeStyle = '#0066ff';
                ctx.lineWidth = 2;
                
                values.forEach((value, index) => {
                    const x = (index / (values.length - 1)) * canvas.width;
                    const y = canvas.height - ((value - minValue) / range) * canvas.height * 0.8 - 20;
                    
                    if (index === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                });
                
                ctx.stroke();
                
                // Draw gradient fill
                ctx.lineTo(canvas.width, canvas.height);
                ctx.lineTo(0, canvas.height);
                ctx.closePath();
                
                const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
                gradient.addColorStop(0, 'rgba(0, 102, 255, 0.3)');
                gradient.addColorStop(1, 'rgba(0, 102, 255, 0.0)');
                ctx.fillStyle = gradient;
                ctx.fill();
            }
        }
        
        // Control functions
        function toggleAudio() {
            isAudioEnabled = !isAudioEnabled;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'toggle_audio',
                    enabled: isAudioEnabled
                }));
            }
        }
        
        function toggleVideo() {
            isVideoEnabled = !isVideoEnabled;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'toggle_video',
                    enabled: isVideoEnabled
                }));
            }
        }
        
        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        }
        
        function showSettings() {
            // TODO: Implement settings modal
            console.log('Settings clicked');
        }
        
        // Update FPS counter
        let lastTime = Date.now();
        let frames = 0;
        
        function updateFPS() {
            frames++;
            const currentTime = Date.now();
            const delta = currentTime - lastTime;
            
            if (delta >= 1000) {
                const fps = Math.round((frames * 1000) / delta);
                document.getElementById('fps-counter').textContent = fps + ' FPS';
                frames = 0;
                lastTime = currentTime;
            }
            
            requestAnimationFrame(updateFPS);
        }
        
        updateFPS();
        
        // Initialize canvases
        updateAvatar({});
        updatePortfolioChart({});
    </script>
</body>
</html>"""
    
    async def run(self):
        """Run the UI application"""
        await self.initialize()
        
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=self.config["api_port"],
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()


async def main():
    """Main entry point"""
    logger.info("Starting AIQ Digital Human UI...")
    
    # Create and run UI
    ui = DigitalHumanUI()
    await ui.run()


if __name__ == "__main__":
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Digital Human UI...")
    except Exception as e:
        logger.error(f"Error running application: {e}")
        sys.exit(1)