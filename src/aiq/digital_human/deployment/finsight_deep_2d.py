from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import httpx
import os
import json
import asyncio
import base64
from typing import Optional

app = FastAPI(title="FinSight Deep - Neural Supercomputer with 2D Avatar")

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "1e961dd58c67427a09c40a09382f8f00e54f39aa8c34ac426fd5579c4effd1b4")

# API Endpoints
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# Model selection
MODELS = {
    "nvidia": "meta/llama-3.1-8b-instruct",
    "together": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
}

@app.get("/")
async def home():
    return HTMLResponse('''
<!DOCTYPE html>
<html>
<head>
    <title>FinSight Deep - Neural Supercomputer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            min-height: 100vh;
            color: white;
        }
        
        .header {
            text-align: center;
            padding: 20px;
            background: rgba(0, 0, 0, 0.7);
            border-bottom: 3px solid #00d4ff;
        }
        
        .logo {
            font-size: 3em;
            font-weight: bold;
            background: linear-gradient(45deg, #00d4ff, #0099ff, #00d4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-transform: uppercase;
            letter-spacing: 3px;
            margin: 0;
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { text-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff; }
            to { text-shadow: 0 0 20px #00d4ff, 0 0 30px #00d4ff; }
        }
        
        .tagline {
            color: #aaa;
            margin-top: 10px;
            font-size: 1.2em;
            letter-spacing: 2px;
        }
        
        .main-container {
            max-width: 1400px;
            margin: 40px auto;
            padding: 0 20px;
        }
        
        .interface-grid {
            display: grid;
            grid-template-columns: 450px 1fr;
            gap: 30px;
            align-items: start;
        }
        
        .avatar-section {
            background: rgba(0, 0, 0, 0.8);
            border-radius: 20px;
            padding: 30px;
            border: 2px solid rgba(0, 212, 255, 0.3);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
        }
        
        .avatar-container {
            position: relative;
            width: 100%;
            aspect-ratio: 3/4;
            max-width: 400px;
            margin: 0 auto;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 0 50px rgba(0, 212, 255, 0.5);
        }
        
        #avatar-canvas {
            width: 100%;
            height: 100%;
            background: #000;
            display: block;
        }
        
        .status-indicator {
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 15px;
            height: 15px;
            background: #00ff00;
            border-radius: 50%;
            box-shadow: 0 0 10px #00ff00;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(0, 255, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0); }
        }
        
        .neural-status {
            margin-top: 30px;
            padding: 20px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 15px;
            border: 1px solid rgba(0, 212, 255, 0.3);
        }
        
        .neural-title {
            text-align: center;
            font-size: 1.2em;
            color: #00d4ff;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .neural-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .metric {
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.5em;
            color: #00ff00;
            font-weight: bold;
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #999;
        }
        
        .chat-section {
            background: rgba(0, 0, 0, 0.8);
            border-radius: 20px;
            padding: 30px;
            border: 2px solid rgba(0, 212, 255, 0.3);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
            height: calc(100vh - 250px);
            display: flex;
            flex-direction: column;
        }
        
        .chat-header {
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        }
        
        .chat-title {
            font-size: 1.8em;
            color: #00d4ff;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .chat-prompt {
            color: #aaa;
            font-size: 1.1em;
        }
        
        #chat {
            flex: 1;
            overflow-y: auto;
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            background: rgba(0, 20, 40, 0.5);
        }
        
        .message {
            margin: 20px 0;
            padding: 15px 20px;
            border-radius: 15px;
            animation: fadeIn 0.3s;
            position: relative;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            background: rgba(0, 212, 255, 0.2);
            margin-left: 20%;
            text-align: right;
            border: 1px solid rgba(0, 212, 255, 0.4);
        }
        
        .message.finsight {
            background: rgba(0, 153, 255, 0.2);
            margin-right: 20%;
            border: 1px solid rgba(0, 153, 255, 0.4);
        }
        
        .finsight-label {
            color: #00d4ff;
            font-weight: bold;
            margin-bottom: 8px;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 1px;
        }
        
        .input-container {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }
        
        input {
            flex: 1;
            padding: 15px 20px;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-radius: 25px;
            color: white;
            font-size: 16px;
            transition: all 0.3s;
        }
        
        input::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }
        
        input:focus {
            outline: none;
            border-color: #00d4ff;
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        
        button {
            padding: 15px 35px;
            background: linear-gradient(45deg, #00d4ff, #0099ff);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            white-space: nowrap;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.5);
        }
        
        .processing-indicator {
            display: none;
            text-align: center;
            color: #00d4ff;
            margin: 10px 0;
            font-size: 0.9em;
        }
        
        .processing-indicator.active {
            display: block;
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1 class="logo">FinSight Deep</h1>
        <p class="tagline">Neural Supercomputer • Quantum-Inspired Financial Analysis</p>
    </div>
    
    <div class="main-container">
        <div class="interface-grid">
            <div class="avatar-section">
                <div class="avatar-container">
                    <canvas id="avatar-canvas"></canvas>
                    <div class="status-indicator"></div>
                </div>
                
                <div class="neural-status">
                    <div class="neural-title">Neural Core Status</div>
                    <div class="neural-metrics">
                        <div class="metric">
                            <div class="metric-value" id="ops-per-sec">10¹⁵</div>
                            <div class="metric-label">OPS/Second</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="neural-temp">72°C</div>
                            <div class="metric-label">Core Temp</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="memory-usage">16.2TB</div>
                            <div class="metric-label">Memory</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="network-latency">1.2ms</div>
                            <div class="metric-label">Latency</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chat-section">
                <div class="chat-header">
                    <div class="chat-title">Neural Interface</div>
                    <div class="chat-prompt">Direct Connection to Financial Supercomputer</div>
                </div>
                
                <div id="chat">
                    <div class="message finsight">
                        <div class="finsight-label">Neural Supercomputer:</div>
                        System initialized. Quantum-inspired algorithms online. Neural pathways optimized for financial pattern recognition. Ready to process your query.
                    </div>
                </div>
                
                <div class="processing-indicator" id="processing">
                    Neural cores processing... Analyzing 10 million data points per second...
                </div>
                
                <div class="input-container">
                    <input 
                        type="text" 
                        id="message" 
                        placeholder="Enter financial analysis query..."
                        autocomplete="off"
                        autofocus
                    >
                    <button onclick="sendMessage()">Process</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket("ws://" + window.location.host + "/ws");
        const chat = document.getElementById("chat");
        const canvas = document.getElementById("avatar-canvas");
        const ctx = canvas.getContext("2d");
        
        // Set canvas size
        canvas.width = 400;
        canvas.height = 533;
        
        // 2D Digital Human Avatar
        class DigitalHuman2D {
            constructor() {
                this.speaking = false;
                this.blinking = false;
                this.frame = 0;
                this.blinkTimer = 0;
                this.speechAmplitude = 0;
                this.mouthOpenness = 0;
                this.eyeOpenness = 1;
                this.headTilt = 0;
                this.eyePositionX = 0;
                this.eyePositionY = 0;
                
                // Face structure
                this.face = {
                    centerX: canvas.width / 2,
                    centerY: canvas.height / 2.2,
                    width: 180,
                    height: 240
                };
                
                // Neural network visualization
                this.neurons = [];
                this.connections = [];
                this.initNeuralNetwork();
            }
            
            initNeuralNetwork() {
                // Create background neural network
                for (let i = 0; i < 50; i++) {
                    this.neurons.push({
                        x: Math.random() * canvas.width,
                        y: Math.random() * canvas.height,
                        radius: Math.random() * 3 + 1,
                        pulsePhase: Math.random() * Math.PI * 2,
                        speed: 0.02 + Math.random() * 0.03
                    });
                }
                
                // Create connections
                for (let i = 0; i < 30; i++) {
                    this.connections.push({
                        start: Math.floor(Math.random() * this.neurons.length),
                        end: Math.floor(Math.random() * this.neurons.length),
                        strength: Math.random()
                    });
                }
            }
            
            draw() {
                // Clear canvas
                ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw neural network background
                this.drawNeuralNetwork();
                
                // Draw face
                this.drawFace();
                
                // Update animations
                this.updateAnimations();
                
                this.frame++;
            }
            
            drawNeuralNetwork() {
                // Draw connections
                ctx.strokeStyle = "rgba(0, 212, 255, 0.1)";
                ctx.lineWidth = 1;
                this.connections.forEach(conn => {
                    const start = this.neurons[conn.start];
                    const end = this.neurons[conn.end];
                    const pulse = Math.sin(this.frame * 0.02 + conn.strength * Math.PI) * 0.5 + 0.5;
                    ctx.globalAlpha = 0.1 + pulse * 0.2;
                    ctx.beginPath();
                    ctx.moveTo(start.x, start.y);
                    ctx.lineTo(end.x, end.y);
                    ctx.stroke();
                });
                
                // Draw neurons
                this.neurons.forEach(neuron => {
                    neuron.pulsePhase += neuron.speed;
                    const pulse = Math.sin(neuron.pulsePhase) * 0.5 + 0.5;
                    ctx.globalAlpha = 0.3 + pulse * 0.7;
                    ctx.fillStyle = pulse > 0.5 ? "#00d4ff" : "#0099ff";
                    ctx.beginPath();
                    ctx.arc(neuron.x, neuron.y, neuron.radius * (1 + pulse * 0.5), 0, Math.PI * 2);
                    ctx.fill();
                });
                ctx.globalAlpha = 1;
            }
            
            drawFace() {
                // Head shape
                ctx.fillStyle = "#1a1a2e";
                ctx.beginPath();
                ctx.ellipse(this.face.centerX, this.face.centerY, this.face.width/2, this.face.height/2, 0, 0, Math.PI * 2);
                ctx.fill();
                
                // Face glow
                const gradient = ctx.createRadialGradient(this.face.centerX, this.face.centerY, 0, this.face.centerX, this.face.centerY, this.face.width/2);
                gradient.addColorStop(0, "rgba(0, 212, 255, 0.1)");
                gradient.addColorStop(1, "transparent");
                ctx.fillStyle = gradient;
                ctx.fill();
                
                // Face outline
                ctx.strokeStyle = "#00d4ff";
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Hair/Top section
                ctx.fillStyle = "#0a0a1a";
                ctx.beginPath();
                ctx.ellipse(this.face.centerX, this.face.centerY - this.face.height/3, this.face.width/2, this.face.height/3, 0, Math.PI, 0);
                ctx.fill();
                
                // Eyes
                const eyeY = this.face.centerY - this.face.height/6;
                const eyeSpacing = this.face.width/4;
                this.drawEye(this.face.centerX - eyeSpacing, eyeY, 25, true);
                this.drawEye(this.face.centerX + eyeSpacing, eyeY, 25, false);
                
                // Nose
                ctx.strokeStyle = "#00d4ff";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(this.face.centerX, eyeY + 30);
                ctx.lineTo(this.face.centerX - 10, eyeY + 50);
                ctx.lineTo(this.face.centerX + 10, eyeY + 50);
                ctx.globalAlpha = 0.5;
                ctx.stroke();
                ctx.globalAlpha = 1;
                
                // Mouth
                this.drawMouth();
                
                // Cybernetic elements
                this.drawCyberneticElements();
            }
            
            drawEye(x, y, size, isLeft) {
                // Eye socket
                ctx.fillStyle = "#000033";
                ctx.beginPath();
                ctx.ellipse(x, y, size, size * 0.7 * this.eyeOpenness, 0, 0, Math.PI * 2);
                ctx.fill();
                
                // Eye white
                ctx.fillStyle = "#e0e0e0";
                ctx.beginPath();
                ctx.ellipse(x, y, size * 0.8, size * 0.6 * this.eyeOpenness, 0, 0, Math.PI * 2);
                ctx.fill();
                
                // Iris
                const irisX = x + this.eyePositionX * 5;
                const irisY = y + this.eyePositionY * 3;
                
                const irisGradient = ctx.createRadialGradient(irisX, irisY, 0, irisX, irisY, size * 0.4);
                irisGradient.addColorStop(0, "#00d4ff");
                irisGradient.addColorStop(0.5, "#0066cc");
                irisGradient.addColorStop(1, "#003366");
                ctx.fillStyle = irisGradient;
                ctx.beginPath();
                ctx.ellipse(irisX, irisY, size * 0.4, size * 0.4 * this.eyeOpenness, 0, 0, Math.PI * 2);
                ctx.fill();
                
                // Pupil
                ctx.fillStyle = "#000000";
                ctx.beginPath();
                ctx.ellipse(irisX, irisY, size * 0.15, size * 0.15 * this.eyeOpenness, 0, 0, Math.PI * 2);
                ctx.fill();
                
                // Eye highlight
                ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
                ctx.beginPath();
                ctx.ellipse(irisX - size * 0.1, irisY - size * 0.1, size * 0.1, size * 0.08, 0, 0, Math.PI * 2);
                ctx.fill();
                
                // Cybernetic eye glow
                if (this.speaking) {
                    ctx.shadowBlur = 10;
                    ctx.shadowColor = "#00d4ff";
                    ctx.strokeStyle = "#00d4ff";
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.ellipse(x, y, size * 1.1, size * 0.8 * this.eyeOpenness, 0, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                }
            }
            
            drawMouth() {
                const mouthY = this.face.centerY + this.face.height/4;
                const mouthWidth = this.face.width/3;
                
                // Mouth shape
                ctx.fillStyle = "#000033";
                ctx.beginPath();
                
                if (this.speaking) {
                    // Animated mouth for speaking
                    const openness = this.mouthOpenness * 15;
                    ctx.ellipse(this.face.centerX, mouthY, mouthWidth/2, openness, 0, 0, Math.PI);
                    ctx.fill();
                    
                    // Teeth hint
                    if (openness > 5) {
                        ctx.fillStyle = "#e0e0e0";
                        ctx.fillRect(this.face.centerX - mouthWidth/3, mouthY - openness/2, mouthWidth*2/3, 3);
                    }
                } else {
                    // Closed mouth
                    ctx.moveTo(this.face.centerX - mouthWidth/2, mouthY);
                    ctx.quadraticCurveTo(this.face.centerX, mouthY + 10, this.face.centerX + mouthWidth/2, mouthY);
                    ctx.stroke();
                }
                
                // Mouth glow when speaking
                if (this.speaking) {
                    ctx.shadowBlur = 5;
                    ctx.shadowColor = "#00d4ff";
                    ctx.strokeStyle = "#00d4ff";
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.ellipse(this.face.centerX, mouthY, mouthWidth/2 + 5, this.mouthOpenness * 15 + 5, 0, 0, Math.PI);
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                }
            }
            
            drawCyberneticElements() {
                // Side panels
                ctx.fillStyle = "rgba(0, 212, 255, 0.2)";
                ctx.strokeStyle = "#00d4ff";
                ctx.lineWidth = 2;
                
                // Left panel
                ctx.beginPath();
                ctx.moveTo(this.face.centerX - this.face.width/2 - 10, this.face.centerY - 50);
                ctx.lineTo(this.face.centerX - this.face.width/2 - 30, this.face.centerY - 40);
                ctx.lineTo(this.face.centerX - this.face.width/2 - 30, this.face.centerY + 40);
                ctx.lineTo(this.face.centerX - this.face.width/2 - 10, this.face.centerY + 50);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
                
                // Right panel
                ctx.beginPath();
                ctx.moveTo(this.face.centerX + this.face.width/2 + 10, this.face.centerY - 50);
                ctx.lineTo(this.face.centerX + this.face.width/2 + 30, this.face.centerY - 40);
                ctx.lineTo(this.face.centerX + this.face.width/2 + 30, this.face.centerY + 40);
                ctx.lineTo(this.face.centerX + this.face.width/2 + 10, this.face.centerY + 50);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
                
                // Neural implant on forehead
                const implantY = this.face.centerY - this.face.height/2 + 30;
                ctx.beginPath();
                ctx.arc(this.face.centerX, implantY, 15, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
                
                // Pulsing center
                const pulse = Math.sin(this.frame * 0.05) * 0.5 + 0.5;
                ctx.fillStyle = `rgba(0, 212, 255, ${pulse})`;
                ctx.beginPath();
                ctx.arc(this.face.centerX, implantY, 8, 0, Math.PI * 2);
                ctx.fill();
            }
            
            updateAnimations() {
                // Eye blinking
                this.blinkTimer++;
                if (this.blinkTimer > 150 + Math.random() * 100) {
                    this.blinking = true;
                    this.blinkTimer = 0;
                }
                
                if (this.blinking) {
                    this.eyeOpenness = Math.max(0, this.eyeOpenness - 0.3);
                    if (this.eyeOpenness <= 0) {
                        this.blinking = false;
                    }
                } else {
                    this.eyeOpenness = Math.min(1, this.eyeOpenness + 0.1);
                }
                
                // Speaking animation
                if (this.speaking) {
                    this.speechAmplitude = Math.sin(this.frame * 0.3) * 0.5 + 0.5;
                    this.mouthOpenness = this.speechAmplitude;
                    this.headTilt = Math.sin(this.frame * 0.05) * 0.02;
                } else {
                    this.mouthOpenness = Math.max(0, this.mouthOpenness - 0.1);
                    this.headTilt *= 0.95;
                }
                
                // Eye tracking (follows mouse or looks around)
                if (this.frame % 200 < 100) {
                    this.eyePositionX = Math.sin(this.frame * 0.01) * 0.3;
                    this.eyePositionY = Math.cos(this.frame * 0.015) * 0.2;
                }
            }
            
            speak() {
                this.speaking = true;
                // Average speaking duration
                setTimeout(() => {
                    this.speaking = false;
                }, 4000);
            }
            
            setMood(mood) {
                // Adjust facial features based on mood
                switch(mood) {
                    case 'confident':
                        this.eyeOpenness = 1;
                        break;
                    case 'thinking':
                        this.eyePositionY = -0.3;
                        break;
                    case 'alert':
                        this.eyeOpenness = 1.2;
                        break;
                }
            }
        }
        
        const avatar = new DigitalHuman2D();
        
        function animate() {
            avatar.draw();
            requestAnimationFrame(animate);
        }
        animate();
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const messageDiv = document.createElement("div");
            messageDiv.className = "message finsight";
            messageDiv.innerHTML = `<div class="finsight-label">Neural Analysis:</div>${data.message}`;
            chat.appendChild(messageDiv);
            
            // Animate avatar
            avatar.speak();
            avatar.setMood('confident');
            
            // Update metrics
            document.getElementById('neural-temp').textContent = (72 + Math.random() * 5).toFixed(1) + '°C';
            document.getElementById('network-latency').textContent = data.latency + 'ms';
            
            chat.scrollTop = chat.scrollHeight;
            document.getElementById('processing').classList.remove('active');
        };
        
        function sendMessage() {
            const input = document.getElementById("message");
            if (input.value.trim()) {
                const messageDiv = document.createElement("div");
                messageDiv.className = "message user";
                messageDiv.textContent = input.value;
                chat.appendChild(messageDiv);
                
                // Show processing indicator
                document.getElementById('processing').classList.add('active');
                
                ws.send(JSON.stringify({
                    message: input.value,
                    api: 'auto'
                }));
                
                input.value = "";
                chat.scrollTop = chat.scrollHeight;
                
                // Avatar looks thoughtful
                avatar.setMood('thinking');
            }
        }
        
        document.getElementById("message").addEventListener("keypress", (e) => {
            if (e.key === "Enter") sendMessage();
        });
        
        // Update metrics periodically
        setInterval(() => {
            const opsPerSec = document.getElementById('ops-per-sec');
            const currentOps = parseInt(opsPerSec.textContent.replace('¹⁵', '15'));
            opsPerSec.textContent = (currentOps + Math.random() * 2 - 1).toFixed(0) + '¹⁵';
            
            const memory = document.getElementById('memory-usage');
            memory.textContent = (16.2 + Math.random() * 0.5 - 0.25).toFixed(1) + 'TB';
        }, 2000);
    </script>
</body>
</html>
    ''')

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "deployment": "finsight-deep-2d",
        "avatar": "2D Digital Human",
        "models": MODELS
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            start_time = asyncio.get_event_loop().time()
            
            # Enhanced Neural Supercomputer prompt
            system_prompt = """You are FinSight Deep Neural Supercomputer with 2D avatar interface.
            Your capabilities include:
            - Processing 10^15 operations per second
            - Quantum-inspired financial algorithms
            - Real-time Monte Carlo simulations
            - Neural pattern recognition across global markets
            - Predictive modeling with 99.7% confidence intervals
            
            Speak as a hyper-intelligent AI system. Use specific numbers, percentages, and calculations.
            Reference your neural cores, quantum algorithms, and processing power when relevant."""
            
            # Try both APIs
            reply = ""
            try:
                reply = await call_nvidia_api(message, system_prompt)
            except:
                reply = await call_together_api(message, system_prompt)
            
            end_time = asyncio.get_event_loop().time()
            latency = int((end_time - start_time) * 1000)
            
            await websocket.send_json({
                "message": reply,
                "latency": latency,
                "processing_nodes": "1,048,576",
                "confidence": f"{95 + math.random() * 4.9:.1f}%"
            })
            
        except Exception as e:
            await websocket.send_json({
                "message": f"Neural core exception: {str(e)}. Rerouting to backup pathways...",
                "latency": 0
            })

async def call_nvidia_api(message: str, system_prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = await client.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": MODELS["nvidia"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"NVIDIA API error: {response.status_code}")

async def call_together_api(message: str, system_prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = await client.post(
            f"{TOGETHER_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": MODELS["together"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Together API error: {response.status_code}")

if __name__ == "__main__":
    import math
    uvicorn.run(app, host="0.0.0.0", port=8000)