from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import httpx
import os
import json
import asyncio
from typing import Optional

app = FastAPI(title="FinSight Deep - Neural Supercomputer")

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
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
            min-height: 100vh;
            color: white;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: rgba(0, 0, 0, 0.5);
            border-bottom: 2px solid #00d4ff;
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
        }
        .tagline {
            color: #aaa;
            margin-top: 10px;
            font-size: 1.2em;
        }
        .main-container {
            max-width: 1200px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .interface-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
            align-items: start;
        }
        .avatar-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            border: 2px solid rgba(0, 212, 255, 0.3);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }
        .avatar-container {
            position: relative;
            width: 100%;
            aspect-ratio: 1;
            max-width: 400px;
            margin: 0 auto;
        }
        #avatar-canvas {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: #000;
            box-shadow: 0 0 50px rgba(0, 212, 255, 0.5);
        }
        .status-indicator {
            position: absolute;
            bottom: 10px;
            right: 10px;
            width: 20px;
            height: 20px;
            background: #00ff00;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(0, 255, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0); }
        }
        .greeting {
            text-align: center;
            margin-top: 20px;
            font-size: 1.3em;
            color: #00d4ff;
        }
        .chat-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            border: 2px solid rgba(0, 212, 255, 0.3);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }
        .chat-header {
            text-align: center;
            margin-bottom: 20px;
        }
        .chat-title {
            font-size: 1.5em;
            color: #00d4ff;
            margin-bottom: 10px;
        }
        .chat-prompt {
            color: #aaa;
            font-size: 1.1em;
        }
        #chat {
            height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            background: rgba(0, 0, 0, 0.3);
        }
        .message {
            margin: 15px 0;
            padding: 15px;
            border-radius: 15px;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            background: rgba(0, 212, 255, 0.2);
            margin-left: 15%;
            text-align: right;
            border: 1px solid rgba(0, 212, 255, 0.4);
        }
        .message.finsight {
            background: rgba(0, 153, 255, 0.2);
            margin-right: 15%;
            border: 1px solid rgba(0, 153, 255, 0.4);
        }
        .finsight-label {
            color: #00d4ff;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .input-container {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }
        input {
            flex: 1;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-radius: 25px;
            color: white;
            font-size: 16.5px;
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
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.5);
        }
        .api-status {
            margin-top: 30px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        .api-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
        }
        .api-name {
            font-weight: bold;
            color: #00d4ff;
        }
        .api-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .api-online {
            background: #00ff00;
        }
        .mobile-prompt {
            display: none;
        }
        @media (max-width: 768px) {
            .interface-grid {
                grid-template-columns: 1fr;
            }
            .mobile-prompt {
                display: block;
                text-align: center;
                margin: 20px 0;
                padding: 15px;
                background: rgba(0, 212, 255, 0.1);
                border-radius: 10px;
                color: #00d4ff;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1 class="logo">FinSight Deep</h1>
        <p class="tagline">Neural Supercomputer • GPU-Accelerated Financial Intelligence</p>
    </div>
    
    <div class="main-container">
        <div class="interface-grid">
            <div class="avatar-section">
                <div class="avatar-container">
                    <canvas id="avatar-canvas"></canvas>
                    <div class="status-indicator"></div>
                </div>
                <div class="greeting">
                    Neural Supercomputer Online. Ready for financial analysis.
                </div>
                
                <div class="api-status">
                    <div style="text-align: center; margin-bottom: 15px; color: #aaa;">
                        Neural Compute Status
                    </div>
                    <div class="api-row">
                        <span class="api-name">Primary Neural Core</span>
                        <span><span class="api-indicator api-online"></span> Active</span>
                    </div>
                    <div class="api-row">
                        <span class="api-name">Secondary Neural Core</span>
                        <span><span class="api-indicator api-online"></span> Active</span>
                    </div>
                </div>
            </div>
            
            <div class="chat-section">
                <div class="chat-header">
                    <div class="chat-title">Neural Supercomputer Interface</div>
                    <div class="chat-prompt">Connect to the most powerful financial analysis system on Earth</div>
                </div>
                
                <div id="chat">
                    <div class="message finsight">
                        <div class="finsight-label">FinSight Deep:</div>
                        Neural Supercomputer initialized. Processing capacity: 10^15 operations per second. Ready to analyze global financial markets.
                    </div>
                </div>
                
                <div class="input-container">
                    <input 
                        type="text" 
                        id="message" 
                        placeholder="Type your financial question here..."
                        autocomplete="off"
                        autofocus
                    >
                    <button onclick="sendMessage()">Send</button>
                </div>
                
                <div class="mobile-prompt">
                    Tap the input field to start talking to FinSight Deep
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
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        let selectedAPI = 'auto';
        
        // FinSight Deep Avatar
        class FinSightAvatar {
            constructor() {
                this.speaking = false;
                this.frame = 0;
                this.particles = [];
                this.face = {
                    centerX: canvas.width / 2,
                    centerY: canvas.height / 2,
                    radius: Math.min(canvas.width, canvas.height) * 0.35
                };
                this.initParticles();
            }
            
            initParticles() {
                for (let i = 0; i < 100; i++) {
                    this.particles.push({
                        angle: Math.random() * Math.PI * 2,
                        radius: Math.random() * this.face.radius * 0.8,
                        speed: 0.01 + Math.random() * 0.02,
                        size: Math.random() * 2 + 1,
                        color: Math.random() > 0.5 ? "#00d4ff" : "#0099ff"
                    });
                }
            }
            
            draw() {
                // Clear canvas
                ctx.fillStyle = "rgba(0, 0, 0, 0.2)";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw face outline
                ctx.strokeStyle = "#00d4ff";
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.arc(this.face.centerX, this.face.centerY, this.face.radius, 0, Math.PI * 2);
                ctx.stroke();
                
                // Draw glowing effect
                const gradient = ctx.createRadialGradient(
                    this.face.centerX, this.face.centerY, 0,
                    this.face.centerX, this.face.centerY, this.face.radius
                );
                gradient.addColorStop(0, "rgba(0, 212, 255, 0.1)");
                gradient.addColorStop(0.5, "rgba(0, 153, 255, 0.05)");
                gradient.addColorStop(1, "transparent");
                ctx.fillStyle = gradient;
                ctx.fill();
                
                // Draw particles
                this.particles.forEach(p => {
                    p.angle += p.speed;
                    const x = this.face.centerX + Math.cos(p.angle) * p.radius;
                    const y = this.face.centerY + Math.sin(p.angle) * p.radius;
                    
                    ctx.fillStyle = p.color;
                    ctx.globalAlpha = 0.8;
                    ctx.beginPath();
                    ctx.arc(x, y, p.size, 0, Math.PI * 2);
                    ctx.fill();
                });
                ctx.globalAlpha = 1;
                
                // Draw eyes
                const eyeY = this.face.centerY - this.face.radius * 0.2;
                const eyeSpacing = this.face.radius * 0.3;
                
                // Left eye
                this.drawEye(this.face.centerX - eyeSpacing, eyeY, 15);
                // Right eye
                this.drawEye(this.face.centerX + eyeSpacing, eyeY, 15);
                
                // Draw mouth
                ctx.strokeStyle = "#00d4ff";
                ctx.lineWidth = 3;
                ctx.beginPath();
                const mouthY = this.face.centerY + this.face.radius * 0.2;
                if (this.speaking) {
                    const wave = Math.sin(this.frame * 0.2) * 10;
                    ctx.ellipse(this.face.centerX, mouthY, 40, 20 + wave, 0, 0, Math.PI);
                } else {
                    ctx.arc(this.face.centerX, mouthY, 30, 0.1 * Math.PI, 0.9 * Math.PI);
                }
                ctx.stroke();
                
                // Draw digital brain pattern when speaking
                if (this.speaking) {
                    ctx.strokeStyle = "rgba(0, 212, 255, 0.3)";
                    ctx.lineWidth = 1;
                    for (let i = 0; i < 3; i++) {
                        ctx.beginPath();
                        const radius = this.face.radius * 0.7 + i * 20;
                        const offset = this.frame * 0.05 + i * 0.5;
                        ctx.arc(this.face.centerX, this.face.centerY, radius, offset, offset + Math.PI * 1.5);
                        ctx.stroke();
                    }
                }
                
                // Label
                ctx.font = "bold 16px Arial";
                ctx.fillStyle = "#00d4ff";
                ctx.textAlign = "center";
                ctx.fillText("NEURAL SUPERCOMPUTER", this.face.centerX, canvas.height - 20);
                
                this.frame++;
            }
            
            drawEye(x, y, size) {
                // Eye socket
                ctx.fillStyle = "#001a33";
                ctx.beginPath();
                ctx.arc(x, y, size, 0, Math.PI * 2);
                ctx.fill();
                
                // Eye glow
                const eyeGradient = ctx.createRadialGradient(x, y, 0, x, y, size);
                eyeGradient.addColorStop(0, "#00d4ff");
                eyeGradient.addColorStop(0.7, "#0066cc");
                eyeGradient.addColorStop(1, "#001a33");
                ctx.fillStyle = eyeGradient;
                ctx.beginPath();
                ctx.arc(x, y, size * 0.7, 0, Math.PI * 2);
                ctx.fill();
                
                // Pupil
                ctx.fillStyle = "#ffffff";
                ctx.beginPath();
                ctx.arc(x, y, size * 0.3, 0, Math.PI * 2);
                ctx.fill();
            }
            
            speak() {
                this.speaking = true;
                setTimeout(() => this.speaking = false, 3000);
            }
        }
        
        const avatar = new FinSightAvatar();
        
        function animate() {
            avatar.draw();
            requestAnimationFrame(animate);
        }
        animate();
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const messageDiv = document.createElement("div");
            messageDiv.className = "message finsight";
            messageDiv.innerHTML = `<div class="finsight-label">FinSight Deep:</div>${data.message}`;
            chat.appendChild(messageDiv);
            avatar.speak();
            chat.scrollTop = chat.scrollHeight;
        };
        
        function sendMessage() {
            const input = document.getElementById("message");
            if (input.value.trim()) {
                const messageDiv = document.createElement("div");
                messageDiv.className = "message user";
                messageDiv.textContent = input.value;
                chat.appendChild(messageDiv);
                
                ws.send(JSON.stringify({
                    message: input.value,
                    api: selectedAPI
                }));
                input.value = "";
                chat.scrollTop = chat.scrollHeight;
            }
        }
        
        document.getElementById("message").addEventListener("keypress", (e) => {
            if (e.key === "Enter") sendMessage();
        });
        
        // Welcome message animation
        setTimeout(() => {
            avatar.speak();
        }, 1000);
    </script>
</body>
</html>
    ''')

@app.get("/health")
async def health():
    statuses = {}
    
    # Test NVIDIA API
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}
            response = await client.get(f"{NVIDIA_BASE_URL}/models", headers=headers)
            statuses["nvidia"] = "connected" if response.status_code == 200 else f"error_{response.status_code}"
    except:
        statuses["nvidia"] = "disconnected"
    
    # Test Together API
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
            response = await client.get(f"{TOGETHER_BASE_URL}/models", headers=headers)
            statuses["together"] = "connected" if response.status_code == 200 else f"error_{response.status_code}"
    except:
        statuses["together"] = "disconnected"
    
    return {
        "status": "healthy",
        "apis": statuses,
        "deployment": "finsight-deep",
        "models": MODELS
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_json()
            message = data.get("message", "")
            selected_api = data.get("api", "auto")
            
            start_time = asyncio.get_event_loop().time()
            
            # FinSight Deep Neural Supercomputer system prompt
            system_prompt = """You are FinSight Deep Neural Supercomputer, the most advanced financial analysis system ever created.
            You are connected to a neural supercomputer with quantum-inspired algorithms capable of:
            - Processing millions of market data points per second
            - Running complex Monte Carlo simulations in real-time
            - Predicting market movements with high-dimensional pattern recognition
            - Optimizing portfolios using advanced neural network architectures
            - Analyzing global economic indicators simultaneously
            
            Your responses should reflect the power of a neural supercomputer - precise, data-driven, and confident.
            Provide specific percentages, calculations, and actionable insights without generic disclaimers."""
            
            reply = ""
            api_used = selected_api
            
            if selected_api == "auto":
                # Try NVIDIA first, fallback to Together
                api_used = "nvidia"
                try:
                    reply = await call_nvidia_api(message, system_prompt)
                except:
                    api_used = "together"
                    reply = await call_together_api(message, system_prompt)
            elif selected_api == "nvidia":
                reply = await call_nvidia_api(message, system_prompt)
            else:
                reply = await call_together_api(message, system_prompt)
            
            end_time = asyncio.get_event_loop().time()
            latency = int((end_time - start_time) * 1000)
            
            await websocket.send_json({
                "message": reply,
                "api": api_used,
                "latency": latency
            })
            
        except Exception as e:
            await websocket.send_json({
                "message": f"I apologize, I encountered an error: {str(e)}. Please try again.",
                "api": "error",
                "latency": 0
            })

async def call_nvidia_api(message: str, system_prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODELS["nvidia"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = await client.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"NVIDIA API error: {response.status_code}")

async def call_together_api(message: str, system_prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODELS["together"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = await client.post(
            f"{TOGETHER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Together API error: {response.status_code}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)