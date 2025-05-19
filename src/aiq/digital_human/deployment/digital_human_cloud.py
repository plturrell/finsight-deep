from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import httpx
import os
import json
import asyncio
from typing import Optional

app = FastAPI(title="Digital Human Financial Advisor - Multi-Cloud GPU")

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
    <title>Digital Human - Multi-Cloud GPU</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 40px 0;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 3.5em;
            margin: 0;
            background: linear-gradient(45deg, #76b900, #00a86b, #0080ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .cloud-badges {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        .cloud-badge {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px 25px;
            border-radius: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .nvidia { border-color: #76b900; }
        .together { border-color: #0080ff; }
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            margin-top: 30px;
        }
        .status-panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .gpu-meter {
            margin: 20px 0;
            padding: 15px;
            background: rgba(0, 255, 0, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(0, 255, 0, 0.3);
        }
        .gpu-meter.nvidia {
            background: rgba(118, 185, 0, 0.1);
            border-color: rgba(118, 185, 0, 0.3);
        }
        .gpu-meter.together {
            background: rgba(0, 128, 255, 0.1);
            border-color: rgba(0, 128, 255, 0.3);
        }
        .chat-panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        #avatar {
            width: 100%;
            height: 350px;
            background: #000;
            border-radius: 15px;
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }
        #chat {
            height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            background: rgba(0, 0, 0, 0.3);
        }
        .message {
            margin: 15px 0;
            padding: 15px;
            border-radius: 10px;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            background: rgba(118, 185, 0, 0.2);
            margin-left: 15%;
            border: 1px solid rgba(118, 185, 0, 0.4);
        }
        .message.ai {
            background: rgba(0, 128, 255, 0.2);
            margin-right: 15%;
            border: 1px solid rgba(0, 128, 255, 0.4);
        }
        .controls {
            display: flex;
            gap: 15px;
            margin: 20px 0;
        }
        input {
            flex: 1;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            color: white;
            font-size: 16px;
        }
        button {
            padding: 15px 30px;
            background: linear-gradient(45deg, #76b900, #0080ff);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(118, 185, 0, 0.5);
        }
        .api-selector {
            display: flex;
            gap: 10px;
            margin: 10px 0;
        }
        .api-option {
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s;
        }
        .api-option.active {
            background: linear-gradient(45deg, #76b900, #0080ff);
            border-color: transparent;
        }
        .performance-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        .stat {
            text-align: center;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #76b900;
        }
        .stat-label {
            color: #888;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Digital Human Financial Advisor</h1>
            <div class="cloud-badges">
                <div class="cloud-badge nvidia">
                    <span style="color: #76b900;">●</span> NVIDIA NIM
                </div>
                <div class="cloud-badge together">
                    <span style="color: #0080ff;">●</span> Together.ai
                </div>
            </div>
        </div>
        
        <div class="main-grid">
            <div class="status-panel">
                <h3>GPU Status</h3>
                
                <div class="gpu-meter nvidia">
                    <strong>NVIDIA DGX Cloud</strong>
                    <div>Status: Online</div>
                    <div>Model: Llama 3.1 8B</div>
                    <div>Latency: <span id="nvidia-latency">-</span>ms</div>
                </div>
                
                <div class="gpu-meter together">
                    <strong>Together.ai GPU</strong>
                    <div>Status: Online</div>
                    <div>Model: Llama 3.1 8B Turbo</div>
                    <div>Latency: <span id="together-latency">-</span>ms</div>
                </div>
                
                <h3>API Selection</h3>
                <div class="api-selector">
                    <div class="api-option active" onclick="selectAPI('nvidia')">NVIDIA</div>
                    <div class="api-option" onclick="selectAPI('together')">Together</div>
                    <div class="api-option" onclick="selectAPI('auto')">Auto</div>
                </div>
                
                <h3>Performance</h3>
                <div class="performance-stats">
                    <div class="stat">
                        <div class="stat-value" id="queries">0</div>
                        <div class="stat-label">Queries</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="uptime">100%</div>
                        <div class="stat-label">Uptime</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="speed">0ms</div>
                        <div class="stat-label">Speed</div>
                    </div>
                </div>
            </div>
            
            <div class="chat-panel">
                <canvas id="avatar"></canvas>
                <div id="chat"></div>
                <div class="controls">
                    <input 
                        type="text" 
                        id="message" 
                        placeholder="Ask about stocks, portfolio optimization, market analysis..."
                        autocomplete="off"
                    >
                    <button onclick="sendMessage()">Send Message</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket("ws://" + window.location.host + "/ws");
        const chat = document.getElementById("chat");
        const canvas = document.getElementById("avatar");
        const ctx = canvas.getContext("2d");
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        let selectedAPI = 'nvidia';
        let queryCount = 0;
        let startTime = Date.now();
        
        // Multi-cloud avatar
        class CloudAvatar {
            constructor() {
                this.speaking = false;
                this.frame = 0;
                this.neurons = [];
                this.initNeurons();
            }
            
            initNeurons() {
                for (let i = 0; i < 200; i++) {
                    this.neurons.push({
                        x: Math.random() * canvas.width,
                        y: Math.random() * canvas.height,
                        vx: (Math.random() - 0.5) * 0.5,
                        vy: (Math.random() - 0.5) * 0.5,
                        size: Math.random() * 2 + 1,
                        color: Math.random() > 0.5 ? "#76b900" : "#0080ff",
                        connections: []
                    });
                }
                
                // Create connections
                for (let i = 0; i < this.neurons.length; i++) {
                    const connections = Math.floor(Math.random() * 3) + 1;
                    for (let j = 0; j < connections; j++) {
                        const target = Math.floor(Math.random() * this.neurons.length);
                        if (target !== i) {
                            this.neurons[i].connections.push(target);
                        }
                    }
                }
            }
            
            draw() {
                ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw connections
                ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
                ctx.lineWidth = 0.5;
                
                for (let i = 0; i < this.neurons.length; i++) {
                    const neuron = this.neurons[i];
                    for (let j = 0; j < neuron.connections.length; j++) {
                        const target = this.neurons[neuron.connections[j]];
                        const distance = Math.sqrt(
                            Math.pow(neuron.x - target.x, 2) + 
                            Math.pow(neuron.y - target.y, 2)
                        );
                        
                        if (distance < 150) {
                            ctx.globalAlpha = (150 - distance) / 150 * 0.3;
                            if (this.speaking) {
                                ctx.strokeStyle = neuron.color;
                            }
                            ctx.beginPath();
                            ctx.moveTo(neuron.x, neuron.y);
                            ctx.lineTo(target.x, target.y);
                            ctx.stroke();
                        }
                    }
                }
                
                // Draw neurons
                for (let neuron of this.neurons) {
                    neuron.x += neuron.vx;
                    neuron.y += neuron.vy;
                    
                    if (neuron.x < 0 || neuron.x > canvas.width) neuron.vx *= -1;
                    if (neuron.y < 0 || neuron.y > canvas.height) neuron.vy *= -1;
                    
                    ctx.globalAlpha = 0.8;
                    ctx.fillStyle = neuron.color;
                    ctx.beginPath();
                    const size = this.speaking ? neuron.size * (1 + Math.sin(this.frame * 0.1) * 0.5) : neuron.size;
                    ctx.arc(neuron.x, neuron.y, size, 0, 2 * Math.PI);
                    ctx.fill();
                }
                
                // Central brain
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                
                const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 100);
                gradient.addColorStop(0, selectedAPI === 'nvidia' ? "#76b900" : "#0080ff");
                gradient.addColorStop(0.5, "rgba(255, 255, 255, 0.1)");
                gradient.addColorStop(1, "transparent");
                
                ctx.globalAlpha = 0.6;
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(centerX, centerY, 100, 0, 2 * Math.PI);
                ctx.fill();
                
                // Cloud labels
                ctx.globalAlpha = 1;
                ctx.font = "bold 16px Arial";
                ctx.fillStyle = "#76b900";
                ctx.fillText("NVIDIA", 20, 30);
                ctx.fillStyle = "#0080ff";
                ctx.fillText("Together.ai", canvas.width - 100, 30);
                
                this.frame++;
            }
            
            speak() {
                this.speaking = true;
                setTimeout(() => this.speaking = false, 3000);
            }
        }
        
        const avatar = new CloudAvatar();
        
        function animate() {
            avatar.draw();
            requestAnimationFrame(animate);
        }
        animate();
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const messageDiv = document.createElement("div");
            messageDiv.className = "message ai";
            messageDiv.innerHTML = `<strong>AI (${data.api}):</strong><br>${data.message}`;
            chat.appendChild(messageDiv);
            avatar.speak();
            chat.scrollTop = chat.scrollHeight;
            
            // Update stats
            if (data.latency) {
                document.getElementById(`${data.api}-latency`).textContent = data.latency;
                document.getElementById('speed').textContent = data.latency + 'ms';
            }
            queryCount++;
            document.getElementById('queries').textContent = queryCount;
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
        
        function selectAPI(api) {
            selectedAPI = api;
            document.querySelectorAll('.api-option').forEach(el => {
                el.classList.remove('active');
            });
            event.target.classList.add('active');
        }
        
        document.getElementById("message").addEventListener("keypress", (e) => {
            if (e.key === "Enter") sendMessage();
        });
        
        // Update uptime
        setInterval(() => {
            const uptime = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
            document.getElementById('uptime').textContent = uptime + 'm';
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
        "deployment": "multi-cloud-gpu",
        "models": MODELS
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_json()
            message = data.get("message", "")
            selected_api = data.get("api", "nvidia")
            
            start_time = asyncio.get_event_loop().time()
            
            # Financial advisor prompt
            system_prompt = """You are a GPU-powered Digital Human Financial Advisor with access to:
            - Real-time market analysis using GPU acceleration
            - Portfolio optimization algorithms
            - Risk assessment and Monte Carlo simulations
            - Market trend predictions using deep learning
            
            Provide expert financial advice and emphasize computational advantages when relevant."""
            
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
                "message": f"Error: {str(e)}",
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
