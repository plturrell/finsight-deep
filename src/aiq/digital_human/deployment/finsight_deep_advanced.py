from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import httpx
import os
import json
import asyncio

app = FastAPI(title="FinSight Deep - Neural Supercomputer")

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable is required")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY environment variable is required")

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
            aspect-ratio: 3/4;
            max-width: 400px;
            margin: 0 auto;
        }
        #avatar-canvas {
            width: 100%;
            height: 100%;
            border-radius: 20px;
            background: #000;
            box-shadow: 0 0 50px rgba(0, 212, 255, 0.5);
        }
        .chat-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            border: 2px solid rgba(0, 212, 255, 0.3);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
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
            padding: 15px 30px;
            background: linear-gradient(45deg, #00d4ff, #0099ff);
            border: none;
            border-radius: 25px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
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
                </div>
            </div>
            
            <div class="chat-section">
                <div id="chat">
                    <div class="message finsight">
                        <div class="finsight-label">FinSight Deep:</div>
                        Neural Supercomputer initialized. Ready for advanced financial analysis.
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
        
        // Realistic Digital Human Avatar
        class RealisticAvatar {
            constructor() {
                this.speaking = false;
                this.blinking = false;
                this.frame = 0;
                this.blinkTimer = 0;
                this.mouthOpenness = 0;
                this.eyeOpenness = 1;
                this.lookX = 0;
                this.lookY = 0;
                this.targetLookX = 0;
                this.targetLookY = 0;
                
                // Face structure
                this.face = {
                    centerX: canvas.width / 2,
                    centerY: canvas.height / 2.3,
                    width: 160,
                    height: 200
                };
            }
            
            draw() {
                // Clear canvas with dark background
                ctx.fillStyle = "#0a0a0a";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw face
                this.drawFace();
                
                // Update animations
                this.updateAnimations();
                
                this.frame++;
            }
            
            drawFace() {
                // Face shadow
                ctx.save();
                ctx.shadowBlur = 20;
                ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
                
                // Face shape (more realistic oval)
                ctx.fillStyle = "#f4d1ae";
                ctx.beginPath();
                ctx.ellipse(this.face.centerX, this.face.centerY, this.face.width/2, this.face.height/2, 0, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
                
                // Forehead
                ctx.fillStyle = "#f8d7b4";
                ctx.beginPath();
                ctx.ellipse(this.face.centerX, this.face.centerY - this.face.height/3, this.face.width/2.2, this.face.height/4, 0, Math.PI, 0);
                ctx.fill();
                
                // Hair
                ctx.fillStyle = "#2a2a2a";
                ctx.beginPath();
                ctx.moveTo(this.face.centerX - this.face.width/2, this.face.centerY - this.face.height/2.5);
                ctx.quadraticCurveTo(this.face.centerX, this.face.centerY - this.face.height/1.8, 
                                    this.face.centerX + this.face.width/2, this.face.centerY - this.face.height/2.5);
                ctx.lineTo(this.face.centerX + this.face.width/2.5, this.face.centerY - this.face.height/3);
                ctx.quadraticCurveTo(this.face.centerX, this.face.centerY - this.face.height/2.2,
                                    this.face.centerX - this.face.width/2.5, this.face.centerY - this.face.height/3);
                ctx.closePath();
                ctx.fill();
                
                // Eyebrows
                ctx.strokeStyle = "#4a3a2a";
                ctx.lineWidth = 5;
                ctx.lineCap = "round";
                
                // Left eyebrow
                ctx.beginPath();
                ctx.moveTo(this.face.centerX - this.face.width/3.5, this.face.centerY - this.face.height/5);
                ctx.quadraticCurveTo(this.face.centerX - this.face.width/6, this.face.centerY - this.face.height/4.2,
                                    this.face.centerX - this.face.width/12, this.face.centerY - this.face.height/5);
                ctx.stroke();
                
                // Right eyebrow
                ctx.beginPath();
                ctx.moveTo(this.face.centerX + this.face.width/3.5, this.face.centerY - this.face.height/5);
                ctx.quadraticCurveTo(this.face.centerX + this.face.width/6, this.face.centerY - this.face.height/4.2,
                                    this.face.centerX + this.face.width/12, this.face.centerY - this.face.height/5);
                ctx.stroke();
                
                // Eyes
                const eyeY = this.face.centerY - this.face.height/8;
                const eyeSpacing = this.face.width/5;
                this.drawRealisticEye(this.face.centerX - eyeSpacing, eyeY, 20, true);
                this.drawRealisticEye(this.face.centerX + eyeSpacing, eyeY, 20, false);
                
                // Nose
                this.drawNose();
                
                // Mouth
                this.drawRealisticMouth();
                
                // Cheek highlights
                ctx.fillStyle = "rgba(255, 200, 180, 0.3)";
                ctx.beginPath();
                ctx.ellipse(this.face.centerX - this.face.width/3, this.face.centerY + this.face.height/8, 
                           this.face.width/8, this.face.height/10, -0.2, 0, Math.PI * 2);
                ctx.fill();
                ctx.beginPath();
                ctx.ellipse(this.face.centerX + this.face.width/3, this.face.centerY + this.face.height/8, 
                           this.face.width/8, this.face.height/10, 0.2, 0, Math.PI * 2);
                ctx.fill();
            }
            
            drawRealisticEye(x, y, size, isLeft) {
                // Eye socket shadow
                ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
                ctx.beginPath();
                ctx.ellipse(x, y + 2, size * 1.1, size * 0.8, 0, 0, Math.PI * 2);
                ctx.fill();
                
                // Eye white
                ctx.fillStyle = "#ffffff";
                ctx.beginPath();
                if (this.eyeOpenness > 0) {
                    ctx.ellipse(x, y, size, size * 0.7 * this.eyeOpenness, 0, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // Eye outline
                    ctx.strokeStyle = "#2a2a2a";
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
                
                // Iris and pupil
                if (this.eyeOpenness > 0.3) {
                    const irisX = x + this.lookX * 5;
                    const irisY = y + this.lookY * 3;
                    
                    // Iris
                    const gradient = ctx.createRadialGradient(irisX, irisY, 0, irisX, irisY, size * 0.35);
                    gradient.addColorStop(0, "#4a90e2");
                    gradient.addColorStop(0.5, "#357abd");
                    gradient.addColorStop(1, "#1e5a9e");
                    ctx.fillStyle = gradient;
                    ctx.beginPath();
                    ctx.arc(irisX, irisY, size * 0.35, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // Pupil
                    ctx.fillStyle = "#000000";
                    ctx.beginPath();
                    ctx.arc(irisX, irisY, size * 0.15, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // Eye reflection
                    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
                    ctx.beginPath();
                    ctx.arc(irisX - size * 0.1, irisY - size * 0.1, size * 0.08, 0, Math.PI * 2);
                    ctx.fill();
                }
                
                // Eyelids
                ctx.fillStyle = "#f4d1ae";
                ctx.beginPath();
                ctx.ellipse(x, y - size * 0.7, size * 1.1, size * 0.3, 0, 0, Math.PI);
                ctx.fill();
                ctx.beginPath();
                ctx.ellipse(x, y + size * 0.7, size * 1.1, size * 0.3, 0, Math.PI, 0);
                ctx.fill();
                
                // Eye lashes
                ctx.strokeStyle = "#2a2a2a";
                ctx.lineWidth = 1;
                ctx.lineCap = "round";
                for (let i = 0; i < 5; i++) {
                    const angle = (i - 2) * 0.1;
                    ctx.beginPath();
                    ctx.moveTo(x + Math.cos(angle) * size * 0.9, y - size * 0.6);
                    ctx.lineTo(x + Math.cos(angle) * size * 1.1, y - size * 0.8);
                    ctx.stroke();
                }
            }
            
            drawNose() {
                ctx.strokeStyle = "#e0b090";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(this.face.centerX, this.face.centerY - this.face.height/12);
                ctx.lineTo(this.face.centerX - 5, this.face.centerY + this.face.height/12);
                ctx.quadraticCurveTo(this.face.centerX, this.face.centerY + this.face.height/10,
                                    this.face.centerX + 5, this.face.centerY + this.face.height/12);
                ctx.stroke();
                
                // Nostrils
                ctx.fillStyle = "rgba(0, 0, 0, 0.2)";
                ctx.beginPath();
                ctx.ellipse(this.face.centerX - 8, this.face.centerY + this.face.height/10, 4, 3, -0.2, 0, Math.PI * 2);
                ctx.fill();
                ctx.beginPath();
                ctx.ellipse(this.face.centerX + 8, this.face.centerY + this.face.height/10, 4, 3, 0.2, 0, Math.PI * 2);
                ctx.fill();
            }
            
            drawRealisticMouth() {
                const mouthY = this.face.centerY + this.face.height/4;
                const mouthWidth = this.face.width/3;
                
                // Upper lip
                ctx.fillStyle = "#d09080";
                ctx.beginPath();
                ctx.moveTo(this.face.centerX - mouthWidth/2, mouthY);
                ctx.quadraticCurveTo(this.face.centerX - mouthWidth/3, mouthY - 5,
                                    this.face.centerX, mouthY + 2);
                ctx.quadraticCurveTo(this.face.centerX + mouthWidth/3, mouthY - 5,
                                    this.face.centerX + mouthWidth/2, mouthY);
                ctx.closePath();
                ctx.fill();
                
                // Lower lip
                ctx.fillStyle = "#e0a090";
                ctx.beginPath();
                ctx.moveTo(this.face.centerX - mouthWidth/2, mouthY);
                if (this.speaking && this.mouthOpenness > 0) {
                    ctx.quadraticCurveTo(this.face.centerX, mouthY + 8 + this.mouthOpenness * 15,
                                        this.face.centerX + mouthWidth/2, mouthY);
                    ctx.fill();
                    
                    // Mouth interior
                    ctx.fillStyle = "#3a2020";
                    ctx.beginPath();
                    ctx.ellipse(this.face.centerX, mouthY + 5, mouthWidth/2.5, this.mouthOpenness * 8, 0, 0, Math.PI);
                    ctx.fill();
                    
                    // Teeth
                    if (this.mouthOpenness > 0.3) {
                        ctx.fillStyle = "#ffffff";
                        ctx.fillRect(this.face.centerX - mouthWidth/3, mouthY - 1, mouthWidth*2/3, 4);
                    }
                } else {
                    ctx.quadraticCurveTo(this.face.centerX, mouthY + 8,
                                        this.face.centerX + mouthWidth/2, mouthY);
                    ctx.closePath();
                    ctx.fill();
                }
                
                // Lip outline
                ctx.strokeStyle = "#c08070";
                ctx.lineWidth = 1;
                ctx.stroke();
            }
            
            updateAnimations() {
                // Eye blinking
                this.blinkTimer++;
                if (this.blinkTimer > 150 + Math.random() * 150) {
                    this.blinking = true;
                    this.blinkTimer = 0;
                }
                
                if (this.blinking) {
                    this.eyeOpenness = Math.max(0, this.eyeOpenness - 0.15);
                    if (this.eyeOpenness <= 0) {
                        this.blinking = false;
                    }
                } else {
                    this.eyeOpenness = Math.min(1, this.eyeOpenness + 0.1);
                }
                
                // Speaking animation
                if (this.speaking) {
                    this.mouthOpenness = 0.3 + Math.sin(this.frame * 0.3) * 0.3 + Math.random() * 0.2;
                } else {
                    this.mouthOpenness = Math.max(0, this.mouthOpenness - 0.1);
                }
                
                // Eye movement
                this.lookX += (this.targetLookX - this.lookX) * 0.1;
                this.lookY += (this.targetLookY - this.lookY) * 0.1;
                
                // Random eye movement
                if (this.frame % 120 === 0) {
                    this.targetLookX = (Math.random() - 0.5) * 0.5;
                    this.targetLookY = (Math.random() - 0.5) * 0.3;
                }
            }
            
            speak() {
                this.speaking = true;
                setTimeout(() => {
                    this.speaking = false;
                }, 3000);
            }
        }
        
        const avatar = new RealisticAvatar();
        
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
                    api: 'auto'
                }));
                input.value = "";
                chat.scrollTop = chat.scrollHeight;
            }
        }
        
        document.getElementById("message").addEventListener("keypress", (e) => {
            if (e.key === "Enter") sendMessage();
        });
    </script>
</body>
</html>
    ''')

@app.get("/health")
async def health():
    return {"status": "healthy", "deployment": "finsight-deep-realistic"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            system_prompt = """You are FinSight Deep Neural Supercomputer, an advanced financial analysis system.
            Provide specific, data-driven insights and recommendations."""
            
            reply = ""
            
            try:
                reply = await call_nvidia_api(message, system_prompt)
            except:
                reply = await call_together_api(message, system_prompt)
            
            await websocket.send_json({
                "message": reply
            })
            
        except Exception as e:
            await websocket.send_json({
                "message": f"I apologize, I encountered an error: {str(e)}"
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
    print("Launching FinSight Deep Neural Supercomputer with Realistic Avatar...")
    print("Access at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)