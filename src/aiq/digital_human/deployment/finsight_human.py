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
        }
        .avatar-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            border: 2px solid rgba(0, 212, 255, 0.3);
        }
        .avatar-container {
            position: relative;
            width: 100%;
            max-width: 400px;
            margin: 0 auto;
            overflow: hidden;
            border-radius: 15px;
            background: #000;
        }
        .human-photo {
            width: 100%;
            height: auto;
            display: block;
        }
        .face-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        .chat-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            border: 2px solid rgba(0, 212, 255, 0.3);
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
        .message.user {
            background: rgba(0, 212, 255, 0.2);
            margin-left: 15%;
            text-align: right;
        }
        .message.finsight {
            background: rgba(0, 153, 255, 0.2);
            margin-right: 15%;
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
        }
        button {
            padding: 15px 30px;
            background: linear-gradient(45deg, #00d4ff, #0099ff);
            border: none;
            border-radius: 25px;
            color: white;
            cursor: pointer;
        }
        .speaking-indicator {
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(0, 212, 255, 0.8);
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            display: none;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1 class="logo">FinSight Deep</h1>
        <p>Neural Supercomputer • Professional Financial Advisor</p>
    </div>
    
    <div class="main-container">
        <div class="interface-grid">
            <div class="avatar-section">
                <h3 style="text-align: center; color: #00d4ff;">AI Financial Advisor</h3>
                <div class="avatar-container">
                    <img class="human-photo" src="https://thispersondoesnotexist.com/" alt="Professional Advisor" crossorigin="anonymous" />
                    <canvas class="face-overlay" id="face-overlay"></canvas>
                    <div class="speaking-indicator" id="speaking-indicator">Speaking...</div>
                </div>
                <p style="text-align: center; margin-top: 10px; color: #888; font-size: 12px;">
                    AI-Generated Professional Avatar
                </p>
            </div>
            
            <div class="chat-section">
                <h3 style="text-align: center; color: #00d4ff;">Financial Analysis Interface</h3>
                <div id="chat">
                    <div class="message finsight">
                        Welcome to FinSight Deep Neural Supercomputer. I'm your AI financial advisor, ready to provide data-driven insights and recommendations.
                    </div>
                </div>
                
                <div class="input-container">
                    <input 
                        type="text" 
                        id="message" 
                        placeholder="Ask about markets, investments, or financial analysis..."
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
        const canvas = document.getElementById("face-overlay");
        const ctx = canvas.getContext("2d");
        const speakingIndicator = document.getElementById("speaking-indicator");
        const avatarContainer = document.querySelector(".avatar-container");
        const humanPhoto = document.querySelector(".human-photo");
        
        // Set canvas size to match photo
        function resizeCanvas() {
            canvas.width = humanPhoto.width;
            canvas.height = humanPhoto.height;
        }
        
        humanPhoto.onload = resizeCanvas;
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
        
        // Facial animation system
        class FacialAnimator {
            constructor() {
                this.speaking = false;
                this.blinking = false;
                this.blinkTimer = 0;
                this.mouthOpenness = 0;
                this.eyeOpenness = 1;
                this.frame = 0;
                
                // Face landmark positions (estimated)
                this.landmarks = {
                    leftEye: { x: 0.35, y: 0.35 },
                    rightEye: { x: 0.65, y: 0.35 },
                    mouth: { x: 0.5, y: 0.65 },
                    mouthWidth: 0.15,
                    eyeWidth: 0.08
                };
            }
            
            draw() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Only draw overlays when speaking or blinking
                if (this.speaking || this.blinking || this.eyeOpenness < 1) {
                    // Eye blink overlay
                    if (this.eyeOpenness < 1) {
                        ctx.fillStyle = 'rgba(0, 0, 0, ' + (1 - this.eyeOpenness) + ')';
                        
                        // Left eye
                        ctx.beginPath();
                        ctx.ellipse(
                            canvas.width * this.landmarks.leftEye.x,
                            canvas.height * this.landmarks.leftEye.y,
                            canvas.width * this.landmarks.eyeWidth,
                            canvas.height * 0.03 * (1 - this.eyeOpenness),
                            0, 0, Math.PI * 2
                        );
                        ctx.fill();
                        
                        // Right eye
                        ctx.beginPath();
                        ctx.ellipse(
                            canvas.width * this.landmarks.rightEye.x,
                            canvas.height * this.landmarks.rightEye.y,
                            canvas.width * this.landmarks.eyeWidth,
                            canvas.height * 0.03 * (1 - this.eyeOpenness),
                            0, 0, Math.PI * 2
                        );
                        ctx.fill();
                    }
                    
                    // Mouth movement overlay (subtle)
                    if (this.speaking && this.mouthOpenness > 0) {
                        // Dark overlay for mouth interior
                        ctx.fillStyle = 'rgba(50, 30, 30, ' + (this.mouthOpenness * 0.5) + ')';
                        ctx.beginPath();
                        ctx.ellipse(
                            canvas.width * this.landmarks.mouth.x,
                            canvas.height * this.landmarks.mouth.y,
                            canvas.width * this.landmarks.mouthWidth * 0.7,
                            canvas.height * 0.02 * this.mouthOpenness,
                            0, 0, Math.PI
                        );
                        ctx.fill();
                    }
                }
                
                this.frame++;
            }
            
            update() {
                // Natural blinking
                this.blinkTimer++;
                if (this.blinkTimer > 180 + Math.random() * 120) {
                    this.blinking = true;
                    this.blinkTimer = 0;
                }
                
                if (this.blinking) {
                    this.eyeOpenness = Math.max(0, this.eyeOpenness - 0.3);
                    if (this.eyeOpenness <= 0) {
                        this.blinking = false;
                    }
                } else {
                    this.eyeOpenness = Math.min(1, this.eyeOpenness + 0.2);
                }
                
                // Speaking animation
                if (this.speaking) {
                    // Natural speech pattern
                    const t = this.frame * 0.15;
                    this.mouthOpenness = Math.abs(Math.sin(t)) * 0.4 + 
                                       Math.abs(Math.sin(t * 2.3)) * 0.3 +
                                       Math.random() * 0.1;
                    speakingIndicator.style.display = 'block';
                } else {
                    this.mouthOpenness = Math.max(0, this.mouthOpenness - 0.1);
                    if (this.mouthOpenness <= 0) {
                        speakingIndicator.style.display = 'none';
                    }
                }
            }
            
            startSpeaking() {
                this.speaking = true;
            }
            
            stopSpeaking() {
                this.speaking = false;
            }
        }
        
        const animator = new FacialAnimator();
        
        // Animation loop
        function animate() {
            animator.update();
            animator.draw();
            requestAnimationFrame(animate);
        }
        animate();
        
        // WebSocket message handler
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const messageDiv = document.createElement("div");
            messageDiv.className = "message finsight";
            messageDiv.innerHTML = data.message;
            chat.appendChild(messageDiv);
            
            // Animate speaking
            animator.startSpeaking();
            const speakDuration = Math.min(data.message.length * 60, 8000); // 60ms per character, max 8 seconds
            setTimeout(() => {
                animator.stopSpeaking();
            }, speakDuration);
            
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
        
        // Reload image periodically for variety (optional)
        // setInterval(() => {
        //     humanPhoto.src = "https://thispersondoesnotexist.com/?" + Date.now();
        // }, 30000);
    </script>
</body>
</html>
    ''')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            system_prompt = """You are FinSight Deep Neural Supercomputer, an advanced financial analysis system 
            with a photorealistic human avatar. You are a professional financial advisor providing specific, 
            data-driven insights and recommendations. Be concise but thorough in your analysis."""
            
            # Generate response
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
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = await client.post(
            f"https://integrate.api.nvidia.com/v1/chat/completions",
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
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = await client.post(
            f"https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Together API error: {response.status_code}")

@app.get("/health")
async def health():
    return {"status": "healthy", "deployment": "finsight-real-human"}

if __name__ == "__main__":
    print("Launching FinSight Deep with Photorealistic Human Avatar...")
    print("Using AI-generated professional human face")
    print("Access at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)