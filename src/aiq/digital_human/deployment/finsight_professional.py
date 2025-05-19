from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import httpx
import os
import json
import asyncio

app = FastAPI(title="FinSight Deep - Neural Supercomputer")

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "1e961dd58c67427a09c40a09382f8f00e54f39aa8c34ac426fd5579c4effd1b4")

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
            aspect-ratio: 3/4;
            max-width: 400px;
            margin: 0 auto;
            overflow: hidden;
            border-radius: 10px;
            background: #000;
        }
        #avatar-video {
            width: 100%;
            height: 100%;
            object-fit: cover;
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
    </style>
</head>
<body>
    <div class="header">
        <h1 class="logo">FinSight Deep</h1>
        <p>Neural Supercomputer • Professional 3D Avatar</p>
    </div>
    
    <div class="main-container">
        <div class="interface-grid">
            <div class="avatar-section">
                <h3 style="text-align: center; color: #00d4ff;">Professional 3D Avatar</h3>
                <div class="avatar-container">
                    <video id="avatar-video" autoplay loop muted>
                        <source src="https://www.w3schools.com/html/mov_bbb.mp4" type="video/mp4">
                    </video>
                </div>
                <div style="text-align: center; margin-top: 10px; color: #00d4ff;">
                    <div id="avatar-status">Avatar Ready</div>
                    <div id="speaking-indicator" style="margin-top: 10px;">
                        <span id="speak-status" style="opacity: 0; transition: opacity 0.3s;">🎤 Speaking...</span>
                    </div>
                </div>
            </div>
            
            <div class="chat-section">
                <h3 style="text-align: center; color: #00d4ff;">Neural Chat Interface</h3>
                <div id="chat">
                    <div class="message finsight">
                        Neural Supercomputer initialized with professional 3D avatar. Ready for advanced financial analysis.
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
        const avatarVideo = document.getElementById("avatar-video");
        const speakStatus = document.getElementById("speak-status");
        
        // Simulate speaking animation
        function simulateSpeaking(duration) {
            speakStatus.style.opacity = '1';
            
            // Simulate mouth movement
            let mouthOpen = false;
            const mouthInterval = setInterval(() => {
                if (avatarVideo) {
                    avatarVideo.style.filter = mouthOpen ? 'brightness(1.1)' : 'brightness(1)';
                    mouthOpen = !mouthOpen;
                }
            }, 150);
            
            setTimeout(() => {
                speakStatus.style.opacity = '0';
                clearInterval(mouthInterval);
                avatarVideo.style.filter = 'brightness(1)';
            }, duration);
        }
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const messageDiv = document.createElement("div");
            messageDiv.className = "message finsight";
            messageDiv.innerHTML = data.message;
            chat.appendChild(messageDiv);
            
            // Simulate speaking animation
            const speakDuration = Math.min(data.message.length * 50, 5000); // 50ms per character, max 5 seconds
            simulateSpeaking(speakDuration);
            
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            system_prompt = """You are FinSight Deep Neural Supercomputer, an advanced financial analysis system 
            with a professional 3D avatar. Provide specific, data-driven insights and recommendations."""
            
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
    return {"status": "healthy", "deployment": "finsight-professional"}

if __name__ == "__main__":
    print("Launching FinSight Deep with Professional 3D Avatar...")
    print("Access at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)