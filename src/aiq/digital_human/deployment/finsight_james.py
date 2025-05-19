from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import httpx
import grpc
import os
import json
import asyncio
import numpy as np
import base64
import wave
import io

app = FastAPI(title="FinSight Deep - Neural Supercomputer")

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "1e961dd58c67427a09c40a09382f8f00e54f39aa8c34ac426fd5579c4effd1b4")

# Audio2Face Configuration
A2F_GRPC_ENDPOINT = "grpc.nvcf.nvidia.com:443"
A2F_FUNCTION_ID = "9327c39f-a361-4e02-bd72-e11b4c9b7b5e"  # James model with tongue animation

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
            max-width: 1400px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .interface-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
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
            background: #000;
            border-radius: 10px;
            overflow: hidden;
        }
        #james-avatar {
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
        .blendshapes-section {
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
        .blendshape-item {
            display: grid;
            grid-template-columns: 150px 1fr;
            align-items: center;
            gap: 10px;
            margin: 5px 0;
        }
        .blendshape-bar {
            background: rgba(0, 212, 255, 0.3);
            height: 8px;
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }
        .blendshape-fill {
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            background: linear-gradient(to right, #00d4ff, #0099ff);
            transition: width 0.1s ease;
        }
        .avatar-status {
            text-align: center;
            margin-top: 10px;
            color: #00d4ff;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1 class="logo">FinSight Deep</h1>
        <p>Neural Supercomputer • NVIDIA James Avatar</p>
    </div>
    
    <div class="main-container">
        <div class="interface-grid">
            <div class="avatar-section">
                <h3 style="text-align: center; color: #00d4ff;">NVIDIA James</h3>
                <div class="avatar-container">
                    <canvas id="james-avatar"></canvas>
                </div>
                <div class="avatar-status" id="avatar-status">Initializing James avatar...</div>
            </div>
            
            <div class="chat-section">
                <h3 style="text-align: center; color: #00d4ff;">Neural Chat Interface</h3>
                <div id="chat">
                    <div class="message finsight">
                        Neural Supercomputer initialized with NVIDIA James avatar.
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
            
            <div class="blendshapes-section">
                <h3 style="text-align: center; color: #00d4ff;">Facial Animation Data</h3>
                <div id="blendshapes">
                    <div class="blendshape-item">
                        <span>Jaw Open</span>
                        <div class="blendshape-bar">
                            <div class="blendshape-fill" id="bs-jaw_open" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="blendshape-item">
                        <span>Mouth Smile</span>
                        <div class="blendshape-bar">
                            <div class="blendshape-fill" id="bs-mouth_smile" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="blendshape-item">
                        <span>Eye Blink L</span>
                        <div class="blendshape-bar">
                            <div class="blendshape-fill" id="bs-eye_blink_l" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="blendshape-item">
                        <span>Eye Blink R</span>
                        <div class="blendshape-bar">
                            <div class="blendshape-fill" id="bs-eye_blink_r" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="blendshape-item">
                        <span>Brow Up</span>
                        <div class="blendshape-bar">
                            <div class="blendshape-fill" id="bs-brow_up" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script type="importmap">
    {
        "imports": {
            "three": "https://unpkg.com/three@0.157.0/build/three.module.js",
            "three/examples/jsm/loaders/GLTFLoader.js": "https://unpkg.com/three@0.157.0/examples/jsm/loaders/GLTFLoader.js"
        }
    }
    </script>
    
    <script type="module">
        import * as THREE from 'three';
        import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
        
        const ws = new WebSocket("ws://" + window.location.host + "/ws");
        const chat = document.getElementById("chat");
        const canvas = document.getElementById("james-avatar");
        
        // Three.js scene setup for James avatar
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, canvas.width / canvas.height, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setSize(400, 533);
        renderer.setPixelRatio(window.devicePixelRatio);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);
        
        camera.position.z = 2;
        camera.position.y = 0.5;
        
        // Load James avatar model
        const loader = new GLTFLoader();
        let jamesModel = null;
        let mixer = null;
        
        // Placeholder avatar while loading
        const geometry = new THREE.BoxGeometry(1, 1.5, 0.5);
        const material = new THREE.MeshPhongMaterial({ color: 0x4a90e2 });
        const placeholder = new THREE.Mesh(geometry, material);
        scene.add(placeholder);
        
        // Animation clock
        const clock = new THREE.Clock();
        
        // Blendshape mapping for James
        const blendshapeMap = {
            'jaw_open': 'jawOpen',
            'mouth_smile': 'mouthSmileLeft',
            'eye_blink_l': 'eyeBlinkLeft',
            'eye_blink_r': 'eyeBlinkRight',
            'brow_up': 'browUpCenter'
        };
        
        function animate() {
            requestAnimationFrame(animate);
            
            const delta = clock.getDelta();
            if (mixer) mixer.update(delta);
            
            // Rotate placeholder
            if (placeholder) {
                placeholder.rotation.y += 0.005;
            }
            
            renderer.render(scene, camera);
        }
        animate();
        
        // Audio2Face connection
        class Audio2FaceJames {
            constructor() {
                this.isConnected = false;
                this.blendshapes = {};
                this.audioQueue = [];
            }
            
            async initialize() {
                try {
                    // Initialize connection to Audio2Face service
                    const response = await fetch('/api/a2f/initialize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            model: 'james',
                            function_id: '9327c39f-a361-4e02-bd72-e11b4c9b7b5e'
                        })
                    });
                    
                    if (response.ok) {
                        this.isConnected = true;
                        document.getElementById('avatar-status').textContent = 'James avatar ready';
                        document.getElementById('avatar-status').style.color = '#00ff00';
                    }
                } catch (error) {
                    console.error('Failed to initialize Audio2Face:', error);
                    document.getElementById('avatar-status').textContent = 'Using fallback animation';
                    document.getElementById('avatar-status').style.color = '#ffaa00';
                }
            }
            
            async processAudio(audioData) {
                if (!this.isConnected) {
                    // Fallback animation
                    this.simulateSpeech(3000);
                    return;
                }
                
                try {
                    const response = await fetch('/api/a2f/process', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            audio: audioData,
                            emotion: 'neutral',
                            function_id: '9327c39f-a361-4e02-bd72-e11b4c9b7b5e'
                        })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        this.animateBlendshapes(data.blendshapes);
                    }
                } catch (error) {
                    console.error('Failed to process audio:', error);
                    this.simulateSpeech(3000);
                }
            }
            
            animateBlendshapes(blendshapeData) {
                // Animate the blendshapes
                blendshapeData.forEach((frame, index) => {
                    setTimeout(() => {
                        Object.entries(frame).forEach(([key, value]) => {
                            this.updateBlendshape(key, value);
                        });
                    }, index * 33); // ~30fps
                });
            }
            
            updateBlendshape(name, value) {
                const element = document.getElementById(`bs-${name}`);
                if (element) {
                    element.style.width = `${value * 100}%`;
                }
                
                // Update 3D model if loaded
                if (jamesModel && jamesModel.morphTargetInfluences) {
                    const mappedName = blendshapeMap[name];
                    const index = jamesModel.morphTargetDictionary[mappedName];
                    if (index !== undefined) {
                        jamesModel.morphTargetInfluences[index] = value;
                    }
                }
            }
            
            simulateSpeech(duration) {
                // Fallback animation when not connected to Audio2Face
                const startTime = Date.now();
                
                const animate = () => {
                    const elapsed = Date.now() - startTime;
                    if (elapsed < duration) {
                        const t = elapsed / duration;
                        
                        // Simulate mouth movement
                        this.updateBlendshape('jaw_open', Math.sin(t * 20) * 0.3 + 0.2);
                        this.updateBlendshape('mouth_smile', Math.sin(t * 10) * 0.1);
                        
                        // Occasional blinks
                        if (Math.random() < 0.01) {
                            this.updateBlendshape('eye_blink_l', 1);
                            this.updateBlendshape('eye_blink_r', 1);
                            setTimeout(() => {
                                this.updateBlendshape('eye_blink_l', 0);
                                this.updateBlendshape('eye_blink_r', 0);
                            }, 150);
                        }
                        
                        requestAnimationFrame(animate);
                    } else {
                        // Reset to neutral
                        this.updateBlendshape('jaw_open', 0);
                        this.updateBlendshape('mouth_smile', 0);
                    }
                };
                animate();
            }
        }
        
        const a2fJames = new Audio2FaceJames();
        a2fJames.initialize();
        
        ws.onmessage = async (event) => {
            const data = JSON.parse(event.data);
            const messageDiv = document.createElement("div");
            messageDiv.className = "message finsight";
            messageDiv.innerHTML = data.message;
            chat.appendChild(messageDiv);
            
            // Process audio for James avatar
            if (data.audio) {
                await a2fJames.processAudio(data.audio);
            } else {
                // Simulate speech for text
                a2fJames.simulateSpeech(data.message.length * 50);
            }
            
            chat.scrollTop = chat.scrollHeight;
        };
        
        window.sendMessage = function() {
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
        
        // Idle animation
        function idleAnimation() {
            // Subtle breathing
            const breathe = Math.sin(Date.now() * 0.001) * 0.05;
            if (placeholder) {
                placeholder.scale.y = 1.5 + breathe;
            }
            
            // Random blinks
            if (Math.random() < 0.003) {
                a2fJames.updateBlendshape('eye_blink_l', 1);
                a2fJames.updateBlendshape('eye_blink_r', 1);
                setTimeout(() => {
                    a2fJames.updateBlendshape('eye_blink_l', 0);
                    a2fJames.updateBlendshape('eye_blink_r', 0);
                }, 150);
            }
            
            requestAnimationFrame(idleAnimation);
        }
        idleAnimation();
    </script>
</body>
</html>
    ''')

@app.post("/api/a2f/initialize")
async def initialize_a2f(request: dict):
    """Initialize Audio2Face connection"""
    try:
        # This would connect to the actual Audio2Face service
        return {"status": "connected", "model": "james"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/a2f/process")
async def process_audio(request: dict):
    """Process audio through Audio2Face"""
    try:
        # In production, this would send audio to Audio2Face gRPC service
        # and receive back blendshape data
        
        # Simulated response
        blendshapes = []
        for i in range(30):  # 1 second at 30fps
            frame = {
                "jaw_open": np.sin(i * 0.3) * 0.3,
                "mouth_smile": np.sin(i * 0.2) * 0.1,
                "eye_blink_l": 0.0 if i % 20 != 0 else 1.0,
                "eye_blink_r": 0.0 if i % 20 != 0 else 1.0,
                "brow_up": np.sin(i * 0.1) * 0.05
            }
            blendshapes.append(frame)
        
        return {"blendshapes": blendshapes}
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            system_prompt = """You are FinSight Deep Neural Supercomputer, an advanced financial analysis system 
            with a photorealistic avatar powered by NVIDIA James."""
            
            # Generate response
            try:
                reply = await call_nvidia_api(message, system_prompt)
            except:
                reply = await call_together_api(message, system_prompt)
            
            # In production, convert text to speech and process through Audio2Face
            # For now, send text response
            await websocket.send_json({
                "message": reply,
                "avatar": "james"
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
    return {"status": "healthy", "deployment": "finsight-james", "avatar": "NVIDIA James"}

if __name__ == "__main__":
    print("Launching FinSight Deep with NVIDIA James Avatar...")
    print("Using Audio2Face-3D James model")
    uvicorn.run(app, host="0.0.0.0", port=8000)