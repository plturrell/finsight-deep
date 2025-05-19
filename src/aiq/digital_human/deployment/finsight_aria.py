from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import httpx
import os
import json
import asyncio
import base64
from typing import Optional

app = FastAPI(title="FinSight Deep - Neural Supercomputer")

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "1e961dd58c67427a09c40a09382f8f00e54f39aa8c34ac426fd5579c4effd1b4")

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
            overflow: hidden;
            border-radius: 20px;
        }
        #avatar-canvas {
            width: 100%;
            height: 100%;
            background: #000;
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
        }
        button {
            padding: 15px 30px;
            background: linear-gradient(45deg, #00d4ff, #0099ff);
            border: none;
            border-radius: 25px;
            color: white;
            font-weight: bold;
            cursor: pointer;
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
        canvas.width = 800;
        canvas.height = 1067;
        
        // Photorealistic Avatar (Aria-style)
        class PhotorealisticAvatar {
            constructor() {
                this.frame = 0;
                this.speaking = false;
                this.blinking = false;
                this.blinkTimer = 0;
                this.mouthOpenness = 0;
                this.eyeOpenness = 1;
                this.lookDirection = {x: 0, y: 0};
                this.emotion = "neutral";
                
                // Face structure
                this.face = {
                    centerX: canvas.width / 2,
                    centerY: canvas.height / 2.2,
                    width: 240,
                    height: 320
                };
                
                // Load base textures
                this.textures = {};
                this.loadTextures();
                
                // Microexpressions
                this.microExpressions = {
                    timer: 0,
                    current: null,
                    intensity: 0
                };
                
                // Hair particles for realistic movement
                this.hairParticles = [];
                this.initHairParticles();
            }
            
            loadTextures() {
                // Base skin texture gradient
                this.skinGradient = ctx.createRadialGradient(
                    this.face.centerX, this.face.centerY - 50, 0,
                    this.face.centerX, this.face.centerY, this.face.height/2
                );
                this.skinGradient.addColorStop(0, '#fdbcb4');
                this.skinGradient.addColorStop(0.3, '#f4a49e');
                this.skinGradient.addColorStop(0.7, '#f1b5a8');
                this.skinGradient.addColorStop(1, '#edb4a1');
                
                // Eye textures
                this.eyeGradients = {
                    iris: ctx.createRadialGradient(0, 0, 0, 0, 0, 20),
                    sclera: ctx.createRadialGradient(0, 0, 0, 0, 0, 35)
                };
                
                // Brown eyes
                this.eyeGradients.iris.addColorStop(0, '#342619');
                this.eyeGradients.iris.addColorStop(0.3, '#6b4423');
                this.eyeGradients.iris.addColorStop(0.7, '#8b5a3c');
                this.eyeGradients.iris.addColorStop(1, '#523823');
                
                this.eyeGradients.sclera.addColorStop(0, '#ffffff');
                this.eyeGradients.sclera.addColorStop(0.8, '#f8f8f8');
                this.eyeGradients.sclera.addColorStop(1, '#f0f0f0');
            }
            
            initHairParticles() {
                // Create hair strands for realistic movement
                for (let i = 0; i < 50; i++) {
                    this.hairParticles.push({
                        x: this.face.centerX + (Math.random() - 0.5) * this.face.width,
                        y: this.face.centerY - this.face.height/2 + Math.random() * 60,
                        length: 80 + Math.random() * 120,
                        thickness: 1 + Math.random() * 2,
                        angle: Math.PI/2 + (Math.random() - 0.5) * 0.5,
                        velocity: 0,
                        color: `hsl(20, 20%, ${15 + Math.random() * 15}%)`
                    });
                }
            }
            
            draw() {
                // Clear canvas
                ctx.fillStyle = "#0a0a0a";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Background gradient
                const bgGradient = ctx.createRadialGradient(
                    canvas.width/2, canvas.height/2, 0,
                    canvas.width/2, canvas.height/2, canvas.width/2
                );
                bgGradient.addColorStop(0, 'rgba(30, 30, 40, 0.5)');
                bgGradient.addColorStop(1, 'rgba(10, 10, 20, 1)');
                ctx.fillStyle = bgGradient;
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                ctx.save();
                
                // Draw layers in order
                this.drawNeck();
                this.drawHair(true); // Back hair
                this.drawFace();
                this.drawEars();
                this.drawFacialFeatures();
                this.drawEyes();
                this.drawNose();
                this.drawMouth();
                this.drawHair(false); // Front hair
                this.drawAccessories();
                
                // Update animations
                this.updateAnimations();
                
                ctx.restore();
                this.frame++;
            }
            
            drawNeck() {
                // Realistic neck
                ctx.fillStyle = '#f4a49e';
                ctx.beginPath();
                ctx.moveTo(this.face.centerX - 60, this.face.centerY + this.face.height/2 - 20);
                ctx.quadraticCurveTo(
                    this.face.centerX, this.face.centerY + this.face.height/2 + 20,
                    this.face.centerX + 60, this.face.centerY + this.face.height/2 - 20
                );
                ctx.lineTo(this.face.centerX + 80, canvas.height);
                ctx.lineTo(this.face.centerX - 80, canvas.height);
                ctx.closePath();
                ctx.fill();
                
                // Neck shadow
                ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
                ctx.beginPath();
                ctx.ellipse(
                    this.face.centerX, this.face.centerY + this.face.height/2,
                    60, 20, 0, 0, Math.PI
                );
                ctx.fill();
            }
            
            drawFace() {
                // Realistic face shape
                ctx.fillStyle = this.skinGradient;
                ctx.beginPath();
                
                // Start from forehead
                ctx.moveTo(this.face.centerX, this.face.centerY - this.face.height/2);
                
                // Right temple
                ctx.quadraticCurveTo(
                    this.face.centerX + this.face.width/3, this.face.centerY - this.face.height/2.2,
                    this.face.centerX + this.face.width/2, this.face.centerY - this.face.height/3
                );
                
                // Right cheek
                ctx.quadraticCurveTo(
                    this.face.centerX + this.face.width/1.8, this.face.centerY,
                    this.face.centerX + this.face.width/2.5, this.face.centerY + this.face.height/4
                );
                
                // Right jaw
                ctx.quadraticCurveTo(
                    this.face.centerX + this.face.width/3, this.face.centerY + this.face.height/2.5,
                    this.face.centerX, this.face.centerY + this.face.height/2
                );
                
                // Left jaw
                ctx.quadraticCurveTo(
                    this.face.centerX - this.face.width/3, this.face.centerY + this.face.height/2.5,
                    this.face.centerX - this.face.width/2.5, this.face.centerY + this.face.height/4
                );
                
                // Left cheek
                ctx.quadraticCurveTo(
                    this.face.centerX - this.face.width/1.8, this.face.centerY,
                    this.face.centerX - this.face.width/2, this.face.centerY - this.face.height/3
                );
                
                // Left temple
                ctx.quadraticCurveTo(
                    this.face.centerX - this.face.width/3, this.face.centerY - this.face.height/2.2,
                    this.face.centerX, this.face.centerY - this.face.height/2
                );
                
                ctx.closePath();
                ctx.fill();
                
                // Face contour shadow
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Cheek highlights
                this.drawCheekHighlights();
            }
            
            drawCheekHighlights() {
                // Subtle cheek highlights for dimension
                ctx.fillStyle = 'rgba(255, 200, 180, 0.15)';
                
                // Right cheek
                ctx.beginPath();
                ctx.ellipse(
                    this.face.centerX + this.face.width/3,
                    this.face.centerY + this.face.height/8,
                    30, 20, 0.3, 0, Math.PI * 2
                );
                ctx.fill();
                
                // Left cheek
                ctx.beginPath();
                ctx.ellipse(
                    this.face.centerX - this.face.width/3,
                    this.face.centerY + this.face.height/8,
                    30, 20, -0.3, 0, Math.PI * 2
                );
                ctx.fill();
            }
            
            drawEars() {
                // Realistic ears
                const earY = this.face.centerY;
                const earSize = 35;
                
                // Left ear
                ctx.fillStyle = '#f4a49e';
                ctx.beginPath();
                ctx.ellipse(
                    this.face.centerX - this.face.width/2 - 5,
                    earY,
                    earSize * 0.7, earSize, -0.2, 0, Math.PI * 2
                );
                ctx.fill();
                
                // Inner ear detail
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(
                    this.face.centerX - this.face.width/2,
                    earY,
                    earSize * 0.6, -Math.PI/3, Math.PI/3
                );
                ctx.stroke();
                
                // Right ear (mirrored)
                ctx.fillStyle = '#f4a49e';
                ctx.beginPath();
                ctx.ellipse(
                    this.face.centerX + this.face.width/2 + 5,
                    earY,
                    earSize * 0.7, earSize, 0.2, 0, Math.PI * 2
                );
                ctx.fill();
            }
            
            drawFacialFeatures() {
                // Eyebrows
                ctx.strokeStyle = '#3d2314';
                ctx.lineWidth = 4;
                ctx.lineCap = 'round';
                
                // Left eyebrow
                ctx.beginPath();
                const leftBrowY = this.face.centerY - this.face.height/5;
                ctx.moveTo(this.face.centerX - this.face.width/4 - 25, leftBrowY + 5);
                ctx.quadraticCurveTo(
                    this.face.centerX - this.face.width/6, leftBrowY - 5,
                    this.face.centerX - this.face.width/8, leftBrowY + 2
                );
                ctx.stroke();
                
                // Right eyebrow
                ctx.beginPath();
                ctx.moveTo(this.face.centerX + this.face.width/4 + 25, leftBrowY + 5);
                ctx.quadraticCurveTo(
                    this.face.centerX + this.face.width/6, leftBrowY - 5,
                    this.face.centerX + this.face.width/8, leftBrowY + 2
                );
                ctx.stroke();
                
                // Forehead details (subtle wrinkles when expressing)
                if (this.emotion === "concerned" || this.emotion === "thinking") {
                    ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
                    ctx.lineWidth = 1;
                    for (let i = 0; i < 3; i++) {
                        ctx.beginPath();
                        const y = this.face.centerY - this.face.height/3 + i * 10;
                        ctx.moveTo(this.face.centerX - 40, y);
                        ctx.quadraticCurveTo(
                            this.face.centerX, y - 3,
                            this.face.centerX + 40, y
                        );
                        ctx.stroke();
                    }
                }
            }
            
            drawEyes() {
                const eyeY = this.face.centerY - this.face.height/8;
                const eyeSpacing = this.face.width/5;
                
                // Draw both eyes
                this.drawEye(this.face.centerX - eyeSpacing, eyeY, 35, 'left');
                this.drawEye(this.face.centerX + eyeSpacing, eyeY, 35, 'right');
                
                // Eye bags/definition for realism
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(
                    this.face.centerX - eyeSpacing, eyeY + 35,
                    40, 0.1, Math.PI - 0.1
                );
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(
                    this.face.centerX + eyeSpacing, eyeY + 35,
                    40, 0.1, Math.PI - 0.1
                );
                ctx.stroke();
            }
            
            drawEye(x, y, size, side) {
                const eyeOffset = this.lookDirection.x * 8 * (side === 'left' ? 1 : -1);
                const openness = this.eyeOpenness;
                
                if (openness > 0) {
                    // Eye socket shadow
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.08)';
                    ctx.beginPath();
                    ctx.ellipse(x, y + 3, size + 5, size * 0.8, 0, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // Sclera (eye white)
                    ctx.save();
                    ctx.translate(x, y);
                    ctx.fillStyle = this.eyeGradients.sclera;
                    ctx.beginPath();
                    ctx.ellipse(0, 0, size, size * 0.7 * openness, 0, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.restore();
                    
                    // Iris
                    const irisX = x + eyeOffset;
                    const irisY = y + this.lookDirection.y * 5;
                    
                    ctx.save();
                    ctx.translate(irisX, irisY);
                    ctx.fillStyle = this.eyeGradients.iris;
                    ctx.beginPath();
                    ctx.arc(0, 0, size * 0.4, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // Pupil
                    ctx.fillStyle = '#000000';
                    ctx.beginPath();
                    ctx.arc(0, 0, size * 0.15, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // Eye reflections
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                    ctx.beginPath();
                    ctx.arc(-size * 0.15, -size * 0.15, size * 0.08, 0, Math.PI * 2);
                    ctx.fill();
                    
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
                    ctx.beginPath();
                    ctx.arc(size * 0.1, size * 0.1, size * 0.05, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.restore();
                    
                    // Iris detail (radial lines)
                    ctx.save();
                    ctx.translate(irisX, irisY);
                    ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
                    ctx.lineWidth = 0.5;
                    for (let i = 0; i < 12; i++) {
                        const angle = (i / 12) * Math.PI * 2;
                        ctx.beginPath();
                        ctx.moveTo(Math.cos(angle) * size * 0.15, Math.sin(angle) * size * 0.15);
                        ctx.lineTo(Math.cos(angle) * size * 0.35, Math.sin(angle) * size * 0.35);
                        ctx.stroke();
                    }
                    ctx.restore();
                }
                
                // Eyelids
                this.drawEyelids(x, y, size, openness);
                
                // Eyelashes
                this.drawEyelashes(x, y, size, side);
            }
            
            drawEyelids(x, y, size, openness) {
                // Upper eyelid
                ctx.fillStyle = '#f4a49e';
                ctx.beginPath();
                ctx.ellipse(x, y - size * 0.7, size + 5, size * 0.4, 0, 0, Math.PI);
                ctx.fill();
                
                ctx.beginPath();
                ctx.ellipse(x, y - size * 0.5, size, size * (1 - openness) * 0.7, 0, 0, Math.PI);
                ctx.fill();
                
                // Lower eyelid
                ctx.beginPath();
                ctx.ellipse(x, y + size * 0.7, size + 5, size * 0.3, 0, Math.PI, 0);
                ctx.fill();
                
                // Eyelid crease
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(x, y - size * 0.4, size * 0.9, 0.2, Math.PI - 0.2);
                ctx.stroke();
            }
            
            drawEyelashes(x, y, size, side) {
                ctx.strokeStyle = '#2a1810';
                ctx.lineWidth = 1.5;
                ctx.lineCap = 'round';
                
                // Upper lashes
                for (let i = 0; i < 8; i++) {
                    const angle = (i / 8) * Math.PI * 0.8 + Math.PI * 0.1;
                    const startX = x + Math.cos(angle) * size * 0.9;
                    const startY = y - size * 0.5 + Math.sin(angle) * size * 0.3;
                    const length = 8 + Math.random() * 4;
                    
                    ctx.beginPath();
                    ctx.moveTo(startX, startY);
                    ctx.lineTo(
                        startX + Math.cos(angle - 0.3) * length,
                        startY - Math.sin(angle - 0.3) * length
                    );
                    ctx.stroke();
                }
                
                // Lower lashes (fewer and smaller)
                ctx.lineWidth = 1;
                for (let i = 0; i < 5; i++) {
                    const angle = (i / 5) * Math.PI * 0.6 + Math.PI * 1.2;
                    const startX = x + Math.cos(angle) * size * 0.8;
                    const startY = y + size * 0.5 + Math.sin(angle) * size * 0.1;
                    const length = 4 + Math.random() * 2;
                    
                    ctx.beginPath();
                    ctx.moveTo(startX, startY);
                    ctx.lineTo(
                        startX + Math.cos(angle + 0.3) * length,
                        startY + Math.sin(angle + 0.3) * length
                    );
                    ctx.stroke();
                }
            }
            
            drawNose() {
                const noseX = this.face.centerX;
                const noseY = this.face.centerY + this.face.height/20;
                
                // Nose bridge
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(noseX - 3, noseY - 40);
                ctx.lineTo(noseX - 5, noseY + 10);
                ctx.stroke();
                
                ctx.beginPath();
                ctx.moveTo(noseX + 3, noseY - 40);
                ctx.lineTo(noseX + 5, noseY + 10);
                ctx.stroke();
                
                // Nose tip
                ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
                ctx.beginPath();
                ctx.ellipse(noseX, noseY + 20, 12, 8, 0, 0, Math.PI);
                ctx.fill();
                
                // Nostrils
                ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                ctx.beginPath();
                ctx.ellipse(noseX - 8, noseY + 20, 5, 3, -0.3, 0, Math.PI * 2);
                ctx.fill();
                
                ctx.beginPath();
                ctx.ellipse(noseX + 8, noseY + 20, 5, 3, 0.3, 0, Math.PI * 2);
                ctx.fill();
                
                // Nose highlight
                ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
                ctx.beginPath();
                ctx.ellipse(noseX, noseY + 5, 6, 4, 0, 0, Math.PI * 2);
                ctx.fill();
            }
            
            drawMouth() {
                const mouthX = this.face.centerX;
                const mouthY = this.face.centerY + this.face.height/4;
                const mouthWidth = this.face.width/3;
                
                // Lip colors
                const upperLipColor = '#d4928b';
                const lowerLipColor = '#e4a19a';
                
                // Upper lip
                ctx.fillStyle = upperLipColor;
                ctx.beginPath();
                ctx.moveTo(mouthX - mouthWidth/2, mouthY);
                
                // Cupid's bow
                ctx.quadraticCurveTo(
                    mouthX - mouthWidth/4, mouthY - 8,
                    mouthX - mouthWidth/6, mouthY - 5
                );
                ctx.quadraticCurveTo(
                    mouthX, mouthY - 10,
                    mouthX + mouthWidth/6, mouthY - 5
                );
                ctx.quadraticCurveTo(
                    mouthX + mouthWidth/4, mouthY - 8,
                    mouthX + mouthWidth/2, mouthY
                );
                
                // Close upper lip
                ctx.lineTo(mouthX + mouthWidth/2.5, mouthY + 3);
                ctx.quadraticCurveTo(
                    mouthX, mouthY + 5,
                    mouthX - mouthWidth/2.5, mouthY + 3
                );
                ctx.closePath();
                ctx.fill();
                
                // Lower lip
                ctx.fillStyle = lowerLipColor;
                ctx.beginPath();
                ctx.moveTo(mouthX - mouthWidth/2.5, mouthY + 3);
                ctx.quadraticCurveTo(
                    mouthX, mouthY + 5,
                    mouthX + mouthWidth/2.5, mouthY + 3
                );
                
                if (this.speaking && this.mouthOpenness > 0) {
                    // Open mouth
                    const openness = this.mouthOpenness * 15;
                    ctx.quadraticCurveTo(
                        mouthX + mouthWidth/3, mouthY + 12 + openness,
                        mouthX, mouthY + 15 + openness
                    );
                    ctx.quadraticCurveTo(
                        mouthX - mouthWidth/3, mouthY + 12 + openness,
                        mouthX - mouthWidth/2.5, mouthY + 3
                    );
                    
                    // Mouth interior
                    ctx.fillStyle = '#3a2020';
                    ctx.beginPath();
                    ctx.ellipse(mouthX, mouthY + 8, mouthWidth/3, openness * 0.7, 0, 0, Math.PI);
                    ctx.fill();
                    
                    // Teeth
                    if (openness > 3) {
                        ctx.fillStyle = '#fafafa';
                        ctx.fillRect(mouthX - mouthWidth/4, mouthY + 1, mouthWidth/2, 4);
                    }
                } else {
                    // Closed mouth
                    ctx.quadraticCurveTo(
                        mouthX, mouthY + 12,
                        mouthX - mouthWidth/2.5, mouthY + 3
                    );
                }
                ctx.closePath();
                ctx.fill();
                
                // Mouth corners (smile lines)
                if (this.emotion === "happy" || this.emotion === "amused") {
                    ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.arc(mouthX - mouthWidth/1.8, mouthY, 8, -Math.PI/4, Math.PI/4);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.arc(mouthX + mouthWidth/1.8, mouthY, 8, Math.PI*3/4, Math.PI*5/4);
                    ctx.stroke();
                }
                
                // Lip gloss effect
                ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
                ctx.beginPath();
                ctx.ellipse(mouthX, mouthY + 8, mouthWidth/5, 3, 0, 0, Math.PI * 2);
                ctx.fill();
            }
            
            drawHair(isBack) {
                const hairColor = '#3d2817';
                
                if (isBack) {
                    // Back hair
                    ctx.fillStyle = hairColor;
                    ctx.beginPath();
                    ctx.moveTo(this.face.centerX - this.face.width/1.8, this.face.centerY - this.face.height/2.3);
                    ctx.quadraticCurveTo(
                        this.face.centerX, this.face.centerY - this.face.height/1.8,
                        this.face.centerX + this.face.width/1.8, this.face.centerY - this.face.height/2.3
                    );
                    ctx.lineTo(this.face.centerX + this.face.width/1.5, this.face.centerY + this.face.height/3);
                    ctx.quadraticCurveTo(
                        this.face.centerX + this.face.width/2, this.face.centerY + this.face.height/2,
                        this.face.centerX, this.face.centerY + this.face.height/1.8
                    );
                    ctx.quadraticCurveTo(
                        this.face.centerX - this.face.width/2, this.face.centerY + this.face.height/2,
                        this.face.centerX - this.face.width/1.5, this.face.centerY + this.face.height/3
                    );
                    ctx.closePath();
                    ctx.fill();
                } else {
                    // Front hair with realistic strands
                    ctx.fillStyle = hairColor;
                    
                    // Main hair shape
                    ctx.beginPath();
                    ctx.moveTo(this.face.centerX - this.face.width/1.8, this.face.centerY - this.face.height/2.5);
                    
                    // Hair parting
                    ctx.quadraticCurveTo(
                        this.face.centerX - this.face.width/4, this.face.centerY - this.face.height/1.9,
                        this.face.centerX, this.face.centerY - this.face.height/2
                    );
                    ctx.quadraticCurveTo(
                        this.face.centerX + this.face.width/4, this.face.centerY - this.face.height/1.9,
                        this.face.centerX + this.face.width/1.8, this.face.centerY - this.face.height/2.5
                    );
                    
                    // Side hair
                    ctx.lineTo(this.face.centerX + this.face.width/2, this.face.centerY - this.face.height/3);
                    ctx.quadraticCurveTo(
                        this.face.centerX + this.face.width/2.2, this.face.centerY - this.face.height/4,
                        this.face.centerX + this.face.width/2.5, this.face.centerY - this.face.height/6
                    );
                    
                    // Bangs
                    ctx.quadraticCurveTo(
                        this.face.centerX + this.face.width/3, this.face.centerY - this.face.height/3.5,
                        this.face.centerX, this.face.centerY - this.face.height/3
                    );
                    ctx.quadraticCurveTo(
                        this.face.centerX - this.face.width/3, this.face.centerY - this.face.height/3.5,
                        this.face.centerX - this.face.width/2.5, this.face.centerY - this.face.height/6
                    );
                    
                    ctx.quadraticCurveTo(
                        this.face.centerX - this.face.width/2.2, this.face.centerY - this.face.height/4,
                        this.face.centerX - this.face.width/2, this.face.centerY - this.face.height/3
                    );
                    
                    ctx.closePath();
                    ctx.fill();
                    
                    // Hair strands for texture
                    ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
                    ctx.lineWidth = 1;
                    for (let i = 0; i < 20; i++) {
                        const x = this.face.centerX + (Math.random() - 0.5) * this.face.width;
                        const y = this.face.centerY - this.face.height/2 + Math.random() * 40;
                        ctx.beginPath();
                        ctx.moveTo(x, y);
                        ctx.quadraticCurveTo(
                            x + (Math.random() - 0.5) * 20,
                            y + 40,
                            x + (Math.random() - 0.5) * 30,
                            y + 80 + Math.random() * 40
                        );
                        ctx.stroke();
                    }
                    
                    // Hair highlights
                    ctx.strokeStyle = 'rgba(139, 69, 19, 0.3)';
                    ctx.lineWidth = 2;
                    for (let i = 0; i < 5; i++) {
                        const x = this.face.centerX + (Math.random() - 0.5) * this.face.width * 0.7;
                        const y = this.face.centerY - this.face.height/2.2 + Math.random() * 20;
                        ctx.beginPath();
                        ctx.moveTo(x, y);
                        ctx.quadraticCurveTo(
                            x + 10,
                            y + 30,
                            x + (Math.random() - 0.5) * 20,
                            y + 60
                        );
                        ctx.stroke();
                    }
                }
                
                // Animated hair particles
                ctx.fillStyle = hairColor;
                this.hairParticles.forEach(particle => {
                    particle.angle += Math.sin(this.frame * 0.01) * 0.001;
                    particle.x += Math.sin(this.frame * 0.02 + particle.angle) * 0.1;
                    
                    ctx.save();
                    ctx.translate(particle.x, particle.y);
                    ctx.rotate(particle.angle);
                    ctx.fillRect(0, 0, particle.thickness, particle.length);
                    ctx.restore();
                });
            }
            
            drawAccessories() {
                // Optional: earrings, glasses, etc.
                if (this.emotion === "professional") {
                    // Simple glasses
                    ctx.strokeStyle = '#333333';
                    ctx.lineWidth = 3;
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
                    
                    const eyeY = this.face.centerY - this.face.height/8;
                    const eyeSpacing = this.face.width/5;
                    
                    // Left lens
                    ctx.beginPath();
                    ctx.ellipse(
                        this.face.centerX - eyeSpacing,
                        eyeY,
                        45, 40, 0, 0, Math.PI * 2
                    );
                    ctx.fill();
                    ctx.stroke();
                    
                    // Right lens
                    ctx.beginPath();
                    ctx.ellipse(
                        this.face.centerX + eyeSpacing,
                        eyeY,
                        45, 40, 0, 0, Math.PI * 2
                    );
                    ctx.fill();
                    ctx.stroke();
                    
                    // Bridge
                    ctx.beginPath();
                    ctx.moveTo(this.face.centerX - eyeSpacing + 35, eyeY);
                    ctx.lineTo(this.face.centerX + eyeSpacing - 35, eyeY);
                    ctx.stroke();
                }
            }
            
            updateAnimations() {
                // Eye blinking
                this.blinkTimer++;
                if (this.blinkTimer > 180 + Math.random() * 120) {
                    this.blinking = true;
                    this.blinkTimer = 0;
                }
                
                if (this.blinking) {
                    this.eyeOpenness = Math.max(0, this.eyeOpenness - 0.2);
                    if (this.eyeOpenness <= 0) {
                        this.blinking = false;
                    }
                } else {
                    this.eyeOpenness = Math.min(1, this.eyeOpenness + 0.15);
                }
                
                // Speaking animation
                if (this.speaking) {
                    // Natural speech patterns
                    const speechPattern = Math.sin(this.frame * 0.3) * 0.4 + 
                                       Math.sin(this.frame * 0.7) * 0.3 +
                                       Math.random() * 0.2;
                    this.mouthOpenness = Math.max(0, Math.min(1, speechPattern));
                } else {
                    this.mouthOpenness = Math.max(0, this.mouthOpenness - 0.1);
                }
                
                // Eye movement (saccades)
                if (this.frame % 120 === 0) {
                    this.lookDirection.x = (Math.random() - 0.5) * 0.6;
                    this.lookDirection.y = (Math.random() - 0.5) * 0.4;
                } else {
                    // Smooth return to center
                    this.lookDirection.x *= 0.98;
                    this.lookDirection.y *= 0.98;
                }
                
                // Microexpressions
                this.microExpressions.timer++;
                if (this.microExpressions.timer > 300 + Math.random() * 200) {
                    this.microExpressions.current = ['eyebrow_raise', 'slight_smile', 'squint'][Math.floor(Math.random() * 3)];
                    this.microExpressions.intensity = 0;
                    this.microExpressions.timer = 0;
                }
                
                if (this.microExpressions.current) {
                    if (this.microExpressions.intensity < 1) {
                        this.microExpressions.intensity += 0.05;
                    } else {
                        this.microExpressions.intensity -= 0.03;
                        if (this.microExpressions.intensity <= 0) {
                            this.microExpressions.current = null;
                        }
                    }
                }
            }
            
            speak() {
                this.speaking = true;
                setTimeout(() => {
                    this.speaking = false;
                }, 3000 + Math.random() * 2000);
            }
            
            setEmotion(emotion) {
                this.emotion = emotion;
                // Adjust facial features based on emotion
                switch(emotion) {
                    case 'happy':
                        this.lookDirection.y = -0.1;
                        break;
                    case 'concerned':
                        this.lookDirection.y = 0.1;
                        break;
                    case 'thinking':
                        this.lookDirection.x = 0.3;
                        this.lookDirection.y = -0.2;
                        break;
                    case 'professional':
                        this.lookDirection.x = 0;
                        this.lookDirection.y = 0;
                        break;
                }
            }
        }
        
        const avatar = new PhotorealisticAvatar();
        
        function animate() {
            avatar.draw();
            requestAnimationFrame(animate);
        }
        animate();
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const messageDiv = document.createElement("div");
            messageDiv.className = "message finsight";
            messageDiv.innerHTML = data.message;
            chat.appendChild(messageDiv);
            
            // Animate avatar
            avatar.speak();
            
            // Detect emotion from response
            if (data.message.toLowerCase().includes('excellent') || 
                data.message.toLowerCase().includes('great')) {
                avatar.setEmotion('happy');
            } else if (data.message.toLowerCase().includes('concern') || 
                       data.message.toLowerCase().includes('risk')) {
                avatar.setEmotion('concerned');
            } else if (data.message.toLowerCase().includes('analyzing') || 
                       data.message.toLowerCase().includes('calculating')) {
                avatar.setEmotion('thinking');
            } else {
                avatar.setEmotion('professional');
            }
            
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
    return {"status": "healthy", "deployment": "finsight-aria"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            system_prompt = """You are FinSight Deep Neural Supercomputer, an advanced financial analysis system.
            Provide specific, data-driven insights and recommendations."""
            
            # Try NVIDIA first, fallback to Together
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
            "model": MODELS["together"],
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

if __name__ == "__main__":
    print("Launching FinSight Aria - Photorealistic Avatar...")
    print("Access at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)