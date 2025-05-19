# Digital Human 2D Avatar Issues and Fixes

## 1. Canvas Sizing Issues

### Problem
- Canvas element doesn't properly size to its container
- Resolution mismatch between CSS size and actual canvas dimensions
- Blurry rendering on high-DPI displays

### Fix
```javascript
// In DigitalHuman2D constructor, add:
initCanvas() {
    // Handle high-DPI displays
    const devicePixelRatio = window.devicePixelRatio || 1;
    
    // Get CSS size
    const rect = this.canvas.getBoundingClientRect();
    
    // Set actual canvas size
    this.canvas.width = rect.width * devicePixelRatio;
    this.canvas.height = rect.height * devicePixelRatio;
    
    // Scale context for high-DPI
    this.ctx.scale(devicePixelRatio, devicePixelRatio);
    
    // Update face center based on actual size
    this.face.centerX = rect.width / 2;
    this.face.centerY = rect.height / 2.2;
}
```

## 2. Missing Avatar 2D Class Implementation

### Problem
- The `DigitalHuman2D` class is referenced but appears incomplete
- Missing dispose method for cleanup
- Canvas context not properly initialized

### Fix
```javascript
// Add to DigitalHuman2D class:
constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    
    if (!this.ctx) {
        console.error('Failed to get 2D context');
        return;
    }
    
    this.initCanvas();
    this.setupResizeHandler();
    // ... rest of initialization
}

setupResizeHandler() {
    this.resizeHandler = () => {
        this.initCanvas();
    };
    window.addEventListener('resize', this.resizeHandler);
}

dispose() {
    this.stop();
    window.removeEventListener('resize', this.resizeHandler);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
}
```

## 3. Socket.io Connection Issues

### Problem
- Wrong WebSocket URL format for Socket.io
- No error handling for connection failures
- Missing reconnection logic

### Fix
```javascript
// In connectWebSocket method:
connectWebSocket() {
    // Correct Socket.io connection URL
    this.socket = io('http://localhost:8000', {
        path: '/socket.io/',
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000
    });
    
    this.socket.on('connect_error', (error) => {
        console.error('Connection error:', error);
        this.updateStatus('error');
    });
    
    // ... rest of socket handlers
}
```

## 4. Missing GSAP Import

### Problem
- Code references GSAP animations but library is not included
- Causes errors when trying to animate 3D avatar

### Fix
- Add GSAP script tag in HTML before the main script:
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.11.4/gsap.min.js"></script>
```

## 5. Memory Leaks in Animation Loop

### Problem
- Animation frames not properly cleaned up
- Potential memory leaks with neural network visualization

### Fix
```javascript
// In DigitalHuman2D class:
animate() {
    if (!this.canvas || !this.ctx) return;
    
    this.draw();
    this.animationId = requestAnimationFrame(() => this.animate());
}

stop() {
    if (this.animationId) {
        cancelAnimationFrame(this.animationId);
        this.animationId = null;
    }
}
```

## 6. 3D Avatar Disposal Issues

### Problem
- THREE.js renderer not properly disposed
- Event listeners not cleaned up

### Fix
```javascript
// In DigitalHumanInterface class:
dispose3DAvatar() {
    if (this.avatarRenderer) {
        const { renderer, scene } = this.avatarRenderer;
        
        // Dispose of all materials and geometries
        scene.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => material.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
        
        // Dispose renderer
        renderer.dispose();
        
        // Remove event listeners
        window.removeEventListener('resize', this.resizeHandler);
    }
}
```

## 7. Chart Update Issues

### Problem
- Chart updates might fail if data format is incorrect
- No error handling for chart operations

### Fix
```javascript
// In updateAllocationChart method:
updateAllocationChart(allocation) {
    try {
        if (this.charts.allocation && allocation) {
            const labels = Object.keys(allocation);
            const data = Object.values(allocation);
            
            if (labels.length > 0 && data.length > 0) {
                this.charts.allocation.data.labels = labels;
                this.charts.allocation.data.datasets[0].data = data;
                this.charts.allocation.update('none'); // Prevent animation lag
            }
        }
    } catch (error) {
        console.error('Error updating allocation chart:', error);
    }
}
```

## 8. Voice Recognition Error Handling

### Problem
- Speech recognition errors not properly handled
- No user feedback for microphone permission issues

### Fix
```javascript
// In initializeSpeechRecognition method:
initializeSpeechRecognition() {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        console.warn('Speech recognition not supported');
        this.disableVoiceButton();
        return;
    }
    
    // ... rest of initialization
    
    this.speechRecognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        
        if (event.error === 'not-allowed') {
            this.displayMessage('Microphone permission denied. Please enable it in your browser settings.', 'assistant');
        } else {
            this.displayMessage('Speech recognition error. Please try again.', 'assistant');
        }
        
        this.toggleVoiceRecognition(false);
    };
}

disableVoiceButton() {
    const voiceBtn = document.getElementById('voiceBtn');
    if (voiceBtn) {
        voiceBtn.disabled = true;
        voiceBtn.style.opacity = '0.5';
        voiceBtn.style.cursor = 'not-allowed';
        voiceBtn.title = 'Speech recognition not supported';
    }
}
```

## 9. Avatar Toggle State Management

### Problem
- Avatar mode toggle doesn't properly clean up previous state
- Canvas might retain previous rendering

### Fix
```javascript
// In toggleAvatarMode method:
toggleAvatarMode() {
    const toggleBtn = document.getElementById('avatarToggleBtn');
    const canvas = document.getElementById('avatarCanvas');
    const context = canvas.getContext('2d');
    
    // Clear canvas before switching
    context.clearRect(0, 0, canvas.width, canvas.height);
    
    if (this.avatarMode === '2d') {
        // Switch to 3D
        this.avatarMode = '3d';
        toggleBtn.querySelector('.toggle-text').textContent = '3D';
        toggleBtn.classList.add('active');
        
        // Properly dispose 2D
        if (this.avatar2D) {
            this.avatar2D.dispose();
            this.avatar2D = null;
        }
        
        // Initialize 3D avatar
        this.initialize3DAvatar(canvas);
    } else {
        // Switch to 2D
        this.avatarMode = '2d';
        toggleBtn.querySelector('.toggle-text').textContent = '2D';
        toggleBtn.classList.remove('active');
        
        // Properly dispose 3D
        if (this.avatarRenderer) {
            this.dispose3DAvatar();
            this.avatarRenderer = null;
        }
        
        // Initialize 2D avatar
        this.avatar2D = new DigitalHuman2D(canvas);
    }
    
    // Store preference
    localStorage.setItem('avatarMode', this.avatarMode);
}
```

## 10. CSS Animation Performance

### Problem
- Multiple overlapping CSS animations causing performance issues
- Typing indicator visibility issue

### Fix
```css
/* Fix typing indicator display */
.typing-indicator {
    display: none;
}

.typing-indicator.show {
    display: inline-flex;
}

/* Optimize animations */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.status-indicator {
    will-change: opacity;
    animation: pulse 2s infinite ease-in-out;
}
```

## 11. Event Listener Cleanup

### Problem
- Event listeners not properly removed when component is destroyed
- Potential memory leaks

### Fix
```javascript
// Add destroy method to DigitalHumanInterface:
destroy() {
    // Disconnect socket
    if (this.socket) {
        this.socket.disconnect();
    }
    
    // Stop speech recognition
    if (this.speechRecognition) {
        this.speechRecognition.stop();
    }
    
    // Dispose avatars
    if (this.avatar2D) {
        this.avatar2D.dispose();
    }
    if (this.avatarRenderer) {
        this.dispose3DAvatar();
    }
    
    // Destroy charts
    Object.values(this.charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    
    // Clear any remaining animations
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
}
```

## 12. Canvas Blur Fix for High-DPI Displays

### Problem
- Canvas appears blurry on Retina/high-DPI displays
- Scaling issues with 2D rendering

### Fix
```javascript
// Complete high-DPI fix for DigitalHuman2D:
draw() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    // Save context state
    this.ctx.save();
    
    // Clear with proper scaling
    this.ctx.clearRect(0, 0, rect.width, rect.height);
    
    // Draw everything
    this.drawNeuralNetwork();
    this.drawFace();
    this.updateAnimations();
    
    // Restore context state
    this.ctx.restore();
    
    this.frame++;
}
```