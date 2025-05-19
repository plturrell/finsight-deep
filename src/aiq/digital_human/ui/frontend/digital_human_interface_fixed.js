// Digital Human Interface JavaScript - Fixed Version
class DigitalHumanInterface {
    constructor() {
        this.socket = null;
        this.sessionId = null;
        this.avatarRenderer = null;
        this.speechRecognition = null;
        this.charts = {};
        this.isVoiceActive = false;
        this.isTyping = false;
        this.avatarMode = localStorage.getItem('avatarMode') || '2d';
        this.avatar2D = null;
        this.avatar3D = null;

        this.initialize();
    }

    async initialize() {
        // Connect to WebSocket
        this.connectWebSocket();

        // Initialize Avatar
        this.initializeAvatar();

        // Initialize charts
        this.initializeCharts();

        // Setup event listeners
        this.setupEventListeners();

        // Initialize speech recognition
        this.initializeSpeechRecognition();

        // Load user data
        await this.loadUserData();
    }

    connectWebSocket() {
        // Fixed Socket.io connection with proper configuration
        this.socket = io('http://localhost:8000', {
            path: '/socket.io/',
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 1000
        });

        this.socket.on('connect', () => {
            console.log('Connected to Digital Human server');
            this.updateStatus('online');
            this.sessionId = this.generateSessionId();
            this.socket.emit('startSession', { sessionId: this.sessionId });
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.updateStatus('error');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateStatus('offline');
        });

        this.socket.on('response', (data) => {
            this.handleResponse(data);
        });

        this.socket.on('avatarUpdate', (data) => {
            this.updateAvatar(data);
        });

        this.socket.on('portfolioUpdate', (data) => {
            this.updatePortfolio(data);
        });

        this.socket.on('analysisUpdate', (data) => {
            this.updateAnalysis(data);
        });

        this.socket.on('error', (error) => {
            this.handleError(error);
        });
    }

    initializeAvatar() {
        const canvas = document.getElementById('avatarCanvas');
        
        if (this.avatarMode === '2d') {
            // Initialize 2D avatar
            this.avatar2D = new DigitalHuman2D(canvas);
            canvas.style.display = 'block';
        } else {
            // Initialize 3D avatar
            this.initialize3DAvatar(canvas);
        }
    }
    
    initialize3DAvatar(canvas) {
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, canvas.width / canvas.height, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });

        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        renderer.setClearColor(0x000000, 0);

        // Add lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(0, 10, 5);
        scene.add(directionalLight);

        // Create placeholder avatar (in production, load from GLTF)
        const geometry = new THREE.SphereGeometry(1, 32, 32);
        const material = new THREE.MeshPhongMaterial({ 
            color: 0x3b82f6,
            emissive: 0x3b82f6,
            emissiveIntensity: 0.2
        });
        const avatar = new THREE.Mesh(geometry, material);
        scene.add(avatar);

        camera.position.z = 3;

        // Animation loop
        const animate = () => {
            requestAnimationFrame(animate);

            // Subtle idle animation
            avatar.rotation.y += 0.005;
            avatar.position.y = Math.sin(Date.now() * 0.001) * 0.1;

            renderer.render(scene, camera);
        };

        animate();

        this.avatarRenderer = { scene, camera, renderer, avatar };

        // Handle resize
        this.resizeHandler = () => {
            const width = canvas.clientWidth;
            const height = canvas.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        };
        window.addEventListener('resize', this.resizeHandler);
    }

    initializeCharts() {
        // Allocation Chart
        const allocationCtx = document.getElementById('allocationChart').getContext('2d');
        this.charts.allocation = new Chart(allocationCtx, {
            type: 'doughnut',
            data: {
                labels: ['Stocks', 'Bonds', 'Cash', 'Crypto'],
                datasets: [{
                    data: [60, 25, 10, 5],
                    backgroundColor: [
                        '#3b82f6',
                        '#4ade80',
                        '#fbbf24',
                        '#ef4444'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#94a3b8',
                            padding: 10
                        }
                    }
                }
            }
        });

        // Sentiment Chart
        const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
        this.charts.sentiment = new Chart(sentimentCtx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Market Sentiment',
                    data: [65, 70, 68, 75, 80, 78],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    }
                }
            }
        });
    }

    setupEventListeners() {
        // Send message
        const sendBtn = document.getElementById('sendBtn');
        const messageInput = document.getElementById('messageInput');

        const sendMessage = () => {
            const message = messageInput.value.trim();
            if (message) {
                this.sendMessage(message);
                messageInput.value = '';
            }
        };

        sendBtn.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Voice control
        const voiceBtn = document.getElementById('voiceBtn');
        voiceBtn.addEventListener('click', () => {
            this.toggleVoiceRecognition();
        });

        // Settings
        const settingsBtn = document.getElementById('settingsBtn');
        const settingsPanel = document.getElementById('settingsPanel');
        
        settingsBtn.addEventListener('click', () => {
            settingsPanel.classList.toggle('open');
        });
        
        // Avatar mode toggle
        const avatarToggleBtn = document.getElementById('avatarToggleBtn');
        avatarToggleBtn.addEventListener('click', () => {
            this.toggleAvatarMode();
        });
        
        // Set initial button state
        if (this.avatarMode === '3d') {
            avatarToggleBtn.querySelector('.toggle-text').textContent = '3D';
            avatarToggleBtn.classList.add('active');
        }

        // Quick actions
        document.querySelectorAll('.quick-action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.target.textContent;
                this.handleQuickAction(action);
            });
        });
    }

    initializeSpeechRecognition() {
        if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
            console.warn('Speech recognition not supported');
            this.disableVoiceButton();
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.speechRecognition = new SpeechRecognition();
        this.speechRecognition.continuous = false;
        this.speechRecognition.interimResults = true;
        this.speechRecognition.lang = 'en-US';

        this.speechRecognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            const isFinal = event.results[0].isFinal;

            if (isFinal) {
                this.sendMessage(transcript);
            } else {
                document.getElementById('messageInput').value = transcript;
            }
        };

        this.speechRecognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            
            if (event.error === 'not-allowed') {
                this.displayMessage('Microphone permission denied. Please enable it in your browser settings.', 'assistant');
            } else {
                this.displayMessage('Speech recognition error. Please try again.', 'assistant');
            }
            
            this.toggleVoiceRecognition(false);
        };

        this.speechRecognition.onend = () => {
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

    toggleVoiceRecognition(force = null) {
        const voiceBtn = document.getElementById('voiceBtn');
        
        if (force !== null) {
            this.isVoiceActive = !force;
        }

        if (this.isVoiceActive) {
            this.speechRecognition.stop();
            voiceBtn.classList.remove('active');
            this.isVoiceActive = false;
        } else {
            this.speechRecognition.start();
            voiceBtn.classList.add('active');
            this.isVoiceActive = true;
        }
    }

    sendMessage(message) {
        // Display user message
        this.displayMessage(message, 'user');

        // Show typing indicator
        this.showTypingIndicator();

        // Send to server
        this.socket.emit('message', {
            sessionId: this.sessionId,
            content: message,
            timestamp: new Date().toISOString()
        });
    }

    handleResponse(data) {
        // Hide typing indicator
        this.hideTypingIndicator();

        // Display assistant message
        this.displayMessage(data.text, 'assistant');

        // Update avatar
        if (data.avatar) {
            this.updateAvatar(data.avatar);
        }

        // Update financial data
        if (data.financialData) {
            this.updateFinancialData(data.financialData);
        }

        // Update analysis
        if (data.analysis) {
            this.updateAnalysis(data.analysis);
        }

        // Handle actions
        if (data.actions) {
            this.handleActions(data.actions);
        }

        // Text-to-speech
        if (data.speak !== false) {
            this.speak(data.text);
        }
    }

    displayMessage(content, sender) {
        const messagesContainer = document.getElementById('messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    showTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        typingIndicator.classList.add('show');
        this.isTyping = true;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        typingIndicator.classList.remove('show');
        this.isTyping = false;
    }

    updateAvatar(data) {
        if (this.avatarMode === '2d') {
            // Handle 2D avatar updates
            if (!this.avatar2D) return;
            
            // Update expression
            if (data.expression) {
                this.avatar2D.setExpression(data.expression);
            }
            
            // Update speaking state
            if (data.isSpeaking !== undefined) {
                this.avatar2D.setSpeaking(data.isSpeaking);
            }
            
            // 2D doesn't need position updates
        } else {
            // Handle 3D avatar updates
            if (!this.avatarRenderer) return;

            const { avatar } = this.avatarRenderer;

            // Update avatar expression
            if (data.expression) {
                this.setAvatarExpression(data.expression);
            }

            // Update avatar gesture
            if (data.gesture) {
                this.setAvatarGesture(data.gesture);
            }

            // Update avatar position
            if (data.position) {
                gsap.to(avatar.position, {
                    x: data.position.x,
                    y: data.position.y,
                    z: data.position.z,
                    duration: 0.5
                });
            }
        }

        // Update status for both modes
        if (data.status) {
            this.updateStatus(data.status);
        }
    }

    setAvatarExpression(expression) {
        // In production, this would update blend shapes or morph targets
        const { avatar } = this.avatarRenderer;
        
        switch (expression) {
            case 'happy':
                avatar.material.color.setHex(0x4ade80);
                break;
            case 'concerned':
                avatar.material.color.setHex(0xfbbf24);
                break;
            case 'neutral':
                avatar.material.color.setHex(0x3b82f6);
                break;
            case 'thinking':
                avatar.material.color.setHex(0x60a5fa);
                break;
        }
    }

    setAvatarGesture(gesture) {
        // In production, this would trigger animations
        const { avatar } = this.avatarRenderer;
        
        switch (gesture) {
            case 'nod':
                gsap.to(avatar.rotation, {
                    x: Math.PI / 8,
                    duration: 0.2,
                    yoyo: true,
                    repeat: 1
                });
                break;
            case 'shake':
                gsap.to(avatar.rotation, {
                    y: Math.PI / 8,
                    duration: 0.1,
                    yoyo: true,
                    repeat: 3
                });
                break;
            case 'present':
                gsap.to(avatar.scale, {
                    x: 1.1,
                    y: 1.1,
                    z: 1.1,
                    duration: 0.3,
                    yoyo: true,
                    repeat: 1
                });
                break;
        }
    }

    updateStatus(status) {
        const statusIndicator = document.querySelector('.status-indicator');
        const statusText = document.querySelector('.avatar-status span');

        statusIndicator.className = 'status-indicator';
        
        switch (status) {
            case 'online':
                statusIndicator.classList.add('online');
                statusText.textContent = 'AI Advisor Online';
                break;
            case 'thinking':
                statusIndicator.classList.add('thinking');
                statusText.textContent = 'Thinking...';
                break;
            case 'error':
                statusIndicator.classList.add('error');
                statusText.textContent = 'Connection Error';
                break;
            case 'offline':
                statusIndicator.classList.add('error');
                statusText.textContent = 'Offline';
                break;
        }
    }

    updatePortfolio(data) {
        // Update portfolio value
        if (data.totalValue) {
            document.querySelector('.portfolio-value').textContent = 
                this.formatCurrency(data.totalValue);
        }

        // Update portfolio change
        if (data.change) {
            const changeElement = document.querySelector('.portfolio-change');
            changeElement.textContent = 
                `${data.change.amount >= 0 ? '+' : ''}${this.formatCurrency(data.change.amount)} (${data.change.percentage.toFixed(2)}%)`;
            changeElement.className = `portfolio-change ${data.change.amount >= 0 ? '' : 'negative'}`;
        }

        // Update metrics
        if (data.metrics) {
            this.updateMetrics(data.metrics);
        }

        // Update allocation chart
        if (data.allocation) {
            this.updateAllocationChart(data.allocation);
        }
    }

    updateMetrics(metrics) {
        // Update individual metric cards
        const metricMapping = {
            'Total Return': metrics.totalReturn,
            'Sharpe Ratio': metrics.sharpeRatio,
            'Risk Level': metrics.riskLevel
        };

        document.querySelectorAll('.metric-card').forEach(card => {
            const label = card.querySelector('.metric-label').textContent;
            if (metricMapping[label] !== undefined) {
                const valueElement = card.querySelector('.metric-value');
                
                if (label === 'Total Return') {
                    valueElement.textContent = `${metrics.totalReturn >= 0 ? '+' : ''}${metrics.totalReturn.toFixed(1)}%`;
                    valueElement.style.color = metrics.totalReturn >= 0 ? '#4ade80' : '#ef4444';
                } else if (label === 'Sharpe Ratio') {
                    valueElement.textContent = metrics.sharpeRatio.toFixed(2);
                } else if (label === 'Risk Level') {
                    const riskBar = card.querySelector('.risk-fill');
                    const riskText = card.querySelector('.risk-indicator span');
                    riskBar.style.width = `${metrics.riskLevel}%`;
                    riskText.textContent = this.getRiskLabel(metrics.riskLevel);
                }
            }
        });
    }

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

    updateAnalysis(data) {
        // Update sentiment chart
        if (data.sentiment && this.charts.sentiment) {
            this.charts.sentiment.data.labels = data.sentiment.labels;
            this.charts.sentiment.data.datasets[0].data = data.sentiment.values;
            this.charts.sentiment.update();
        }

        // Update recommendations
        if (data.recommendations) {
            const recommendationElements = document.querySelectorAll('.recommendation');
            data.recommendations.forEach((rec, index) => {
                if (recommendationElements[index]) {
                    recommendationElements[index].innerHTML = 
                        `<strong>Recommendation:</strong> ${rec.text}`;
                }
            });
        }

        // Update risk analysis
        if (data.riskMetrics) {
            const varElement = document.querySelector('.metric-value[style*="color: #ef4444"]');
            const drawdownElement = document.querySelector('.metric-value[style*="color: #fbbf24"]');
            
            if (varElement) varElement.textContent = this.formatCurrency(data.riskMetrics.valueAtRisk);
            if (drawdownElement) drawdownElement.textContent = `${data.riskMetrics.maxDrawdown.toFixed(1)}%`;
        }
    }

    handleActions(actions) {
        actions.forEach(action => {
            switch (action.type) {
                case 'showChart':
                    this.showChart(action.data);
                    break;
                case 'updateView':
                    this.updateView(action.view);
                    break;
                case 'executeTransaction':
                    this.confirmTransaction(action.data);
                    break;
                case 'showDetails':
                    this.showDetails(action.data);
                    break;
            }
        });
    }

    handleQuickAction(action) {
        const actionMap = {
            'Rebalance Portfolio': 'Please rebalance my portfolio',
            'View Performance': 'Show me my portfolio performance',
            'Tax Optimization': 'Help me optimize for taxes',
            'Risk Assessment': 'Run a complete risk assessment'
        };

        const message = actionMap[action] || `Execute ${action}`;
        this.sendMessage(message);
    }

    speak(text) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 0.8;
            
            // Cancel any ongoing speech
            window.speechSynthesis.cancel();
            
            // Speak the new text
            window.speechSynthesis.speak(utterance);
        }
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 0,
            maximumFractionDigits: 2
        }).format(amount);
    }

    getRiskLabel(riskLevel) {
        if (riskLevel < 33) return 'Conservative';
        if (riskLevel < 66) return 'Moderate';
        return 'Aggressive';
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    async loadUserData() {
        try {
            const response = await fetch('/api/user/data', {
                headers: {
                    'Authorization': `Bearer ${this.getAuthToken()}`
                }
            });

            if (response.ok) {
                const data = await response.json();
                this.updatePortfolio(data.portfolio);
                this.updateAnalysis(data.analysis);
            }
        } catch (error) {
            console.error('Failed to load user data:', error);
        }
    }

    getAuthToken() {
        // In production, this would retrieve the auth token from storage
        return localStorage.getItem('authToken') || 'demo_token';
    }

    handleError(error) {
        console.error('Error:', error);
        this.displayMessage('I apologize, but I encountered an error. Please try again.', 'assistant');
        this.updateStatus('error');
    }

    confirmTransaction(data) {
        // In production, this would show a confirmation dialog
        const confirmed = confirm(`Confirm transaction: ${data.description}?`);
        if (confirmed) {
            this.socket.emit('executeTransaction', {
                sessionId: this.sessionId,
                transactionId: data.id,
                confirmed: true
            });
        }
    }

    showChart(data) {
        // Create and display a new chart in a modal
        const modal = document.createElement('div');
        modal.className = 'chart-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <h3>${data.title}</h3>
                <canvas id="modalChart"></canvas>
                <button class="close-btn">Close</button>
            </div>
        `;
        document.body.appendChild(modal);

        // Create the chart
        const ctx = modal.querySelector('#modalChart').getContext('2d');
        new Chart(ctx, data.config);

        // Close button
        modal.querySelector('.close-btn').addEventListener('click', () => {
            modal.remove();
        });
    }

    updateView(view) {
        // Update the interface to show a specific view
        switch (view) {
            case 'portfolio':
                // Focus on portfolio panel
                document.querySelector('.left-panel').scrollIntoView();
                break;
            case 'analysis':
                // Focus on analysis panel
                document.querySelector('.right-panel').scrollIntoView();
                break;
            case 'fullscreen':
                // Expand avatar to fullscreen
                this.toggleFullscreenAvatar();
                break;
        }
    }

    toggleFullscreenAvatar() {
        const avatarContainer = document.querySelector('.avatar-container');
        avatarContainer.classList.toggle('fullscreen');
        
        if (avatarContainer.classList.contains('fullscreen')) {
            avatarContainer.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }

    showDetails(data) {
        // Show detailed information in a side panel
        const detailsPanel = document.createElement('div');
        detailsPanel.className = 'details-panel';
        detailsPanel.innerHTML = `
            <h3>${data.title}</h3>
            <div class="details-content">
                ${data.content}
            </div>
            <button class="close-btn">Close</button>
        `;
        
        document.body.appendChild(detailsPanel);
        
        // Animate in
        setTimeout(() => {
            detailsPanel.classList.add('open');
        }, 10);
        
        // Close button
        detailsPanel.querySelector('.close-btn').addEventListener('click', () => {
            detailsPanel.classList.remove('open');
            setTimeout(() => {
                detailsPanel.remove();
            }, 300);
        });
    }
    
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
}

// 2D Avatar Implementation - Fixed Version
class DigitalHuman2D {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        
        if (!this.ctx) {
            console.error('Failed to get 2D context');
            return;
        }
        
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
        this.animationId = null;
        
        // Face structure
        this.face = {
            centerX: this.canvas.width / 2,
            centerY: this.canvas.height / 2.2,
            width: 180,
            height: 240
        };
        
        // Neural network visualization
        this.neurons = [];
        this.connections = [];
        this.initNeuralNetwork();
        
        // Initialize canvas for high-DPI displays
        this.initCanvas();
        this.setupResizeHandler();
        
        // Start animation loop
        this.animate();
    }
    
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
    
    setupResizeHandler() {
        this.resizeHandler = () => {
            this.initCanvas();
        };
        window.addEventListener('resize', this.resizeHandler);
    }
    
    initNeuralNetwork() {
        // Create background neural network
        for (let i = 0; i < 50; i++) {
            this.neurons.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
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
    
    dispose() {
        this.stop();
        window.removeEventListener('resize', this.resizeHandler);
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    draw() {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        
        // Save context state
        this.ctx.save();
        
        // Clear with proper scaling
        this.ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
        this.ctx.fillRect(0, 0, rect.width, rect.height);
        
        // Draw everything
        this.drawNeuralNetwork();
        this.drawFace();
        this.updateAnimations();
        
        // Restore context state
        this.ctx.restore();
        
        this.frame++;
    }
    
    drawNeuralNetwork() {
        // Draw connections
        this.ctx.strokeStyle = "rgba(0, 212, 255, 0.1)";
        this.ctx.lineWidth = 1;
        this.connections.forEach(conn => {
            const start = this.neurons[conn.start];
            const end = this.neurons[conn.end];
            const pulse = Math.sin(this.frame * 0.02 + conn.strength * Math.PI) * 0.5 + 0.5;
            this.ctx.globalAlpha = 0.1 + pulse * 0.2;
            this.ctx.beginPath();
            this.ctx.moveTo(start.x, start.y);
            this.ctx.lineTo(end.x, end.y);
            this.ctx.stroke();
        });
        
        // Draw neurons
        this.neurons.forEach(neuron => {
            neuron.pulsePhase += neuron.speed;
            const pulse = Math.sin(neuron.pulsePhase) * 0.5 + 0.5;
            this.ctx.globalAlpha = 0.3 + pulse * 0.7;
            this.ctx.fillStyle = pulse > 0.5 ? "#00d4ff" : "#0099ff";
            this.ctx.beginPath();
            this.ctx.arc(neuron.x, neuron.y, neuron.radius * (1 + pulse * 0.5), 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.globalAlpha = 1;
    }
    
    drawFace() {
        // Head shape
        this.ctx.fillStyle = "#1a1a2e";
        this.ctx.beginPath();
        this.ctx.ellipse(this.face.centerX, this.face.centerY, this.face.width/2, this.face.height/2, 0, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Face glow
        const gradient = this.ctx.createRadialGradient(this.face.centerX, this.face.centerY, 0, this.face.centerX, this.face.centerY, this.face.width/2);
        gradient.addColorStop(0, "rgba(0, 212, 255, 0.1)");
        gradient.addColorStop(1, "transparent");
        this.ctx.fillStyle = gradient;
        this.ctx.fill();
        
        // Face outline
        this.ctx.strokeStyle = "#00d4ff";
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Hair/Top section
        this.ctx.fillStyle = "#0a0a1a";
        this.ctx.beginPath();
        this.ctx.ellipse(this.face.centerX, this.face.centerY - this.face.height/3, this.face.width/2, this.face.height/3, 0, Math.PI, 0);
        this.ctx.fill();
        
        // Eyes
        const eyeY = this.face.centerY - this.face.height/6;
        const eyeSpacing = this.face.width/4;
        this.drawEye(this.face.centerX - eyeSpacing, eyeY, 25, true);
        this.drawEye(this.face.centerX + eyeSpacing, eyeY, 25, false);
        
        // Nose
        this.ctx.strokeStyle = "#00d4ff";
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(this.face.centerX, eyeY + 30);
        this.ctx.lineTo(this.face.centerX - 10, eyeY + 50);
        this.ctx.lineTo(this.face.centerX + 10, eyeY + 50);
        this.ctx.globalAlpha = 0.5;
        this.ctx.stroke();
        this.ctx.globalAlpha = 1;
        
        // Mouth
        this.drawMouth();
        
        // Cybernetic elements
        this.drawCyberneticElements();
    }
    
    drawEye(x, y, size, isLeft) {
        // Eye socket
        this.ctx.fillStyle = "#000033";
        this.ctx.beginPath();
        this.ctx.ellipse(x, y, size, size * 0.7 * this.eyeOpenness, 0, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Eye white
        this.ctx.fillStyle = "#e0e0e0";
        this.ctx.beginPath();
        this.ctx.ellipse(x, y, size * 0.8, size * 0.6 * this.eyeOpenness, 0, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Iris
        const irisX = x + this.eyePositionX * 5;
        const irisY = y + this.eyePositionY * 3;
        
        const irisGradient = this.ctx.createRadialGradient(irisX, irisY, 0, irisX, irisY, size * 0.4);
        irisGradient.addColorStop(0, "#00d4ff");
        irisGradient.addColorStop(0.5, "#0066cc");
        irisGradient.addColorStop(1, "#003366");
        this.ctx.fillStyle = irisGradient;
        this.ctx.beginPath();
        this.ctx.ellipse(irisX, irisY, size * 0.4, size * 0.4 * this.eyeOpenness, 0, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Pupil
        this.ctx.fillStyle = "#000000";
        this.ctx.beginPath();
        this.ctx.ellipse(irisX, irisY, size * 0.15, size * 0.15 * this.eyeOpenness, 0, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Eye highlight
        this.ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        this.ctx.beginPath();
        this.ctx.ellipse(irisX - size * 0.1, irisY - size * 0.1, size * 0.1, size * 0.08, 0, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Cybernetic eye glow
        if (this.speaking) {
            this.ctx.shadowBlur = 10;
            this.ctx.shadowColor = "#00d4ff";
            this.ctx.strokeStyle = "#00d4ff";
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.ellipse(x, y, size * 1.1, size * 0.8 * this.eyeOpenness, 0, 0, Math.PI * 2);
            this.ctx.stroke();
            this.ctx.shadowBlur = 0;
        }
    }
    
    drawMouth() {
        const mouthY = this.face.centerY + this.face.height/4;
        const mouthWidth = this.face.width/3;
        
        // Mouth shape
        this.ctx.fillStyle = "#000033";
        this.ctx.beginPath();
        
        if (this.speaking) {
            // Animated mouth for speaking
            const openness = this.mouthOpenness * 15;
            this.ctx.ellipse(this.face.centerX, mouthY, mouthWidth/2, openness, 0, 0, Math.PI);
            this.ctx.fill();
            
            // Teeth hint
            if (openness > 5) {
                this.ctx.fillStyle = "#e0e0e0";
                this.ctx.fillRect(this.face.centerX - mouthWidth/3, mouthY - openness/2, mouthWidth*2/3, 3);
            }
        } else {
            // Closed mouth
            this.ctx.moveTo(this.face.centerX - mouthWidth/2, mouthY);
            this.ctx.quadraticCurveTo(this.face.centerX, mouthY + 10, this.face.centerX + mouthWidth/2, mouthY);
            this.ctx.stroke();
        }
        
        // Mouth glow when speaking
        if (this.speaking) {
            this.ctx.shadowBlur = 5;
            this.ctx.shadowColor = "#00d4ff";
            this.ctx.strokeStyle = "#00d4ff";
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.ellipse(this.face.centerX, mouthY, mouthWidth/2 + 5, this.mouthOpenness * 15 + 5, 0, 0, Math.PI);
            this.ctx.stroke();
            this.ctx.shadowBlur = 0;
        }
    }
    
    drawCyberneticElements() {
        // Side panels
        this.ctx.fillStyle = "rgba(0, 212, 255, 0.2)";
        this.ctx.strokeStyle = "#00d4ff";
        this.ctx.lineWidth = 2;
        
        // Left panel
        this.ctx.beginPath();
        this.ctx.moveTo(this.face.centerX - this.face.width/2 - 10, this.face.centerY - 50);
        this.ctx.lineTo(this.face.centerX - this.face.width/2 - 30, this.face.centerY - 40);
        this.ctx.lineTo(this.face.centerX - this.face.width/2 - 30, this.face.centerY + 40);
        this.ctx.lineTo(this.face.centerX - this.face.width/2 - 10, this.face.centerY + 50);
        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.stroke();
        
        // Right panel
        this.ctx.beginPath();
        this.ctx.moveTo(this.face.centerX + this.face.width/2 + 10, this.face.centerY - 50);
        this.ctx.lineTo(this.face.centerX + this.face.width/2 + 30, this.face.centerY - 40);
        this.ctx.lineTo(this.face.centerX + this.face.width/2 + 30, this.face.centerY + 40);
        this.ctx.lineTo(this.face.centerX + this.face.width/2 + 10, this.face.centerY + 50);
        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.stroke();
        
        // Forehead element
        this.ctx.beginPath();
        this.ctx.arc(this.face.centerX, this.face.centerY - this.face.height/2.5, 10, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.stroke();
        
        // Add pulsing effect
        const pulse = Math.sin(this.frame * 0.05) * 0.5 + 0.5;
        this.ctx.shadowBlur = 10 * pulse;
        this.ctx.shadowColor = "#00d4ff";
        this.ctx.beginPath();
        this.ctx.arc(this.face.centerX, this.face.centerY - this.face.height/2.5, 15, 0, Math.PI * 2);
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
    }
    
    updateAnimations() {
        // Idle eye movement
        this.eyePositionX = Math.sin(this.frame * 0.01) * 0.3;
        this.eyePositionY = Math.sin(this.frame * 0.015) * 0.2;
        
        // Blinking
        this.blinkTimer++;
        if (this.blinkTimer > 180 && !this.blinking) {
            this.blinking = true;
            this.blinkTimer = 0;
        }
        
        if (this.blinking) {
            this.eyeOpenness = Math.max(0, this.eyeOpenness - 0.2);
            if (this.eyeOpenness <= 0) {
                this.blinking = false;
            }
        } else {
            this.eyeOpenness = Math.min(1, this.eyeOpenness + 0.1);
        }
        
        // Speaking animation
        if (this.speaking) {
            this.speechAmplitude = Math.sin(this.frame * 0.1) * 0.5 + 0.5;
            this.mouthOpenness = 0.3 + this.speechAmplitude * 0.7;
        } else {
            this.mouthOpenness = Math.max(0, this.mouthOpenness - 0.1);
        }
        
        // Head tilt
        this.headTilt = Math.sin(this.frame * 0.005) * 0.02;
    }
    
    setSpeaking(isSpeaking) {
        this.speaking = isSpeaking;
    }
    
    setExpression(expression) {
        // Set different expressions
        switch(expression) {
            case 'happy':
                this.eyeOpenness = 0.8;
                break;
            case 'thinking':
                this.eyeOpenness = 0.7;
                this.eyePositionY = -0.3;
                break;
            case 'surprised':
                this.eyeOpenness = 1.3;
                break;
        }
    }
}

// Initialize the interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.digitalHuman = new DigitalHumanInterface();
});