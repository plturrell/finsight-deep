# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Monitoring dashboard for distributed AIQToolkit
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
from typing import Dict, Any, List
import json
from datetime import datetime
import logging

from aiq.distributed.node_manager import NodeManager
from aiq.distributed.task_scheduler import TaskScheduler
from aiq.distributed.monitoring.metrics import MetricsAggregator

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Web-based monitoring dashboard for distributed cluster"""
    
    def __init__(self, 
                 node_manager: NodeManager,
                 task_scheduler: TaskScheduler,
                 port: int = 8080):
        self.node_manager = node_manager
        self.task_scheduler = task_scheduler
        self.port = port
        
        # Flask app setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'aiqtoolkit-monitoring'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Metrics aggregator
        self.metrics_aggregator = MetricsAggregator()
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio()
        
        # Background update thread
        self.update_thread = None
        self.running = False
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/cluster/status')
        def cluster_status():
            status = self.node_manager.get_cluster_status()
            return jsonify(status)
        
        @self.app.route('/api/tasks/status')
        def tasks_status():
            status = self.task_scheduler.get_queue_status()
            return jsonify(status)
        
        @self.app.route('/api/tasks/history')
        def tasks_history():
            history = self.task_scheduler.get_task_history(limit=50)
            return jsonify(history)
        
        @self.app.route('/api/nodes/<node_id>')
        def node_details(node_id):
            node = self.node_manager.get_node_info(node_id)
            if node:
                return jsonify(node.__dict__)
            return jsonify({"error": "Node not found"}), 404
    
    def _setup_socketio(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Client connected to monitoring dashboard")
            emit('connected', {'data': 'Connected to monitoring server'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("Client disconnected from monitoring dashboard")
    
    def start(self):
        """Start the monitoring dashboard"""
        self.running = True
        
        # Start background update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Start Flask app
        logger.info(f"Starting monitoring dashboard on port {self.port}")
        self.socketio.run(self.app, host='0.0.0.0', port=self.port)
    
    def stop(self):
        """Stop the monitoring dashboard"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
    
    def _update_loop(self):
        """Background thread to push updates to clients"""
        while self.running:
            try:
                # Get current status
                cluster_status = self.node_manager.get_cluster_status()
                task_status = self.task_scheduler.get_queue_status()
                
                # Get metrics summary
                metrics_summary = self.metrics_aggregator.get_cluster_summary()
                
                # Prepare update data
                update_data = {
                    'timestamp': datetime.now().isoformat(),
                    'cluster': cluster_status,
                    'tasks': task_status,
                    'metrics': metrics_summary
                }
                
                # Emit update to all connected clients
                self.socketio.emit('status_update', update_data)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
            
            time.sleep(5)  # Update every 5 seconds


# HTML template for the dashboard
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>AIQToolkit Distributed Monitoring</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .metric {
            flex: 1;
            min-width: 200px;
            margin: 10px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .node-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }
        .node-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
        }
        .node-online {
            border-left: 4px solid #28a745;
        }
        .node-offline {
            border-left: 4px solid #dc3545;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-online {
            background-color: #28a745;
        }
        .status-offline {
            background-color: #dc3545;
        }
        .charts {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .chart-container {
            position: relative;
            height: 300px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AIQToolkit Distributed Cluster Monitor</h1>
        
        <div class="card">
            <h2>Cluster Overview</h2>
            <div class="metrics" id="cluster-metrics">
                <div class="metric">
                    <div class="metric-value" id="total-nodes">0</div>
                    <div class="metric-label">Total Nodes</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="online-nodes">0</div>
                    <div class="metric-label">Online Nodes</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="total-gpus">0</div>
                    <div class="metric-label">Total GPUs</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="available-gpus">0</div>
                    <div class="metric-label">Available GPUs</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Task Queue</h2>
            <div class="metrics" id="task-metrics">
                <div class="metric">
                    <div class="metric-value" id="queue-size">0</div>
                    <div class="metric-label">Queue Size</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="pending-tasks">0</div>
                    <div class="metric-label">Pending</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="running-tasks">0</div>
                    <div class="metric-label">Running</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="completed-tasks">0</div>
                    <div class="metric-label">Completed</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Resource Usage</h2>
            <div class="charts">
                <div class="chart-container">
                    <canvas id="cpu-chart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="gpu-chart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Nodes</h2>
            <div class="node-list" id="node-list">
                <!-- Node cards will be added here -->
            </div>
        </div>
    </div>
    
    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // Chart setup
        const cpuCtx = document.getElementById('cpu-chart').getContext('2d');
        const gpuCtx = document.getElementById('gpu-chart').getContext('2d');
        
        const cpuChart = new Chart(cpuCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Average CPU Usage (%)',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
        
        const gpuChart = new Chart(gpuCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'GPU Utilization (%)',
                    data: [],
                    backgroundColor: 'rgba(54, 162, 235, 0.5)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
        
        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to monitoring server');
        });
        
        socket.on('status_update', function(data) {
            updateMetrics(data);
            updateNodeList(data.cluster);
            updateCharts(data);
        });
        
        function updateMetrics(data) {
            // Update cluster metrics
            const clusterSummary = data.cluster.summary;
            document.getElementById('total-nodes').textContent = clusterSummary.total_nodes;
            document.getElementById('online-nodes').textContent = clusterSummary.online_nodes;
            document.getElementById('total-gpus').textContent = clusterSummary.total_gpus;
            document.getElementById('available-gpus').textContent = clusterSummary.available_gpus;
            
            // Update task metrics
            const taskStatus = data.tasks;
            document.getElementById('queue-size').textContent = taskStatus.queue_size;
            document.getElementById('pending-tasks').textContent = taskStatus.status_counts.pending || 0;
            document.getElementById('running-tasks').textContent = taskStatus.status_counts.running || 0;
            document.getElementById('completed-tasks').textContent = taskStatus.status_counts.completed || 0;
        }
        
        function updateNodeList(clusterData) {
            const nodeList = document.getElementById('node-list');
            nodeList.innerHTML = '';
            
            for (const [nodeId, nodeInfo] of Object.entries(clusterData.nodes)) {
                const nodeCard = document.createElement('div');
                nodeCard.className = `node-card ${nodeInfo.status === 'online' ? 'node-online' : 'node-offline'}`;
                
                nodeCard.innerHTML = `
                    <h4>
                        <span class="status-indicator ${nodeInfo.status === 'online' ? 'status-online' : 'status-offline'}"></span>
                        ${nodeInfo.hostname}
                    </h4>
                    <p><strong>ID:</strong> ${nodeId}</p>
                    <p><strong>GPUs:</strong> ${nodeInfo.num_gpus} (${nodeInfo.free_gpus} free)</p>
                    <p><strong>Tasks:</strong> ${nodeInfo.current_tasks.length}</p>
                    <p><strong>Last Heartbeat:</strong> ${new Date(nodeInfo.last_heartbeat).toLocaleTimeString()}</p>
                `;
                
                nodeList.appendChild(nodeCard);
            }
        }
        
        function updateCharts(data) {
            // Update CPU chart
            const timestamp = new Date(data.timestamp).toLocaleTimeString();
            if (cpuChart.data.labels.length > 20) {
                cpuChart.data.labels.shift();
                cpuChart.data.datasets[0].data.shift();
            }
            cpuChart.data.labels.push(timestamp);
            cpuChart.data.datasets[0].data.push(data.metrics.avg_cpu_percent || 0);
            cpuChart.update();
            
            // Update GPU chart (placeholder data)
            // In real implementation, this would show per-GPU utilization
            gpuChart.data.labels = ['GPU Cluster Average'];
            gpuChart.data.datasets[0].data = [Math.random() * 100]; // Replace with real data
            gpuChart.update();
        }
    </script>
</body>
</html>
'''


def create_dashboard_app(node_manager: NodeManager, 
                       task_scheduler: TaskScheduler) -> Flask:
    """Create Flask app for dashboard"""
    dashboard = MonitoringDashboard(node_manager, task_scheduler)
    
    # Add the HTML template
    @dashboard.app.route('/dashboard')
    def serve_dashboard():
        return DASHBOARD_HTML
    
    return dashboard


# Example usage
if __name__ == "__main__":
    # This would normally be initialized with real node manager and scheduler
    from unittest.mock import Mock
    
    mock_node_manager = Mock()
    mock_task_scheduler = Mock()
    
    dashboard = MonitoringDashboard(mock_node_manager, mock_task_scheduler)
    dashboard.start()