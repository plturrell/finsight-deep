# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Metrics collection for distributed AIQToolkit
"""

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import logging
import time
import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    update_interval: float = 5.0  # seconds
    namespace: str = "aiqtoolkit"
    subsystem: str = "distributed"


class ClusterMetrics:
    """Prometheus metrics for distributed cluster"""
    
    def __init__(self, config: MetricsConfig = None):
        self.config = config or MetricsConfig()
        self.enabled = self.config.enable_metrics
        
        if not self.enabled:
            return
        
        # Define metrics
        self._define_metrics()
        
        # Start metrics server
        if self.config.metrics_port > 0:
            start_http_server(self.config.metrics_port)
            logger.info(f"Metrics server started on port {self.config.metrics_port}")
    
    def _define_metrics(self):
        """Define Prometheus metrics"""
        ns = self.config.namespace
        sub = self.config.subsystem
        
        # Cluster metrics
        self.cluster_info = Info(
            f"{ns}_cluster_info",
            "Cluster information",
            namespace=ns
        )
        
        self.nodes_total = Gauge(
            "nodes_total",
            "Total number of nodes in cluster",
            namespace=ns,
            subsystem=sub
        )
        
        self.nodes_online = Gauge(
            "nodes_online",
            "Number of online nodes",
            namespace=ns,
            subsystem=sub
        )
        
        self.gpus_total = Gauge(
            "gpus_total",
            "Total GPUs in cluster",
            namespace=ns,
            subsystem=sub
        )
        
        self.gpus_available = Gauge(
            "gpus_available",
            "Available GPUs in cluster",
            namespace=ns,
            subsystem=sub
        )
        
        # Task metrics
        self.tasks_submitted = Counter(
            "tasks_submitted_total",
            "Total tasks submitted",
            ["function_name"],
            namespace=ns,
            subsystem=sub
        )
        
        self.tasks_completed = Counter(
            "tasks_completed_total",
            "Total tasks completed",
            ["function_name", "status"],
            namespace=ns,
            subsystem=sub
        )
        
        self.task_duration = Histogram(
            "task_duration_seconds",
            "Task execution duration",
            ["function_name"],
            namespace=ns,
            subsystem=sub,
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
        )
        
        self.task_queue_size = Gauge(
            "task_queue_size",
            "Number of tasks in queue",
            namespace=ns,
            subsystem=sub
        )
        
        # Node metrics
        self.node_cpu_usage = Gauge(
            "node_cpu_usage_percent",
            "Node CPU usage percentage",
            ["node_id"],
            namespace=ns,
            subsystem=sub
        )
        
        self.node_memory_usage = Gauge(
            "node_memory_usage_bytes",
            "Node memory usage in bytes",
            ["node_id"],
            namespace=ns,
            subsystem=sub
        )
        
        self.node_gpu_memory = Gauge(
            "node_gpu_memory_bytes",
            "GPU memory usage per device",
            ["node_id", "gpu_id"],
            namespace=ns,
            subsystem=sub
        )
        
        self.node_gpu_utilization = Gauge(
            "node_gpu_utilization_percent",
            "GPU utilization percentage",
            ["node_id", "gpu_id"],
            namespace=ns,
            subsystem=sub
        )
        
        # Communication metrics
        self.grpc_requests = Counter(
            "grpc_requests_total",
            "Total gRPC requests",
            ["method", "status"],
            namespace=ns,
            subsystem=sub
        )
        
        self.grpc_request_duration = Histogram(
            "grpc_request_duration_seconds",
            "gRPC request duration",
            ["method"],
            namespace=ns,
            subsystem=sub
        )
        
        # Error metrics
        self.errors_total = Counter(
            "errors_total",
            "Total errors",
            ["type", "node_id"],
            namespace=ns,
            subsystem=sub
        )
    
    def update_cluster_metrics(self, cluster_status: Dict[str, Any]):
        """Update cluster-wide metrics"""
        if not self.enabled:
            return
            
        summary = cluster_status.get("summary", {})
        
        self.nodes_total.set(summary.get("total_nodes", 0))
        self.nodes_online.set(summary.get("online_nodes", 0))
        self.gpus_total.set(summary.get("total_gpus", 0))
        self.gpus_available.set(summary.get("available_gpus", 0))
        
        # Update cluster info
        self.cluster_info.info({
            "cluster_id": cluster_status.get("cluster_id", "unknown"),
            "timestamp": cluster_status.get("timestamp", "")
        })
    
    def record_task_submitted(self, function_name: str):
        """Record task submission"""
        if not self.enabled:
            return
        self.tasks_submitted.labels(function_name=function_name).inc()
    
    def record_task_completed(self, function_name: str, status: str, duration: float):
        """Record task completion"""
        if not self.enabled:
            return
            
        self.tasks_completed.labels(
            function_name=function_name,
            status=status
        ).inc()
        
        self.task_duration.labels(function_name=function_name).observe(duration)
    
    def update_task_queue_size(self, size: int):
        """Update task queue size"""
        if not self.enabled:
            return
        self.task_queue_size.set(size)
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Update metrics for a specific node"""
        if not self.enabled:
            return
            
        # CPU and memory
        cpu_percent = metrics.get("cpu_percent", 0)
        memory_bytes = metrics.get("memory_bytes", 0)
        
        self.node_cpu_usage.labels(node_id=node_id).set(cpu_percent)
        self.node_memory_usage.labels(node_id=node_id).set(memory_bytes)
        
        # GPU metrics
        gpu_metrics = metrics.get("gpu_metrics", {})
        for gpu_id, gpu_data in gpu_metrics.items():
            self.node_gpu_memory.labels(
                node_id=node_id, 
                gpu_id=str(gpu_id)
            ).set(gpu_data.get("memory_used", 0))
            
            self.node_gpu_utilization.labels(
                node_id=node_id,
                gpu_id=str(gpu_id)
            ).set(gpu_data.get("utilization", 0))
    
    def record_grpc_request(self, method: str, status: str, duration: float):
        """Record gRPC request metrics"""
        if not self.enabled:
            return
            
        self.grpc_requests.labels(method=method, status=status).inc()
        self.grpc_request_duration.labels(method=method).observe(duration)
    
    def record_error(self, error_type: str, node_id: str):
        """Record error occurrence"""
        if not self.enabled:
            return
        self.errors_total.labels(type=error_type, node_id=node_id).inc()


class NodeMetricsCollector:
    """Collects metrics from a single node"""
    
    def __init__(self, node_id: str, gpu_manager=None):
        self.node_id = node_id
        self.gpu_manager = gpu_manager
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current node metrics"""
        metrics = {
            "timestamp": time.time(),
            "node_id": self.node_id,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_bytes": psutil.virtual_memory().used,
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "network_io": self._get_network_io(),
            "gpu_metrics": self._collect_gpu_metrics()
        }
        
        return metrics
    
    def _get_network_io(self) -> Dict[str, int]:
        """Get network I/O statistics"""
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
    
    def _collect_gpu_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Collect GPU metrics if available"""
        gpu_metrics = {}
        
        if not self.gpu_manager or not torch.cuda.is_available():
            return gpu_metrics
            
        try:
            gpu_infos = self.gpu_manager.get_gpu_info()
            
            for gpu_info in gpu_infos:
                gpu_metrics[gpu_info.device_id] = {
                    "name": gpu_info.name,
                    "memory_total": gpu_info.memory_total,
                    "memory_used": gpu_info.memory_used,
                    "memory_free": gpu_info.memory_free,
                    "utilization": gpu_info.utilization,
                    "temperature": gpu_info.temperature,
                    "power_draw": gpu_info.power_draw
                }
                
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
            
        return gpu_metrics


class MetricsAggregator:
    """Aggregates metrics from multiple nodes"""
    
    def __init__(self):
        self.node_metrics: Dict[str, Dict[str, Any]] = {}
        self.task_metrics: Dict[str, Dict[str, Any]] = {}
        
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Update metrics for a node"""
        self.node_metrics[node_id] = metrics
        
    def update_task_metrics(self, task_id: str, metrics: Dict[str, Any]):
        """Update metrics for a task"""
        self.task_metrics[task_id] = metrics
        
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get aggregated cluster metrics"""
        total_cpu = 0
        total_memory = 0
        total_gpus = 0
        available_gpus = 0
        
        for node_metrics in self.node_metrics.values():
            total_cpu += node_metrics.get("cpu_percent", 0)
            total_memory += node_metrics.get("memory_bytes", 0)
            
            gpu_metrics = node_metrics.get("gpu_metrics", {})
            total_gpus += len(gpu_metrics)
            
            # Count available GPUs (utilization < 80%)
            for gpu_data in gpu_metrics.values():
                if gpu_data.get("utilization", 0) < 80:
                    available_gpus += 1
        
        return {
            "timestamp": time.time(),
            "nodes_count": len(self.node_metrics),
            "avg_cpu_percent": total_cpu / max(len(self.node_metrics), 1),
            "total_memory_bytes": total_memory,
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "task_count": len(self.task_metrics)
        }
    
    def get_node_rankings(self) -> List[Tuple[str, float]]:
        """Get nodes ranked by available resources"""
        rankings = []
        
        for node_id, metrics in self.node_metrics.items():
            # Simple scoring based on available resources
            cpu_available = 100 - metrics.get("cpu_percent", 100)
            memory_available = 100 - metrics.get("memory_percent", 100)
            
            # Factor in GPU availability
            gpu_score = 0
            gpu_metrics = metrics.get("gpu_metrics", {})
            for gpu_data in gpu_metrics.values():
                gpu_available = 100 - gpu_data.get("utilization", 100)
                gpu_score += gpu_available
                
            # Combined score
            score = (cpu_available + memory_available + gpu_score) / 3
            rankings.append((node_id, score))
        
        # Sort by score (higher is better)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


# Example usage
if __name__ == "__main__":
    # Create metrics collector
    config = MetricsConfig(enable_metrics=True, metrics_port=9090)
    metrics = ClusterMetrics(config)
    
    # Simulate some metrics
    metrics.record_task_submitted("text_analysis")
    metrics.record_task_completed("text_analysis", "success", 2.5)
    
    metrics.update_cluster_metrics({
        "summary": {
            "total_nodes": 5,
            "online_nodes": 4,
            "total_gpus": 20,
            "available_gpus": 15
        }
    })
    
    print(f"Metrics server running on port {config.metrics_port}")
    print("Visit http://localhost:9090/metrics to see metrics")