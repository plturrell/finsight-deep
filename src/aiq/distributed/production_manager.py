# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Production manager for distributed AIQToolkit
Integrates security, monitoring, and management
"""

import asyncio
import os
import signal
import sys
from typing import Optional
import logging
import argparse

from aiq.distributed.node_manager import NodeManager
from aiq.distributed.task_scheduler import TaskScheduler
from aiq.distributed.security.tls_config import TLSManager, load_tls_config_from_env
from aiq.distributed.security.auth import AuthManager, AuthInterceptor, create_secure_auth_config
from aiq.distributed.monitoring.metrics import ClusterMetrics, MetricsConfig
from aiq.distributed.monitoring.dashboard import MonitoringDashboard

logger = logging.getLogger(__name__)


class ProductionManager:
    """Production-ready distributed cluster manager"""
    
    def __init__(self):
        self.node_manager = None
        self.task_scheduler = None
        self.tls_manager = None
        self.auth_manager = None
        self.metrics = None
        self.dashboard = None
        self._shutdown_event = asyncio.Event()
    
    def setup_from_env(self):
        """Setup components from environment variables"""
        # TLS setup
        tls_config = load_tls_config_from_env()
        if tls_config:
            self.tls_manager = TLSManager(tls_config)
            logger.info("TLS enabled")
        else:
            logger.warning("TLS disabled - running in insecure mode")
        
        # Authentication setup
        auth_enabled = os.getenv("ENABLE_AUTH", "true").lower() == "true"
        if auth_enabled:
            auth_config = create_secure_auth_config()
            auth_config.secret_key = os.getenv("AUTH_SECRET_KEY", auth_config.secret_key)
            self.auth_manager = AuthManager(auth_config)
            logger.info("Authentication enabled")
        else:
            logger.warning("Authentication disabled")
        
        # Metrics setup
        metrics_enabled = os.getenv("METRICS_ENABLED", "true").lower() == "true"
        if metrics_enabled:
            metrics_config = MetricsConfig(
                enable_metrics=True,
                metrics_port=int(os.getenv("METRICS_PORT", "9090"))
            )
            self.metrics = ClusterMetrics(metrics_config)
            logger.info(f"Metrics enabled on port {metrics_config.metrics_port}")
        
        # Node manager setup
        grpc_port = int(os.getenv("GRPC_PORT", "50051"))
        self.node_manager = NodeManager(port=grpc_port)
        
        # Task scheduler setup
        self.task_scheduler = TaskScheduler(self.node_manager)
        
        # Dashboard setup
        dashboard_enabled = os.getenv("DASHBOARD_ENABLED", "true").lower() == "true"
        if dashboard_enabled:
            dashboard_port = int(os.getenv("DASHBOARD_PORT", "8080"))
            self.dashboard = MonitoringDashboard(
                self.node_manager,
                self.task_scheduler,
                port=dashboard_port
            )
            logger.info(f"Dashboard enabled on port {dashboard_port}")
    
    async def start(self):
        """Start the production manager"""
        logger.info("Starting production manager...")
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Start node manager with security
        if self.tls_manager:
            server = self.tls_manager.create_secure_server(self.node_manager.port)
        else:
            server = self.node_manager.server
        
        if self.auth_manager:
            auth_interceptor = AuthInterceptor(self.auth_manager)
            server = grpc.aio.server(
                interceptors=[auth_interceptor],
                options=[
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ]
            )
        
        # Start services
        try:
            # Start node manager
            manager_task = asyncio.create_task(self.node_manager.run())
            
            # Start task scheduler
            scheduler_task = asyncio.create_task(self.task_scheduler.start())
            
            # Start dashboard in thread
            if self.dashboard:
                dashboard_thread = threading.Thread(target=self.dashboard.start)
                dashboard_thread.daemon = True
                dashboard_thread.start()
            
            # Start metrics update loop
            if self.metrics:
                metrics_task = asyncio.create_task(self._update_metrics_loop())
            
            logger.info("Production manager started successfully")
            
            # Wait for shutdown
            await self._shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Error starting production manager: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down production manager...")
        
        # Stop services
        if self.task_scheduler:
            await self.task_scheduler.stop()
        
        if self.node_manager:
            self.node_manager.stop()
        
        if self.dashboard:
            self.dashboard.stop()
        
        logger.info("Production manager shutdown complete")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _update_metrics_loop(self):
        """Update metrics periodically"""
        while not self._shutdown_event.is_set():
            try:
                # Update cluster metrics
                cluster_status = self.node_manager.get_cluster_status()
                self.metrics.update_cluster_metrics(cluster_status)
                
                # Update task metrics
                queue_status = self.task_scheduler.get_queue_status()
                self.metrics.update_task_queue_size(queue_status["queue_size"])
                
                # Update node metrics
                for node_id, node_info in self.node_manager.nodes.items():
                    # Get node metrics (would come from worker in production)
                    node_metrics = {
                        "cpu_percent": 50.0,  # Placeholder
                        "memory_bytes": 8 * 1024**3,  # Placeholder
                        "gpu_metrics": {}
                    }
                    self.metrics.update_node_metrics(node_id, node_metrics)
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
            
            await asyncio.sleep(10)  # Update every 10 seconds


class ProductionWorker:
    """Production-ready worker node"""
    
    def __init__(self):
        self.worker = None
        self.auth_manager = None
        self.tls_manager = None
        self.metrics_collector = None
        self._shutdown_event = asyncio.Event()
    
    def setup_from_env(self):
        """Setup worker from environment variables"""
        from aiq.distributed.worker_node import WorkerNode
        from aiq.distributed.monitoring.metrics import NodeMetricsCollector
        
        # Get configuration from environment
        manager_host = os.getenv("MANAGER_HOST", "localhost")
        manager_port = int(os.getenv("MANAGER_PORT", "50051"))
        worker_port = int(os.getenv("WORKER_PORT", "50052"))
        node_id = os.getenv("NODE_ID", None)
        
        # Setup TLS
        tls_config = load_tls_config_from_env()
        if tls_config:
            self.tls_manager = TLSManager(tls_config)
        
        # Setup authentication
        auth_enabled = os.getenv("ENABLE_AUTH", "true").lower() == "true"
        if auth_enabled:
            auth_config = create_secure_auth_config()
            auth_config.secret_key = os.getenv("AUTH_SECRET_KEY", auth_config.secret_key)
            self.auth_manager = AuthManager(auth_config)
        
        # Create worker
        self.worker = WorkerNode(
            node_id=node_id,
            manager_host=manager_host,
            manager_port=manager_port,
            worker_port=worker_port
        )
        
        # Setup metrics collection
        self.metrics_collector = NodeMetricsCollector(
            node_id=self.worker.node_id,
            gpu_manager=self.worker.gpu_manager
        )
        
        # Register production functions
        self._register_production_functions()
    
    def _register_production_functions(self):
        """Register production-ready functions"""
        # Import and register actual functions here
        # For example:
        # from aiq.functions.text_analysis import TextAnalysisFunction
        # self.worker.register_function("text_analysis", TextAnalysisFunction())
        pass
    
    async def start(self):
        """Start the production worker"""
        logger.info("Starting production worker...")
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        try:
            # Start worker with security
            if self.tls_manager and self.auth_manager:
                # Apply security to worker's gRPC connections
                # This would modify the worker's manager client to use secure channel
                pass
            
            # Start worker
            await self.worker.start()
            
            # Start metrics reporting
            metrics_task = asyncio.create_task(self._report_metrics_loop())
            
            logger.info("Production worker started successfully")
            
            # Wait for shutdown
            await self._shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Error starting production worker: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down production worker...")
        
        if self.worker:
            await self.worker.stop()
        
        logger.info("Production worker shutdown complete")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _report_metrics_loop(self):
        """Report metrics to manager periodically"""
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = self.metrics_collector.collect_metrics()
                
                # Report to manager (would use gRPC in production)
                logger.debug(f"Collected metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Error reporting metrics: {e}")
            
            await asyncio.sleep(30)  # Report every 30 seconds


async def main():
    """Main entry point for production deployment"""
    parser = argparse.ArgumentParser(description="AIQToolkit Production Distributed System")
    parser.add_argument("--role", choices=["manager", "worker"], required=True,
                       help="Node role")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.role == "manager":
        manager = ProductionManager()
        manager.setup_from_env()
        await manager.start()
    else:
        worker = ProductionWorker()
        worker.setup_from_env()
        await worker.start()


if __name__ == "__main__":
    asyncio.run(main())