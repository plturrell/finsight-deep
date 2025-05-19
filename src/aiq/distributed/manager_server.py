# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed Manager Server Entry Point"""

import os
import sys
import asyncio
import logging
import argparse
from typing import Optional

from aiq.distributed.node_manager import NodeManager
from aiq.distributed.servers.manager_grpc_server import ManagerGRPCServer
from aiq.distributed.security.auth import AuthManager, AuthConfig
from aiq.distributed.monitoring.prometheus_metrics import PrometheusMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DistributedManagerServer:
    """Main distributed manager server"""
    
    def __init__(self, port: int = 50051, metrics_port: int = 9090, dashboard_port: int = 8080):
        self.port = port
        self.metrics_port = metrics_port
        self.dashboard_port = dashboard_port
        
        # Initialize components
        self.node_manager = NodeManager()
        self.auth_manager = AuthManager(AuthConfig(
            jwt_secret=os.environ.get('JWT_SECRET', 'default-secret'),
            tls_enabled=os.environ.get('ENABLE_TLS', 'true').lower() == 'true'
        ))
        
        # Initialize monitoring
        self.metrics = PrometheusMetrics(port=metrics_port)
        
        # Initialize gRPC server
        self.grpc_server = ManagerGRPCServer(
            node_manager=self.node_manager,
            auth_manager=self.auth_manager,
            port=port
        )
    
    async def start(self):
        """Start the distributed manager server"""
        try:
            logger.info(f"Starting distributed manager server on port {self.port}")
            
            # Start metrics server
            self.metrics.start()
            logger.info(f"Metrics server started on port {self.metrics_port}")
            
            # Start gRPC server
            await self.grpc_server.start()
            logger.info(f"gRPC server started on port {self.port}")
            
            # Keep server running
            await asyncio.get_event_loop().create_future()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the server gracefully"""
        logger.info("Shutting down distributed manager server...")
        await self.grpc_server.stop()
        self.metrics.stop()
        logger.info("Server shutdown complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AIQToolkit Distributed Manager Server")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--metrics-port", type=int, default=9090, help="Metrics server port")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Dashboard port")
    
    args = parser.parse_args()
    
    # Create and start server
    server = DistributedManagerServer(
        port=args.port,
        metrics_port=args.metrics_port,
        dashboard_port=args.dashboard_port
    )
    
    # Run server
    asyncio.run(server.start())

if __name__ == "__main__":
    main()