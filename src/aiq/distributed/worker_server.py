# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed Worker Server Entry Point"""

import os
import sys
import asyncio
import logging
import argparse
import socket
from typing import Optional

import torch

from aiq.distributed.worker import DistributedWorker
from aiq.distributed.servers.worker_grpc_server import WorkerGRPCServer
from aiq.distributed.security.auth import AuthManager, AuthConfig
from aiq.gpu.multi_gpu_manager import MultiGPUManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DistributedWorkerServer:
    """Main distributed worker server"""
    
    def __init__(self, 
                 manager_host: str = "localhost",
                 manager_port: int = 50051,
                 worker_port: int = 50052,
                 worker_id: Optional[str] = None):
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.worker_port = worker_port
        self.worker_id = worker_id or f"worker-{socket.gethostname()}-{os.getpid()}"
        
        # Initialize GPU manager
        self.gpu_manager = MultiGPUManager()
        
        # Initialize worker
        self.worker = DistributedWorker(
            worker_id=self.worker_id,
            gpu_manager=self.gpu_manager
        )
        
        # Initialize auth
        self.auth_manager = AuthManager(AuthConfig(
            jwt_secret=os.environ.get('JWT_SECRET', 'default-secret'),
            tls_enabled=os.environ.get('ENABLE_TLS', 'true').lower() == 'true'
        ))
        
        # Initialize gRPC server
        self.grpc_server = WorkerGRPCServer(
            worker=self.worker,
            auth_manager=self.auth_manager,
            port=worker_port
        )
    
    async def start(self):
        """Start the distributed worker server"""
        try:
            logger.info(f"Starting distributed worker {self.worker_id}")
            logger.info(f"Manager: {self.manager_host}:{self.manager_port}")
            logger.info(f"Worker port: {self.worker_port}")
            
            # Log GPU information
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"GPUs available: {gpu_count}")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    logger.info(f"  GPU {i}: {gpu_name}")
            else:
                logger.warning("No GPUs available, running on CPU")
            
            # Connect to manager
            await self.worker.connect_to_manager(self.manager_host, self.manager_port)
            logger.info("Connected to manager")
            
            # Start gRPC server
            await self.grpc_server.start()
            logger.info(f"Worker gRPC server started on port {self.worker_port}")
            
            # Keep server running
            await asyncio.get_event_loop().create_future()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Error starting worker: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the worker gracefully"""
        logger.info("Shutting down distributed worker...")
        await self.worker.disconnect()
        await self.grpc_server.stop()
        logger.info("Worker shutdown complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AIQToolkit Distributed Worker Server")
    parser.add_argument("--manager-host", type=str, default="localhost", help="Manager host")
    parser.add_argument("--manager-port", type=int, default=50051, help="Manager port")
    parser.add_argument("--worker-port", type=int, default=50052, help="Worker port")
    parser.add_argument("--worker-id", type=str, help="Worker ID (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Create and start worker
    worker = DistributedWorkerServer(
        manager_host=args.manager_host,
        manager_port=args.manager_port,
        worker_port=args.worker_port,
        worker_id=args.worker_id
    )
    
    # Run worker
    asyncio.run(worker.start())

if __name__ == "__main__":
    main()