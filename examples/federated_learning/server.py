"""Federated Learning Server Example"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from typing import Dict, Any

from aiq.distributed.federated.federated_learning import (
    FederatedLearningServer,
    FederatedConfig,
    AggregationStrategy,
    ClientSelection
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageClassificationModel(nn.Module):
    """Example model for federated learning"""
    
    def __init__(self, input_size: int = 784, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class FederatedServer:
    """Federated learning server application"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.fl_config = self._create_fl_config(self.config['federated_config'])
        self.server = FederatedLearningServer(self.fl_config)
        self.model = None
        self._running = False
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _create_fl_config(self, config_dict: Dict[str, Any]) -> FederatedConfig:
        """Create FederatedConfig from dictionary"""
        # Map string values to enums
        if 'aggregation_strategy' in config_dict:
            config_dict['aggregation_strategy'] = AggregationStrategy[
                config_dict['aggregation_strategy'].upper()
            ]
            
        if 'client_selection' in config_dict:
            config_dict['client_selection'] = ClientSelection[
                config_dict['client_selection'].upper()
            ]
            
        return FederatedConfig(**config_dict)
        
    async def start(self):
        """Start the federated learning server"""
        logger.info("Starting Federated Learning Server")
        
        # Initialize model
        model_config = self.config.get('model', {})
        self.model = ImageClassificationModel(
            input_size=model_config.get('input_size', 784),
            num_classes=model_config.get('num_classes', 10)
        )
        
        await self.server.initialize_global_model(self.model)
        
        # Start API server for client connections
        api_config = self.config.get('api', {})
        api_host = api_config.get('host', '0.0.0.0')
        api_port = api_config.get('port', 8080)
        
        # Run both API server and training loop
        self._running = True
        
        await asyncio.gather(
            self._run_api_server(api_host, api_port),
            self._run_training_loop(),
            self._monitor_progress()
        )
        
    async def _run_api_server(self, host: str, port: int):
        """Run API server for client connections"""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_post('/register', self._handle_registration)
        app.router.add_get('/model/{round_number}', self._handle_model_request)
        app.router.add_post('/update/{round_number}', self._handle_update)
        app.router.add_get('/status', self._handle_status)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        
        logger.info(f"API server listening on {host}:{port}")
        await site.start()
        
        while self._running:
            await asyncio.sleep(1)
            
    async def _run_training_loop(self):
        """Run the federated learning training loop"""
        try:
            await self.server.run_training()
        except Exception as e:
            logger.error(f"Training error: {e}")
        finally:
            self._running = False
            
    async def _monitor_progress(self):
        """Monitor and log training progress"""
        while self._running:
            if self.server.metrics_history:
                latest_metrics = self.server.metrics_history[-1]
                logger.info(f"Round {self.server.current_round}: {latest_metrics}")
                
                # Check privacy budget if using differential privacy
                if self.server.privacy_manager:
                    epsilon, delta = self.server.privacy_manager.get_privacy_spent()
                    logger.info(f"Privacy spent: ε={epsilon:.2f}, δ={delta:.2e}")
                    
            await asyncio.sleep(30)  # Log every 30 seconds
            
    async def _handle_registration(self, request):
        """Handle client registration"""
        try:
            data = await request.json()
            client_id = await self.server.register_client(data)
            
            return web.json_response({
                "status": "success",
                "client_id": client_id
            })
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=400)
            
    async def _handle_model_request(self, request):
        """Handle model download request"""
        round_number = int(request.match_info['round_number'])
        
        if round_number != self.server.current_round:
            return web.json_response({
                "status": "error",
                "message": "Invalid round number"
            }, status=400)
            
        # Serialize model state
        model_state = self.server.global_model.state_dict()
        
        # Convert to transferable format
        serialized = {}
        for name, tensor in model_state.items():
            serialized[name] = {
                "data": tensor.cpu().numpy().tolist(),
                "shape": list(tensor.shape)
            }
            
        return web.json_response({
            "status": "success",
            "round": round_number,
            "model": serialized
        })
        
    async def _handle_update(self, request):
        """Handle client update submission"""
        try:
            data = await request.json()
            round_number = int(request.match_info['round_number'])
            
            # Deserialize model updates
            model_updates = {}
            for name, tensor_data in data['model_updates'].items():
                tensor = torch.tensor(tensor_data['data'])
                tensor = tensor.reshape(tensor_data['shape'])
                model_updates[name] = tensor
                
            # Create client update
            from aiq.distributed.federated.federated_learning import ClientUpdate
            update = ClientUpdate(
                client_id=data['client_id'],
                round_number=round_number,
                model_updates=model_updates,
                metrics=data['metrics'],
                sample_count=data['sample_count'],
                computation_time=data['computation_time'],
                privacy_noise_added=data.get('privacy_noise_added', False)
            )
            
            # Add to server's update queue
            self.server.round_updates.append(update)
            
            return web.json_response({"status": "success"})
            
        except Exception as e:
            logger.error(f"Update error: {e}")
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=400)
            
    async def _handle_status(self, request):
        """Handle status request"""
        return web.json_response({
            "status": "running",
            "current_round": self.server.current_round,
            "total_rounds": self.server.config.rounds,
            "registered_clients": len(self.server.client_registry),
            "metrics": self.server.metrics_history[-1] if self.server.metrics_history else None
        })
        
    def stop(self):
        """Stop the server"""
        logger.info("Stopping Federated Learning Server")
        self._running = False


async def main():
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/federated_server.yml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    server = FederatedServer(args.config)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received signal, shutting down...")
        server.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await server.start()
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())