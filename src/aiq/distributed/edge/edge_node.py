"""Edge computing node implementation for AIQToolkit distributed processing"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import threading
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import websockets
import aiohttp
import yaml

from aiq.data_models.config import AIQConfig
from aiq.runtime.runner import AIQRunner
from aiq.distributed.node_manager import NodeInfo
from aiq.distributed.security.auth import AuthManager, AuthConfig

logger = logging.getLogger(__name__)


class EdgeNodeStatus(Enum):
    """Edge node connectivity status"""
    ONLINE = "online"
    OFFLINE = "offline"  
    SYNCING = "syncing"
    ERROR = "error"


class EdgeMode(Enum):
    """Edge node operation mode"""
    ALWAYS_CONNECTED = "always_connected"
    INTERMITTENT = "intermittent"
    OFFLINE_FIRST = "offline_first"


@dataclass
class EdgeNodeConfig:
    """Configuration for edge node"""
    node_id: str
    device_type: str  # "mobile", "iot", "embedded", "workstation"
    mode: EdgeMode
    sync_interval: int = 3600  # seconds
    cache_size_mb: int = 100
    offline_queue_size: int = 1000
    bandwidth_limit_mbps: Optional[float] = None
    power_mode: str = "balanced"  # "balanced", "low_power", "performance"
    local_model_path: Optional[Path] = None
    security_level: str = "standard"  # "standard", "high", "maximum"


@dataclass
class ModelSyncConfig:
    """Configuration for model synchronization"""
    sync_frequency: int = 3600  # seconds
    compression_enabled: bool = True
    differential_sync: bool = True
    encryption_enabled: bool = True
    chunk_size_mb: int = 10
    retry_attempts: int = 3


class EdgeNode:
    """Edge computing node for distributed AIQToolkit"""
    
    def __init__(self, config: EdgeNodeConfig):
        self.config = config
        self.status = EdgeNodeStatus.OFFLINE
        self.local_models: Dict[str, nn.Module] = {}
        self.offline_queue: List[Dict[str, Any]] = []
        self.sync_manager = ModelSyncManager(config)
        self.cache_manager = EdgeCacheManager(config.cache_size_mb)
        self.power_manager = PowerManager(config.power_mode)
        self.auth_manager = AuthManager(AuthConfig(secret_key="dummy_key"))
        self._running = False
        self._sync_thread = None
        self._manager_url: Optional[str] = None
        
    async def connect_to_manager(self, manager_url: str, auth_token: str) -> bool:
        """Connect to central manager node"""
        try:
            self._manager_url = manager_url
            
            # Verify authentication
            if not self.auth_manager.verify_token(auth_token):
                logger.error("Invalid authentication token")
                return False
                
            # Test connection
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {auth_token}"}
                async with session.get(f"{manager_url}/health", headers=headers) as resp:
                    if resp.status == 200:
                        self.status = EdgeNodeStatus.ONLINE
                        logger.info(f"Connected to manager at {manager_url}")
                        return True
                        
        except Exception as e:
            logger.error(f"Failed to connect to manager: {e}")
            self.status = EdgeNodeStatus.ERROR
            
        return False
        
    async def run_offline_workflow(self, workflow_config: AIQConfig) -> Dict[str, Any]:
        """Run workflow locally on edge node"""
        try:
            # Check if models are available locally
            required_models = self._extract_required_models(workflow_config)
            for model_id in required_models:
                if model_id not in self.local_models:
                    # Try to load from cache
                    if not await self._load_cached_model(model_id):
                        return {
                            "status": "error",
                            "message": f"Model {model_id} not available offline"
                        }
            
            # Run workflow locally
            runner = AIQRunner()
            result = await runner.run(workflow_config)
            
            # Queue result for sync if in offline mode
            if self.status == EdgeNodeStatus.OFFLINE:
                self._queue_for_sync({
                    "type": "workflow_result",
                    "workflow_id": workflow_config.id,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to run offline workflow: {e}")
            return {"status": "error", "message": str(e)}
            
    async def sync_with_manager(self) -> bool:
        """Synchronize with central manager"""
        if self.status != EdgeNodeStatus.ONLINE:
            return False
            
        try:
            self.status = EdgeNodeStatus.SYNCING
            
            # Upload queued results
            await self._upload_queued_results()
            
            # Sync models
            await self.sync_manager.sync_models(self._manager_url)
            
            # Download updates
            await self._download_updates()
            
            self.status = EdgeNodeStatus.ONLINE
            return True
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.status = EdgeNodeStatus.ERROR
            return False
            
    async def deploy_model(self, model_id: str, model_data: bytes) -> bool:
        """Deploy model to edge node"""
        try:
            # Save model locally
            model_path = self.config.local_model_path / f"{model_id}.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(model_path, 'wb') as f:
                f.write(model_data)
                
            # Load model
            model = torch.load(model_path, map_location=self._get_device())
            self.local_models[model_id] = model
            
            # Update cache
            self.cache_manager.cache_model(model_id, model_data)
            
            logger.info(f"Deployed model {model_id} to edge node")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False
            
    def start_background_sync(self):
        """Start background synchronization thread"""
        if self._sync_thread is not None:
            return
            
        self._running = True
        self._sync_thread = threading.Thread(target=self._background_sync_loop)
        self._sync_thread.start()
        
    def stop_background_sync(self):
        """Stop background synchronization"""
        self._running = False
        if self._sync_thread:
            self._sync_thread.join()
            self._sync_thread = None
            
    def _background_sync_loop(self):
        """Background sync loop"""
        while self._running:
            try:
                if self.config.mode != EdgeMode.OFFLINE_FIRST:
                    asyncio.run(self.sync_with_manager())
                    
            except Exception as e:
                logger.error(f"Background sync error: {e}")
                
            time.sleep(self.config.sync_interval)
            
    def _queue_for_sync(self, data: Dict[str, Any]):
        """Queue data for later synchronization"""
        if len(self.offline_queue) >= self.config.offline_queue_size:
            # Remove oldest items
            self.offline_queue.pop(0)
            
        self.offline_queue.append(data)
        
    async def _upload_queued_results(self):
        """Upload queued results to manager"""
        if not self.offline_queue:
            return
            
        async with aiohttp.ClientSession() as session:
            for item in self.offline_queue[:]:  # Copy to avoid modification during iteration
                try:
                    headers = {"Authorization": f"Bearer {self.jwt_manager.current_token}"}
                    async with session.post(
                        f"{self._manager_url}/edge/sync",
                        json=item,
                        headers=headers
                    ) as resp:
                        if resp.status == 200:
                            self.offline_queue.remove(item)
                        else:
                            logger.warning(f"Failed to sync item: {resp.status}")
                            
                except Exception as e:
                    logger.error(f"Failed to upload queued item: {e}")
                    
    async def _download_updates(self):
        """Download updates from manager"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.jwt_manager.current_token}"}
                async with session.get(
                    f"{self._manager_url}/edge/updates/{self.config.node_id}",
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        updates = await resp.json()
                        await self._apply_updates(updates)
                        
        except Exception as e:
            logger.error(f"Failed to download updates: {e}")
            
    async def _apply_updates(self, updates: List[Dict[str, Any]]):
        """Apply downloaded updates"""
        for update in updates:
            try:
                if update["type"] == "model":
                    await self.deploy_model(update["model_id"], update["model_data"])
                elif update["type"] == "config":
                    self._update_config(update["config"])
                elif update["type"] == "workflow":
                    self._update_workflow(update["workflow"])
                    
            except Exception as e:
                logger.error(f"Failed to apply update: {e}")
                
    def _get_device(self) -> torch.device:
        """Get appropriate device for edge node"""
        if torch.cuda.is_available() and self.config.power_mode != "low_power":
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
            
    def _extract_required_models(self, workflow_config: AIQConfig) -> List[str]:
        """Extract required model IDs from workflow config"""
        models = []
        for component in workflow_config.components:
            if hasattr(component, "model_id"):
                models.append(component.model_id)
        return models
        
    async def _load_cached_model(self, model_id: str) -> bool:
        """Load model from cache"""
        try:
            model_data = self.cache_manager.get_model(model_id)
            if model_data:
                model = torch.load(model_data, map_location=self._get_device())
                self.local_models[model_id] = model
                return True
        except Exception as e:
            logger.error(f"Failed to load cached model: {e}")
        return False


class ModelSyncManager:
    """Manages model synchronization for edge nodes"""
    
    def __init__(self, edge_config: EdgeNodeConfig):
        self.edge_config = edge_config
        self.sync_config = ModelSyncConfig()
        self.model_versions: Dict[str, str] = {}
        
    async def sync_models(self, manager_url: str) -> bool:
        """Synchronize models with manager"""
        try:
            # Get model manifest from manager
            manifest = await self._get_model_manifest(manager_url)
            
            # Compare with local versions
            updates_needed = self._check_updates_needed(manifest)
            
            # Download updated models
            for model_id in updates_needed:
                await self._download_model(manager_url, model_id)
                
            return True
            
        except Exception as e:
            logger.error(f"Model sync failed: {e}")
            return False
            
    async def _get_model_manifest(self, manager_url: str) -> Dict[str, str]:
        """Get model manifest from manager"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{manager_url}/models/manifest") as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    raise Exception(f"Failed to get manifest: {resp.status}")
                    
    def _check_updates_needed(self, manifest: Dict[str, str]) -> List[str]:
        """Check which models need updates"""
        updates = []
        for model_id, version in manifest.items():
            if model_id not in self.model_versions or \
               self.model_versions[model_id] != version:
                updates.append(model_id)
        return updates
        
    async def _download_model(self, manager_url: str, model_id: str):
        """Download model from manager"""
        try:
            if self.sync_config.differential_sync:
                # Download only deltas
                await self._download_model_delta(manager_url, model_id)
            else:
                # Download full model
                await self._download_full_model(manager_url, model_id)
                
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise
            
    async def _download_model_delta(self, manager_url: str, model_id: str):
        """Download model delta for differential sync"""
        # Implementation for differential model sync
        # This would download only the changed parameters
        pass
        
    async def _download_full_model(self, manager_url: str, model_id: str):
        """Download full model"""
        # Implementation for full model download
        pass


class EdgeCacheManager:
    """Manages local cache for edge node"""
    
    def __init__(self, cache_size_mb: int):
        self.cache_size_mb = cache_size_mb
        self.cache_path = Path.home() / ".aiq" / "edge_cache"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_index: Dict[str, Dict[str, Any]] = {}
        self._load_cache_index()
        
    def cache_model(self, model_id: str, model_data: bytes):
        """Cache model locally"""
        try:
            # Check cache size
            if self._get_cache_size() + len(model_data) > self.cache_size_mb * 1024 * 1024:
                self._evict_oldest()
                
            # Save to cache
            cache_file = self.cache_path / f"{model_id}.cache"
            with open(cache_file, 'wb') as f:
                f.write(model_data)
                
            # Update index
            self.cache_index[model_id] = {
                "size": len(model_data),
                "timestamp": datetime.now().isoformat(),
                "access_count": 0
            }
            self._save_cache_index()
            
        except Exception as e:
            logger.error(f"Failed to cache model: {e}")
            
    def get_model(self, model_id: str) -> Optional[bytes]:
        """Get model from cache"""
        try:
            cache_file = self.cache_path / f"{model_id}.cache"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = f.read()
                    
                # Update access count
                if model_id in self.cache_index:
                    self.cache_index[model_id]["access_count"] += 1
                    self._save_cache_index()
                    
                return data
                
        except Exception as e:
            logger.error(f"Failed to get cached model: {e}")
            
        return None
        
    def _get_cache_size(self) -> int:
        """Get current cache size in bytes"""
        total_size = 0
        for item in self.cache_index.values():
            total_size += item["size"]
        return total_size
        
    def _evict_oldest(self):
        """Evict oldest cached items"""
        # Simple LRU eviction
        if not self.cache_index:
            return
            
        oldest_id = min(
            self.cache_index.keys(),
            key=lambda k: self.cache_index[k]["timestamp"]
        )
        
        # Remove from cache
        cache_file = self.cache_path / f"{oldest_id}.cache"
        if cache_file.exists():
            cache_file.unlink()
            
        del self.cache_index[oldest_id]
        self._save_cache_index()
        
    def _load_cache_index(self):
        """Load cache index from disk"""
        index_file = self.cache_path / "index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache index: {e}")
                self.cache_index = {}
                
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_file = self.cache_path / "index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")


class PowerManager:
    """Manages power consumption for edge devices"""
    
    def __init__(self, power_mode: str):
        self.power_mode = power_mode
        self.cpu_limit = self._get_cpu_limit()
        self.gpu_limit = self._get_gpu_limit()
        
    def apply_power_limits(self):
        """Apply power consumption limits"""
        try:
            if self.power_mode == "low_power":
                # Limit CPU frequency
                if hasattr(os, 'sched_setaffinity'):
                    os.sched_setaffinity(0, {0, 1})  # Use only 2 cores
                    
                # Set GPU power limit if available
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)
                    # This would require nvidia-ml-py for actual implementation
                    
        except Exception as e:
            logger.warning(f"Failed to apply power limits: {e}")
            
    def _get_cpu_limit(self) -> float:
        """Get CPU utilization limit based on power mode"""
        limits = {
            "low_power": 0.25,
            "balanced": 0.75,
            "performance": 1.0
        }
        return limits.get(self.power_mode, 0.75)
        
    def _get_gpu_limit(self) -> float:
        """Get GPU utilization limit based on power mode"""
        limits = {
            "low_power": 0.3,
            "balanced": 0.7,
            "performance": 1.0
        }
        return limits.get(self.power_mode, 0.7)