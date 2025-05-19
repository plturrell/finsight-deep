"""
Load Balancer and Scalability Manager for Digital Human System
Handles multi-instance deployment, request routing, and auto-scaling
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import hashlib
import aiohttp
import redis.asyncio as aioredis
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from aiq.utils.debugging_utils import log_function_call


logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE = "adaptive"  # ML-based routing


@dataclass
class Instance:
    """Represents a Digital Human service instance"""
    instance_id: str
    host: str
    port: int
    weight: float = 1.0
    active_connections: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    response_time_avg: float = 0.0
    health_score: float = 1.0
    last_health_check: datetime = None
    is_healthy: bool = True
    zone: str = "default"
    capabilities: Set[str] = None


class LoadBalancer:
    """
    Intelligent load balancer for Digital Human system
    Features:
    - Multiple load balancing strategies
    - Health checking
    - Auto-scaling
    - Session affinity
    - Circuit breaker
    - Request routing based on capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy = LoadBalancingStrategy(
            config.get("strategy", LoadBalancingStrategy.ADAPTIVE)
        )
        
        # Instance management
        self.instances: Dict[str, Instance] = {}
        self.instance_order: List[str] = []
        self.current_index = 0
        
        # Session affinity
        self.session_affinity_enabled = config.get("session_affinity", True)
        self.session_map: Dict[str, str] = {}  # session_id -> instance_id
        
        # Health checking
        self.health_check_interval = config.get("health_check_interval", 10)
        self.health_check_timeout = config.get("health_check_timeout", 5)
        self.unhealthy_threshold = config.get("unhealthy_threshold", 3)
        self.health_check_failures: Dict[str, int] = {}
        
        # Circuit breaker
        self.circuit_breaker_enabled = config.get("circuit_breaker", True)
        self.circuit_breaker_threshold = config.get("circuit_breaker_threshold", 5)
        self.circuit_breaker_timeout = config.get("circuit_breaker_timeout", 60)
        self.circuit_breaker_state: Dict[str, Dict] = {}
        
        # Auto-scaling
        self.auto_scaling_enabled = config.get("auto_scaling", True)
        self.min_instances = config.get("min_instances", 2)
        self.max_instances = config.get("max_instances", 10)
        self.scale_up_threshold = config.get("scale_up_threshold", 0.8)
        self.scale_down_threshold = config.get("scale_down_threshold", 0.3)
        self.scale_cooldown = config.get("scale_cooldown", 300)
        self.last_scale_time = datetime.now()
        
        # Metrics for adaptive routing
        self.request_history: List[Dict] = []
        self.routing_model = None
        self.metrics_window = config.get("metrics_window", 300)  # 5 minutes
        
        # Redis for distributed state
        self.redis_client = None
        self.redis_url = config.get("redis_url", "redis://localhost:6379/0")
        
        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.health_check_task = None
        self.auto_scaling_task = None
        self.metrics_cleanup_task = None
        
        # Consistent hashing ring
        self.hash_ring: List[Tuple[int, str]] = []
        self.virtual_nodes = config.get("virtual_nodes", 150)
        
        logger.info(f"Initialized load balancer with strategy: {self.strategy}")
    
    async def initialize(self):
        """Initialize the load balancer"""
        # Connect to Redis
        self.redis_client = await aioredis.from_url(self.redis_url)
        
        # Start background tasks
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.auto_scaling_task = asyncio.create_task(self._auto_scaling_loop())
        self.metrics_cleanup_task = asyncio.create_task(self._metrics_cleanup_loop())
        
        # Initialize routing model for adaptive strategy
        if self.strategy == LoadBalancingStrategy.ADAPTIVE:
            self._initialize_routing_model()
        
        logger.info("Load balancer initialized successfully")
    
    def add_instance(self, instance: Instance):
        """Add a new instance to the pool"""
        self.instances[instance.instance_id] = instance
        self.instance_order.append(instance.instance_id)
        
        # Update consistent hash ring
        if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            self._update_hash_ring()
        
        logger.info(f"Added instance: {instance.instance_id} at {instance.host}:{instance.port}")
    
    def remove_instance(self, instance_id: str):
        """Remove an instance from the pool"""
        if instance_id in self.instances:
            del self.instances[instance_id]
            self.instance_order.remove(instance_id)
            
            # Update consistent hash ring
            if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                self._update_hash_ring()
            
            # Clear session affinity for this instance
            sessions_to_clear = [
                session_id for session_id, inst_id in self.session_map.items()
                if inst_id == instance_id
            ]
            for session_id in sessions_to_clear:
                del self.session_map[session_id]
            
            logger.info(f"Removed instance: {instance_id}")
    
    async def route_request(
        self,
        session_id: Optional[str] = None,
        request_type: str = "general",
        required_capabilities: Set[str] = None
    ) -> Optional[Instance]:
        """Route a request to an appropriate instance"""
        # Check session affinity first
        if self.session_affinity_enabled and session_id:
            if session_id in self.session_map:
                instance_id = self.session_map[session_id]
                instance = self.instances.get(instance_id)
                if instance and instance.is_healthy:
                    return instance
        
        # Filter instances by capabilities if required
        available_instances = self._filter_instances_by_capabilities(required_capabilities)
        
        if not available_instances:
            logger.error("No healthy instances available")
            return None
        
        # Route based on strategy
        instance = None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            instance = self._round_robin_select(available_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            instance = self._least_connections_select(available_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            instance = self._weighted_round_robin_select(available_instances)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            instance = self._consistent_hash_select(session_id or request_type, available_instances)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            instance = await self._adaptive_select(request_type, available_instances)
        
        # Update session affinity
        if instance and session_id and self.session_affinity_enabled:
            self.session_map[session_id] = instance.instance_id
            await self._update_session_map_redis(session_id, instance.instance_id)
        
        # Update metrics
        if instance:
            instance.active_connections += 1
            self._record_routing_decision(instance, request_type)
        
        return instance
    
    def _filter_instances_by_capabilities(
        self,
        required_capabilities: Optional[Set[str]]
    ) -> List[Instance]:
        """Filter instances by required capabilities"""
        available_instances = []
        
        for instance in self.instances.values():
            if not instance.is_healthy:
                continue
                
            # Check circuit breaker
            if self.circuit_breaker_enabled and self._is_circuit_open(instance.instance_id):
                continue
            
            # Check capabilities
            if required_capabilities:
                if not instance.capabilities or not required_capabilities.issubset(instance.capabilities):
                    continue
            
            available_instances.append(instance)
        
        return available_instances
    
    def _round_robin_select(self, instances: List[Instance]) -> Instance:
        """Round-robin selection"""
        if not instances:
            return None
            
        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance
    
    def _least_connections_select(self, instances: List[Instance]) -> Instance:
        """Select instance with least active connections"""
        return min(instances, key=lambda x: x.active_connections)
    
    def _weighted_round_robin_select(self, instances: List[Instance]) -> Instance:
        """Weighted round-robin selection"""
        total_weight = sum(inst.weight for inst in instances)
        
        if total_weight == 0:
            return self._round_robin_select(instances)
        
        random_weight = np.random.uniform(0, total_weight)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.weight
            if current_weight >= random_weight:
                return instance
        
        return instances[-1]
    
    def _consistent_hash_select(self, key: str, instances: List[Instance]) -> Instance:
        """Consistent hashing selection"""
        if not self.hash_ring:
            return self._round_robin_select(instances)
        
        # Hash the key
        key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        # Find the next server in the ring
        for ring_hash, instance_id in self.hash_ring:
            if ring_hash >= key_hash:
                instance = self.instances.get(instance_id)
                if instance and instance in instances:
                    return instance
        
        # Wrap around to the first server
        instance_id = self.hash_ring[0][1]
        instance = self.instances.get(instance_id)
        if instance and instance in instances:
            return instance
        
        return self._round_robin_select(instances)
    
    async def _adaptive_select(
        self,
        request_type: str,
        instances: List[Instance]
    ) -> Instance:
        """ML-based adaptive routing"""
        if not self.routing_model or not instances:
            return self._weighted_round_robin_select(instances)
        
        # Prepare features for each instance
        features = []
        for instance in instances:
            feature_vector = [
                instance.cpu_usage / 100.0,
                instance.memory_usage / 100.0,
                instance.gpu_usage / 100.0,
                instance.active_connections / 100.0,
                instance.response_time_avg / 1000.0,  # Convert to seconds
                instance.health_score,
                1.0 if request_type == "inference" else 0.0,
                1.0 if instance.zone == self.config.get("preferred_zone", "default") else 0.0
            ]
            features.append(feature_vector)
        
        # Predict best instance
        try:
            features_array = np.array(features)
            scores = self.routing_model.predict(features_array)
            best_index = np.argmax(scores)
            return instances[best_index]
        except Exception as e:
            logger.error(f"Adaptive routing failed: {e}")
            return self._weighted_round_robin_select(instances)
    
    def _update_hash_ring(self):
        """Update consistent hashing ring"""
        self.hash_ring = []
        
        for instance_id, instance in self.instances.items():
            if not instance.is_healthy:
                continue
                
            # Add virtual nodes for better distribution
            for i in range(self.virtual_nodes):
                virtual_key = f"{instance_id}:{i}"
                ring_hash = int(hashlib.md5(virtual_key.encode()).hexdigest(), 16)
                self.hash_ring.append((ring_hash, instance_id))
        
        # Sort the ring
        self.hash_ring.sort(key=lambda x: x[0])
    
    async def _health_check_loop(self):
        """Background health checking"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                tasks = []
                for instance in self.instances.values():
                    tasks.append(self._check_instance_health(instance))
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _check_instance_health(self, instance: Instance):
        """Check health of a single instance"""
        health_url = f"http://{instance.host}:{instance.port}/health"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    health_url,
                    timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)
                ) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Update instance metrics
                        instance.cpu_usage = health_data.get("cpu_usage", 0)
                        instance.memory_usage = health_data.get("memory_usage", 0)
                        instance.gpu_usage = health_data.get("gpu_usage", 0)
                        instance.response_time_avg = health_data.get("response_time_avg", 0)
                        instance.health_score = self._calculate_health_score(health_data)
                        instance.last_health_check = datetime.now()
                        instance.is_healthy = True
                        
                        # Reset failure count
                        self.health_check_failures[instance.instance_id] = 0
                        
                        # Reset circuit breaker if needed
                        if instance.instance_id in self.circuit_breaker_state:
                            self.circuit_breaker_state[instance.instance_id]["failures"] = 0
                    else:
                        self._handle_health_check_failure(instance)
                        
        except Exception as e:
            self._handle_health_check_failure(instance)
            logger.error(f"Health check failed for {instance.instance_id}: {e}")
    
    def _handle_health_check_failure(self, instance: Instance):
        """Handle health check failure"""
        instance.health_score = 0.0
        instance.last_health_check = datetime.now()
        
        # Increment failure count
        failures = self.health_check_failures.get(instance.instance_id, 0) + 1
        self.health_check_failures[instance.instance_id] = failures
        
        # Mark unhealthy if threshold exceeded
        if failures >= self.unhealthy_threshold:
            instance.is_healthy = False
            logger.warning(f"Instance {instance.instance_id} marked unhealthy")
            
            # Update circuit breaker
            if self.circuit_breaker_enabled:
                if instance.instance_id not in self.circuit_breaker_state:
                    self.circuit_breaker_state[instance.instance_id] = {
                        "failures": 0,
                        "last_failure": None,
                        "state": "closed"
                    }
                
                cb_state = self.circuit_breaker_state[instance.instance_id]
                cb_state["failures"] += 1
                cb_state["last_failure"] = datetime.now()
                
                if cb_state["failures"] >= self.circuit_breaker_threshold:
                    cb_state["state"] = "open"
                    logger.warning(f"Circuit breaker opened for {instance.instance_id}")
    
    def _calculate_health_score(self, health_data: Dict[str, Any]) -> float:
        """Calculate overall health score for an instance"""
        # Weighted health score based on various metrics
        weights = {
            "cpu": 0.25,
            "memory": 0.25,
            "gpu": 0.20,
            "response_time": 0.20,
            "error_rate": 0.10
        }
        
        score = 0.0
        
        # CPU score (inverted - lower is better)
        cpu_score = 1.0 - (health_data.get("cpu_usage", 0) / 100.0)
        score += weights["cpu"] * cpu_score
        
        # Memory score (inverted - lower is better)
        memory_score = 1.0 - (health_data.get("memory_usage", 0) / 100.0)
        score += weights["memory"] * memory_score
        
        # GPU score (inverted - lower is better)
        gpu_score = 1.0 - (health_data.get("gpu_usage", 0) / 100.0)
        score += weights["gpu"] * gpu_score
        
        # Response time score (inverted and normalized)
        response_time = health_data.get("response_time_avg", 0)
        response_time_score = max(0, 1.0 - (response_time / 1000.0))  # Assume 1s is bad
        score += weights["response_time"] * response_time_score
        
        # Error rate score (inverted)
        error_rate = health_data.get("error_rate", 0)
        error_rate_score = 1.0 - min(1.0, error_rate / 10.0)  # 10% error rate is maximum bad
        score += weights["error_rate"] * error_rate_score
        
        return max(0.0, min(1.0, score))
    
    def _is_circuit_open(self, instance_id: str) -> bool:
        """Check if circuit breaker is open for an instance"""
        if instance_id not in self.circuit_breaker_state:
            return False
        
        cb_state = self.circuit_breaker_state[instance_id]
        
        if cb_state["state"] != "open":
            return False
        
        # Check if timeout has passed
        if cb_state["last_failure"]:
            time_since_failure = (datetime.now() - cb_state["last_failure"]).total_seconds()
            if time_since_failure > self.circuit_breaker_timeout:
                cb_state["state"] = "half-open"
                return False
        
        return True
    
    async def _auto_scaling_loop(self):
        """Background auto-scaling"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not self.auto_scaling_enabled:
                    continue
                
                # Check if cooldown period has passed
                time_since_last_scale = (datetime.now() - self.last_scale_time).total_seconds()
                if time_since_last_scale < self.scale_cooldown:
                    continue
                
                # Calculate average metrics
                avg_metrics = self._calculate_average_metrics()
                
                # Determine if scaling is needed
                if avg_metrics["utilization"] > self.scale_up_threshold:
                    await self._scale_up()
                elif avg_metrics["utilization"] < self.scale_down_threshold:
                    await self._scale_down()
                
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
    
    def _calculate_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics across all instances"""
        if not self.instances:
            return {"utilization": 0.0}
        
        total_cpu = sum(inst.cpu_usage for inst in self.instances.values())
        total_memory = sum(inst.memory_usage for inst in self.instances.values())
        total_gpu = sum(inst.gpu_usage for inst in self.instances.values())
        total_connections = sum(inst.active_connections for inst in self.instances.values())
        
        num_instances = len(self.instances)
        
        # Calculate overall utilization (weighted average)
        utilization = (
            (total_cpu / num_instances) * 0.3 +
            (total_memory / num_instances) * 0.3 +
            (total_gpu / num_instances) * 0.2 +
            (total_connections / (num_instances * 100)) * 0.2  # Assume 100 connections is 100%
        ) / 100.0
        
        return {
            "utilization": utilization,
            "avg_cpu": total_cpu / num_instances,
            "avg_memory": total_memory / num_instances,
            "avg_gpu": total_gpu / num_instances,
            "avg_connections": total_connections / num_instances
        }
    
    async def _scale_up(self):
        """Scale up by adding more instances"""
        current_count = len(self.instances)
        
        if current_count >= self.max_instances:
            logger.info("Already at maximum instances")
            return
        
        # Signal external orchestrator to add instance
        await self._notify_scaling_event("scale_up", current_count + 1)
        
        self.last_scale_time = datetime.now()
        logger.info(f"Scaling up from {current_count} to {current_count + 1} instances")
    
    async def _scale_down(self):
        """Scale down by removing instances"""
        current_count = len(self.instances)
        
        if current_count <= self.min_instances:
            logger.info("Already at minimum instances")
            return
        
        # Find least loaded instance to remove
        least_loaded = min(
            self.instances.values(),
            key=lambda x: x.active_connections + x.cpu_usage + x.memory_usage
        )
        
        # Signal external orchestrator to remove instance
        await self._notify_scaling_event("scale_down", current_count - 1, least_loaded.instance_id)
        
        self.last_scale_time = datetime.now()
        logger.info(f"Scaling down from {current_count} to {current_count - 1} instances")
    
    async def _notify_scaling_event(
        self,
        event_type: str,
        target_count: int,
        instance_id: Optional[str] = None
    ):
        """Notify external systems about scaling events"""
        event = {
            "event_type": event_type,
            "target_count": target_count,
            "instance_id": instance_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # In production, this would notify Kubernetes, AWS Auto Scaling, etc.
        if self.redis_client:
            await self.redis_client.publish("scaling_events", json.dumps(event))
    
    async def _update_session_map_redis(self, session_id: str, instance_id: str):
        """Update session mapping in Redis for distributed coordination"""
        if self.redis_client:
            await self.redis_client.setex(
                f"session_map:{session_id}",
                3600,  # 1 hour TTL
                instance_id
            )
    
    def _record_routing_decision(self, instance: Instance, request_type: str):
        """Record routing decision for analytics and ML training"""
        decision = {
            "timestamp": time.time(),
            "instance_id": instance.instance_id,
            "request_type": request_type,
            "instance_metrics": {
                "cpu_usage": instance.cpu_usage,
                "memory_usage": instance.memory_usage,
                "gpu_usage": instance.gpu_usage,
                "active_connections": instance.active_connections,
                "response_time_avg": instance.response_time_avg,
                "health_score": instance.health_score
            }
        }
        
        self.request_history.append(decision)
        
        # Keep history size manageable
        if len(self.request_history) > 10000:
            self.request_history = self.request_history[-5000:]
    
    async def _metrics_cleanup_loop(self):
        """Clean up old metrics periodically"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Clean old request history
                cutoff_time = time.time() - self.metrics_window
                self.request_history = [
                    record for record in self.request_history
                    if record["timestamp"] > cutoff_time
                ]
                
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
    
    def _initialize_routing_model(self):
        """Initialize ML model for adaptive routing"""
        # In production, this would be a more sophisticated model
        # For demo, we use a simple scoring function
        self.routing_model = SimpleRoutingModel()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current load balancer metrics"""
        metrics = {
            "instances": len(self.instances),
            "healthy_instances": sum(1 for inst in self.instances.values() if inst.is_healthy),
            "total_connections": sum(inst.active_connections for inst in self.instances.values()),
            "average_metrics": self._calculate_average_metrics(),
            "routing_strategy": self.strategy.value,
            "session_count": len(self.session_map),
            "circuit_breakers_open": sum(
                1 for state in self.circuit_breaker_state.values()
                if state["state"] == "open"
            )
        }
        
        return metrics
    
    async def shutdown(self):
        """Graceful shutdown"""
        # Cancel background tasks
        if self.health_check_task:
            self.health_check_task.cancel()
        if self.auto_scaling_task:
            self.auto_scaling_task.cancel()
        if self.metrics_cleanup_task:
            self.metrics_cleanup_task.cancel()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Load balancer shut down successfully")


class SimpleRoutingModel:
    """Simple ML model for routing decisions"""
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Score instances based on features"""
        # Simple weighted scoring
        weights = np.array([
            -0.3,  # CPU usage (negative - lower is better)
            -0.3,  # Memory usage
            -0.2,  # GPU usage
            -0.1,  # Active connections
            -0.1,  # Response time
            1.0,   # Health score
            0.1,   # Request type match
            0.1    # Zone preference
        ])
        
        scores = np.dot(features, weights)
        return scores