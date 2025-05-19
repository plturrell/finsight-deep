# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from typing import Dict, Any


class MetricsCollector:
    """Production metrics with Prometheus integration"""
    
    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            'agent_requests_total',
            'Total number of agent requests',
            ['agent_id', 'status']
        )
        
        self.request_duration = Histogram(
            'agent_request_duration_seconds',
            'Agent request duration',
            ['agent_id'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Connection metrics
        self.active_connections = Gauge(
            'agent_active_connections',
            'Number of active connections',
            ['agent_id']
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'agent_cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.cache_misses = Counter(
            'agent_cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'agent_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['agent_id']
        )
        
        # Error metrics
        self.error_count = Counter(
            'agent_errors_total',
            'Total number of errors',
            ['agent_id', 'error_type']
        )
    
    def track_request(self, agent_id: str):
        """Track a request"""
        return RequestTracker(self, agent_id)
    
    def track_cache(self, cache_type: str):
        """Track cache operations"""
        return CacheTracker(self, cache_type)
    
    def update_circuit_breaker(self, agent_id: str, state: str):
        """Update circuit breaker state"""
        state_map = {"closed": 0, "open": 1, "half-open": 2}
        self.circuit_breaker_state.labels(agent_id=agent_id).set(state_map.get(state, 0))
    
    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        return generate_latest()


class RequestTracker:
    """Context manager for tracking request metrics"""
    
    def __init__(self, collector: MetricsCollector, agent_id: str):
        self.collector = collector
        self.agent_id = agent_id
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.collector.active_connections.labels(agent_id=self.agent_id).inc()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        # Update metrics
        self.collector.request_duration.labels(agent_id=self.agent_id).observe(duration)
        self.collector.active_connections.labels(agent_id=self.agent_id).dec()
        
        if exc_type is None:
            self.collector.request_count.labels(agent_id=self.agent_id, status="success").inc()
        else:
            self.collector.request_count.labels(agent_id=self.agent_id, status="error").inc()
            self.collector.error_count.labels(
                agent_id=self.agent_id, 
                error_type=exc_type.__name__
            ).inc()


class CacheTracker:
    """Track cache hits/misses"""
    
    def __init__(self, collector: MetricsCollector, cache_type: str):
        self.collector = collector
        self.cache_type = cache_type
    
    def hit(self):
        """Record cache hit"""
        self.collector.cache_hits.labels(cache_type=self.cache_type).inc()
    
    def miss(self):
        """Record cache miss"""
        self.collector.cache_misses.labels(cache_type=self.cache_type).inc()


# Global metrics instance
metrics = MetricsCollector()