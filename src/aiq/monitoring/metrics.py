"""
Enhanced monitoring and metrics collection for AIQToolkit.

This module provides:
- Prometheus metrics collection
- Custom metrics for AI operations
- Performance monitoring
- Alert management
- Distributed tracing support
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import json
import threading

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry
)
from fastapi import FastAPI, Response
from opentelemetry import trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import set_meter_provider, get_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from aiq.utils.exception_handlers import handle_errors, async_handle_errors
from aiq.settings.security_config import get_security_config


# Initialize metrics registry
registry = CollectorRegistry()

# System metrics
system_info = Info('aiqtoolkit_info', 'AIQToolkit system information', registry=registry)
uptime_gauge = Gauge('aiqtoolkit_uptime_seconds', 'Uptime in seconds', registry=registry)
active_sessions = Gauge('aiqtoolkit_active_sessions', 'Number of active sessions', registry=registry)

# API metrics
api_requests_total = Counter(
    'aiqtoolkit_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)
api_request_duration = Histogram(
    'aiqtoolkit_api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint'],
    registry=registry
)
api_errors_total = Counter(
    'aiqtoolkit_api_errors_total',
    'Total API errors',
    ['method', 'endpoint', 'error_type'],
    registry=registry
)

# Workflow metrics
workflow_executions_total = Counter(
    'aiqtoolkit_workflow_executions_total',
    'Total workflow executions',
    ['workflow_name', 'status'],
    registry=registry
)
workflow_duration = Histogram(
    'aiqtoolkit_workflow_duration_seconds',
    'Workflow execution duration',
    ['workflow_name'],
    registry=registry
)
workflow_errors_total = Counter(
    'aiqtoolkit_workflow_errors_total',
    'Total workflow errors',
    ['workflow_name', 'error_type'],
    registry=registry
)

# LLM metrics
llm_requests_total = Counter(
    'aiqtoolkit_llm_requests_total',
    'Total LLM API requests',
    ['provider', 'model', 'status'],
    registry=registry
)
llm_tokens_used = Counter(
    'aiqtoolkit_llm_tokens_used_total',
    'Total tokens used by LLM',
    ['provider', 'model', 'token_type'],
    registry=registry
)
llm_request_duration = Histogram(
    'aiqtoolkit_llm_request_duration_seconds',
    'LLM request duration',
    ['provider', 'model'],
    registry=registry
)
llm_cost_estimate = Counter(
    'aiqtoolkit_llm_cost_estimate_dollars',
    'Estimated LLM API cost',
    ['provider', 'model'],
    registry=registry
)

# GPU metrics
gpu_utilization = Gauge(
    'aiqtoolkit_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id'],
    registry=registry
)
gpu_memory_used = Gauge(
    'aiqtoolkit_gpu_memory_used_bytes',
    'GPU memory used',
    ['gpu_id'],
    registry=registry
)
gpu_temperature = Gauge(
    'aiqtoolkit_gpu_temperature_celsius',
    'GPU temperature',
    ['gpu_id'],
    registry=registry
)

# Security metrics
auth_attempts_total = Counter(
    'aiqtoolkit_auth_attempts_total',
    'Total authentication attempts',
    ['result', 'method'],
    registry=registry
)
security_events_total = Counter(
    'aiqtoolkit_security_events_total',
    'Total security events',
    ['event_type', 'severity'],
    registry=registry
)
rate_limit_hits_total = Counter(
    'aiqtoolkit_rate_limit_hits_total',
    'Total rate limit hits',
    ['endpoint'],
    registry=registry
)

# Cache metrics
cache_hits_total = Counter(
    'aiqtoolkit_cache_hits_total',
    'Total cache hits',
    ['cache_name'],
    registry=registry
)
cache_misses_total = Counter(
    'aiqtoolkit_cache_misses_total',
    'Total cache misses',
    ['cache_name'],
    registry=registry
)
cache_size_bytes = Gauge(
    'aiqtoolkit_cache_size_bytes',
    'Cache size in bytes',
    ['cache_name'],
    registry=registry
)

# Database metrics
db_connections_active = Gauge(
    'aiqtoolkit_db_connections_active',
    'Active database connections',
    ['database'],
    registry=registry
)
db_queries_total = Counter(
    'aiqtoolkit_db_queries_total',
    'Total database queries',
    ['database', 'operation'],
    registry=registry
)
db_query_duration = Histogram(
    'aiqtoolkit_db_query_duration_seconds',
    'Database query duration',
    ['database', 'operation'],
    registry=registry
)


class MetricsCollector:
    """Centralized metrics collection"""
    
    def __init__(self, service_name: str = "aiqtoolkit"):
        self.service_name = service_name
        self.start_time = time.time()
        self.custom_metrics: Dict[str, Any] = {}
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        self.alert_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Simple counter for basic operations
        self._counters: Dict[str, int] = defaultdict(int)
        
        # Initialize tracing
        self.tracer_provider = TracerProvider(
            resource=Resource.create({
                "service.name": service_name,
                "service.version": "2.0.0"
            })
        )
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(__name__)
        
        # Update system info
        system_info.info({
            'version': '2.0.0',
            'python_version': '3.12',
            'environment': 'production'
        })
    
    def update_uptime(self):
        """Update uptime metric"""
        uptime_gauge.set(time.time() - self.start_time)
    
    def track_api_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """Track API request metrics"""
        api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def track_workflow_execution(
        self,
        workflow_name: str,
        status: str,
        duration: float,
        error_type: Optional[str] = None
    ):
        """Track workflow execution metrics"""
        workflow_executions_total.labels(
            workflow_name=workflow_name,
            status=status
        ).inc()
        
        if status == "success":
            workflow_duration.labels(
                workflow_name=workflow_name
            ).observe(duration)
        
        if error_type:
            workflow_errors_total.labels(
                workflow_name=workflow_name,
                error_type=error_type
            ).inc()
    
    def track_llm_request(
        self,
        provider: str,
        model: str,
        status: str,
        duration: float,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        cost_estimate: float = 0.0
    ):
        """Track LLM request metrics"""
        llm_requests_total.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()
        
        llm_request_duration.labels(
            provider=provider,
            model=model
        ).observe(duration)
        
        if tokens_prompt > 0:
            llm_tokens_used.labels(
                provider=provider,
                model=model,
                token_type="prompt"
            ).inc(tokens_prompt)
        
        if tokens_completion > 0:
            llm_tokens_used.labels(
                provider=provider,
                model=model,
                token_type="completion"
            ).inc(tokens_completion)
        
        if cost_estimate > 0:
            llm_cost_estimate.labels(
                provider=provider,
                model=model
            ).inc(cost_estimate)
    
    def update_gpu_metrics(
        self,
        gpu_id: int,
        utilization: float,
        memory_used: int,
        temperature: float
    ):
        """Update GPU metrics"""
        gpu_utilization.labels(gpu_id=str(gpu_id)).set(utilization)
        gpu_memory_used.labels(gpu_id=str(gpu_id)).set(memory_used)
        gpu_temperature.labels(gpu_id=str(gpu_id)).set(temperature)
    
    def track_security_event(
        self,
        event_type: str,
        severity: str,
        auth_result: Optional[str] = None,
        auth_method: Optional[str] = None
    ):
        """Track security metrics"""
        security_events_total.labels(
            event_type=event_type,
            severity=severity
        ).inc()
        
        if auth_result and auth_method:
            auth_attempts_total.labels(
                result=auth_result,
                method=auth_method
            ).inc()
    
    def track_cache_operation(
        self,
        cache_name: str,
        hit: bool,
        size_bytes: Optional[int] = None
    ):
        """Track cache operations"""
        if hit:
            cache_hits_total.labels(cache_name=cache_name).inc()
        else:
            cache_misses_total.labels(cache_name=cache_name).inc()
        
        if size_bytes is not None:
            cache_size_bytes.labels(cache_name=cache_name).set(size_bytes)
    
    def track_database_operation(
        self,
        database: str,
        operation: str,
        duration: float,
        active_connections: Optional[int] = None
    ):
        """Track database operations"""
        db_queries_total.labels(
            database=database,
            operation=operation
        ).inc()
        
        db_query_duration.labels(
            database=database,
            operation=operation
        ).observe(duration)
        
        if active_connections is not None:
            db_connections_active.labels(database=database).set(active_connections)
    
    def set_alert_threshold(
        self,
        metric_name: str,
        threshold_type: str,  # 'min', 'max'
        value: float
    ):
        """Set alert threshold for a metric"""
        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = {}
        self.alert_thresholds[metric_name][threshold_type] = value
    
    def register_alert_callback(
        self,
        metric_name: str,
        callback: Callable[[str, float, Dict[str, Any]], None]
    ):
        """Register callback for metric alerts"""
        self.alert_callbacks[metric_name].append(callback)
    
    def check_alerts(self):
        """Check metrics against alert thresholds"""
        # This would be called periodically to check thresholds
        # Implementation depends on specific metrics to monitor
        pass
    
    @handle_errors(default_return=None)
    def create_custom_metric(
        self,
        name: str,
        metric_type: str,  # 'counter', 'gauge', 'histogram', 'summary'
        description: str,
        labels: Optional[List[str]] = None
    ):
        """Create custom metric"""
        with self._lock:
            if name in self.custom_metrics:
                return self.custom_metrics[name]
            
            labels = labels or []
            
            if metric_type == 'counter':
                metric = Counter(name, description, labels, registry=registry)
            elif metric_type == 'gauge':
                metric = Gauge(name, description, labels, registry=registry)
            elif metric_type == 'histogram':
                metric = Histogram(name, description, labels, registry=registry)
            elif metric_type == 'summary':
                metric = Summary(name, description, labels, registry=registry)
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")
            
            self.custom_metrics[name] = metric
            return metric
    
    def increment(self, metric_name: str, value: int = 1):
        """Simple counter increment for basic tracking"""
        with self._lock:
            self._counters[metric_name] += value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get simple summary of counters"""
        with self._lock:
            return dict(self._counters)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "uptime": time.time() - self.start_time,
            "counters": self.get_summary(),
            "metrics": {},
            "alerts": self.alert_thresholds
        }
        
        # Add metric summaries
        for metric_name, metric in self.custom_metrics.items():
            summary["metrics"][metric_name] = {
                "type": type(metric).__name__,
                "description": metric._documentation
            }
        
        return summary


# Global metrics collector
metrics_collector = MetricsCollector()


# FastAPI endpoints for metrics
def setup_metrics_endpoints(app: FastAPI):
    """Setup Prometheus metrics endpoint"""
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        metrics_collector.update_uptime()
        return Response(
            content=generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.get("/metrics/summary")
    async def metrics_summary():
        """Get metrics summary"""
        return metrics_collector.get_metrics_summary()


# Decorators for automatic metric collection
def track_execution_time(metric_name: str = None):
    """Decorator to track function execution time"""
    def decorator(func):
        nonlocal metric_name
        if not metric_name:
            metric_name = f"aiqtoolkit_{func.__name__}_duration_seconds"
        
        # Create histogram if it doesn't exist
        histogram = metrics_collector.create_custom_metric(
            metric_name,
            'histogram',
            f"Execution time for {func.__name__}"
        )
        
        @async_handle_errors(reraise=True)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                histogram.observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                histogram.observe(duration)
                raise
        
        @handle_errors(reraise=True)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                histogram.observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                histogram.observe(duration)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def track_counter(metric_name: str = None, labels: Dict[str, str] = None):
    """Decorator to track function call count"""
    def decorator(func):
        nonlocal metric_name
        if not metric_name:
            metric_name = f"aiqtoolkit_{func.__name__}_total"
        
        # Create counter if it doesn't exist
        counter = metrics_collector.create_custom_metric(
            metric_name,
            'counter',
            f"Call count for {func.__name__}",
            list(labels.keys()) if labels else []
        )
        
        @async_handle_errors(reraise=True)
        async def async_wrapper(*args, **kwargs):
            if labels:
                counter.labels(**labels).inc()
            else:
                counter.inc()
            return await func(*args, **kwargs)
        
        @handle_errors(reraise=True)
        def sync_wrapper(*args, **kwargs):
            if labels:
                counter.labels(**labels).inc()
            else:
                counter.inc()
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Export public interface
__all__ = [
    "MetricsCollector",
    "metrics_collector",
    "setup_metrics_endpoints",
    "track_execution_time",
    "track_counter",
    
    # Metric types
    "Counter",
    "Histogram",
    "Gauge",
    "Summary",
    
    # Specific metrics
    "api_requests_total",
    "api_request_duration",
    "workflow_executions_total",
    "workflow_duration",
    "llm_requests_total",
    "llm_tokens_used",
    "gpu_utilization",
    "security_events_total"
]