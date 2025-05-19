"""
Production Monitoring and Alerting for Digital Human System
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
import threading

from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from aiq.data_models.common import BaseModel
from aiq.utils.debugging_utils import log_function_call


# Configure structured logging
class ProductionLogger:
    """Enhanced logger for production environments"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter for structured logs
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "name": "%(name)s", '
            '"level": "%(levelname)s", "message": "%(message)s", '
            '"module": "%(module)s", "function": "%(funcName)s", '
            '"line": %(lineno)d}'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistence
        log_file = os.getenv("LOG_FILE", "/var/log/digital_human/app.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_event(self, event_type: str, details: Dict[str, Any], level: str = "info"):
        """Log structured event"""
        event_data = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **details
        }
        
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(json.dumps(event_data))


# Prometheus metrics
session_counter = Counter('digital_human_sessions_total', 'Total number of sessions', ['status'])
response_time = Histogram('digital_human_response_time_seconds', 'Response time distribution', ['endpoint'])
active_users = Gauge('digital_human_active_users', 'Currently active users')
error_counter = Counter('digital_human_errors_total', 'Total errors', ['error_type', 'severity'])
gpu_utilization = Gauge('digital_human_gpu_utilization_percent', 'GPU utilization percentage', ['gpu_id'])
memory_usage = Gauge('digital_human_memory_usage_bytes', 'Memory usage in bytes', ['type'])
model_inference_time = Summary('digital_human_model_inference_seconds', 'Model inference time')
cache_hit_rate = Gauge('digital_human_cache_hit_rate', 'Cache hit rate percentage')


class ProductionMonitor:
    """
    Comprehensive monitoring for Digital Human system including:
    - Performance metrics
    - Error tracking
    - Health checks
    - Alerting
    - Distributed tracing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = ProductionLogger("digital_human.monitor")
        
        # Initialize monitoring backends
        self._setup_prometheus()
        self._setup_opentelemetry()
        self._setup_sentry()
        self._setup_alerting()
        
        # Metrics storage
        self.metrics_buffer = defaultdict(list)
        self.alert_thresholds = config.get("alert_thresholds", {})
        
        # Start background monitoring tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics server"""
        port = self.config.get("prometheus_port", 9090)
        start_http_server(port)
        self.logger.log_event("prometheus_started", {"port": port})
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry for distributed tracing"""
        # Configure tracing
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config.get("otlp_endpoint", "localhost:4317"),
            insecure=True
        )
        
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Configure metrics
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=self.config.get("otlp_endpoint", "localhost:4317")),
            export_interval_millis=10000
        )
        
        metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
        
        # Instrument libraries
        FastAPIInstrumentor.instrument()
        SQLAlchemyInstrumentor.instrument()
        RedisInstrumentor.instrument()
        
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
    
    def _setup_sentry(self):
        """Setup Sentry for error tracking"""
        sentry_dsn = self.config.get("sentry_dsn")
        if sentry_dsn:
            sentry_sdk.init(
                dsn=sentry_dsn,
                integrations=[
                    FastApiIntegration(transaction_style="endpoint"),
                    SqlalchemyIntegration()
                ],
                traces_sample_rate=0.1,
                environment=self.config.get("environment", "production")
            )
            self.logger.log_event("sentry_initialized", {"environment": self.config.get("environment")})
    
    def _setup_alerting(self):
        """Setup alerting channels"""
        self.slack_client = None
        slack_token = self.config.get("slack_token")
        if slack_token:
            self.slack_client = WebClient(token=slack_token)
            self.alert_channel = self.config.get("alert_channel", "#alerts")
        
        self.pagerduty_key = self.config.get("pagerduty_key")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await self._collect_system_metrics()
                await self._check_health()
                await self._process_alerts()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.log_event("monitoring_error", {"error": str(e)}, level="error")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # GPU metrics
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    utilization = torch.cuda.utilization(i)
                    gpu_utilization.labels(gpu_id=str(i)).set(utilization)
            
            # Memory metrics
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage.labels(type="rss").set(memory_info.rss)
            memory_usage.labels(type="vms").set(memory_info.vms)
            
            # CPU metrics
            cpu_percent = process.cpu_percent(interval=1)
            cpu_usage = Gauge('digital_human_cpu_usage_percent', 'CPU usage percentage')
            cpu_usage.set(cpu_percent)
            
        except Exception as e:
            self.logger.log_event("metrics_collection_error", {"error": str(e)}, level="error")
    
    async def _check_health(self):
        """Check system health and trigger alerts if needed"""
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "checks": {}
        }
        
        # Check GPU availability
        if self.config.get("require_gpu", True):
            gpu_available = torch.cuda.is_available()
            health_status["checks"]["gpu"] = {
                "status": "healthy" if gpu_available else "unhealthy",
                "available": gpu_available,
                "count": torch.cuda.device_count() if gpu_available else 0
            }
            
            if not gpu_available:
                await self._send_alert(
                    "GPU not available",
                    "The system requires GPU but none is available",
                    "critical"
                )
        
        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        memory_threshold = self.alert_thresholds.get("memory_percent", 90)
        
        health_status["checks"]["memory"] = {
            "status": "healthy" if memory.percent < memory_threshold else "unhealthy",
            "percent": memory.percent,
            "available": memory.available
        }
        
        if memory.percent > memory_threshold:
            await self._send_alert(
                "High memory usage",
                f"Memory usage is at {memory.percent}%",
                "warning"
            )
        
        # Check response times
        avg_response_time = self._calculate_avg_response_time()
        response_threshold = self.alert_thresholds.get("response_time_seconds", 1.0)
        
        health_status["checks"]["response_time"] = {
            "status": "healthy" if avg_response_time < response_threshold else "unhealthy",
            "average": avg_response_time
        }
        
        if avg_response_time > response_threshold:
            await self._send_alert(
                "High response time",
                f"Average response time is {avg_response_time:.2f}s",
                "warning"
            )
        
        # Update overall status
        unhealthy_checks = [
            check for check in health_status["checks"].values()
            if check["status"] == "unhealthy"
        ]
        
        if unhealthy_checks:
            health_status["status"] = "unhealthy"
        
        self.logger.log_event("health_check", health_status)
        
        return health_status
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from recent metrics"""
        if not self.metrics_buffer["response_times"]:
            return 0.0
        
        recent_times = self.metrics_buffer["response_times"][-100:]  # Last 100 requests
        return sum(recent_times) / len(recent_times)
    
    async def _process_alerts(self):
        """Process and send alerts based on metrics"""
        # Check error rates
        error_rate = self._calculate_error_rate()
        error_threshold = self.alert_thresholds.get("error_rate_percent", 5.0)
        
        if error_rate > error_threshold:
            await self._send_alert(
                "High error rate",
                f"Error rate is at {error_rate:.1f}%",
                "critical"
            )
        
        # Check active sessions
        max_sessions = self.alert_thresholds.get("max_sessions", 1000)
        current_sessions = active_users._value.get()
        
        if current_sessions > max_sessions:
            await self._send_alert(
                "High session count",
                f"Active sessions: {current_sessions}",
                "warning"
            )
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        if not self.metrics_buffer["requests"]:
            return 0.0
            
        recent_requests = self.metrics_buffer["requests"][-1000:]  # Last 1000 requests
        errors = sum(1 for r in recent_requests if r.get("error"))
        
        return (errors / len(recent_requests)) * 100
    
    async def _send_alert(self, title: str, message: str, severity: str = "info"):
        """Send alert through configured channels"""
        alert_data = {
            "title": title,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "environment": self.config.get("environment", "production")
        }
        
        # Log the alert
        self.logger.log_event("alert", alert_data, level="warning")
        
        # Send to Slack
        if self.slack_client:
            try:
                await self._send_slack_alert(alert_data)
            except Exception as e:
                self.logger.log_event("slack_alert_error", {"error": str(e)}, level="error")
        
        # Send to PagerDuty for critical alerts
        if severity == "critical" and self.pagerduty_key:
            await self._send_pagerduty_alert(alert_data)
    
    async def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """Send alert to Slack"""
        color = {
            "info": "#36a64f",
            "warning": "#ff9800",
            "critical": "#ff0000"
        }.get(alert_data["severity"], "#808080")
        
        attachment = {
            "color": color,
            "title": alert_data["title"],
            "text": alert_data["message"],
            "fields": [
                {
                    "title": "Environment",
                    "value": alert_data["environment"],
                    "short": True
                },
                {
                    "title": "Severity",
                    "value": alert_data["severity"],
                    "short": True
                },
                {
                    "title": "Time",
                    "value": alert_data["timestamp"],
                    "short": False
                }
            ]
        }
        
        self.slack_client.chat_postMessage(
            channel=self.alert_channel,
            attachments=[attachment]
        )
    
    async def _send_pagerduty_alert(self, alert_data: Dict[str, Any]):
        """Send alert to PagerDuty"""
        # Implementation would depend on PagerDuty API
        pass
    
    def track_request(self, endpoint: str, duration: float, error: Optional[str] = None):
        """Track API request metrics"""
        response_time.labels(endpoint=endpoint).observe(duration)
        
        request_data = {
            "endpoint": endpoint,
            "duration": duration,
            "timestamp": time.time(),
            "error": error
        }
        
        self.metrics_buffer["requests"].append(request_data)
        self.metrics_buffer["response_times"].append(duration)
        
        # Keep buffer size manageable
        if len(self.metrics_buffer["requests"]) > 10000:
            self.metrics_buffer["requests"] = self.metrics_buffer["requests"][-5000:]
            self.metrics_buffer["response_times"] = self.metrics_buffer["response_times"][-5000:]
    
    def track_error(self, error_type: str, severity: str = "error"):
        """Track error occurrences"""
        error_counter.labels(error_type=error_type, severity=severity).inc()
        
        error_data = {
            "type": error_type,
            "severity": severity,
            "timestamp": time.time()
        }
        
        self.metrics_buffer["errors"].append(error_data)
    
    def track_session(self, action: str):
        """Track session lifecycle"""
        if action == "start":
            active_users.inc()
            session_counter.labels(status="started").inc()
        elif action == "end":
            active_users.dec()
            session_counter.labels(status="ended").inc()
    
    def track_model_inference(self, duration: float):
        """Track model inference performance"""
        model_inference_time.observe(duration)
    
    def track_cache_hit(self, hit: bool):
        """Track cache performance"""
        if "cache_hits" not in self.metrics_buffer:
            self.metrics_buffer["cache_hits"] = {"hits": 0, "total": 0}
        
        self.metrics_buffer["cache_hits"]["total"] += 1
        if hit:
            self.metrics_buffer["cache_hits"]["hits"] += 1
        
        # Update cache hit rate
        hit_rate = (
            self.metrics_buffer["cache_hits"]["hits"] / 
            self.metrics_buffer["cache_hits"]["total"]
        ) * 100
        
        cache_hit_rate.set(hit_rate)
    
    async def shutdown(self):
        """Graceful shutdown of monitoring"""
        self.monitoring_task.cancel()
        self.logger.log_event("monitoring_shutdown", {"status": "completed"})


def setup_production_logging(name: str) -> ProductionLogger:
    """Setup production logging for a module"""
    return ProductionLogger(name)