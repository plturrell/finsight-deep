# Google Agent API - Traced System Production Guide

## Overview

This guide covers the fully traced, production-ready Google Agent API system with distributed tracing, comprehensive monitoring, and enterprise-grade security features.

## Architecture

The traced system includes:

1. **Core Components**
   - `TracedGoogleAgentClient`: Agent client with full tracing integration
   - `TracedAgentToAgentConnector`: Multi-agent orchestrator with trace propagation
   - `DistributedTracer`: OpenTelemetry-based distributed tracing
   - `AuthenticationManager`: JWT-based authentication with RSA signatures
   - `SecretManager`: Multi-provider secret management
   - `InputValidator`: Comprehensive input validation

2. **Observability Stack**
   - **Jaeger**: Distributed trace collection and visualization
   - **Prometheus**: Metrics collection and alerting
   - **Grafana**: Dashboards and visualization
   - **Fluentd**: Log aggregation
   - **OpenTelemetry**: Trace and metric instrumentation

3. **Security Features**
   - JWT authentication with RSA signatures
   - Role-based access control (RBAC)
   - Secure secret management (Vault, AWS Secrets Manager)
   - Input validation (XSS, SQL injection, command injection protection)
   - TLS 1.3 encryption
   - Rate limiting and circuit breakers

## Quick Start

### Local Development

1. **Start the traced environment:**
   ```bash
   cd examples/google_agent_api
   docker-compose -f docker-compose.traced.yml up -d
   ```

2. **Run the traced example:**
   ```bash
   python traced_example.py
   ```

3. **View traces in Jaeger:**
   - Open http://localhost:16686
   - Select "google-agent-api" service
   - View distributed traces

4. **View metrics in Grafana:**
   - Open http://localhost:3000
   - Login with admin/admin
   - Navigate to the "Google Agent API - Traced System" dashboard

### Production Deployment

1. **Configure production settings:**
   ```bash
   # Edit production_config.yaml with your environment values
   vim production_config.yaml
   ```

2. **Deploy to Kubernetes:**
   ```bash
   # Apply the deployment
   kubectl apply -f kubernetes_deployment.yaml
   
   # Verify deployment
   kubectl get pods -n google-agent-api
   ```

3. **Configure monitoring:**
   ```bash
   # Deploy Jaeger
   kubectl apply -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/master/deploy/crds/jaegertracing.io_jaegers_crd.yaml
   kubectl apply -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/master/deploy/service_account.yaml
   
   # Deploy Prometheus
   helm install prometheus prometheus-community/prometheus -n monitoring
   
   # Deploy Grafana
   helm install grafana grafana/grafana -n monitoring
   ```

## Tracing Implementation

### Adding Traces to New Code

```python
from aiq.tool.google_agent_api.tracing import DistributedTracer

class MyNewComponent:
    def __init__(self, tracer: DistributedTracer):
        self.tracer = tracer
    
    async def my_operation(self):
        async with self.tracer.trace_async_operation(
            "my_operation",
            kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute("custom.attribute", "value")
            
            # Your operation code here
            result = await self._do_work()
            
            span.set_attribute("result.size", len(result))
            return result
```

### Trace Context Propagation

```python
# Extracting trace context
context = self.tracer.extract_context(headers)

# Injecting trace context
headers = {}
self.tracer.inject_context(headers)

# Propagating context to child operations
with self.tracer.create_span("parent_operation") as parent:
    # Context automatically propagated to children
    with self.tracer.create_span("child_operation") as child:
        # Child span linked to parent
        pass
```

## Security Configuration

### Authentication Setup

```python
# Initialize authentication
auth_manager = AuthenticationManager(
    secret_key=os.environ["AUTH_SECRET_KEY"],
    issuer="aiq-agent-system",
    audience="agent-api"
)

# Create token with permissions
token = await auth_manager.create_token(
    user_id="user123",
    permissions={Permission.READ, Permission.EXECUTE},
    roles={"developer"}
)

# Verify token
auth_token = await auth_manager.verify_token(token)
if auth_token and Permission.EXECUTE in auth_token.permissions:
    # Authorized to execute
    pass
```

### Secret Management

```python
# Initialize secret manager
secret_manager = SecretManager()
await secret_manager.initialize()

# Store a secret
await secret_manager.store_secret(
    key="api_key",
    value="sensitive_value",
    metadata={"rotation": "monthly"}
)

# Retrieve a secret
api_key = await secret_manager.get_secret("api_key")

# Rotate secrets
await secret_manager.rotate_secret("api_key", new_value)
```

## Monitoring and Alerts

### Key Metrics

1. **Request Metrics**
   - `http_requests_total`: Total request count
   - `http_request_duration_seconds`: Request latency histogram
   - `http_requests_in_flight`: Current active requests

2. **Agent Metrics**
   - `agent_calls_total`: Agent invocation count
   - `agent_call_duration_seconds`: Agent call latency
   - `agent_errors_total`: Agent error count

3. **System Metrics**
   - `circuit_breaker_state`: Circuit breaker status
   - `cache_hits_total`: Cache hit rate
   - `auth_failures_total`: Authentication failures

### Alert Examples

```yaml
# High error rate
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  for: 5m
  annotations:
    summary: "Error rate above 5%"

# Circuit breaker open
- alert: CircuitBreakerOpen
  expr: circuit_breaker_state{state="open"} == 1
  for: 1m
  annotations:
    summary: "Circuit breaker is open for {{ $labels.agent }}"

# High latency
- alert: HighLatency
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
  for: 5m
  annotations:
    summary: "95th percentile latency above 1s"
```

## Performance Tuning

### Connection Pool Configuration

```yaml
connection_pool:
  max_connections: 100      # Maximum connections per agent
  min_connections: 10       # Minimum idle connections
  connection_timeout: 5000  # Connection timeout (ms)
  idle_timeout: 600000     # Idle connection timeout (ms)
```

### Circuit Breaker Settings

```yaml
circuit_breaker:
  failure_threshold: 5      # Failures before opening
  recovery_timeout: 60000   # Recovery timeout (ms)
  half_open_requests: 3     # Requests in half-open state
```

### Rate Limiting

```yaml
rate_limiting:
  requests_per_minute: 1000  # Per-client rate limit
  burst_size: 100           # Burst capacity
  global_limit: 10000       # Global rate limit
```

## Troubleshooting

### Common Issues

1. **High Latency**
   - Check circuit breaker status
   - Verify connection pool health
   - Review trace spans for bottlenecks

2. **Authentication Failures**
   - Verify JWT token expiration
   - Check RSA key configuration
   - Review permission requirements

3. **Missing Traces**
   - Verify Jaeger connectivity
   - Check sampling rate configuration
   - Ensure trace context propagation

### Debug Commands

```bash
# Check pod logs
kubectl logs -n google-agent-api deployment/google-agent-api

# View Jaeger traces
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686

# Check Prometheus metrics
kubectl port-forward -n monitoring svc/prometheus-server 9090:9090

# Test connectivity
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/health
```

## Best Practices

1. **Tracing**
   - Use meaningful span names
   - Add relevant attributes
   - Set proper span status
   - Use appropriate span kinds

2. **Security**
   - Rotate secrets regularly
   - Use least privilege principles
   - Monitor authentication failures
   - Enable TLS everywhere

3. **Performance**
   - Configure connection pools appropriately
   - Set reasonable timeouts
   - Use caching effectively
   - Monitor resource usage

4. **Monitoring**
   - Set up comprehensive alerts
   - Create meaningful dashboards
   - Review traces regularly
   - Analyze slow operations

## Next Steps

1. **Advanced Features**
   - Implement trace sampling strategies
   - Add custom metrics exporters
   - Create SLO dashboards
   - Implement chaos engineering tests

2. **Integration**
   - Connect to existing monitoring systems
   - Integrate with CI/CD pipelines
   - Add automated performance tests
   - Implement A/B testing with traces

3. **Optimization**
   - Profile critical paths
   - Optimize database queries
   - Implement request batching
   - Add response caching