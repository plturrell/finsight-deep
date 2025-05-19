# Production Deployment Guide for Digital Human System

## Overview

This guide provides step-by-step instructions for deploying the Digital Human System to production with all real service integrations.

## Prerequisites

### Required Services
1. **NVIDIA Services**
   - NVIDIA API key with access to ACE, Riva, NeMo, and Tokkio
   - GPU infrastructure (minimum RTX A4000)

2. **Neural Supercomputer**
   - Production endpoint URL
   - API key for authentication

3. **Web Search APIs**
   - Google Custom Search API key and CSE ID
   - Yahoo News API key

4. **Financial Data APIs**
   - Alpha Vantage API key
   - Polygon.io API key
   - Quandl API key

5. **Infrastructure**
   - Kubernetes cluster with GPU nodes
   - PostgreSQL database
   - Redis cluster
   - Milvus vector database

## Environment Setup

1. **Export Required Environment Variables**
```bash
# NVIDIA Services
export NVIDIA_API_KEY="your-nvidia-api-key"

# Neural Supercomputer
export NEURAL_SUPERCOMPUTER_ENDPOINT="https://your-neural-endpoint.com"
export NEURAL_SUPERCOMPUTER_API_KEY="your-neural-api-key"

# Web Search
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CSE_ID="your-custom-search-engine-id"
export YAHOO_API_KEY="your-yahoo-api-key"

# Financial Data
export ALPHA_VANTAGE_API_KEY="your-alpha-vantage-key"
export POLYGON_API_KEY="your-polygon-key"
export QUANDL_API_KEY="your-quandl-key"

# Database Credentials
export POSTGRES_HOST="your-postgres-host"
export POSTGRES_USER="digitaluser"
export POSTGRES_PASSWORD="secure-password"
export REDIS_HOST="your-redis-host"
export REDIS_PASSWORD="secure-password"
export MILVUS_HOST="your-milvus-host"

# Security
export JWT_SECRET_KEY="your-jwt-secret"
export API_KEY_ENCRYPTION_KEY="your-32-byte-encryption-key"
export SSL_CERT_FILE="/path/to/cert.pem"
export SSL_KEY_FILE="/path/to/key.pem"
```

2. **Validate Configuration**
```bash
# Run configuration validator
python -m aiq.digital_human.deployment.validate_config
```

## Deployment Steps

### 1. Build Production Images
```bash
# Build with production Dockerfile
docker build -f docker/Dockerfile.digital_human_production \
  -t aiqtoolkit/digital-human-prod:latest \
  --build-arg PRODUCTION=true .

# Push to registry
docker push aiqtoolkit/digital-human-prod:latest
```

### 2. Deploy Infrastructure
```bash
# Run deployment script
./src/aiq/digital_human/deployment/deploy_production.sh

# Or deploy manually:

# Create namespace
kubectl create namespace digital-human-prod

# Deploy databases
helm install postgresql bitnami/postgresql -n digital-human-prod
helm install redis bitnami/redis -n digital-human-prod
helm install milvus milvus/milvus -n digital-human-prod
```

### 3. Initialize Databases
```bash
# Create database schema
kubectl exec -it postgresql-0 -n digital-human-prod -- psql -U postgres -c "
CREATE DATABASE digital_human;
CREATE USER digitaluser WITH PASSWORD '$POSTGRES_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE digital_human TO digitaluser;
"

# Run migrations
kubectl run migrations --rm -it \
  --image=aiqtoolkit/digital-human-prod:latest \
  --namespace=digital-human-prod \
  -- python -m aiq.digital_human.deployment.migrate
```

### 4. Deploy Application
```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/digital-human-deployment.yaml
kubectl apply -f kubernetes/digital-human-service.yaml
kubectl apply -f kubernetes/digital-human-ingress.yaml

# Configure autoscaling
kubectl autoscale deployment digital-human \
  --cpu-percent=70 \
  --min=3 \
  --max=10 \
  -n digital-human-prod
```

### 5. Configure SSL/TLS
```bash
# Create SSL certificate secret
kubectl create secret tls digital-human-tls \
  --cert=$SSL_CERT_FILE \
  --key=$SSL_KEY_FILE \
  -n digital-human-prod
```

### 6. Deploy Monitoring
```bash
# Deploy Prometheus and Grafana
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n digital-human-prod

# Import dashboards
kubectl apply -f docker/monitoring/grafana/dashboards/

# Deploy Elasticsearch and Kibana for logs
helm install elasticsearch elastic/elasticsearch -n digital-human-prod
helm install kibana elastic/kibana -n digital-human-prod
```

## Verification

### 1. Health Checks
```bash
# Check pod status
kubectl get pods -n digital-human-prod

# Check service endpoints
kubectl get svc -n digital-human-prod

# Test health endpoint
SERVICE_IP=$(kubectl get svc digital-human -n digital-human-prod -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -k https://$SERVICE_IP/health
```

### 2. Run Integration Tests
```bash
# Run full test suite
kubectl run integration-tests --rm -it \
  --image=aiqtoolkit/digital-human-prod:latest \
  --namespace=digital-human-prod \
  -- python -m pytest tests/integration/ -v
```

### 3. Performance Testing
```bash
# Run load tests
kubectl run load-test --rm -it \
  --image=loadimpact/k6 \
  --namespace=digital-human-prod \
  -- run /tests/load_test.js
```

## Production Checklist

- [ ] All environment variables configured
- [ ] SSL certificates installed
- [ ] Databases initialized and connected
- [ ] NVIDIA services authenticated
- [ ] Neural supercomputer connected
- [ ] Web search APIs validated
- [ ] Financial data sources working
- [ ] Monitoring dashboards accessible
- [ ] Health checks passing
- [ ] Integration tests passing
- [ ] Load tests meeting SLAs
- [ ] Backup procedures in place
- [ ] Disaster recovery tested

## Monitoring

### Grafana Dashboards
- System metrics: http://[GRAFANA_IP]:3000/d/digital-human
- GPU utilization: http://[GRAFANA_IP]:3000/d/gpu-metrics
- API performance: http://[GRAFANA_IP]:3000/d/api-metrics

### Kibana Logs
- Application logs: http://[KIBANA_IP]:5601
- Error tracking: http://[KIBANA_IP]:5601/app/apm

### Alerts
Configure alerts for:
- Response time > 5 seconds
- Error rate > 1%
- GPU utilization > 90%
- Memory usage > 80%

## Troubleshooting

### Common Issues

1. **NVIDIA Service Connection Failed**
   - Verify API key is valid
   - Check network connectivity
   - Ensure GPU drivers are installed

2. **Neural Supercomputer Timeout**
   - Check endpoint URL
   - Verify API key
   - Monitor network latency

3. **Database Connection Issues**
   - Verify credentials
   - Check network security groups
   - Ensure SSL certificates are valid

4. **GPU Out of Memory**
   - Reduce batch size
   - Scale horizontally
   - Check for memory leaks

## Maintenance

### Daily Tasks
- Monitor error logs
- Check system metrics
- Verify backup completion

### Weekly Tasks
- Review performance metrics
- Update dependencies
- Test disaster recovery

### Monthly Tasks
- Security audit
- Cost optimization review
- Capacity planning

## Support

For production support:
- Email: support@aiqtoolkit.com
- Slack: #digital-human-support
- On-call: +1-xxx-xxx-xxxx

## Security Considerations

1. **API Keys**
   - Rotate keys quarterly
   - Use separate keys for each environment
   - Never commit keys to source control

2. **Network Security**
   - Use VPN for admin access
   - Implement WAF rules
   - Enable DDoS protection

3. **Data Protection**
   - Encrypt data at rest
   - Use TLS for all connections
   - Implement PII masking

## Scaling Guidelines

### Horizontal Scaling
- Add nodes for increased throughput
- Use GPU node pools for compute
- Implement session affinity

### Vertical Scaling
- Upgrade to larger GPU instances
- Increase memory allocation
- Optimize batch processing

## Cost Optimization

1. **GPU Usage**
   - Use spot instances for batch processing
   - Implement request queuing
   - Share GPU resources when possible

2. **Data Storage**
   - Archive old sessions
   - Compress log files
   - Use tiered storage

3. **API Calls**
   - Implement caching
   - Batch requests
   - Use webhooks instead of polling