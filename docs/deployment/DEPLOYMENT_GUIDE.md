# AIQToolkit Production Deployment Guide

## Overview

This guide covers the complete deployment process for AIQToolkit, including security configuration, environment setup, and production deployment.

## Prerequisites

- Docker and Docker Compose installed
- PostgreSQL 15+ (or use Docker)
- Redis 7+ (or use Docker)
- Python 3.11+
- Node.js 18+ (for UI)
- NVIDIA GPU with CUDA support (for digital human features)

## Security Setup

### 1. Environment Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your secure values
nano .env
```

Required environment variables:
- `AIQ_API_KEY`: API authentication key
- `AIQ_JWT_SECRET`: JWT signing secret
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `OPENAI_API_KEY`: OpenAI API key (if using)
- `NIM_API_KEY`: NVIDIA NIM API key (if using)
- `CONSENSUS_PRIVATE_KEY`: Ethereum private key (if using consensus)

### 2. Generate Secure Keys

```bash
# Generate API key
openssl rand -hex 32

# Generate JWT secret
openssl rand -hex 64

# Generate consensus private key (if needed)
python -c "from eth_account import Account; print(Account.create().key.hex())"
```

## Deployment Steps

### 1. Local Development

```bash
# Install dependencies
uv sync --all-extras

# Start services
docker-compose -f docker/docker-compose.yml up -d postgres redis milvus

# Run migrations
python -m alembic upgrade head

# Start API server
python -m aiq.digital_human.ui.api_server_complete

# Start UI (in separate terminal)
cd external/aiqtoolkit-opensource-ui
npm install
npm run dev
```

### 2. Docker Deployment

```bash
# Build images
docker-compose -f docker/docker-compose.production.yml build

# Start all services
docker-compose -f docker/docker-compose.production.yml up -d

# Check logs
docker-compose -f docker/docker-compose.production.yml logs -f
```

### 3. Production Deployment

```bash
# Run deployment script
./scripts/deploy_production.sh

# Or manually:
# 1. Set environment variables
export $(grep -v '^#' .env | xargs)

# 2. Deploy infrastructure
docker-compose -f docker/docker-compose.production.yml up -d

# 3. Run migrations
docker-compose exec api python -m alembic upgrade head

# 4. Deploy smart contracts (if using consensus)
python scripts/deploy_consensus_contracts.py

# 5. Start application services
docker-compose -f docker/docker-compose.production.yml up -d api digital-human consensus ui
```

### 4. Cloud Deployment (AWS)

```bash
# Deploy to AWS ECS
aws ecs create-cluster --cluster-name aiqtoolkit-cluster

# Create task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service \
    --cluster aiqtoolkit-cluster \
    --service-name aiqtoolkit-api \
    --task-definition aiqtoolkit:1 \
    --desired-count 2 \
    --launch-type FARGATE
```

## Configuration

### API Server Configuration

```python
# config.py
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4")),
    "reload": os.getenv("API_RELOAD", "false").lower() == "true",
    "log_level": os.getenv("LOG_LEVEL", "info")
}
```

### Database Configuration

```python
# database.py
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL"),
    "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
    "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
    "echo": os.getenv("DB_ECHO", "false").lower() == "true"
}
```

### Consensus Configuration

```python
# consensus.py
CONSENSUS_CONFIG = {
    "contract_address": os.getenv("CONSENSUS_CONTRACT_ADDRESS"),
    "private_key": os.getenv("CONSENSUS_PRIVATE_KEY"),
    "rpc_url": os.getenv("CONSENSUS_RPC_URL", "http://localhost:8545"),
    "gas_limit": int(os.getenv("CONSENSUS_GAS_LIMIT", "1000000")),
    "gas_price": int(os.getenv("CONSENSUS_GAS_PRICE", "20"))
}
```

## Testing

### 1. Run Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/integration/test_end_to_end_flow.py

# Run with coverage
pytest --cov=aiq tests/
```

### 2. Run Integration Tests

```bash
# Start test environment
docker-compose -f docker/docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/ -v

# Cleanup
docker-compose -f docker/docker-compose.test.yml down
```

### 3. End-to-End Testing

```bash
# Run E2E test
python tests/integration/test_end_to_end_flow.py

# Or with pytest
pytest tests/integration/test_end_to_end_flow.py::test_end_to_end_integration
```

## Monitoring

### 1. Prometheus Setup

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'aiqtoolkit-api'
    static_configs:
      - targets: ['api:8000']
  
  - job_name: 'aiqtoolkit-consensus'
    static_configs:
      - targets: ['consensus:8090']
```

### 2. Grafana Dashboards

Import dashboards from `docker/monitoring/grafana/dashboards/`:
- API Performance Dashboard
- Consensus Metrics Dashboard
- Digital Human Dashboard

### 3. Logging

```python
# Configure logging
LOGGING_CONFIG = {
    "version": 1,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "json"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "logs/aiqtoolkit.log",
            "level": "DEBUG",
            "formatter": "json"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```

## Troubleshooting

### Common Issues

1. **API Not Starting**
   ```bash
   # Check logs
   docker logs aiqtoolkit_api_1
   
   # Verify environment variables
   docker exec aiqtoolkit_api_1 env
   ```

2. **Database Connection Failed**
   ```bash
   # Test connection
   docker exec aiqtoolkit_api_1 python -c "import psycopg2; psycopg2.connect('$DATABASE_URL')"
   
   # Check database logs
   docker logs aiqtoolkit_postgres_1
   ```

3. **Consensus Not Working**
   ```bash
   # Check contract deployment
   python scripts/verify_contract.py
   
   # Test Ethereum connection
   curl -X POST $CONSENSUS_RPC_URL -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Run with verbose output
python -m aiq.digital_human.ui.api_server_complete --log-level debug
```

## Backup and Recovery

### 1. Database Backup

```bash
# Backup PostgreSQL
docker exec aiqtoolkit_postgres_1 pg_dump -U aiqtoolkit aiqtoolkit > backup.sql

# Restore
docker exec -i aiqtoolkit_postgres_1 psql -U aiqtoolkit aiqtoolkit < backup.sql
```

### 2. Redis Backup

```bash
# Save Redis data
docker exec aiqtoolkit_redis_1 redis-cli BGSAVE

# Copy backup
docker cp aiqtoolkit_redis_1:/data/dump.rdb ./redis-backup.rdb
```

### 3. Configuration Backup

```bash
# Backup configuration
tar -czf config-backup.tar.gz .env docker/ scripts/

# Restore
tar -xzf config-backup.tar.gz
```

## Performance Optimization

### 1. API Optimization

```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)
```

### 2. Caching Strategy

```python
# Redis caching
from functools import lru_cache
import redis

redis_client = redis.from_url(REDIS_URL)

@lru_cache(maxsize=1000)
def get_cached_data(key: str):
    return redis_client.get(key)
```

### 3. GPU Optimization

```python
# CUDA optimization
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

## Security Best Practices

1. **Regular Updates**
   ```bash
   # Update dependencies
   uv sync --all-extras
   
   # Update Docker images
   docker-compose pull
   ```

2. **Access Control**
   - Use IAM roles for cloud deployments
   - Implement least privilege principle
   - Regular permission audits

3. **Monitoring**
   - Set up alerts for suspicious activity
   - Monitor API usage patterns
   - Track authentication failures

## Support

For issues and questions:
- GitHub Issues: https://github.com/NVIDIA/aiqtoolkit/issues
- Documentation: https://docs.aiqtoolkit.com
- Community: https://discord.gg/aiqtoolkit

## License

See [LICENSE.md](LICENSE.md) for details.