# NVIDIA Distributed Deployment Guide

This guide will help you deploy AIQToolkit's distributed components on NVIDIA infrastructure.

## Prerequisites

1. **NVIDIA GPU**: At least one NVIDIA GPU with CUDA support
2. **NVIDIA Drivers**: Latest NVIDIA drivers installed
3. **Docker**: Docker with NVIDIA runtime support
4. **API Keys**: NVIDIA NGC and NIM API keys

## Quick Start

### 1. Set Up Environment

Run the interactive setup wizard:

```bash
./scripts/setup_nvidia_deployment.sh
```

This will:
- Configure your NVIDIA API keys
- Set deployment parameters
- Create a `.env.production` file
- Generate a test script

### 2. Test NVIDIA API Connection

Verify your API keys are working:

```bash
python3 test_nvidia_api.py
```

### 3. Deploy Distributed System

Deploy the distributed components:

```bash
./scripts/deploy_nvidia_distributed.sh
```

This will:
- Build Docker images for manager and workers
- Deploy with Docker Compose
- Start monitoring services (Prometheus, Grafana)
- Run initial deployment tests

### 4. Run Distributed Inference

Test the system with example inference:

```bash
python3 examples/distributed/run_distributed_inference.py
```

## Architecture

The distributed system consists of:

- **Manager Node**: Coordinates tasks across workers
- **Worker Nodes**: Execute inference tasks on GPUs
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Security**: TLS encryption and JWT authentication

## API Endpoints

After deployment, these endpoints will be available:

- **Manager gRPC**: `localhost:50051`
- **Dashboard**: `http://localhost:8080`
- **Metrics**: `http://localhost:9090`
- **Grafana**: `http://localhost:3001` (admin/admin)

## Scaling

To scale the number of workers:

```bash
docker compose -f docker/docker-compose.nvidia.yml scale worker=10
```

## Monitoring

View system metrics and performance:

1. Open Grafana: `http://localhost:3001`
2. Login with admin/admin
3. Navigate to the AIQToolkit dashboard

## Troubleshooting

### No GPUs Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### API Connection Issues

```bash
# Test API directly
curl -X POST https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NIM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/llama-3.1-nemotron-70b-instruct", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}'
```

### View Logs

```bash
# Manager logs
docker logs aiq-distributed-manager

# Worker logs
docker logs aiq-distributed-worker-1
docker logs aiq-distributed-worker-2
```

## Getting API Keys

### NVIDIA NGC API Key
1. Visit https://ngc.nvidia.com/setup/api-key
2. Log in or create an account
3. Generate a new API key
4. Copy the key and save it securely

### NIM API Key
1. Visit https://build.nvidia.com/
2. Log in with your NVIDIA account
3. Navigate to API Keys section
4. Generate a new key for NIM services
5. Copy and save the key

## Security Notes

- Keep your `.env.production` file secure
- Never commit API keys to version control
- Add `.env.production` to `.gitignore`
- Use environment-specific keys for production

## Advanced Configuration

Edit `.env.production` to customize:

```bash
# GPU allocation
GPU_COUNT=8
NODE_COUNT=4

# Model selection
DEFAULT_MODEL=nvidia/mistral-nemo-minitron-8b-8k-instruct

# Security settings
TLS_ENABLED=true
JWT_SECRET=your-secure-secret

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

## Support

For issues or questions:
- Check the logs: `docker logs <container-name>`
- Run tests: `python -m pytest tests/aiq/distributed/`
- Open an issue on GitHub

## Next Steps

1. Explore the distributed inference examples
2. Integrate with your own workflows
3. Monitor performance with Grafana
4. Scale based on your workload