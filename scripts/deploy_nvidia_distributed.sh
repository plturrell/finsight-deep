#!/bin/bash
# Deploy AIQToolkit distributed system to NVIDIA GPU infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
NVIDIA_NGC_API_KEY="${NVIDIA_NGC_API_KEY:-}"
NIM_API_KEY="${NIM_API_KEY:-}"
BASE_URL="${BASE_URL:-https://integrate.api.nvidia.com/v1}"
GPU_COUNT="${GPU_COUNT:-4}"
NODE_COUNT="${NODE_COUNT:-2}"
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-production}"

echo -e "${BLUE}AIQToolkit Distributed NVIDIA Deployment${NC}"
echo "GPU Count: ${GPU_COUNT}"
echo "Node Count: ${NODE_COUNT}"
echo "Deployment Mode: ${DEPLOYMENT_MODE}"
echo

# Check for required environment variables
if [ -z "$NVIDIA_NGC_API_KEY" ] || [ -z "$NIM_API_KEY" ]; then
    echo -e "${RED}Error: NVIDIA API keys not found${NC}"
    echo "Please set NVIDIA_NGC_API_KEY and NIM_API_KEY environment variables"
    echo "You can get these from:"
    echo "  - NGC API Key: https://ngc.nvidia.com/setup/api-key"
    echo "  - NIM API Key: https://build.nvidia.com/"
    exit 1
fi

# Create environment configuration
echo -e "${GREEN}Creating environment configuration...${NC}"
cat > .env.production << EOF
# NVIDIA Configuration
NVIDIA_NGC_API_KEY=${NVIDIA_NGC_API_KEY}
NIM_API_KEY=${NIM_API_KEY}
BASE_URL=${BASE_URL}

# Distributed Configuration
GPU_COUNT=${GPU_COUNT}
NODE_COUNT=${NODE_COUNT}
MANAGER_HOST=localhost
MANAGER_PORT=50051

# Model Configuration
DEFAULT_MODEL=nvidia/llama-3.1-nemotron-70b-instruct
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5

# Security
JWT_SECRET=$(openssl rand -hex 32)
TLS_ENABLED=true

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
EOF

echo -e "${GREEN}Building Docker images...${NC}"
docker build -f docker/Dockerfile.distributed_manager -t aiqtoolkit/distributed-manager:latest .
docker build -f docker/Dockerfile.distributed_worker -t aiqtoolkit/distributed-worker:latest .

# Create NVIDIA deployment configuration
echo -e "${GREEN}Creating NVIDIA deployment configuration...${NC}"
cat > docker/docker-compose.nvidia.yml << 'EOF'
version: '3.8'

services:
  # Distributed Manager
  manager:
    image: aiqtoolkit/distributed-manager:latest
    container_name: aiq-distributed-manager
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - CUDA_VISIBLE_DEVICES=0
    env_file:
      - ../.env.production
    ports:
      - "50051:50051"  # gRPC
      - "8080:8080"    # Dashboard
      - "9090:9090"    # Metrics
    volumes:
      - ../src:/app/src
      - manager_data:/data
      - ../logs:/app/logs
    networks:
      - aiq_distributed
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python -m aiq.distributed.manager_server

  # Distributed Workers
  worker-1:
    image: aiqtoolkit/distributed-worker:latest
    container_name: aiq-distributed-worker-1
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - CUDA_VISIBLE_DEVICES=0
      - WORKER_ID=worker-1
      - MANAGER_HOST=manager
      - MANAGER_PORT=50051
    env_file:
      - ../.env.production
    volumes:
      - ../src:/app/src
      - worker1_data:/data
    depends_on:
      - manager
    networks:
      - aiq_distributed
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python -m aiq.distributed.worker_server

  worker-2:
    image: aiqtoolkit/distributed-worker:latest
    container_name: aiq-distributed-worker-2
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=2
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - CUDA_VISIBLE_DEVICES=0
      - WORKER_ID=worker-2
      - MANAGER_HOST=manager
      - MANAGER_PORT=50051
    env_file:
      - ../.env.production
    volumes:
      - ../src:/app/src
      - worker2_data:/data
    depends_on:
      - manager
    networks:
      - aiq_distributed
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python -m aiq.distributed.worker_server

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - aiq_distributed

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - grafana_data:/var/lib/grafana
    networks:
      - aiq_distributed

volumes:
  manager_data:
  worker1_data:
  worker2_data:
  prometheus_data:
  grafana_data:

networks:
  aiq_distributed:
    driver: bridge
EOF

# Deploy with Docker Compose
echo -e "${GREEN}Starting NVIDIA distributed deployment...${NC}"
docker compose -f docker/docker-compose.nvidia.yml up -d

# Wait for services to be ready
echo -e "${GREEN}Waiting for services to be ready...${NC}"
sleep 10

# Check deployment status
echo -e "${GREEN}Checking deployment status...${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Run initial tests
echo -e "${GREEN}Running initial tests...${NC}"
python -m pytest tests/aiq/distributed/test_distributed_deployment.py -v

# Display endpoints
echo
echo -e "${GREEN}Deployment successful!${NC}"
echo
echo "Endpoints:"
echo "  Manager gRPC: localhost:50051"
echo "  Dashboard: http://localhost:8080"
echo "  Metrics: http://localhost:9090"
echo "  Grafana: http://localhost:3001 (admin/admin)"
echo
echo "To view logs:"
echo "  docker logs aiq-distributed-manager"
echo "  docker logs aiq-distributed-worker-1"
echo "  docker logs aiq-distributed-worker-2"
echo
echo "To scale workers:"
echo "  docker compose -f docker/docker-compose.nvidia.yml scale worker=5"
echo
echo "To run distributed inference:"
echo "  python examples/distributed/run_distributed_inference.py"
