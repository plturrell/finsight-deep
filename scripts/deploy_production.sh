#!/bin/bash

# AIQToolkit Production Deployment Script
# Handles environment setup and security configuration

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env file found. Creating from template...${NC}"
    cp .env.template .env
    echo -e "${GREEN}.env file created from template${NC}"
    echo -e "${RED}Please edit .env and add your configuration before continuing${NC}"
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Validate required environment variables
echo -e "${YELLOW}Validating environment configuration...${NC}"

required_vars=(
    "AIQ_API_KEY"
    "DATABASE_URL"
    "REDIS_URL"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=($var)
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo -e "${RED}Missing required environment variables:${NC}"
    printf '%s\n' "${missing_vars[@]}"
    exit 1
fi

echo -e "${GREEN}Environment validation passed${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Please install Docker first.${NC}"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose not found. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p logs
mkdir -p data
mkdir -p .secrets

# Generate JWT secret if not set
if [ -z "$AIQ_JWT_SECRET" ]; then
    echo -e "${YELLOW}Generating JWT secret...${NC}"
    JWT_SECRET=$(openssl rand -hex 32)
    echo "AIQ_JWT_SECRET=$JWT_SECRET" >> .env
    echo -e "${GREEN}JWT secret generated${NC}"
fi

# Build Docker images
echo -e "${YELLOW}Building Docker images...${NC}"
docker-compose -f docker/docker-compose.yml build

# Start infrastructure services
echo -e "${YELLOW}Starting infrastructure services...${NC}"
docker-compose -f docker/docker-compose.yml up -d postgres redis milvus

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Run database migrations
echo -e "${YELLOW}Running database migrations...${NC}"
docker-compose -f docker/docker-compose.yml run --rm api python -m alembic upgrade head

# Deploy consensus smart contracts if enabled
if [ "$ENABLE_CONSENSUS" = "true" ]; then
    echo -e "${YELLOW}Deploying consensus smart contracts...${NC}"
    
    if [ -z "$CONSENSUS_PRIVATE_KEY" ]; then
        echo -e "${RED}CONSENSUS_PRIVATE_KEY not set. Skipping contract deployment.${NC}"
    else
        python scripts/deploy_consensus_contracts.py
    fi
fi

# Start application services
echo -e "${YELLOW}Starting application services...${NC}"
docker-compose -f docker/docker-compose.yml up -d api digital-human consensus

# Start monitoring if enabled
if [ "$ENABLE_MONITORING" = "true" ]; then
    echo -e "${YELLOW}Starting monitoring services...${NC}"
    docker-compose -f docker/docker-compose.digital_human.yml up -d prometheus grafana
fi

# Health check
echo -e "${YELLOW}Performing health check...${NC}"
sleep 5

# Check API health
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ $response -eq 200 ]; then
    echo -e "${GREEN}API is healthy${NC}"
else
    echo -e "${RED}API health check failed${NC}"
    exit 1
fi

# Run integration tests
if [ "$RUN_TESTS" = "true" ]; then
    echo -e "${YELLOW}Running integration tests...${NC}"
    docker-compose -f docker/docker-compose.yml run --rm api pytest tests/integration/test_end_to_end_flow.py
fi

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${YELLOW}Service URLs:${NC}"
echo "  API: http://localhost:8000"
echo "  UI: http://localhost:3000"
echo "  Consensus UI: http://localhost:3000/consensus"
if [ "$ENABLE_MONITORING" = "true" ]; then
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana: http://localhost:3001"
fi

# Show logs
echo -e "${YELLOW}Showing recent logs:${NC}"
docker-compose -f docker/docker-compose.yml logs --tail=50