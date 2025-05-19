#!/bin/bash

# AIQToolkit Hackathon Quick Start Script
# Sets up everything needed for NVIDIA Hackathon demo in <5 minutes

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 AIQToolkit Hackathon Quick Start${NC}"
echo -e "${YELLOW}Setting up NVIDIA GPU-powered consensus platform...${NC}\n"

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA GPU not detected. Please ensure CUDA is installed.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.11+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python installed${NC}"

# Create environment file
echo -e "\n${YELLOW}Setting up environment...${NC}"
if [ ! -f .env ]; then
    cp .env.template .env
    
    # Generate secure keys
    API_KEY=$(openssl rand -hex 32)
    JWT_SECRET=$(openssl rand -hex 64)
    
    # Update .env with generated values
    sed -i '' "s/AIQ_API_KEY=/AIQ_API_KEY=$API_KEY/" .env
    sed -i '' "s/AIQ_JWT_SECRET=/AIQ_JWT_SECRET=$JWT_SECRET/" .env
    
    echo -e "${GREEN}✓ Environment configured${NC}"
else
    echo -e "${GREEN}✓ Environment file exists${NC}"
fi

# Create necessary directories
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p logs data benchmarks/results docs/benchmarks

# Install Python dependencies
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
if command -v uv &> /dev/null; then
    uv sync --all-extras
else
    pip install -e ".[all]"
fi
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Build Docker images
echo -e "\n${YELLOW}Building Docker images...${NC}"
docker-compose -f docker/docker-compose.hackathon.yml build --parallel
echo -e "${GREEN}✓ Docker images built${NC}"

# Start services
echo -e "\n${YELLOW}Starting services...${NC}"
docker-compose -f docker/docker-compose.hackathon.yml up -d
echo -e "${GREEN}✓ Services started${NC}"

# Wait for services
echo -e "\n${YELLOW}Waiting for services to be ready...${NC}"
./scripts/wait_for_services.sh
echo -e "${GREEN}✓ All services ready${NC}"

# Run quick benchmarks
echo -e "\n${YELLOW}Running GPU benchmarks...${NC}"
python benchmarks/gpu_performance.py --quick
echo -e "${GREEN}✓ Benchmarks complete${NC}"

# Display access information
echo -e "\n${GREEN}🎉 Setup Complete!${NC}"
echo -e "\n${YELLOW}Access Points:${NC}"
echo -e "  Web UI: ${GREEN}http://localhost:3000${NC}"
echo -e "  API Docs: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  Consensus Dashboard: ${GREEN}http://localhost:3000/consensus${NC}"
echo -e "  Monitoring: ${GREEN}http://localhost:3001${NC} (admin/admin)"

echo -e "\n${YELLOW}Quick Demo:${NC}"
echo -e "  Run: ${GREEN}python examples/hackathon_demo.py${NC}"
echo -e "  Or visit: ${GREEN}http://localhost:3000/demo${NC}"

echo -e "\n${YELLOW}GPU Status:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo -e "\n${GREEN}Ready for NVIDIA Hackathon! 🚀${NC}"