#!/bin/bash
# Quick test script for NVIDIA distributed deployment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}AIQToolkit NVIDIA Deployment Test${NC}"
echo

# Check for NVIDIA GPU
echo -e "${GREEN}1. Checking NVIDIA GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo -e "${GREEN}✓ NVIDIA drivers installed${NC}"
else
    echo -e "${RED}✗ nvidia-smi not found. Please install NVIDIA drivers.${NC}"
fi
echo

# Check Docker with NVIDIA runtime
echo -e "${GREEN}2. Checking Docker NVIDIA runtime...${NC}"
if docker info | grep -q nvidia; then
    echo -e "${GREEN}✓ Docker NVIDIA runtime available${NC}"
else
    echo -e "${RED}✗ Docker NVIDIA runtime not found. Install nvidia-docker2${NC}"
fi
echo

# Check environment variables
echo -e "${GREEN}3. Checking environment variables...${NC}"
if [ -n "$NVIDIA_NGC_API_KEY" ]; then
    echo -e "${GREEN}✓ NVIDIA_NGC_API_KEY set${NC}"
else
    echo -e "${RED}✗ NVIDIA_NGC_API_KEY not set${NC}"
    echo "Get your key from: https://ngc.nvidia.com/setup/api-key"
fi

if [ -n "$NIM_API_KEY" ]; then
    echo -e "${GREEN}✓ NIM_API_KEY set${NC}"
else
    echo -e "${RED}✗ NIM_API_KEY not set${NC}" 
    echo "Get your key from: https://build.nvidia.com/"
fi
echo

# Test distributed components
echo -e "${GREEN}4. Testing distributed components...${NC}"
cd "$(dirname "$0")/.."

# Run component tests
python -m pytest tests/aiq/distributed/test_distributed_deployment.py -v
echo

# Show deployment commands
echo -e "${BLUE}Ready to deploy!${NC}"
echo
echo "To deploy the distributed system:"
echo "  ./scripts/deploy_nvidia_distributed.sh"
echo
echo "To run distributed inference:"
echo "  python examples/distributed/run_distributed_inference.py"