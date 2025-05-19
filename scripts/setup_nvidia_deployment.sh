#!/bin/bash
# Interactive setup script for NVIDIA deployment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     AIQToolkit NVIDIA Deployment Setup Wizard        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo

# Check if .env file exists
ENV_FILE=".env.production"
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Found existing $ENV_FILE${NC}"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Using existing configuration."
        exit 0
    fi
fi

echo -e "${GREEN}Step 1: NVIDIA API Credentials${NC}"
echo "You'll need API keys from NVIDIA to use their services."
echo

# NVIDIA NGC API Key
echo -e "${BLUE}1.1 NVIDIA NGC API Key${NC}"
echo "Get your key from: ${GREEN}https://ngc.nvidia.com/setup/api-key${NC}"
echo -n "Enter your NVIDIA NGC API Key: "
read -s NVIDIA_NGC_API_KEY
echo
if [ -z "$NVIDIA_NGC_API_KEY" ]; then
    echo -e "${RED}Error: NGC API Key is required${NC}"
    exit 1
fi

# NIM API Key
echo -e "${BLUE}1.2 NIM API Key${NC}"
echo "Get your key from: ${GREEN}https://build.nvidia.com/${NC}"
echo -n "Enter your NIM API Key: "
read -s NIM_API_KEY
echo
if [ -z "$NIM_API_KEY" ]; then
    echo -e "${RED}Error: NIM API Key is required${NC}"
    exit 1
fi

echo
echo -e "${GREEN}Step 2: Deployment Configuration${NC}"

# GPU count
echo -n "Number of GPUs to use (default: 4): "
read GPU_COUNT
GPU_COUNT=${GPU_COUNT:-4}

# Node count
echo -n "Number of nodes (default: 2): "
read NODE_COUNT
NODE_COUNT=${NODE_COUNT:-2}

# Model selection
echo
echo -e "${BLUE}Step 3: Model Selection${NC}"
echo "Available models:"
echo "1. nvidia/llama-3.1-nemotron-70b-instruct (default)"
echo "2. nvidia/mistral-nemo-minitron-8b-8k-instruct"
echo "3. nvidia/nv-llama2-70b-rlhf"
echo -n "Select model (1-3, default: 1): "
read MODEL_CHOICE

case $MODEL_CHOICE in
    2) DEFAULT_MODEL="nvidia/mistral-nemo-minitron-8b-8k-instruct";;
    3) DEFAULT_MODEL="nvidia/nv-llama2-70b-rlhf";;
    *) DEFAULT_MODEL="nvidia/llama-3.1-nemotron-70b-instruct";;
esac

# Create .env.production file
echo
echo -e "${GREEN}Creating $ENV_FILE...${NC}"
cat > $ENV_FILE << EOF
# NVIDIA Configuration
NVIDIA_NGC_API_KEY=$NVIDIA_NGC_API_KEY
NIM_API_KEY=$NIM_API_KEY
BASE_URL=https://integrate.api.nvidia.com/v1

# Distributed Configuration
GPU_COUNT=$GPU_COUNT
NODE_COUNT=$NODE_COUNT
MANAGER_HOST=localhost
MANAGER_PORT=50051

# Model Configuration
DEFAULT_MODEL=$DEFAULT_MODEL
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5

# Security
JWT_SECRET=$(openssl rand -hex 32)
TLS_ENABLED=true

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# Environment
AIQ_ENV=production
EOF

echo -e "${GREEN}✓ Configuration saved to $ENV_FILE${NC}"
echo

# Create example test script
echo -e "${GREEN}Creating example test script...${NC}"
cat > test_nvidia_api.py << 'EOF'
#!/usr/bin/env python3
"""Test NVIDIA API connectivity"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.production')

# Test NVIDIA API
def test_nvidia_api():
    api_key = os.getenv('NIM_API_KEY')
    base_url = os.getenv('BASE_URL', 'https://integrate.api.nvidia.com/v1')
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Test with a simple completion
    data = {
        'model': os.getenv('DEFAULT_MODEL'),
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'max_tokens': 10
    }
    
    try:
        response = requests.post(f'{base_url}/chat/completions', 
                                headers=headers, 
                                json=data,
                                timeout=10)
        if response.status_code == 200:
            print("✓ NVIDIA API connection successful!")
            print("Response:", response.json()['choices'][0]['message']['content'])
        else:
            print(f"✗ API Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"✗ Connection error: {e}")

if __name__ == "__main__":
    test_nvidia_api()
EOF

chmod +x test_nvidia_api.py

echo
echo -e "${BLUE}Setup Complete!${NC}"
echo
echo "Next steps:"
echo "1. Test NVIDIA API connection:"
echo "   python3 test_nvidia_api.py"
echo
echo "2. Deploy the distributed system:"
echo "   ./scripts/deploy_nvidia_distributed.sh"
echo
echo "3. Run distributed inference:"
echo "   python3 examples/distributed/run_distributed_inference.py"
echo
echo -e "${GREEN}Your API keys are stored in $ENV_FILE${NC}"
echo -e "${YELLOW}Keep this file secure and don't commit it to git!${NC}"