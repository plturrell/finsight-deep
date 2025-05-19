#!/bin/bash
# Docker deployment with NVIDIA API
# This script runs Docker containers with NVIDIA API integration

set -e

echo "Docker + NVIDIA API Deployment"
echo "==============================="
echo ""

# Check if NIM_API_KEY is set
if [ -z "$NIM_API_KEY" ]; then
    echo "⚠️  WARNING: NIM_API_KEY environment variable not set"
    echo "Please set your NVIDIA API key using:"
    echo "export NIM_API_KEY='your-key-here'"
    echo ""
fi

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    echo "   On macOS: open -a Docker"
    exit 1
fi

echo "✅ Docker is running"

# Create a minimal Dockerfile that uses NVIDIA API
cat > Dockerfile.nvidia_api << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install requests

# Create the API test script
COPY nvidia_api_test.py .
COPY .env.production .

# Run the test
CMD ["python", "nvidia_api_test.py"]
EOF

# Create the test script
cat > nvidia_api_test.py << 'EOF'
import os
import json
import requests
import time

# Load environment
with open('.env.production', 'r') as f:
    for line in f:
        if '=' in line and not line.startswith('#'):
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

API_KEY = os.environ.get('NIM_API_KEY')
BASE_URL = "https://integrate.api.nvidia.com/v1"

def test_nvidia_api():
    """Test NVIDIA API in Docker"""
    print("Testing NVIDIA API from Docker container...")
    print(f"API Key: {API_KEY[:10]}...{API_KEY[-5:]}")
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    # Multiple test prompts
    prompts = [
        "What is Docker?",
        "Explain containerization",
        "How does GPU acceleration work?"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        data = {
            'model': 'meta/llama-3.1-8b-instruct',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 50,
            'temperature': 0.7
        }
        
        try:
            start = time.time()
            response = requests.post(
                f'{BASE_URL}/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            duration = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print(f"Response: {content[:100]}...")
                print(f"Time: {duration:.2f}s")
                print("✅ Success")
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print("\n=== Docker + NVIDIA API Test Complete ===")

if __name__ == "__main__":
    test_nvidia_api()
EOF

# Build Docker image
echo "Building Docker image..."
docker build -f Dockerfile.nvidia_api -t nvidia-api-test .

# Run container
echo "Running container with NVIDIA API..."
docker run --rm nvidia-api-test

echo ""
echo "To run again: docker run --rm nvidia-api-test"
echo "To run interactively: docker run -it --rm nvidia-api-test /bin/bash"