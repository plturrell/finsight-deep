#!/bin/bash
# Docker + NVIDIA API test
# Uses environment variables for API keys

echo "Testing Docker + NVIDIA API Integration"
echo "====================================="

# Check if API key is set
if [ -z "$NIM_API_KEY" ]; then
    echo "❌ Error: NIM_API_KEY environment variable not set"
    echo "Please set it using: export NIM_API_KEY='your-key-here'"
    exit 1
fi

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker is running"
echo "✅ API key is set"

# Create a simple test container that uses NVIDIA API
cat > test_nvidia_docker.py << 'EOF'
import os
import json
import urllib.request
import ssl

# Get API key from environment
api_key = os.environ.get('NIM_API_KEY', '')
if not api_key:
    print("❌ Error: NIM_API_KEY not found in container environment")
    exit(1)

ssl_context = ssl.create_default_context()

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

data = {
    'model': 'meta/llama-3.1-8b-instruct',
    'messages': [{'role': 'user', 'content': 'What is Docker?'}],
    'max_tokens': 50
}

request = urllib.request.Request(
    'https://integrate.api.nvidia.com/v1/chat/completions',
    data=json.dumps(data).encode('utf-8'),
    headers=headers,
    method='POST'
)

try:
    with urllib.request.urlopen(request, context=ssl_context, timeout=30) as response:
        result = json.loads(response.read().decode('utf-8'))
        print(f"✅ NVIDIA API Response: {result['choices'][0]['message']['content'][:100]}...")
        print(f"✅ This is running inside Docker container!")
except Exception as e:
    print(f"❌ Error: {e}")
EOF

# Create minimal Dockerfile
cat > Dockerfile.test << 'EOF'
FROM python:3.9-slim
WORKDIR /app
COPY test_nvidia_docker.py .
CMD ["python", "test_nvidia_docker.py"]
EOF

# Build and run the container
echo
echo "Building Docker container..."
docker build -f Dockerfile.test -t nvidia-api-test .

echo
echo "Running container with NVIDIA API..."
docker run --rm -e NIM_API_KEY="$NIM_API_KEY" nvidia-api-test

echo
echo "====================================="
echo "Test complete. API key was passed securely via environment variable."