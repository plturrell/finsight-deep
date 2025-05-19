#!/bin/bash
# Simple real test: Docker + NVIDIA API
# NOTICE: This script has been moved to scripts/docker/ for better organization

echo "Simple Docker + NVIDIA API Test"
echo "=============================="
echo "⚠️ This script has been moved to scripts/docker/simple_nvidia_docker.sh"
echo "⚠️ Please use that version going forward - it now uses environment variables for API keys"
echo "⚠️ Set NIM_API_KEY environment variable before running:"
echo "export NIM_API_KEY='your-nvidia-api-key'"

# 1. Check if Docker is running
docker --version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker and try again:"
    echo "  macOS: open -a Docker"
    echo "  Linux: sudo systemctl start docker"
    exit 1
fi

echo "✅ Docker is running"

# 2. Create a simple Python script that calls NVIDIA API
cat > test_nvidia_docker.py << 'EOF'
import urllib.request
import json
import ssl

# NVIDIA API credentials
API_KEY = "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"
URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Test the API
print("Testing NVIDIA API from Docker...")
print(f"Key: {API_KEY[:10]}...{API_KEY[-5:]}")

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

data = {
    'model': 'meta/llama-3.1-8b-instruct',
    'messages': [{'role': 'user', 'content': 'Hello from Docker!'}],
    'max_tokens': 30
}

request = urllib.request.Request(
    URL,
    data=json.dumps(data).encode('utf-8'),
    headers=headers
)

try:
    with urllib.request.urlopen(request, context=ssl.create_default_context()) as response:
        result = json.loads(response.read().decode('utf-8'))
        print("✅ API Response:", result['choices'][0]['message']['content'])
except Exception as e:
    print("❌ Error:", e)
EOF

# 3. Create Dockerfile
cat > Dockerfile.simple << 'EOF'
FROM python:3.10-alpine
WORKDIR /app
COPY test_nvidia_docker.py .
CMD ["python", "test_nvidia_docker.py"]
EOF

# 4. Build and run
echo ""
echo "Building Docker image..."
docker build -f Dockerfile.simple -t nvidia-test .

echo ""
echo "Running Docker container with NVIDIA API..."
docker run --rm nvidia-test