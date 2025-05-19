#!/bin/bash
# Start real Docker test with NVIDIA API

echo "Starting Real Docker + NVIDIA API Test"
echo "===================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop:"
    echo "   - On macOS: open -a Docker"
    echo "   - Wait for Docker to fully start"
    echo "   - Then run this script again"
    exit 1
fi

echo "✅ Docker is running"

# Clean up any existing containers
echo "Cleaning up..."
docker-compose -f docker-compose-nvidia-test.yml down 2>/dev/null

# Build and start containers
echo "Building containers..."
docker-compose -f docker-compose-nvidia-test.yml build

echo "Starting containers..."
docker-compose -f docker-compose-nvidia-test.yml up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 5

# Check logs
echo "Worker logs:"
docker-compose -f docker-compose-nvidia-test.yml logs nvidia-api-worker

echo "Test logs:"
docker-compose -f docker-compose-nvidia-test.yml logs nvidia-api-tester

echo
echo "===================================="
echo "To test the API:"
echo "curl http://localhost:8000/health"
echo
echo "To see real-time logs:"
echo "docker-compose -f docker-compose-nvidia-test.yml logs -f"
echo
echo "To stop:"
echo "docker-compose -f docker-compose-nvidia-test.yml down"