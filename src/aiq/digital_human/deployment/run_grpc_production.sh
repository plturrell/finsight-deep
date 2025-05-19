#!/bin/bash

# FinSight Deep gRPC Production Runner

echo "🚀 Starting FinSight Deep with NVIDIA gRPC Integration..."

# Check for required environment variables
if [ -z "$NVIDIA_API_KEY" ]; then
    echo "❌ Error: NVIDIA_API_KEY environment variable is not set"
    exit 1
fi

if [ -z "$TOGETHER_API_KEY" ]; then
    echo "❌ Error: TOGETHER_API_KEY environment variable is not set"
    exit 1
fi

# Compile proto files if needed
if [ ! -d "generated" ]; then
    echo "📦 Compiling proto files..."
    python audio2face_proto_compiler.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to compile proto files"
        exit 1
    fi
fi

# Install dependencies if needed
if ! python -c "import grpc" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the application
echo "🎮 Starting FinSight Deep gRPC server..."
echo "🌐 Access the interface at: http://localhost:8000"
echo "🤖 Using NVIDIA James photorealistic avatar"
echo "🔌 Connected to: grpc.nvcf.nvidia.com:443"

python finsight_grpc_production.py