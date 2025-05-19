#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Script to build and run AIQToolkit Docker containers

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODE="gpu"
BUILD=false
INTERACTIVE=true
DAEMON=false
COMPOSE=false

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --mode MODE      Container mode: 'gpu' or 'cpu' (default: gpu)"
    echo "  -b, --build          Build the container before running"
    echo "  -d, --daemon         Run in daemon mode (background)"
    echo "  -c, --compose        Use docker-compose"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -b -m gpu         Build and run GPU container"
    echo "  $0 -c                Run with docker-compose"
    echo "  $0 -m cpu -d         Run CPU container in background"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -d|--daemon)
            DAEMON=true
            INTERACTIVE=false
            shift
            ;;
        -c|--compose)
            COMPOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "gpu" && "$MODE" != "cpu" ]]; then
    echo -e "${RED}Error: Invalid mode '$MODE'. Must be 'gpu' or 'cpu'${NC}"
    exit 1
fi

# Change to project root
cd "$(dirname "$0")/.."

# Docker compose mode
if [[ "$COMPOSE" == true ]]; then
    echo -e "${GREEN}Starting AIQToolkit with docker-compose...${NC}"
    
    if [[ "$BUILD" == true ]]; then
        docker-compose -f docker/docker-compose.yml build
    fi
    
    if [[ "$DAEMON" == true ]]; then
        docker-compose -f docker/docker-compose.yml up -d
    else
        docker-compose -f docker/docker-compose.yml up
    fi
    
    exit 0
fi

# Single container mode
if [[ "$MODE" == "gpu" ]]; then
    DOCKERFILE="docker/Dockerfile.gpu"
    IMAGE_TAG="aiqtoolkit:gpu"
    RUNTIME_ARGS="--gpus all"
else
    DOCKERFILE="docker/Dockerfile.cpu"
    IMAGE_TAG="aiqtoolkit:cpu"
    RUNTIME_ARGS=""
fi

# Build if requested
if [[ "$BUILD" == true ]]; then
    echo -e "${GREEN}Building $MODE container...${NC}"
    docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" .
fi

# Check if image exists
if ! docker image inspect "$IMAGE_TAG" &> /dev/null; then
    echo -e "${YELLOW}Image $IMAGE_TAG not found. Building...${NC}"
    docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" .
fi

# Run container
echo -e "${GREEN}Running $MODE container...${NC}"

RUN_ARGS="$RUNTIME_ARGS"
RUN_ARGS="$RUN_ARGS -v $(pwd)/src:/app/src"
RUN_ARGS="$RUN_ARGS -v $(pwd)/examples:/app/examples"
RUN_ARGS="$RUN_ARGS -v $(pwd)/tests:/app/tests"
RUN_ARGS="$RUN_ARGS -p 8000:8000"
RUN_ARGS="$RUN_ARGS -p 8080:8080"

# Add display support for GUI apps
if [[ ! -z "$DISPLAY" ]]; then
    RUN_ARGS="$RUN_ARGS -e DISPLAY=$DISPLAY"
    RUN_ARGS="$RUN_ARGS -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
fi

# Interactive or daemon mode
if [[ "$INTERACTIVE" == true ]]; then
    RUN_ARGS="$RUN_ARGS -it --rm"
    CMD="/bin/bash"
else
    RUN_ARGS="$RUN_ARGS -d"
    CMD="tail -f /dev/null"
fi

# Run the container
docker run $RUN_ARGS "$IMAGE_TAG" $CMD

if [[ "$DAEMON" == true ]]; then
    echo -e "${GREEN}Container started in background${NC}"
    echo "To access the container: docker exec -it \$(docker ps -q -f ancestor=$IMAGE_TAG) bash"
    echo "To stop the container: docker stop \$(docker ps -q -f ancestor=$IMAGE_TAG)"
fi