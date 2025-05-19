#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start a distributed AIQToolkit cluster

set -e

# Configuration
MANAGER_HOST="${MANAGER_HOST:-localhost}"
MANAGER_PORT="${MANAGER_PORT:-50051}"
NUM_WORKERS="${NUM_WORKERS:-3}"
BASE_WORKER_PORT="${BASE_WORKER_PORT:-50052}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting AIQToolkit Distributed Cluster${NC}"
echo "Manager: ${MANAGER_HOST}:${MANAGER_PORT}"
echo "Workers: ${NUM_WORKERS}"
echo

# Start manager node
echo -e "${GREEN}Starting manager node...${NC}"
python -m aiq.distributed.multi_node_example --mode manager --manager-port ${MANAGER_PORT} &
MANAGER_PID=$!
echo "Manager PID: ${MANAGER_PID}"

# Wait for manager to start
sleep 5

# Start worker nodes
echo -e "${GREEN}Starting ${NUM_WORKERS} worker nodes...${NC}"
for i in $(seq 1 ${NUM_WORKERS}); do
    WORKER_PORT=$((BASE_WORKER_PORT + i - 1))
    echo "Starting worker ${i} on port ${WORKER_PORT}..."
    python -m aiq.distributed.multi_node_example \
        --mode worker \
        --manager-host ${MANAGER_HOST} \
        --manager-port ${MANAGER_PORT} \
        --worker-port ${WORKER_PORT} &
    WORKER_PID=$!
    echo "Worker ${i} PID: ${WORKER_PID}"
done

echo
echo -e "${GREEN}Cluster started successfully!${NC}"
echo
echo "To run a distributed workflow:"
echo "  python examples/distributed_processing/multi_node_example.py --mode workflow"
echo
echo "To stop the cluster:"
echo "  pkill -f 'aiq.distributed.multi_node_example'"
echo

# Wait for interrupt
trap "echo -e '\n${RED}Stopping cluster...${NC}'; pkill -f 'aiq.distributed.multi_node_example'; exit" INT TERM

echo "Press Ctrl+C to stop the cluster..."
wait