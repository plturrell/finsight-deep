#!/bin/bash

# Federated Learning Demo Script

set -e

echo "Starting Federated Learning Demo"

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    kill $SERVER_PID 2>/dev/null || true
    kill $CLIENT1_PID 2>/dev/null || true
    kill $CLIENT2_PID 2>/dev/null || true
    kill $CLIENT3_PID 2>/dev/null || true
}

trap cleanup EXIT

# Start the federated learning server
echo "Starting server..."
python server.py --config configs/federated_server.yml &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Start multiple clients
echo "Starting client 1 (workstation)..."
python client.py --config configs/edge_client.yml --client-id edge_1 &
CLIENT1_PID=$!

echo "Starting client 2 (workstation)..."
python client.py --config configs/edge_client.yml --client-id edge_2 &
CLIENT2_PID=$!

echo "Starting client 3 (mobile)..."
python client.py --config configs/edge_mobile.yml --client-id mobile_1 &
CLIENT3_PID=$!

echo "Federated learning demo running..."
echo "Press Ctrl+C to stop"

# Wait for user interrupt
wait