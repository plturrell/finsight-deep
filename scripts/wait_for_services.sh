#!/bin/bash

# Wait for all services to be ready
# Shows progress while waiting

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Waiting for services to start...${NC}"

# Function to check service
check_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "Checking $name"
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    echo -e " ${RED}✗${NC}"
    return 1
}

# Check each service
services_ready=true

# API Server
if ! check_service "http://localhost:8000/health" "API Server"; then
    services_ready=false
fi

# Frontend UI
if ! check_service "http://localhost:3000" "Frontend UI"; then
    services_ready=false
fi

# Prometheus (optional)
if check_service "http://localhost:9090/-/ready" "Prometheus"; then
    :
fi

# Grafana (optional)
if check_service "http://localhost:3001/api/health" "Grafana"; then
    :
fi

# PostgreSQL
echo -n "Checking PostgreSQL"
for i in {1..30}; do
    if docker-compose -f docker/docker-compose.hackathon.yml exec -T postgres pg_isready -U aiqtoolkit > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Redis
echo -n "Checking Redis"
for i in {1..30}; do
    if docker-compose -f docker/docker-compose.hackathon.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Overall status
echo
if [ "$services_ready" = true ]; then
    echo -e "${GREEN}All services are ready!${NC}"
    exit 0
else
    echo -e "${RED}Some services failed to start${NC}"
    exit 1
fi