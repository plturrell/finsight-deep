#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Deploy AIQToolkit distributed system to production

set -e

# Configuration
NAMESPACE="${NAMESPACE:-aiqtoolkit}"
HELM_RELEASE="${HELM_RELEASE:-aiqtoolkit-prod}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io/aiqtoolkit}"
VERSION="${VERSION:-latest}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Deploying AIQToolkit Distributed System${NC}"
echo "Namespace: ${NAMESPACE}"
echo "Release: ${HELM_RELEASE}"
echo "Registry: ${DOCKER_REGISTRY}"
echo "Version: ${VERSION}"
echo

# Create namespace
echo -e "${GREEN}Creating namespace...${NC}"
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Generate TLS certificates
echo -e "${GREEN}Generating TLS certificates...${NC}"
if ! kubectl get secret aiqtoolkit-tls-certs -n ${NAMESPACE} >/dev/null 2>&1; then
    ./scripts/generate_tls_certs.sh ${NAMESPACE}
fi

# Build and push Docker images
echo -e "${GREEN}Building Docker images...${NC}"
docker build -f docker/Dockerfile.distributed_manager -t ${DOCKER_REGISTRY}/distributed-manager:${VERSION} .
docker build -f docker/Dockerfile.distributed_worker -t ${DOCKER_REGISTRY}/distributed-worker:${VERSION} .

echo -e "${GREEN}Pushing Docker images...${NC}"
docker push ${DOCKER_REGISTRY}/distributed-manager:${VERSION}
docker push ${DOCKER_REGISTRY}/distributed-worker:${VERSION}

# Deploy with Helm
echo -e "${GREEN}Deploying with Helm...${NC}"
helm upgrade --install ${HELM_RELEASE} ./helm/aiqtoolkit \
    --namespace ${NAMESPACE} \
    --set global.imageRegistry=${DOCKER_REGISTRY} \
    --set manager.image.tag=${VERSION} \
    --set worker.image.tag=${VERSION} \
    --set worker.replicas=5 \
    --set security.tls.enabled=true \
    --set security.auth.enabled=true \
    --set monitoring.enabled=true \
    --set monitoring.prometheus.enabled=true \
    --set monitoring.grafana.enabled=true \
    --wait

# Wait for deployment
echo -e "${GREEN}Waiting for deployment to be ready...${NC}"
kubectl wait --namespace ${NAMESPACE} \
    --for=condition=available \
    --timeout=600s \
    deployment/aiqtoolkit-manager

# Get service endpoints
echo -e "${GREEN}Service endpoints:${NC}"
MANAGER_IP=$(kubectl get svc aiqtoolkit-manager -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Manager gRPC: ${MANAGER_IP}:50051"
echo "Metrics: ${MANAGER_IP}:9090"
echo "Dashboard: ${MANAGER_IP}:8080"

# Set up port forwarding for local access
echo -e "${GREEN}Setting up port forwarding...${NC}"
echo "Dashboard: http://localhost:8080"
echo "Metrics: http://localhost:9090"

# Create port-forward in background
kubectl port-forward -n ${NAMESPACE} svc/aiqtoolkit-manager 8080:8080 9090:9090 &
PF_PID=$!

echo
echo -e "${GREEN}Deployment complete!${NC}"
echo
echo "To access the dashboard: http://localhost:8080"
echo "To view metrics: http://localhost:9090/metrics"
echo "To stop port-forwarding: kill ${PF_PID}"
echo
echo "To scale workers:"
echo "  kubectl scale statefulset aiqtoolkit-worker -n ${NAMESPACE} --replicas=10"
echo
echo "To view logs:"
echo "  kubectl logs -n ${NAMESPACE} -l app=aiqtoolkit-manager"
echo "  kubectl logs -n ${NAMESPACE} -l app=aiqtoolkit-worker"