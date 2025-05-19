#!/bin/bash

# Build and Deploy Digital Human with Docker to AWS EKS
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Source environment variables
source production_env.sh

# Configuration
DOCKER_REGISTRY="public.ecr.aws/z9g5m6x8"  # Public ECR registry
IMAGE_NAME="digital-human-llama3"
IMAGE_TAG="latest"
FULL_IMAGE="${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# Step 1: Build Docker image
build_docker_image() {
    log "Building Docker image..."
    
    cd /Users/apple/projects/AIQToolkit
    
    # Build the production image
    docker build \
        -f docker/Dockerfile.digital_human_production \
        -t ${IMAGE_NAME}:${IMAGE_TAG} \
        --build-arg PRODUCTION=true \
        .
    
    # Tag for ECR
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${FULL_IMAGE}
    
    info "Docker image built successfully: ${FULL_IMAGE}"
}

# Step 2: Push to ECR
push_to_ecr() {
    log "Pushing image to Amazon ECR..."
    
    # Get ECR login token (for public registry)
    aws ecr-public get-login-password --region us-east-1 | \
        docker login --username AWS --password-stdin public.ecr.aws
    
    # Create repository if it doesn't exist
    aws ecr-public create-repository \
        --repository-name ${IMAGE_NAME} \
        --region us-east-1 || true
    
    # Push the image
    docker push ${FULL_IMAGE}
    
    info "Image pushed successfully to ECR"
}

# Step 3: Create EKS cluster
create_eks_cluster() {
    log "Creating EKS cluster..."
    
    eksctl create cluster \
        --name ${CLUSTER_NAME} \
        --region ${AWS_REGION} \
        --with-oidc \
        --nodegroup-name gpu-nodes \
        --node-type g4dn.xlarge \
        --nodes 2 \
        --nodes-min 1 \
        --nodes-max 4 \
        --managed \
        --ssh-access
    
    # Install GPU device plugin
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml
    
    info "EKS cluster created with GPU support"
}

# Step 4: Deploy application
deploy_application() {
    log "Deploying Digital Human application..."
    
    # Create namespace
    kubectl create namespace ${NAMESPACE} || true
    
    # Create secrets
    kubectl create secret generic nvidia-api-key \
        --from-literal=api-key="${NVIDIA_API_KEY}" \
        --namespace ${NAMESPACE} || true
    
    # Apply Kubernetes manifests
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: digital-human-llama3
  namespace: ${NAMESPACE}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: digital-human-llama3
  template:
    metadata:
      labels:
        app: digital-human-llama3
    spec:
      containers:
      - name: digital-human
        image: ${FULL_IMAGE}
        ports:
        - containerPort: 8000
        env:
        - name: NVIDIA_API_KEY
          valueFrom:
            secretKeyRef:
              name: nvidia-api-key
              key: api-key
        - name: LLM_MODEL
          value: "meta/llama3-8b-instruct"
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "24Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 90
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: digital-human-service
  namespace: ${NAMESPACE}
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  - port: 443
    targetPort: 8000
    protocol: TCP  
    name: https
  selector:
    app: digital-human-llama3
EOF

    info "Application deployed successfully"
}

# Step 5: Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Install metrics server
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
    
    # Install Prometheus and Grafana using Helm
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=30d \
        --set grafana.adminPassword=admin
    
    info "Monitoring stack deployed"
}

# Step 6: Get application URL
get_application_url() {
    log "Getting application URL..."
    
    # Wait for load balancer
    echo "Waiting for load balancer to be ready..."
    kubectl wait --for=condition=ready \
        --timeout=300s \
        -n ${NAMESPACE} \
        service/digital-human-service
    
    # Get the URL
    APP_URL=$(kubectl get service digital-human-service \
        -n ${NAMESPACE} \
        -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    
    info "Application deployed successfully!"
    echo ""
    echo "==================================="
    echo "Digital Human Interface URL: http://${APP_URL}"
    echo "API Endpoint: http://${APP_URL}/api"
    echo "Health Check: http://${APP_URL}/health"
    echo "==================================="
}

# Main execution
main() {
    log "Starting Digital Human deployment..."
    
    case "$1" in
        build)
            build_docker_image
            ;;
        push)
            push_to_ecr
            ;;
        deploy)
            build_docker_image
            push_to_ecr
            create_eks_cluster
            deploy_application
            setup_monitoring
            get_application_url
            ;;
        quick)
            # Quick deployment - assumes cluster exists
            build_docker_image
            push_to_ecr
            deploy_application
            get_application_url
            ;;
        destroy)
            eksctl delete cluster --name ${CLUSTER_NAME} --region ${AWS_REGION}
            ;;
        *)
            echo "Usage: $0 {build|push|deploy|quick|destroy}"
            exit 1
            ;;
    esac
}

main "$@"