#!/bin/bash

# Optimized AWS Deployment for Digital Human
# Uses provided credentials and NVIDIA Audio2Face-3D API

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Configuration
export AWS_REGION="us-east-1"
export CLUSTER_NAME="digital-human-hackathon"
export NAMESPACE="digital-human"

# Set credentials from environment
export AWS_ACCESS_KEY_ID="IKIDZ8Qn16xb1GKBobDcnjW7VedPxKWVHp8F"
export AWS_SECRET_ACCESS_KEY="xgBarxvfDAyYv40pTwCTeopf1HYheOxL"
export NVIDIA_API_KEY="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"

# Quick deployment for hackathon
quick_deploy() {
    log "Starting quick deployment for hackathon..."
    
    # 1. Setup AWS credentials
    aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
    aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
    aws configure set region $AWS_REGION
    
    # 2. Create EKS cluster with GPU support
    log "Creating EKS cluster..."
    eksctl create cluster \
        --name $CLUSTER_NAME \
        --region $AWS_REGION \
        --nodegroup-name gpu-nodes \
        --node-type g4dn.xlarge \
        --nodes 3 \
        --nodes-min 1 \
        --nodes-max 5 \
        --managed \
        --spot
    
    # 3. Install NVIDIA device plugin
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
    
    # 4. Create namespace
    kubectl create namespace $NAMESPACE
    
    # 5. Store secrets
    log "Storing secrets..."
    python3 src/aiq/digital_human/deployment/secure_secrets_manager.py nvidia
    
    # 6. Deploy application
    deploy_app
    
    # 7. Setup load balancer
    setup_load_balancer
    
    log "Deployment complete!"
}

deploy_app() {
    log "Deploying Digital Human application..."
    
    # Get ECR repository
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REPO="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/digital-human"
    
    # Create ECR repository
    aws ecr create-repository --repository-name digital-human --region $AWS_REGION || true
    
    # Login to ECR
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO
    
    # Build and push image
    docker build -f docker/Dockerfile.digital_human_production -t digital-human:latest .
    docker tag digital-human:latest $ECR_REPO:latest
    docker push $ECR_REPO:latest
    
    # Deploy to K8s
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: nvidia-secret
  namespace: $NAMESPACE
type: Opaque
stringData:
  api-key: "$NVIDIA_API_KEY"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: digital-human
  namespace: $NAMESPACE
spec:
  replicas: 3
  selector:
    matchLabels:
      app: digital-human
  template:
    metadata:
      labels:
        app: digital-human
    spec:
      nodeSelector:
        workload-type: gpu
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: digital-human
        image: $ECR_REPO:latest
        ports:
        - containerPort: 8000
        env:
        - name: NVIDIA_API_KEY
          valueFrom:
            secretKeyRef:
              name: nvidia-secret
              key: api-key
        - name: AWS_REGION
          value: $AWS_REGION
        - name: PRODUCTION
          value: "true"
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: digital-human
  namespace: $NAMESPACE
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
    app: digital-human
EOF
}

setup_load_balancer() {
    log "Setting up load balancer..."
    
    # Wait for load balancer
    kubectl wait --for=condition=ready pod -l app=digital-human -n $NAMESPACE --timeout=300s
    
    # Get load balancer URL
    LB_URL=""
    while [ -z "$LB_URL" ]; do
        sleep 10
        LB_URL=$(kubectl get svc digital-human -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    done
    
    log "Load balancer ready: $LB_URL"
    
    # Create Route53 record (optional)
    # aws route53 change-resource-record-sets ...
    
    # Output access info
    cat <<EOF

========================================
Digital Human Deployment Complete!
========================================

Access URL: http://$LB_URL

Test endpoints:
- Health: http://$LB_URL/health
- API Docs: http://$LB_URL/docs

NVIDIA Audio2Face-3D is configured and ready!

To view logs:
kubectl logs -f deployment/digital-human -n $NAMESPACE

To scale:
kubectl scale deployment digital-human --replicas=5 -n $NAMESPACE

EOF
}

# Validate deployment
validate() {
    log "Validating deployment..."
    
    # Check pods
    kubectl get pods -n $NAMESPACE
    
    # Check GPU allocation
    kubectl describe nodes | grep -A5 "Allocated resources"
    
    # Test health endpoint
    LB_URL=$(kubectl get svc digital-human -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    curl -f http://$LB_URL/health || error "Health check failed"
    
    log "Validation complete!"
}

# Cleanup function
cleanup() {
    log "Cleaning up resources..."
    eksctl delete cluster --name $CLUSTER_NAME --region $AWS_REGION
    aws ecr delete-repository --repository-name digital-human --region $AWS_REGION --force
}

# Main
case "${1:-deploy}" in
    deploy)
        quick_deploy
        validate
        ;;
    validate)
        validate
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 {deploy|validate|cleanup}"
        exit 1
        ;;
esac