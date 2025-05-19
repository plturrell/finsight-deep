#!/bin/bash

# Quick deployment script for Digital Human - uses pre-built components
set -e

# Colors
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

# Source environment
source production_env.sh

# Quick validation
if [ -z "$NVIDIA_API_KEY" ]; then
    error "NVIDIA_API_KEY not set"
fi

if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    error "AWS credentials not set"
fi

# Deploy using pre-built NVIDIA NIM containers
deploy_with_nim() {
    log "Deploying Digital Human with NVIDIA NIM containers..."
    
    # Create deployment manifest
    cat > deployment.yaml <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: digital-human
---
apiVersion: v1
kind: Secret
metadata:
  name: nvidia-api-key
  namespace: digital-human
type: Opaque
stringData:
  api-key: "${NVIDIA_API_KEY}"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: digital-human-llama3
  namespace: digital-human
spec:
  replicas: 1
  selector:
    matchLabels:
      app: digital-human
  template:
    metadata:
      labels:
        app: digital-human
    spec:
      containers:
      - name: llama3-nim
        image: nvcr.io/nim/meta/llama3-8b-instruct:latest
        env:
        - name: NGC_API_KEY
          valueFrom:
            secretKeyRef:
              name: nvidia-api-key
              key: api-key
        - name: NIM_MODE
          value: "local"
        - name: NIM_MODEL_NAME
          value: "meta/llama3-8b-instruct"
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "24Gi"
            cpu: "8"
            nvidia.com/gpu: 1
      - name: digital-human-ui
        image: python:3.10-slim
        env:
        - name: NVIDIA_API_KEY
          valueFrom:
            secretKeyRef:
              name: nvidia-api-key
              key: api-key
        - name: LLM_ENDPOINT
          value: "http://localhost:8000/v1"
        - name: LLM_MODEL
          value: "meta/llama3-8b-instruct"
        ports:
        - containerPort: 8080
        command: ["/bin/bash", "-c"]
        args:
        - |
          pip install fastapi uvicorn websockets requests
          python -c "
import uvicorn
from fastapi import FastAPI, WebSocket
import httpx

app = FastAPI()

@app.get('/health')
async def health():
    return {'status': 'healthy'}

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        async with httpx.AsyncClient() as client:
            response = await client.post('http://localhost:8000/v1/chat/completions', 
                json={'model': 'meta/llama3-8b-instruct', 'messages': [{'role': 'user', 'content': data}]})
        await websocket.send_json(response.json())

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
          "
---
apiVersion: v1
kind: Service
metadata:
  name: digital-human-service
  namespace: digital-human
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: digital-human
EOF

    # Apply to Kubernetes
    kubectl apply -f deployment.yaml
    
    log "Deployment initiated!"
}

# Create minimal EKS cluster
create_minimal_cluster() {
    log "Creating minimal EKS cluster..."
    
    eksctl create cluster \
        --name digital-human-quick \
        --region us-east-1 \
        --nodegroup-name gpu-node \
        --node-type g4dn.xlarge \
        --nodes 1 \
        --nodes-min 1 \
        --nodes-max 2 \
        --managed \
        --spot
    
    # Install NVIDIA GPU operator
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/v23.9.0/deployments/gpu-operator/nvidia.yaml
    
    log "Cluster created!"
}

# Main deployment
main() {
    log "Starting quick deployment..."
    
    case "${1:-deploy}" in
        cluster)
            create_minimal_cluster
            ;;
        app)
            deploy_with_nim
            ;;
        deploy)
            create_minimal_cluster
            sleep 60  # Wait for cluster to be ready
            deploy_with_nim
            ;;
        status)
            kubectl get pods -n digital-human
            kubectl get service -n digital-human
            ;;
        url)
            URL=$(kubectl get service digital-human-service -n digital-human -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
            echo "Digital Human URL: http://${URL}"
            ;;
        destroy)
            eksctl delete cluster --name digital-human-quick --region us-east-1
            ;;
        *)
            echo "Usage: $0 {cluster|app|deploy|status|url|destroy}"
            exit 1
            ;;
    esac
}

main "$@"