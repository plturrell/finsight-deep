#!/bin/bash

# Deploy Digital Human on GPU-enabled cloud instance
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check GPU instance quotas
check_gpu_quotas() {
    log "Checking GPU instance quotas..."
    
    # Check current quotas for GPU instances
    QUOTA=$(aws service-quotas get-service-quota \
        --service-code ec2 \
        --quota-code L-1216C47A \
        --region us-east-1 \
        --query 'Quota.Value' \
        --output text 2>/dev/null || echo "0")
    
    if [ "$QUOTA" == "0" ]; then
        warn "GPU instance quota is 0. You'll need to request a quota increase."
        echo ""
        echo "To request GPU instance quota:"
        echo "1. Go to AWS Service Quotas"
        echo "2. Search for 'Running On-Demand G and VT instances'"
        echo "3. Request increase to at least 4 vCPUs"
        echo ""
        error "Cannot proceed without GPU quota"
    fi
    
    info "GPU quota available: $QUOTA vCPUs"
}

# Alternative: Use cloud GPU services
suggest_alternatives() {
    echo ""
    echo "=== Alternative GPU Deployment Options ==="
    echo ""
    echo "Since AWS GPU quotas are limited, here are alternatives:"
    echo ""
    echo "1. Google Cloud Platform (GCP):"
    echo "   - T4 GPUs available in free tier"
    echo "   - Use: gcloud compute instances create"
    echo ""
    echo "2. Azure:"
    echo "   - NC-series VMs with NVIDIA GPUs"
    echo "   - Use: az vm create --size Standard_NC6"
    echo ""
    echo "3. Paperspace:"
    echo "   - Dedicated GPU cloud platform"
    echo "   - Gradient free tier available"
    echo ""
    echo "4. Google Colab Pro:"
    echo "   - GPU runtime for notebooks"
    echo "   - Can host FastAPI apps with ngrok"
    echo ""
    echo "5. NVIDIA Cloud:"
    echo "   - Optimized for NVIDIA SDKs"
    echo "   - Best for ACE platform"
    echo ""
}

# Create deployment for NVIDIA Cloud
create_nvidia_cloud_deployment() {
    log "Creating NVIDIA Cloud deployment configuration..."
    
    cat > nvidia_cloud_deploy.yaml <<EOF
# NVIDIA Cloud Deployment for Digital Human
apiVersion: v1
kind: Service
metadata:
  name: digital-human
spec:
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: digital-human
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
      - name: digital-human
        image: nvcr.io/nvidia/base/cuda:12.2.0-runtime-ubuntu22.04
        env:
        - name: NVIDIA_API_KEY
          value: "${NVIDIA_API_KEY}"
        - name: LLM_MODEL
          value: "meta/llama3-8b-instruct"
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1
        command: ["/bin/bash", "-c"]
        args:
          - |
            apt-get update
            apt-get install -y python3 python3-pip
            pip3 install fastapi uvicorn httpx nvidia-nim-client
            python3 -c "
import uvicorn
from fastapi import FastAPI
import httpx
import os

app = FastAPI()

NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

@app.get('/')
async def root():
    return {'message': 'Digital Human Financial Advisor', 'gpu': 'enabled'}

@app.post('/chat')
async def chat(message: dict):
    # Use NVIDIA NIM API
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'https://api.nvidia.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {NVIDIA_API_KEY}'},
            json={
                'model': 'meta/llama3-8b-instruct',
                'messages': [{'role': 'user', 'content': message['content']}]
            }
        )
    return response.json()

uvicorn.run(app, host='0.0.0.0', port=8000)
            "
EOF
    
    info "NVIDIA Cloud deployment configuration created"
}

# Create GCP deployment script
create_gcp_deployment() {
    log "Creating GCP deployment script..."
    
    cat > deploy_gcp.sh <<'EOF'
#!/bin/bash
# Deploy on Google Cloud Platform with T4 GPU

PROJECT_ID="your-project-id"
ZONE="us-central1-a"
INSTANCE_NAME="digital-human-gpu"

# Create GPU instance
gcloud compute instances create $INSTANCE_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --maintenance-policy=TERMINATE \
  --preemptible \
  --metadata startup-script='#!/bin/bash
apt-get update
apt-get install -y docker.io nvidia-container-toolkit
systemctl restart docker

docker run -d \
  --gpus all \
  -p 80:8000 \
  -e NVIDIA_API_KEY="'$NVIDIA_API_KEY'" \
  -e LLM_MODEL="meta/llama3-8b-instruct" \
  python:3.10 \
  bash -c "pip install fastapi uvicorn httpx && python app.py"
'

echo "GPU instance created. Get IP with:"
echo "gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE"
EOF
    
    chmod +x deploy_gcp.sh
    info "GCP deployment script created: deploy_gcp.sh"
}

# Main execution
main() {
    source production_env.sh
    
    log "Checking GPU deployment options..."
    
    # Check AWS GPU quotas
    check_gpu_quotas
    
    # If we get here, quotas are available
    log "Proceeding with GPU deployment..."
    
    # For now, suggest alternatives
    suggest_alternatives
    
    # Create deployment files for different platforms
    create_nvidia_cloud_deployment
    create_gcp_deployment
    
    echo ""
    echo "=== Deployment Files Created ==="
    echo "1. nvidia_cloud_deploy.yaml - For NVIDIA Cloud"
    echo "2. deploy_gcp.sh - For Google Cloud Platform"
    echo ""
    echo "To proceed:"
    echo "- For NVIDIA Cloud: Apply the YAML to your cluster"
    echo "- For GCP: Run ./deploy_gcp.sh after setting up gcloud"
    echo "- For AWS: Request GPU quota increase first"
}

main "$@"