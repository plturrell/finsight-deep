#!/bin/bash

# Deploy Digital Human on EC2 instances
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

# Source environment
source production_env.sh

# Create EC2 user data script
create_user_data() {
    cat > user_data.sh <<'EOF'
#!/bin/bash
# Install Docker
apt-get update
apt-get install -y docker.io docker-compose curl

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Create deployment directory
mkdir -p /app/digital-human
cd /app/digital-human

# Create Docker Compose file
cat > docker-compose.yml <<'COMPOSE'
version: '3.8'

services:
  digital-human:
    image: python:3.10
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
      - LLM_MODEL=meta/llama3-8b-instruct
    ports:
      - "80:8000"
    command: |
      bash -c "
      pip install fastapi uvicorn httpx &&
      python -c '
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import httpx
import os

app = FastAPI(title=\"Digital Human Financial Advisor\")

NVIDIA_API_KEY = os.getenv(\"NVIDIA_API_KEY\")
LLM_MODEL = os.getenv(\"LLM_MODEL\")

@app.get(\"/\")
async def home():
    return HTMLResponse(content=\"\"\"
    <!DOCTYPE html>
    <html>
    <head>
        <title>Digital Human Financial Advisor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            #chat { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin: 20px 0; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background: #e3f2fd; }
            .ai { background: #f5f5f5; }
            input { width: 70%; padding: 10px; }
            button { padding: 10px 20px; background: #1976d2; color: white; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>Digital Human Financial Advisor</h1>
        <div id=\"chat\"></div>
        <div>
            <input type=\"text\" id=\"message\" placeholder=\"Ask about stocks, portfolio, markets...\">
            <button onclick=\"sendMessage()\">Send</button>
        </div>
        <script>
            const chat = document.getElementById(\"chat\");
            const input = document.getElementById(\"message\");
            
            async function sendMessage() {
                const message = input.value;
                if (!message) return;
                
                chat.innerHTML += \"<div class=\\\"message user\\\">You: \" + message + \"</div>\";
                input.value = \"\";
                
                try {
                    const response = await fetch(\"/chat\", {
                        method: \"POST\",
                        headers: { \"Content-Type\": \"application/json\" },
                        body: JSON.stringify({ content: message })
                    });
                    const data = await response.json();
                    const aiMessage = data.choices?.[0]?.message?.content || \"I apologize, I could not generate a response.\";
                    chat.innerHTML += \"<div class=\\\"message ai\\\">Digital Human: \" + aiMessage + \"</div>\";
                    chat.scrollTop = chat.scrollHeight;
                } catch (error) {
                    chat.innerHTML += \"<div class=\\\"message ai\\\">Error: \" + error.message + \"</div>\";
                }
            }
            
            input.addEventListener(\"keypress\", (e) => {
                if (e.key === \"Enter\") sendMessage();
            });
        </script>
    </body>
    </html>
    \"\"\")

@app.get(\"/health\")
async def health():
    return {\"status\": \"healthy\", \"model\": LLM_MODEL}

@app.post(\"/chat\")
async def chat(message: dict):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                \"https://api.nvidia.com/v1/chat/completions\",
                headers={
                    \"Authorization\": f\"Bearer {NVIDIA_API_KEY}\",
                    \"Content-Type\": \"application/json\"
                },
                json={
                    \"model\": LLM_MODEL,
                    \"messages\": [{\"role\": \"user\", \"content\": message.get(\"content\", \"\")}],
                    \"temperature\": 0.7,
                    \"max_tokens\": 2048
                }
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == \"__main__\":
    uvicorn.run(app, host=\"0.0.0.0\", port=8000)
      '
      "
COMPOSE

# Set environment variables
export NVIDIA_API_KEY="${NVIDIA_API_KEY}"

# Run Docker Compose
docker-compose up -d

# Check status
docker-compose ps
EOF
}

# Launch EC2 instance
launch_instance() {
    log "Launching EC2 instance..."
    
    # Create security group
    SECURITY_GROUP=$(aws ec2 create-security-group \
        --group-name digital-human-sg \
        --description "Security group for Digital Human" \
        --query 'GroupId' \
        --output text 2>/dev/null || echo "sg-existing")
    
    # Add ingress rules
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0 2>/dev/null || true
    
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 2>/dev/null || true
    
    # Create user data
    create_user_data
    
    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id ami-0e2c8caa4b6378d8c \
        --instance-type g4dn.xlarge \
        --security-group-ids $SECURITY_GROUP \
        --user-data file://user_data.sh \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=digital-human-prod}]' \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    log "Instance launched: $INSTANCE_ID"
    
    # Wait for instance to be running
    log "Waiting for instance to start..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    info "Digital Human deployed!"
    echo ""
    echo "==================================="
    echo "Digital Human Financial Advisor"
    echo "==================================="
    echo "Instance ID: $INSTANCE_ID"
    echo "Public IP: $PUBLIC_IP"
    echo "URL: http://$PUBLIC_IP"
    echo "==================================="
    echo ""
    echo "Note: It may take 5-10 minutes for the service to be fully ready."
    echo "To check status: ssh ubuntu@$PUBLIC_IP"
    echo "To terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
}

# Main function
main() {
    case "${1:-deploy}" in
        deploy)
            launch_instance
            ;;
        terminate)
            if [ -z "$2" ]; then
                echo "Usage: $0 terminate <instance-id>"
                exit 1
            fi
            aws ec2 terminate-instances --instance-ids $2
            ;;
        *)
            echo "Usage: $0 {deploy|terminate}"
            exit 1
            ;;
    esac
}

main "$@"