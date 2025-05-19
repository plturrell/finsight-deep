#!/bin/bash

# Simple EC2 deployment without GPU
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

# Create user data script
create_user_data() {
    cat > user_data.sh <<EOF
#!/bin/bash
apt-get update
apt-get install -y docker.io

# Run the Digital Human container
docker run -d \
  --name digital-human \
  -p 80:8000 \
  -e NVIDIA_API_KEY="${NVIDIA_API_KEY}" \
  -e LLM_MODEL="meta/llama3-8b-instruct" \
  --restart=always \
  python:3.10 \
  bash -c "
pip install fastapi uvicorn httpx &&
cat > app.py <<'PYTHON'
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import httpx
import os

app = FastAPI(title='Digital Human Financial Advisor')

NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
LLM_MODEL = os.getenv('LLM_MODEL')

@app.get('/')
async def home():
    return HTMLResponse(content='''
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
        <h1>Digital Human Financial Advisor (Hackathon Demo)</h1>
        <div id="chat"></div>
        <div>
            <input type="text" id="message" placeholder="Ask about stocks, portfolio, markets...">
            <button onclick="sendMessage()">Send</button>
        </div>
        <script>
            const chat = document.getElementById("chat");
            const input = document.getElementById("message");
            
            async function sendMessage() {
                const message = input.value;
                if (!message) return;
                
                chat.innerHTML += '<div class="message user">You: ' + message + '</div>';
                input.value = "";
                
                try {
                    const response = await fetch("/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ content: message })
                    });
                    const data = await response.json();
                    const aiMessage = data.choices?.[0]?.message?.content || "I apologize, I could not generate a response.";
                    chat.innerHTML += '<div class="message ai">Financial Advisor: ' + aiMessage + '</div>';
                    chat.scrollTop = chat.scrollHeight;
                } catch (error) {
                    chat.innerHTML += '<div class="message ai">Error: ' + error.message + '</div>';
                }
            }
            
            input.addEventListener("keypress", (e) => {
                if (e.key === "Enter") sendMessage();
            });
        </script>
    </body>
    </html>
    ''')

@app.get('/health')
async def health():
    return {"status": "healthy", "model": LLM_MODEL, "deployment": "hackathon"}

@app.post('/chat')
async def chat(message: dict):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.nvidia.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a Digital Human Financial Advisor. Provide helpful financial advice and market analysis."},
                        {"role": "user", "content": message.get("content", "")}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
PYTHON

python app.py
"
EOF
}

# Launch EC2 instance
launch_instance() {
    log "Launching EC2 instance (CPU-only)..."
    
    # Use existing security group or create new one
    SECURITY_GROUP="sg-07e14a53a0b703277"  # Using the existing one created earlier
    
    # Create user data
    create_user_data
    
    # Launch a small instance (t2.micro is in free tier)
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id ami-0866a3c8686eaeeba \
        --instance-type t2.micro \
        --security-group-ids $SECURITY_GROUP \
        --user-data file://user_data.sh \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=digital-human-hackathon}]' \
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
    
    info "Digital Human deployed successfully!"
    echo ""
    echo "============================================"
    echo "Digital Human Financial Advisor - Hackathon"
    echo "============================================"
    echo "Instance ID: $INSTANCE_ID"
    echo "Public IP: $PUBLIC_IP"
    echo "URL: http://$PUBLIC_IP"
    echo "Health Check: http://$PUBLIC_IP/health"
    echo "============================================"
    echo ""
    echo "Note: Service will be ready in 3-5 minutes"
    echo "To check logs: ssh ubuntu@$PUBLIC_IP"
    echo "To terminate: ./deploy_simple_ec2.sh terminate $INSTANCE_ID"
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
            log "Terminating instance $2"
            ;;
        status)
            if [ -z "$2" ]; then
                echo "Usage: $0 status <instance-id>"
                exit 1
            fi
            aws ec2 describe-instances --instance-ids $2
            ;;
        *)
            echo "Usage: $0 {deploy|terminate|status}"
            exit 1
            ;;
    esac
}

main "$@"