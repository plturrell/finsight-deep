#!/bin/bash

# Local deployment using Docker Compose - production ready
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

# Create docker-compose.production.yml
create_compose_file() {
    log "Creating production Docker Compose configuration..."
    
    cat > docker-compose.production.yml <<EOF
version: '3.8'

services:
  # Redis for caching
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD:-changeme}
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Milvus vector database
  milvus:
    image: milvusdb/milvus:latest
    environment:
      ETCD_USE_EMBED: "true"
      ETCD_DATA_DIR: "/var/lib/milvus/etcd"
    ports:
      - "19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Digital Human API with NVIDIA integrations
  digital-human-api:
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
      - LLM_MODEL=meta/llama3-8b-instruct
      - REDIS_URL=redis://:${REDIS_PASSWORD:-changeme}@redis:6379/0
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
      - PYTHONUNBUFFERED=1
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - milvus
    volumes:
      - ./:/app
    working_dir: /app
    command: >
      bash -c "
      apt-get update && apt-get install -y python3 python3-pip &&
      pip3 install fastapi uvicorn httpx redis pymilvus websockets &&
      python3 -c '
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
import httpx
import json
import redis
import os

app = FastAPI(title=\"Digital Human API\")

# Redis client
redis_client = redis.from_url(os.getenv(\"REDIS_URL\"))

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv(\"NVIDIA_API_KEY\")
LLM_MODEL = os.getenv(\"LLM_MODEL\")

@app.get(\"/health\")
async def health():
    return {\"status\": \"healthy\", \"model\": LLM_MODEL}

@app.post(\"/chat\")
async def chat(message: dict):
    try:
        # Check cache
        cache_key = f\"chat:{message.get(\"content\", \"\")}\"
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Call NVIDIA API
        async with httpx.AsyncClient() as client:
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
            result = response.json()
            
            # Cache result
            redis_client.setex(cache_key, 300, json.dumps(result))
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket(\"/ws\")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            result = await chat({\"content\": data})
            await websocket.send_json(result)
    except Exception as e:
        await websocket.close(code=1000)

if __name__ == \"__main__\":
    uvicorn.run(app, host=\"0.0.0.0\", port=8000)
      '
      "
    runtime: nvidia

  # Web UI
  digital-human-ui:
    image: node:18-alpine
    environment:
      - API_URL=http://digital-human-api:8000
    ports:
      - "3000:3000"
    depends_on:
      - digital-human-api
    volumes:
      - ./ui:/app
    working_dir: /app
    command: >
      sh -c "
      npm install express http-proxy-middleware &&
      node -e '
      const express = require(\"express\");
      const { createProxyMiddleware } = require(\"http-proxy-middleware\");
      
      const app = express();
      
      app.use(\"/api\", createProxyMiddleware({
        target: process.env.API_URL,
        changeOrigin: true,
        pathRewrite: { \"^/api\": \"\" }
      }));
      
      app.get(\"/\", (req, res) => {
        res.send(\`
          <!DOCTYPE html>
          <html>
          <head>
            <title>Digital Human Financial Advisor</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 20px; }
              #chat { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; }
              input { width: 70%; padding: 10px; }
              button { padding: 10px 20px; }
            </style>
          </head>
          <body>
            <h1>Digital Human Financial Advisor</h1>
            <div id=\"chat\"></div>
            <input type=\"text\" id=\"message\" placeholder=\"Ask about financial markets...\">
            <button onclick=\"sendMessage()\">Send</button>
            <script>
              const ws = new WebSocket(\"ws://localhost:8000/ws\");
              const chat = document.getElementById(\"chat\");
              
              ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                const message = data.choices?.[0]?.message?.content || \"No response\";
                chat.innerHTML += \"<p><b>AI:</b> \" + message + \"</p>\";
              };
              
              function sendMessage() {
                const input = document.getElementById(\"message\");
                const message = input.value;
                if (message) {
                  chat.innerHTML += \"<p><b>You:</b> \" + message + \"</p>\";
                  ws.send(message);
                  input.value = \"\";
                }
              }
            </script>
          </body>
          </html>
        \`);
      });
      
      app.listen(3000, () => console.log(\"UI running on port 3000\"));
      '
      "

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards

volumes:
  redis_data:
  milvus_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
EOF

    log "Docker Compose file created!"
}

# Main deployment
deploy() {
    log "Starting local production deployment..."
    
    # Create compose file
    create_compose_file
    
    # Start services
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services
    log "Waiting for services to start..."
    sleep 30
    
    # Check status
    docker-compose -f docker-compose.production.yml ps
    
    info "Deployment complete!"
    echo ""
    echo "==================================="
    echo "Digital Human Financial Advisor"
    echo "==================================="
    echo "Web UI: http://localhost:3000"
    echo "API: http://localhost:8000"
    echo "API Health: http://localhost:8000/health"
    echo "WebSocket: ws://localhost:8000/ws"
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3001 (admin/admin)"
    echo "==================================="
    echo ""
    echo "To view logs: docker-compose -f docker-compose.production.yml logs -f"
    echo "To stop: docker-compose -f docker-compose.production.yml down"
}

case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    stop)
        docker-compose -f docker-compose.production.yml down
        ;;
    logs)
        docker-compose -f docker-compose.production.yml logs -f
        ;;
    status)
        docker-compose -f docker-compose.production.yml ps
        ;;
    *)
        echo "Usage: $0 {deploy|stop|logs|status}"
        exit 1
        ;;
esac