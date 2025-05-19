#!/bin/bash
apt-get update
apt-get install -y ubuntu-drivers-common docker.io
ubuntu-drivers autoinstall

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
echo "deb https://nvidia.github.io/libnvidia-container/stable/ubuntu22.04/amd64 /" > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Save startup script
cat > /root/start_digital_human.sh <<'SCRIPT'
#!/bin/bash
docker run -d --gpus all --restart=always -p 80:8000 \
  -e NVIDIA_API_KEY="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" \
  -e LLM_MODEL="meta/llama3-8b-instruct" \
  --name digital-human \
  python:3.10 bash -c "
pip install fastapi uvicorn httpx websockets torch &&
cat > app.py <<'PY'
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import httpx
import os
import json
import torch

app = FastAPI()

@app.get('/')
async def home():
    gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu else 'No GPU'
    return HTMLResponse(f'''
<html>
<head>
<title>Digital Human - GPU</title>
<style>
body {{ font-family: Arial; margin: 40px; background: #f5f5f5; }}
.container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
.gpu {{ background: #4CAF50; color: white; padding: 10px; border-radius: 5px; }}
#chat {{ height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin: 20px 0; }}
input {{ width: 70%; padding: 12px; }}
button {{ padding: 12px 24px; background: #2196F3; color: white; border: none; cursor: pointer; }}
</style>
</head>
<body>
<div class="container">
<h1>Digital Human Financial Advisor</h1>
<div class="gpu">GPU: {gpu_name}</div>
<div id="chat"></div>
<input id="msg" placeholder="Ask about financial markets...">
<button onclick="send()">Send</button>
</div>
<script>
const ws = new WebSocket("ws://" + location.host + "/ws");
ws.onmessage = (e) => {{
  const data = JSON.parse(e.data);
  document.getElementById("chat").innerHTML += "<p><b>AI:</b> " + data.message + "</p>";
}};
function send() {{
  const input = document.getElementById("msg");
  document.getElementById("chat").innerHTML += "<p><b>You:</b> " + input.value + "</p>";
  ws.send(input.value);
  input.value = "";
}}
</script>
</body>
</html>
    ''')

@app.get('/health')
async def health():
    return {{
        "status": "healthy",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": os.getenv("LLM_MODEL")
    }}

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    api_key = os.getenv('NVIDIA_API_KEY')
    
    while True:
        message = await websocket.receive_text()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                'https://api.nvidia.com/v1/chat/completions',
                headers={{'Authorization': f'Bearer {api_key}'}},
                json={{
                    'model': 'meta/llama3-8b-instruct',
                    'messages': [
                        {{'role': 'system', 'content': 'You are a GPU-powered financial advisor.'}},
                        {{'role': 'user', 'content': message}}
                    ],
                    'temperature': 0.7,
                    'max_tokens': 500
                }}
            )
        
        result = response.json()
        reply = result.get('choices', [{{}}])[0].get('message', {{}}).get('content', 'Error')
        await websocket.send_json({{"message": reply}})

uvicorn.run(app, host='0.0.0.0', port=8000)
PY
python3 app.py"
SCRIPT

chmod +x /root/start_digital_human.sh
/root/start_digital_human.sh
