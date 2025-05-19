import urllib.request
import json
import ssl

# NVIDIA API credentials - get from environment
import os

# Get API key from environment variable or use placeholder
API_KEY = os.environ.get('NIM_API_KEY', 'YOUR_NVIDIA_API_KEY')
if API_KEY == 'YOUR_NVIDIA_API_KEY':
    print("⚠️ Warning: Using placeholder API key. Please set NIM_API_KEY environment variable.")

URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Test the API
print("Testing NVIDIA API from Docker...")
print(f"Key: {API_KEY[:10]}...{API_KEY[-5:]}")

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

data = {
    'model': 'meta/llama-3.1-8b-instruct',
    'messages': [{'role': 'user', 'content': 'Hello from Docker!'}],
    'max_tokens': 30
}

request = urllib.request.Request(
    URL,
    data=json.dumps(data).encode('utf-8'),
    headers=headers
)

try:
    with urllib.request.urlopen(request, context=ssl.create_default_context()) as response:
        result = json.loads(response.read().decode('utf-8'))
        print("✅ API Response:", result['choices'][0]['message']['content'])
except Exception as e:
    print("❌ Error:", e)
