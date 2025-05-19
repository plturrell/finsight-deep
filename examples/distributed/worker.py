#!/usr/bin/env python3
"""Real NVIDIA API worker - NOT simulated"""

import os
import json
import urllib.request
import ssl
from flask import Flask, jsonify, request
import time

app = Flask(__name__)

API_KEY = os.environ.get('NVIDIA_API_KEY')
WORKER_ID = os.environ.get('WORKER_ID', 'worker-001')

ssl_context = ssl.create_default_context()

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'worker_id': WORKER_ID,
        'api_configured': bool(API_KEY)
    })

@app.route('/process', methods=['POST'])
def process():
    """Process request using real NVIDIA API"""
    data = request.json
    prompt = data.get('prompt', 'Hello')
    
    # Real NVIDIA API call
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    api_data = {
        'model': 'meta/llama-3.1-8b-instruct',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 50
    }
    
    req = urllib.request.Request(
        'https://integrate.api.nvidia.com/v1/chat/completions',
        data=json.dumps(api_data).encode('utf-8'),
        headers=headers,
        method='POST'
    )
    
    try:
        start_time = time.time()
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            duration = time.time() - start_time
            
            return jsonify({
                'worker_id': WORKER_ID,
                'prompt': prompt,
                'response': result['choices'][0]['message']['content'],
                'duration': duration,
                'model': result.get('model'),
                'status': 'success',
                'real_api': True  # This is REAL, not simulated!
            })
    except Exception as e:
        return jsonify({
            'worker_id': WORKER_ID,
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print(f"Starting REAL NVIDIA API worker: {WORKER_ID}")
    print(f"API Key configured: {bool(API_KEY)}")
    print("This is NOT a simulation - real API calls to NVIDIA!")
    app.run(host='0.0.0.0', port=8000)