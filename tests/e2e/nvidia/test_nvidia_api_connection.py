#!/usr/bin/env python3
"""Test NVIDIA API connectivity without Docker"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.production')

def test_nvidia_nim_api():
    """Test NVIDIA NIM API connectivity"""
    api_key = os.getenv('NIM_API_KEY')
    base_url = os.getenv('BASE_URL', 'https://integrate.api.nvidia.com/v1')
    
    print("Testing NVIDIA NIM API...")
    print(f"API Key: {api_key[:10]}...{api_key[-5:]}")
    print(f"Base URL: {base_url}")
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Test with a simple completion
    data = {
        'model': 'meta/llama-3.1-8b-instruct',
        'messages': [{'role': 'user', 'content': 'Hello, NVIDIA!'}],
        'max_tokens': 50,
        'temperature': 0.7
    }
    
    try:
        response = requests.post(
            f'{base_url}/chat/completions', 
            headers=headers, 
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ NVIDIA API connection successful!")
            print("Response:", result['choices'][0]['message']['content'])
            return True
        else:
            print(f"✗ API Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

def test_together_api():
    """Test Together.ai API as backup"""
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        print("No Together.ai API key configured")
        return False
    
    print("\nTesting Together.ai API...")
    print(f"API Key: {api_key[:10]}...{api_key[-5:]}")
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        'messages': [{'role': 'user', 'content': 'Hello, Together.ai!'}],
        'max_tokens': 50,
        'temperature': 0.7
    }
    
    try:
        response = requests.post(
            'https://api.together.ai/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Together.ai connection successful!")
            print("Response:", result['choices'][0]['message']['content'])
            return True
        else:
            print(f"✗ API Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

def simulate_distributed_deployment():
    """Simulate distributed deployment with cloud APIs"""
    print("\n=== Simulating Distributed Deployment ===")
    
    # Test manager configuration
    print("\n1. Manager Node Configuration:")
    print(f"  Manager Host: {os.getenv('MANAGER_HOST', 'localhost')}")
    print(f"  Manager Port: {os.getenv('MANAGER_PORT', '50051')}")
    print(f"  GPU Count: {os.getenv('GPU_COUNT', '4')}")
    print(f"  Node Count: {os.getenv('NODE_COUNT', '2')}")
    
    # Test worker configuration
    print("\n2. Worker Node Configuration:")
    for i in range(int(os.getenv('NODE_COUNT', '2'))):
        print(f"  Worker {i+1}:")
        print(f"    - GPU Device: {i}")
        print(f"    - Model: {os.getenv('DEFAULT_MODEL')}")
        print(f"    - Status: Ready")
    
    # Test distributed inference
    print("\n3. Testing Distributed Inference:")
    
    prompts = [
        "Explain distributed computing",
        "What is GPU acceleration?",
        "Describe neural networks",
        "How does AI work?"
    ]
    
    api_key = os.getenv('NIM_API_KEY')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    for i, prompt in enumerate(prompts):
        worker_id = i % int(os.getenv('NODE_COUNT', '2'))
        print(f"\n  Task {i+1} -> Worker {worker_id + 1}")
        print(f"  Prompt: {prompt}")
        
        data = {
            'model': os.getenv('DEFAULT_MODEL'),
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 30,
            'temperature': 0.7
        }
        
        try:
            response = requests.post(
                f"{os.getenv('BASE_URL')}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print(f"  Response: {content[:50]}...")
                print(f"  Status: ✓ Success")
            else:
                print(f"  Status: ✗ Error {response.status_code}")
        except Exception as e:
            print(f"  Status: ✗ Failed - {e}")
    
    print("\n=== Deployment Simulation Complete ===")

def main():
    """Run all tests"""
    print("AIQToolkit NVIDIA API Connection Test")
    print("=" * 40)
    
    # Test primary NVIDIA API
    nvidia_ok = test_nvidia_nim_api()
    
    # Test backup API
    together_ok = test_together_api()
    
    # Simulate deployment
    if nvidia_ok or together_ok:
        simulate_distributed_deployment()
    else:
        print("\n✗ No working API connections found")
    
    print("\n" + "=" * 40)
    print("Test Complete")

if __name__ == "__main__":
    main()