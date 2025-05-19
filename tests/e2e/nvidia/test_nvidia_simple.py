#!/usr/bin/env python3
"""Simple NVIDIA API test using urllib"""

import os
import json
import urllib.request
import urllib.error
import ssl

# Create SSL context to handle certificates
ssl_context = ssl.create_default_context()

def test_nvidia_api():
    """Test NVIDIA API using only standard library"""
    # Read from environment file
    env_vars = {}
    try:
        with open('.env.production', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
    except FileNotFoundError:
        print("❌ .env.production not found")
        return False
    
    api_key = env_vars.get('NIM_API_KEY')
    base_url = env_vars.get('BASE_URL', 'https://integrate.api.nvidia.com/v1')
    
    print("NVIDIA API Connection Test")
    print("=" * 30)
    print(f"API Key: {api_key[:10]}...{api_key[-5:]}")
    print(f"Base URL: {base_url}")
    print()
    
    # Prepare request
    url = f"{base_url}/chat/completions"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'meta/llama-3.1-8b-instruct',
        'messages': [{'role': 'user', 'content': 'Say hello'}],
        'max_tokens': 10,
        'temperature': 0.7
    }
    
    # Create request
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers=headers,
        method='POST'
    )
    
    try:
        # Send request
        with urllib.request.urlopen(request, context=ssl_context, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            print("✅ NVIDIA API Connected Successfully!")
            print(f"Response: {result['choices'][0]['message']['content']}")
            return True
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error: {e.code}")
        print(f"Response: {e.read().decode('utf-8')}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_deployment_readiness():
    """Check if we're ready for deployment"""
    print("\nDeployment Readiness Check")
    print("=" * 30)
    
    # Check environment file
    if os.path.exists('.env.production'):
        print("✅ Environment file exists")
    else:
        print("❌ Environment file missing")
    
    # Check deployment scripts
    scripts = [
        'scripts/deploy_nvidia_distributed.sh',
        'scripts/setup_nvidia_deployment.sh',
        'scripts/test_nvidia_deployment.sh'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script} exists")
        else:
            print(f"❌ {script} missing")
    
    # Check Docker files
    docker_files = [
        'docker/Dockerfile.distributed_manager',
        'docker/Dockerfile.distributed_worker'
    ]
    
    for file in docker_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
    
    print("\nDeployment Commands:")
    print("1. Start Docker daemon")
    print("2. Run: ./scripts/deploy_nvidia_distributed.sh")
    print("3. Monitor: http://localhost:8080 (dashboard)")
    print("4. Test: python examples/distributed/run_distributed_inference.py")

if __name__ == "__main__":
    if test_nvidia_api():
        check_deployment_readiness()
    else:
        print("\n❌ Failed to connect to NVIDIA API")
        print("Please check your API keys and try again")